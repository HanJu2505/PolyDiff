import torch
from typing import Optional, Dict, Any
from einops import rearrange
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention import Attention
from diffusers.utils import deprecate
import torch.nn.functional as F

class CubeDiffAttnProcessor:
    r"""
    A custom processor for CubeDiff that uses PyTorch 2.0+ scaled dot-product attention
    without modifying or reshaping the attention mask.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CubeDiffAttnProcessor requires PyTorch 2.0+. Please upgrade PyTorch to use this.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        ip_adapter_image_embeds: Optional[list] = None,  # Accept but ignore for attn1
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # ip_adapter_image_embeds is accepted in signature to prevent diffusers warnings,
        # but ignored here - this processor is for Self-Attention (attn1).
        # IP-Adapter embeddings are handled by attn2's CubeDiffIPAdapterAttnProcessor.
        
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = hidden_states.shape[0]

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = key.shape[-1] // attn.heads

        # [B, H, L, D]
        # [B L D] -> [B, L, H, D] -> [B, H, L, D]
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Output proj
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class CubeDiffIPAdapterAttnProcessor:
    """
    Attention Processor for Cross-Attention (attn2) that supports IP-Adapter's decoupled attention.
    
    This processor computes:
        output = text_attention + scale * ip_adapter_attention
    
    Where ip_adapter_attention uses separate to_k_ip and to_v_ip projections.
    """

    def __init__(self, hidden_size: int, cross_attention_dim: int, num_tokens: int = 4, scale: float = 1.0):
        """
        Args:
            hidden_size: The hidden dimension of the attention layer
            cross_attention_dim: The dimension of the cross-attention input (e.g., text embeddings)
            num_tokens: Number of image tokens (4 for IP-Adapter, 16 for IP-Adapter Plus)
            scale: IP-Adapter strength (0.0-1.0)
        """
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CubeDiffIPAdapterAttnProcessor requires PyTorch 2.0+")
        
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.scale = scale
        
        # IP-Adapter specific projection layers (will be loaded from weights)
        self.to_k_ip = None  # nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip = None  # nn.Linear(cross_attention_dim, hidden_size, bias=False)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        ip_adapter_image_embeds: Optional[list] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # Handle deprecation warnings
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # Query from hidden states
        query = attn.to_q(hidden_states)

        # Use encoder_hidden_states for cross-attention
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # Handle tuple case: UNet's process_encoder_hidden_states may return
            # (encoder_hidden_states, encoder_hidden_states_image) when IP-Adapter is configured
            if isinstance(encoder_hidden_states, tuple):
                encoder_hidden_states = encoder_hidden_states[0]
            
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # Standard text Key/Value
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = query.shape[-1] // attn.heads

        # Reshape for multi-head attention: [B, L, D] -> [B, H, L, D/H]
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 1. Compute text attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # 2. Compute IP-Adapter attention (decoupled)
        if ip_adapter_image_embeds is not None and self.to_k_ip is not None and self.to_v_ip is not None:
            # ip_adapter_image_embeds is a list with shape [num_ip_adapters][batch, num_tokens, dim]
            ip_embeds = ip_adapter_image_embeds[0] if isinstance(ip_adapter_image_embeds, list) else ip_adapter_image_embeds
            
            # Project through IP-Adapter specific layers
            ip_key = self.to_k_ip(ip_embeds)
            ip_value = self.to_v_ip(ip_embeds)
            
            # Reshape for multi-head attention
            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            
            # Compute IP-Adapter attention
            ip_attention = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            
            # Add weighted IP-Adapter attention (decoupled fusion)
            hidden_states = hidden_states + self.scale * ip_attention

        # Reshape back: [B, H, L, D/H] -> [B, L, D]
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class CubeDiffTransformerBlock(BasicTransformerBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_faces = 6
        self.attn1.set_processor(CubeDiffAttnProcessor())

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention

        bt, hw, _ = hidden_states.shape

        T = self.num_faces
        B = bt // T

        # Normalization layer; by default should be layer norm on the hidden states 
        # which is the case for stable diffusion we are adapting, but we leave the if-else for flexibility and keeping the original code intact
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif self.norm_type == "ada_norm_single":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(bt, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

  
        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        # reshape to attend to all faces
        norm_hidden_states = rearrange(norm_hidden_states, "(b t) (hw) c -> b (t hw) c", b=B, t=T, hw=hw)

        front_face_drop = cross_attention_kwargs.pop("front_face_drop", False)

        if front_face_drop:
            # Right now it's a bit hacky, because we drop front face 
            # For the whole minibatch with probability 10%, as opposed to 
            # Dropping the front face for each sample independently. This is because
            # Using the mask would cause the backend to always use math mode instead of flashattention, which is much slower.
            with torch.no_grad():
                # [B, H, Q, K]
                # This should work.... Since it broadcasts
                self_attention_mask = torch.ones((1, 1, 1, T*hw), dtype=torch.bool, device=hidden_states.device)
                self_attention_mask[:, :, :, :hw] = False
        else:
            self_attention_mask = None

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=self_attention_mask,
            **cross_attention_kwargs,
        )

        # Delete the attention mask to save memory
        del self_attention_mask
 
        # reshape back to (B*T, C, H, W) post attention
        attn_output = rearrange(attn_output, "b (t hw) c -> (b t) (hw) c", b=B, t=T, hw=hw)

        if self.norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states

        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 1.2 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )

            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        # i2vgen doesn't have this norm 🤷‍♂️
        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == "ada_norm_zero":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        # -------- SMALL MODIFICATION AS WE DO NOT HAVE THE CHUNK FUNCTION ----------

        ff_output = self.ff(norm_hidden_states)

        # -------- END OF MODIFICATION ----------

        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states

        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states



