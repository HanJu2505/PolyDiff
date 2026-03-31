import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention import Attention
from diffusers.utils import deprecate
import torch.nn.functional as F


def get_sinusoidal_pe(
    uv_coords: torch.Tensor,
    seq_len: int,
    head_dim: int,
    num_faces: int = 6,
) -> torch.Tensor:
    """
    Generate sinusoidal positional encoding from (u, v) coordinates.

    Key insight: attn1 in CubeDiffTransformerBlock receives hidden_states that have been
    rearranged from (B*T, hw, C) → (B, T*hw, C), so seq_len = T * hw (e.g. 6*4096=24576).
    We split seq_len back into per-face hw, generate PE per face, then rearrange back.

    Args:
        uv_coords: (B*T, 2, H_base, W_base) — base-resolution UV coordinate map.
        seq_len:   T * hw — total sequence length seen by attn1 after face concatenation.
        head_dim:  Dimension per attention head.
        num_faces: Number of cubemap faces (default 6).

    Returns:
        (B, seq_len, head_dim) — PE tensor ready to broadcast over heads.
    """
    bt = uv_coords.shape[0]       # B*T
    T = num_faces
    B = bt // T
    device = uv_coords.device

    # Recover per-face spatial size: seq_len = T * hw
    assert seq_len % T == 0, f"seq_len={seq_len} is not divisible by num_faces={T}"
    hw = seq_len // T
    h = w = int(math.sqrt(hw))
    assert h * w == hw, f"per-face hw={hw} is not a perfect square"

    # Downsample UV map to current feature resolution: (B*T, 2, H_base, W_base) -> (B*T, 2, h, w)
    uv_float = uv_coords.float()
    uv_resized = F.interpolate(uv_float, size=(h, w), mode='bilinear', align_corners=False)

    # Flatten spatial dims: (B*T, 2, h, w) -> (B*T, hw, 2)
    uv_flat = uv_resized.permute(0, 2, 3, 1).reshape(bt, hw, 2)

    # NeRF-style sinusoidal encoding:
    # head_dim split into 4 parts (u_sin, u_cos, v_sin, v_cos) × num_bands
    num_bands = head_dim // 4
    remainder = head_dim - num_bands * 4

    if num_bands > 0:
        freq_bands = (2.0 ** torch.arange(num_bands, dtype=torch.float32, device=device)) * math.pi
        # angles: (B*T, hw, 2, num_bands)
        angles = uv_flat.unsqueeze(-1) * freq_bands.view(1, 1, 1, -1)
        sin_enc = torch.sin(angles)
        cos_enc = torch.cos(angles)
        # Interleave sin/cos: (B*T, hw, 2, num_bands, 2) -> (B*T, hw, 4*num_bands)
        pe_per_face = torch.stack([sin_enc, cos_enc], dim=-1).reshape(bt, hw, 4 * num_bands)
    else:
        pe_per_face = uv_flat.new_zeros(bt, hw, 0)

    if remainder > 0:
        pe_per_face = torch.cat([pe_per_face, uv_flat.new_zeros(bt, hw, remainder)], dim=-1)
    # pe_per_face: (B*T, hw, head_dim)

    # Rearrange to match CubeDiffTransformerBlock's rearrange order:
    # (B*T, hw, head_dim) -> (B, T, hw, head_dim) -> (B, T*hw, head_dim)
    pe = pe_per_face.reshape(B, T, hw, head_dim).reshape(B, T * hw, head_dim)

    return pe  # (B, seq_len, head_dim), float32



class CubeDiffAttnProcessor:
    r"""
    A custom processor for CubeDiff that uses PyTorch 2.0+ scaled dot-product attention.
    Injects sinusoidal positional encoding into Query and Key for cross-view self-attention.
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
        uv_coords: Optional[torch.Tensor] = None,  # Explicit param: Diffusers filters unknown kwargs!
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # uv_coords: (B*T, 2, H, W) UV coordinate map for sinusoidal PE injection into Q/K.
        # Must be in explicit signature — Diffusers inspects __call__ and silently drops
        # any cross_attention_kwargs keys that are not declared here.
        
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

        # === Inject sinusoidal PE into Q and K ===
        if uv_coords is not None:
            seq_len = query.shape[2]  # T * hw (faces are concatenated before attn1)
            pe = get_sinusoidal_pe(uv_coords, seq_len, head_dim)
            # pe: (B, seq_len, head_dim) — cast to query dtype/device
            pe = pe.to(dtype=query.dtype, device=query.device)
            # Unsqueeze for heads: (B, 1, seq_len, head_dim) broadcasts over (B, heads, seq_len, head_dim)
            pe = pe.unsqueeze(1)
            query = query + pe
            key = key + pe

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


class CubeDiffAppearanceAttnProcessor(nn.Module):
    """
    Cross-attention processor with a dedicated appearance branch:

        output = text_attention + scale * appearance_attention

    Appearance tokens are projected with their own K/V layers so the branch can
    specialize without hijacking the text branch.
    """

    def __init__(self, hidden_size: int, cross_attention_dim: int, scale: float = 1.0):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CubeDiffAppearanceAttnProcessor requires PyTorch 2.0+")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = float(scale)
        self.to_k_app = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_app = nn.Linear(cross_attention_dim, hidden_size, bias=False)

    def initialize_from_attention(self, attn: Attention) -> None:
        with torch.no_grad():
            self.to_k_app.weight.copy_(attn.to_k.weight)
            self.to_v_app.weight.copy_(attn.to_v.weight)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        appearance_tokens: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
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
        else:
            if isinstance(encoder_hidden_states, tuple):
                encoder_hidden_states = encoder_hidden_states[0]
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = query.shape[-1] // attn.heads
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

        if appearance_tokens is not None:
            app_key = self.to_k_app(appearance_tokens)
            app_value = self.to_v_app(appearance_tokens)
            app_key = app_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            app_value = app_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_k is not None:
                app_key = attn.norm_k(app_key)

            app_hidden = F.scaled_dot_product_attention(
                query, app_key, app_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states + self.scale * app_hidden

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

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

        # Extract uv_coords safely — do NOT pop from original dict (shared across blocks)
        uv_coords = cross_attention_kwargs.get("uv_coords", None)

        # reshape to attend to all faces
        norm_hidden_states = norm_hidden_states.reshape(B, T, hw, -1).reshape(B, T * hw, -1)

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

        # Build kwargs for attn1 — pass uv_coords for PE injection
        attn1_kwargs = {
            k: v
            for k, v in cross_attention_kwargs.items()
            if k not in ("uv_coords", "front_face_drop", "appearance_tokens")
        }
        attn1_kwargs["uv_coords"] = uv_coords

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=self_attention_mask,
            **attn1_kwargs,
        )

        # Delete the attention mask to save memory
        del self_attention_mask
 
        # reshape back to (B*T, C, H, W) post attention
        attn_output = attn_output.reshape(B, T, hw, -1).reshape(B * T, hw, -1)

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

            # Build clean kwargs for attn2 — exclude uv_coords (not needed for cross-attention)
            attn2_kwargs = {
                k: v
                for k, v in cross_attention_kwargs.items()
                if k not in ("uv_coords", "front_face_drop")
            }

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **attn2_kwargs,
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


def _resolve_layer_scale(module_name: str, layer_scales: Dict[str, float]) -> float:
    if module_name.startswith("down_blocks.0") or module_name.startswith("down_blocks.1"):
        return layer_scales["shallow"]
    if module_name.startswith("up_blocks.2") or module_name.startswith("up_blocks.3"):
        return layer_scales["shallow"]
    if module_name.startswith("down_blocks.2") or module_name.startswith("up_blocks.1"):
        return layer_scales["mid"]
    if module_name.startswith("down_blocks.3") or module_name.startswith("up_blocks.0"):
        return layer_scales["deep"]
    if module_name.startswith("mid_block"):
        return layer_scales["deep"]
    return layer_scales["mid"]


def install_appearance_processors(
    unet: nn.Module,
    *,
    base_scale: float = 0.3,
    layer_scales: Optional[Dict[str, float]] = None,
) -> None:
    if layer_scales is None:
        layer_scales = {"shallow": 1.0, "mid": 0.7, "deep": 0.4}

    for module_name, module in unet.named_modules():
        if not hasattr(module, "attn2") or module.attn2 is None:
            continue

        attn2 = module.attn2
        hidden_size = attn2.inner_dim
        cross_attention_dim = attn2.cross_attention_dim or hidden_size
        processor = CubeDiffAppearanceAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            scale=base_scale * _resolve_layer_scale(module_name, layer_scales),
        )
        processor.initialize_from_attention(attn2)
        attn2.set_processor(processor)
