from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import torch
import numpy as np
from tqdm.auto import tqdm

from ..compat import ensure_torch_xpu_compat

ensure_torch_xpu_compat()

from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import BaseOutput
from diffusers.image_processor import PipelineImageInput
from ..modules.extra_channels import get_uv_tensors, make_extra_channels_tensor
from ..modules.utils import patch_groupnorm, patch_unet, swap_transformer_blocks, load_sliced_unet_weights
from .postprocessing import postprocess_outputs
from dataclasses import dataclass


@dataclass
class CubeDiffPipelineOutput(BaseOutput):
    faces: np.ndarray
    faces_cropped: np.ndarray
    equirectangular: np.ndarray


class CubeDiffPipeline(StableDiffusionPipeline):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, cubediff_weights_path=None, **kwargs):
        """
        Load CubeDiffPipeline from a base SD1.5 model, apply CubeDiff attention patches,
        and optionally load CubeDiff pretrained weights (with 7→4 channel slicing).
        
        Args:
            pretrained_model_name_or_path: Path to base SD1.5 model (must be 4-channel).
            cubediff_weights_path: Optional path to CubeDiff checkpoint (.bin/.safetensors).
                                   If provided, weights are loaded with 7→4 channel slicing.
        """

        # Load the base pipeline (must be a standard 4-channel SD1.5 model)
        pipeline = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # Apply CubeDiff attention patches (swap BasicTransformerBlock -> CubeDiffTransformerBlock)
        swap_transformer_blocks(pipeline.unet)
        
        # Apply groupnorm patches (GroupNorm -> CubeDiffGroupNorm)
        patch_groupnorm(pipeline.vae)
        
        # Load CubeDiff pretrained weights with 7→4 channel slicing if provided
        if cubediff_weights_path is not None:
            import os
            if cubediff_weights_path.endswith('.safetensors'):
                from safetensors.torch import load_file
                state_dict = load_file(cubediff_weights_path, device="cpu")
            else:
                state_dict = torch.load(cubediff_weights_path, map_location="cpu")
            load_sliced_unet_weights(pipeline.unet, state_dict)
            print(f"[CubeDiff] Loaded CubeDiff weights from {os.path.basename(cubediff_weights_path)}")
    
        return pipeline

    def load_ip_adapter(
        self,
        pretrained_model_name_or_path_or_dict,
        subfolder: Optional[str] = None,
        weight_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Load IP-Adapter weights and migrate them to CubeDiffIPAdapterAttnProcessor.
        
        This method:
        1. Calls parent's load_ip_adapter to load weights into standard processors
        2. Migrates the to_k_ip/to_v_ip weights to our custom processor for attn2
        3. Restores CubeDiffAttnProcessor for attn1 (preserving Cross-View Self-Attention)
        """
        from ..modules.attention import CubeDiffAttnProcessor, CubeDiffIPAdapterAttnProcessor
        
        # 1. Call parent method to load IP-Adapter weights
        super().load_ip_adapter(
            pretrained_model_name_or_path_or_dict,
            subfolder=subfolder,
            weight_name=weight_name,
            **kwargs,
        )
        
        # 2. Migrate IP-Adapter weights from diffusers processors to our custom processors
        self._migrate_ip_adapter_to_cubediff()
        
        print("[CubeDiff] IP-Adapter weights migrated to CubeDiffIPAdapterAttnProcessor")

    def _migrate_ip_adapter_to_cubediff(self):
        """
        Traverse UNet and:
        - Restore CubeDiffAttnProcessor for attn1 (Self-Attention)
        - Install CubeDiffIPAdapterAttnProcessor for attn2 (Cross-Attention) with migrated weights
        """
        from ..modules.attention import CubeDiffAttnProcessor, CubeDiffIPAdapterAttnProcessor
        
        # Debug counters
        attn1_count = 0
        attn2_migrated = 0
        attn2_skipped = 0
        
        def process_transformer_block(transformer_block):
            """Process a single transformer block."""
            nonlocal attn1_count, attn2_migrated, attn2_skipped
            
            # Restore attn1 to CubeDiffAttnProcessor (critical for Cross-View Self-Attention)
            if hasattr(transformer_block, 'attn1'):
                transformer_block.attn1.set_processor(CubeDiffAttnProcessor())
                attn1_count += 1
            
            # Handle attn2 - migrate IP-Adapter weights to our processor
            if hasattr(transformer_block, 'attn2') and transformer_block.attn2 is not None:
                attn2 = transformer_block.attn2
                old_processor = attn2.processor
                
                # Check if the old processor has IP-Adapter weights
                if hasattr(old_processor, 'to_k_ip') and old_processor.to_k_ip is not None:
                    # Get dimensions
                    hidden_size = attn2.inner_dim
                    cross_attention_dim = attn2.cross_attention_dim or hidden_size
                    
                    # Determine num_tokens and scale from old processor
                    num_tokens = getattr(old_processor, 'num_tokens', 4)
                    scale = getattr(old_processor, 'scale', 1.0)
                    # diffusers 使用列表支持多个 IP-Adapter，我们只用第一个
                    if isinstance(scale, (list, tuple)):
                        scale = scale[0] if len(scale) > 0 else 1.0
                    
                    # Create our custom processor
                    new_processor = CubeDiffIPAdapterAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        num_tokens=num_tokens,
                        scale=float(scale),  # 确保是 float
                    )
                    
                    # Migrate weights (to_k_ip and to_v_ip are nn.ModuleList in diffusers)
                    if isinstance(old_processor.to_k_ip, torch.nn.ModuleList):
                        # IP-Adapter uses ModuleList for multiple adapters
                        new_processor.to_k_ip = old_processor.to_k_ip[0]
                        new_processor.to_v_ip = old_processor.to_v_ip[0]
                    else:
                        new_processor.to_k_ip = old_processor.to_k_ip
                        new_processor.to_v_ip = old_processor.to_v_ip
                    
                    # Set the new processor
                    attn2.set_processor(new_processor)
                    attn2_migrated += 1
                else:
                    attn2_skipped += 1
        
        # Traverse all transformer blocks in UNet
        def traverse_module(module):
            for child in module.children():
                # Check if this is a CubeDiffTransformerBlock or BasicTransformerBlock
                if hasattr(child, 'attn1') and hasattr(child, 'attn2'):
                    process_transformer_block(child)
                else:
                    traverse_module(child)
        
        traverse_module(self.unet)
        
        # Print migration summary
        print(f"[CubeDiff] Migration summary:")
        print(f"  - attn1 (Self-Attention): {attn1_count} restored to CubeDiffAttnProcessor")
        print(f"  - attn2 (Cross-Attention): {attn2_migrated} migrated to CubeDiffIPAdapterAttnProcessor")
        print(f"  - attn2 (skipped, no IP-Adapter weights): {attn2_skipped}")

    def set_ip_adapter_scale(self, scale: float):
        """
        Set the IP-Adapter scale for all CubeDiffIPAdapterAttnProcessor instances.
        
        Args:
            scale: The weight for IP-Adapter influence (0.0 = no IP-Adapter, 1.0 = full strength)
        """
        from ..modules.attention import CubeDiffIPAdapterAttnProcessor
        
        def traverse_and_set_scale(module):
            for child in module.children():
                if hasattr(child, 'attn2') and child.attn2 is not None:
                    processor = child.attn2.processor
                    if isinstance(processor, CubeDiffIPAdapterAttnProcessor):
                        processor.scale = scale
                else:
                    traverse_and_set_scale(child)
        
        traverse_and_set_scale(self.unet)
        print(f"[CubeDiff] IP-Adapter scale set to {scale}")

    @torch.no_grad()
    def __call__(
        self,
        prompts: Union[str, List[str]],
        *,
        conditioning_image: torch.Tensor,  # (C,H,W)
        ip_adapter_image: PipelineImageInput = None,  # Accept image or list of 6 images
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        cfg_scale: float = 3.5,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Generate 6 cubemap faces with optional per-face IP-Adapter control.
        
        Args:
            prompts: Single prompt (applied to all faces) or list of 6 prompts
            conditioning_image: Reference image for front face conditioning
            ip_adapter_image: Optional reference image(s) for IP-Adapter.
                              Can be a single image (applied to all faces) or
                              a list of 6 images (one per face in order: Front, Back, Left, Right, Top, Bottom)
            num_inference_steps: Number of denoising steps
            generator: Random generator for reproducibility
            cfg_scale: Classifier-free guidance scale
            cross_attention_kwargs: Additional kwargs for cross attention
        """
        # Prefer the live UNet parameter device over `_execution_device`.
        # During validation we may swap in an unwrapped accelerator model that
        # already lives on CUDA even if `_execution_device` still reports CPU.
        try:
            device = next(self.unet.parameters()).device
        except StopIteration:
            device = self._execution_device
        T = 6  # faces

        # 1. Process prompts
        if isinstance(prompts, str):
            prompts = [prompts] * T
        
        if len(prompts) != T:
            raise ValueError(f"Expected 6 prompts, got {len(prompts)}")

        text_inputs = self.tokenizer(
            prompts,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            return_tensors="pt",
        )
        encoder_hidden_states = self.text_encoder(text_inputs.input_ids.to(device))[0]

        uncond_inputs = self.tokenizer(
            [""] * T,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_inputs.input_ids.to(device))[0]

        # 2. Prepare IP-Adapter Image Embeddings (per-face, independent)
        cond_image_embeds = None
        uncond_image_embeds = None
        cond_raw_for_unet = None
        uncond_raw_for_unet = None

        if ip_adapter_image is not None:
            if not hasattr(self, 'image_encoder') or self.image_encoder is None:
                raise ValueError("IP-Adapter not loaded. Call load_ip_adapter() first.")

            # Normalize input: accept a single image or a list of T images
            if not isinstance(ip_adapter_image, (list, tuple)):
                ip_adapter_image = [ip_adapter_image] * T
            if len(ip_adapter_image) != T:
                raise ValueError(f"Expected {T} ip_adapter_image(s) (one per face), got {len(ip_adapter_image)}")

            # Encode each face image independently through CLIP image encoder
            # This bypasses prepare_ip_adapter_image_embeds which mistakes a list of 6
            # images for "6 IP Adapters" rather than "6 face references for 1 IP Adapter".
            face_embeds = []
            for img in ip_adapter_image:
                pv = self.feature_extractor(images=img, return_tensors="pt").pixel_values
                pv = pv.to(device=device, dtype=self.image_encoder.dtype)
                emb = self.image_encoder(pv).image_embeds  # (1, 1024)
                face_embeds.append(emb)

            cond_raw = torch.cat(face_embeds, dim=0)                    # (T, 1024)
            uncond_raw = torch.zeros_like(cond_raw)                     # (T, 1024) zeros = no style

            cond_raw_for_unet = cond_raw
            uncond_raw_for_unet = uncond_raw

            # Project to cross-attention dim for our CubeDiffIPAdapterAttnProcessor
            if hasattr(self.unet, 'encoder_hid_proj') and self.unet.encoder_hid_proj is not None:
                cond_proj = self.unet.encoder_hid_proj(cond_raw)        # (T, num_tokens, 768) or (T,1,num_tokens,768)
                uncond_proj = self.unet.encoder_hid_proj(uncond_raw)

                # Handle extra dim if present: (T,1,num_tokens,768) → (T,num_tokens,768)
                if isinstance(cond_proj, list):   cond_proj   = cond_proj[0]
                if isinstance(uncond_proj, list): uncond_proj = uncond_proj[0]
                if cond_proj.ndim   == 4: cond_proj   = cond_proj.squeeze(1)
                if uncond_proj.ndim == 4: uncond_proj = uncond_proj.squeeze(1)

                cond_image_embeds   = [cond_proj]     # (T, num_tokens, 768)
                uncond_image_embeds = [uncond_proj]
            else:
                cond_image_embeds   = [cond_raw]
                uncond_image_embeds = [uncond_raw]


        # 3. Initialize cross_attention_kwargs
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}

        # --- scheduler / latents -------------------------------------------
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        sample_size = self.unet.config.sample_size
        latents = torch.randn(
            (T, 4, sample_size, sample_size),
            generator=generator,
            device=device,
            dtype=self.unet.dtype,
        )
        latents *= self.scheduler.init_noise_sigma

        # Generate UV coordinates for PE injection (computed once, reused every step)
        uv_coords = get_uv_tensors(1, sample_size, sample_size).to(device, dtype=self.unet.dtype)

        if conditioning_image.ndim == 3:
            conditioning_image = conditioning_image.unsqueeze(0)
        conditioning_image = conditioning_image.to(device, dtype=self.unet.dtype)
        ref_lat = self.vae.encode(conditioning_image).latent_dist.mean[0]
        ref_lat *= self.vae.config.scaling_factor

        # --- Denoising loop with progress bar ---
        progress_bar = tqdm(
            self.scheduler.timesteps,
            desc="Generating 360° panorama",
            total=len(self.scheduler.timesteps),
            unit="step"
        )
        
        for i, t in enumerate(progress_bar):
            latents[0] = ref_lat  # keep front face fixed
            latents_scaled = self.scheduler.scale_model_input(latents, t)
            if self.unet.conv_in.weight.shape[1] == 7:
                extra_channels = make_extra_channels_tensor(1, sample_size, sample_size).to(
                    device=device, dtype=self.unet.dtype
                )
                latents_input = torch.cat([latents_scaled, extra_channels], dim=1)
            else:
                latents_input = latents_scaled

            # 4. Conditional Forward (with cond IP-Adapter embeds)
            iter_kwargs = cross_attention_kwargs.copy()
            iter_kwargs["uv_coords"] = uv_coords  # Pass UV coords for PE injection
            if "style_cond" in iter_kwargs:
                if "style_scale" not in iter_kwargs:
                    raise ValueError("style_cond requires style_scale in cross_attention_kwargs")
                base_style_scale = iter_kwargs["style_scale"]
                override_time_scale = iter_kwargs.pop("style_time_scale_override", None)
                if override_time_scale is None:
                    progress = i / max(1, len(self.scheduler.timesteps) - 1)
                    time_scale = 0.2 if progress < 0.25 else 0.5
                else:
                    time_scale = float(override_time_scale)
                iter_kwargs["style_scale"] = base_style_scale.to(device=device, dtype=self.unet.dtype) * time_scale

            # Prepare added_cond_kwargs with RAW image_embeds (UNet requires this for encoder_hid_dim_type='ip_image_proj')
            # Also pass PROJECTED embeds via cross_attention_kwargs for our decoupled attention processor
            added_cond = {}
            if cond_image_embeds is not None:
                iter_kwargs["ip_adapter_image_embeds"] = cond_image_embeds  # Projected, for decoupled attention
                added_cond["image_embeds"] = cond_raw_for_unet  # Raw, for UNet's concatenation path
            elif getattr(self.unet.config, 'encoder_hid_dim_type', None) == 'ip_image_proj':
                # UNet config requires image_embeds even when no IP-Adapter image is given
                # Pass zero vector as placeholder (6 faces × 1024-dim CLIP embedding)
                added_cond["image_embeds"] = torch.zeros(T, 1024, device=device, dtype=self.unet.dtype)

            
            noise_pred = self.unet(
                latents_input, 
                t, 
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=iter_kwargs if iter_kwargs else None,
                added_cond_kwargs=added_cond if added_cond else {},
            ).sample

            # 5. Unconditional Forward: drop style guidance so CFG acts on both text and style.
            iter_uncond_kwargs = {
                k: v
                for k, v in cross_attention_kwargs.items()
                if k not in ("style_cond", "style_scale", "style_time_scale_override")
            }
            iter_uncond_kwargs["front_face_drop"] = True  # CubeDiff specific
            iter_uncond_kwargs["uv_coords"] = uv_coords  # Pass UV coords for PE injection
            
            # Prepare added_cond_kwargs for unconditional pass
            added_uncond = {}
            if uncond_image_embeds is not None:
                iter_uncond_kwargs["ip_adapter_image_embeds"] = uncond_image_embeds  # Projected
                added_uncond["image_embeds"] = uncond_raw_for_unet  # Raw
            elif getattr(self.unet.config, 'encoder_hid_dim_type', None) == 'ip_image_proj':
                added_uncond["image_embeds"] = torch.zeros(T, 1024, device=device, dtype=self.unet.dtype)

            
            noise_pred_uncond = self.unet(
                latents_input,
                t,
                encoder_hidden_states=uncond_embeddings,
                cross_attention_kwargs=iter_uncond_kwargs,
                added_cond_kwargs=added_uncond if added_uncond else {},
            ).sample

            combined = noise_pred_uncond + cfg_scale * (noise_pred - noise_pred_uncond)
            latents[1:] = self.scheduler.step(combined[1:], t, latents[1:]).prev_sample
            
            # Update progress bar with current step info
            progress_bar.set_postfix({
                'timestep': f'{t.item():.0f}',
                'step': f'{i+1}/{len(self.scheduler.timesteps)}'
            })

        # --- decode ---------------------------------------------------------
        imgs = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        equirec, uncropped, cropped = postprocess_outputs(imgs)

        return CubeDiffPipelineOutput(
            faces=uncropped,
            faces_cropped=cropped,
            equirectangular=equirec,
        )
