from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import torch
import numpy as np
from tqdm.auto import tqdm

from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import BaseOutput
from diffusers.image_processor import PipelineImageInput
from ..modules.extra_channels import make_extra_channels_tensor
from ..modules.utils import patch_groupnorm, patch_unet, swap_transformer_blocks
from .postprocessing import postprocess_outputs
from dataclasses import dataclass


@dataclass
class CubeDiffPipelineOutput(BaseOutput):
    faces: np.ndarray
    faces_cropped: np.ndarray
    equirectangular: np.ndarray


class CubeDiffPipeline(StableDiffusionPipeline):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load CubeDiffPipeline from pretrained model and automatically apply CubeDiff patches.
        
        The pretrained model should already have the correct input conv layer (7 channels),
        but we still need to patch the attention mechanisms and group norms.
        """

        # Load the base pipeline
        pipeline = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        if pipeline.unet.config.in_channels != 7:
            # Is a base SD model, patch input conv as well
            patch_unet(pipeline.unet, in_channels=7)
        else:
            # Apply attention patches (swap BasicTransformerBlock -> CubeDiffTransformerBlock)
            swap_transformer_blocks(pipeline.unet)
        
        # Apply groupnorm patches (GroupNorm -> CubeDiffGroupNorm)
        patch_groupnorm(pipeline.vae)
    
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

        # 2. Prepare IP-Adapter Image Embeddings
        cond_image_embeds = None
        uncond_image_embeds = None
        
        if ip_adapter_image is not None:
            # Check if IP-Adapter is loaded
            if not hasattr(self, 'image_encoder') or self.image_encoder is None:
                raise ValueError("IP-Adapter not loaded. Call load_ip_adapter() first.")
            
            # prepare_ip_adapter_image_embeds handles list input automatically
            # If input is 6 images, it generates embeddings for each
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                None,  # negative image (will use zeros)
                device,
                1,  # 【关键修改】每张图对应1份特征，而非6份
                do_classifier_free_guidance=(cfg_scale > 1.0),
            )
            
            # ===== DEBUG: 打印嵌入形状 =====
            print(f"\n[DEBUG] IP-Adapter image_embeds type: {type(image_embeds)}")
            if isinstance(image_embeds, list):
                print(f"[DEBUG] image_embeds list length: {len(image_embeds)}")
                for i, emb in enumerate(image_embeds):
                    print(f"[DEBUG] image_embeds[{i}] shape: {emb.shape}")
            else:
                print(f"[DEBUG] image_embeds shape: {image_embeds.shape}")
            # ================================
            
            # image_embeds is a list with one element per IP-Adapter
            # With batch_size=1 and 6 images + CFG, shape is [2, 6, num_tokens, dim]
            # Index 0 = uncond (negative), Index 1 = cond (positive)
            if isinstance(image_embeds, list):
                image_embeds = image_embeds[0]  # Get first IP-Adapter's embeddings
            
            print(f"[DEBUG] After extracting from list, image_embeds shape: {image_embeds.shape}")
            
            # Split embeddings: [0] for uncond, [1] for cond
            # Each has shape [6, dim] after indexing (raw CLIP output)
            uncond_raw = image_embeds[0]  # [6, 1024]
            cond_raw = image_embeds[1]    # [6, 1024]
            
            # Store raw embeddings for added_cond_kwargs (UNet requires this)
            uncond_raw_for_unet = uncond_raw
            cond_raw_for_unet = cond_raw
            
            # Project through encoder_hid_proj to get [6, num_tokens, cross_attention_dim]
            # This is required for the to_k_ip/to_v_ip layers in our processor
            if hasattr(self.unet, 'encoder_hid_proj') and self.unet.encoder_hid_proj is not None:
                uncond_projected = self.unet.encoder_hid_proj(uncond_raw)
                cond_projected = self.unet.encoder_hid_proj(cond_raw)
                
                # encoder_hid_proj may return a list - extract tensor if so
                if isinstance(uncond_projected, list):
                    uncond_projected = uncond_projected[0]
                if isinstance(cond_projected, list):
                    cond_projected = cond_projected[0]
                
                # Reshape: remove extra dimension if present [6,1,4,768] -> [6,4,768]
                if uncond_projected.ndim == 4:
                    uncond_projected = uncond_projected.squeeze(1)
                if cond_projected.ndim == 4:
                    cond_projected = cond_projected.squeeze(1)
                
                uncond_image_embeds = [uncond_projected]  # For cross_attention_kwargs (decoupled)
                cond_image_embeds = [cond_projected]      # For cross_attention_kwargs (decoupled)
                
                print(f"[DEBUG] After projection - uncond shape: {uncond_projected.shape}")
                print(f"[DEBUG] After projection - cond shape: {cond_projected.shape}\n")
            else:
                # Fallback: use raw embeddings (may not work correctly)
                uncond_image_embeds = [uncond_raw]
                cond_image_embeds = [cond_raw]
                uncond_raw_for_unet = uncond_raw
                cond_raw_for_unet = cond_raw
                print(f"[DEBUG] No encoder_hid_proj found, using raw embeddings\n")

        # 3. Initialize cross_attention_kwargs
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}

        # --- scheduler / latents -------------------------------------------
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        latents = torch.randn(
            (T, 4, self.unet.config.sample_size, self.unet.config.sample_size),
            generator=generator,
            device=device,
            dtype=self.unet.dtype,
        )
        latents *= self.scheduler.init_noise_sigma

        static_extra = make_extra_channels_tensor(1, self.unet.config.sample_size, self.unet.config.sample_size).to(device, dtype=self.unet.dtype)

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
            latents_input = torch.cat([latents_scaled, static_extra], dim=1)

            # 4. Conditional Forward (with cond IP-Adapter embeds)
            iter_kwargs = cross_attention_kwargs.copy()
            
            # Prepare added_cond_kwargs with RAW image_embeds (UNet requires this for encoder_hid_dim_type='ip_image_proj')
            # Also pass PROJECTED embeds via cross_attention_kwargs for our decoupled attention processor
            added_cond = {}
            if cond_image_embeds is not None:
                iter_kwargs["ip_adapter_image_embeds"] = cond_image_embeds  # Projected, for decoupled attention
                added_cond["image_embeds"] = cond_raw_for_unet  # Raw, for UNet's concatenation path
            
            noise_pred = self.unet(
                latents_input, 
                t, 
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=iter_kwargs if iter_kwargs else None,
                added_cond_kwargs=added_cond if added_cond else {},
            ).sample

            # 5. Unconditional Forward (with uncond IP-Adapter embeds)
            iter_uncond_kwargs = cross_attention_kwargs.copy()
            iter_uncond_kwargs["front_face_drop"] = True  # CubeDiff specific
            
            # Prepare added_cond_kwargs for unconditional pass
            added_uncond = {}
            if uncond_image_embeds is not None:
                iter_uncond_kwargs["ip_adapter_image_embeds"] = uncond_image_embeds  # Projected
                added_uncond["image_embeds"] = uncond_raw_for_unet  # Raw
            
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
