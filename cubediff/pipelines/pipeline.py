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
            # Each has shape [6, num_tokens, dim] after indexing
            uncond_image_embeds = [image_embeds[0]]  # List format for cross_attention_kwargs
            cond_image_embeds = [image_embeds[1]]
            
            print(f"[DEBUG] uncond_image_embeds[0] shape: {uncond_image_embeds[0].shape}")
            print(f"[DEBUG] cond_image_embeds[0] shape: {cond_image_embeds[0].shape}\n")

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
            
            # Prepare added_cond_kwargs with image_embeds for IP-Adapter
            added_cond = {}
            if cond_image_embeds is not None:
                iter_kwargs["ip_adapter_image_embeds"] = cond_image_embeds
                # Extract tensor from list for added_cond_kwargs
                added_cond["image_embeds"] = cond_image_embeds[0]
            
            noise_pred = self.unet(
                latents_input, 
                t, 
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=iter_kwargs if iter_kwargs else None,
                added_cond_kwargs=added_cond,
            ).sample

            # 5. Unconditional Forward (with uncond IP-Adapter embeds)
            iter_uncond_kwargs = cross_attention_kwargs.copy()
            iter_uncond_kwargs["front_face_drop"] = True  # CubeDiff specific
            
            # Prepare added_cond_kwargs for unconditional pass
            added_uncond = {}
            if uncond_image_embeds is not None:
                iter_uncond_kwargs["ip_adapter_image_embeds"] = uncond_image_embeds
                # Extract tensor from list for added_cond_kwargs
                added_uncond["image_embeds"] = uncond_image_embeds[0]
            
            noise_pred_uncond = self.unet(
                latents_input,
                t,
                encoder_hidden_states=uncond_embeddings,
                cross_attention_kwargs=iter_uncond_kwargs,
                added_cond_kwargs=added_uncond,
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
