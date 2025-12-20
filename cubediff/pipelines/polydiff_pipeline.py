"""
PolyDiff-18 Pipeline

An 18-view extension of CubeDiff for seamless 360° panorama generation.
"""

from __future__ import annotations

from typing import List, Optional, Union, Dict
import torch
import numpy as np
from tqdm.auto import tqdm

from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import BaseOutput
from ..modules.extra_channels import make_extra_channels_tensor_18
from ..modules.utils import patch_groupnorm, patch_unet, swap_transformer_blocks
from ..modules.geometry import VIEW_CONFIG_18, VIEW_NAMES_18, assign_prompts_18
from .fusion import gaussian_spherical_fusion_fast, inverse_mapping_fusion
from dataclasses import dataclass


@dataclass
class PolyDiffPipelineOutput(BaseOutput):
    """Output of PolyDiffPipeline.
    
    Attributes:
        views: All 18 uncropped perspective views [18, H, W, 3]
        equirectangular: Fused ERP panorama [erp_H, erp_W, 3]
    """
    views: np.ndarray
    equirectangular: np.ndarray


class PolyDiffPipeline(StableDiffusionPipeline):
    """
    PolyDiff-18: 18-view panorama generation pipeline.
    
    Extends CubeDiff from 6 cubemap faces to 18 over-complete views:
    - 6 main views (front, right, back, left, top, bottom)
    - 4 equator seam views (45°, 135°, 225°, 315° yaw)
    - 4 top-polar seam views (45° pitch)
    - 4 bottom-polar seam views (-45° pitch)
    
    Uses Gaussian-weighted spherical fusion to blend views into seamless ERP.
    """
    
    NUM_VIEWS = 18
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, num_faces: int = 18, **kwargs):
        """
        Load PolyDiffPipeline from pretrained model and apply patches.
        
        Args:
            pretrained_model_name_or_path: Path to pretrained CubeDiff or SD model
            num_faces: Number of views (18 for PolyDiff, 6 for original CubeDiff)
        """
        # Load the base pipeline
        pipeline = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        if pipeline.unet.config.in_channels != 7:
            # Is a base SD model, patch input conv as well
            patch_unet(pipeline.unet, in_channels=7)
        else:
            # Apply attention patches with 18-view configuration
            swap_transformer_blocks(pipeline.unet, num_faces=num_faces)
        
        # Apply groupnorm patches for 18 views
        patch_groupnorm(pipeline.vae, num_faces=num_faces)
    
        return pipeline

    @torch.no_grad()
    def __call__(
        self,
        prompts: Union[str, List[str], Dict[str, str]],
        *,
        conditioning_image: torch.Tensor,  # (C,H,W) or (1,C,H,W)
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        cfg_scale: float = 3.5,
        fov_deg: float = 95.0,
        erp_height: int = 1024,
        erp_width: int = 2048,
        fusion_mode: str = "wta",
        effective_fov_deg: float = 70.0,
    ):
        """
        Generate 360° panorama from conditioning image and prompts.
        
        Args:
            prompts: Either:
                - A single string (applied to all views)
                - A list of 18 strings (one per view)
                - A dict with keys 'Front', 'Right', 'Back', 'Left', 'Top', 'Bottom', 'Overall'
            conditioning_image: Front view anchor image (C,H,W) or (1,C,H,W)
            num_inference_steps: Number of diffusion steps
            generator: Optional torch generator for reproducibility
            cfg_scale: Classifier-free guidance scale
            fov_deg: Field of view for ERP fusion
            erp_height, erp_width: Output ERP dimensions
        
        Returns:
            PolyDiffPipelineOutput with views and equirectangular panorama
        """
        device = self._execution_device
        T = self.NUM_VIEWS

        # Handle prompts
        if isinstance(prompts, str):
            prompts = [prompts] * T
        elif isinstance(prompts, dict):
            prompts = assign_prompts_18(prompts)
        
        if len(prompts) != T:
            raise ValueError(f"Expected {T} prompts, got {len(prompts)}")

        # Tokenize and encode prompts
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

        # --- scheduler / latents -------------------------------------------
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        latents = torch.randn(
            (T, 4, self.unet.config.sample_size, self.unet.config.sample_size),
            generator=generator,
            device=device,
            dtype=self.unet.dtype,
        )
        latents *= self.scheduler.init_noise_sigma

        # Build extra channels tensor for 18 views
        static_extra = make_extra_channels_tensor_18(
            1, self.unet.config.sample_size, self.unet.config.sample_size
        ).to(device, dtype=self.unet.dtype)

        # Encode conditioning image (anchor for front view)
        if conditioning_image.ndim == 3:
            conditioning_image = conditioning_image.unsqueeze(0)
        conditioning_image = conditioning_image.to(device, dtype=self.unet.dtype)
        ref_lat = self.vae.encode(conditioning_image).latent_dist.mean[0]
        ref_lat *= self.vae.config.scaling_factor

        # --- Denoising loop with progress bar ---
        progress_bar = tqdm(
            self.scheduler.timesteps,
            desc="Generating 18-view panorama",
            total=len(self.scheduler.timesteps),
            unit="step"
        )
        
        for i, t in enumerate(progress_bar):
            # Keep front face fixed to anchor
            latents[0] = ref_lat
            
            latents_scaled = self.scheduler.scale_model_input(latents, t)
            latents_input = torch.cat([latents_scaled, static_extra], dim=1)

            # Conditional prediction
            noise_pred = self.unet(
                latents_input, t, encoder_hidden_states=encoder_hidden_states
            ).sample
            
            # Unconditional prediction (with front face drop for CFG)
            noise_pred_uncond = self.unet(
                latents_input,
                t,
                encoder_hidden_states=uncond_embeddings,
                cross_attention_kwargs={"front_face_drop": True},
            ).sample

            # CFG combination
            combined = noise_pred_uncond + cfg_scale * (noise_pred - noise_pred_uncond)
            
            # Update all views except anchor (view 0)
            latents[1:] = self.scheduler.step(combined[1:], t, latents[1:]).prev_sample
            
            progress_bar.set_postfix({
                'timestep': f'{t.item():.0f}',
                'step': f'{i+1}/{len(self.scheduler.timesteps)}'
            })

        # --- decode ---------------------------------------------------------
        print("\n[INFO] Decoding 18 views...")
        imgs = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        # Convert to numpy (18, H, W, 3)
        views_list = []
        for i in range(T):
            img = imgs[i].detach().cpu().permute(1, 2, 0).float().numpy()
            img = (img * 255).astype(np.uint8)
            views_list.append(img)
        
        views_np = np.array(views_list)

        # --- Inverse mapping fusion ---
        print(f"[INFO] Fusing views into ERP panorama (mode={fusion_mode})...")
        equirec = inverse_mapping_fusion(
            views_list, 
            VIEW_CONFIG_18, 
            fov_deg=fov_deg,
            erp_height=erp_height,
            erp_width=erp_width,
            mode=fusion_mode,
            effective_fov_deg=effective_fov_deg,
        )

        return PolyDiffPipelineOutput(
            views=views_np,
            equirectangular=equirec,
        )
