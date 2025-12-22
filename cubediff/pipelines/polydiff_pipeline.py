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
from .fusion import gaussian_spherical_fusion_fast, inverse_mapping_fusion, tiered_fusion
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
        two_stage: bool = True,
    ):
        """
        Generate 360° panorama from conditioning image and prompts.
        """
        device = self._execution_device
        T = self.NUM_VIEWS
        
        # --- Handle prompts ---
        if isinstance(prompts, str):
            prompts = [prompts] * T
        elif isinstance(prompts, dict):
            prompts = assign_prompts_18(prompts)
        
        if len(prompts) != T:
            raise ValueError(f"Expected {T} prompts, got {len(prompts)}")

        # Split prompts for two-stage
        main_prompts = prompts[:6]
        full_prompts = prompts

        # --- Scheduler ---
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        # --- Encode conditioning image (anchor) ---
        if conditioning_image.ndim == 3:
            conditioning_image = conditioning_image.unsqueeze(0)
        conditioning_image = conditioning_image.to(device, dtype=self.unet.dtype)
        ref_lat = self.vae.encode(conditioning_image).latent_dist.mean[0]
        ref_lat *= self.vae.config.scaling_factor

        # --- Build extra channels for 18 views ---
        static_extra_18 = make_extra_channels_tensor_18(
            1, self.unet.config.sample_size, self.unet.config.sample_size
        ).to(device, dtype=self.unet.dtype)
        
        # Build extra channels for 6 views (subset of 18)
        static_extra_6 = static_extra_18[:6]

        # =========================================================
        # STAGE 1: Generate 6 Main Views (Sharp & Independent)
        # =========================================================
        print("\n[Stage 1] Generating 6 Main Views (Front, Right, Back, Left, Top, Bottom)...")
        
        # Encode main prompts
        text_inputs_6 = self.tokenizer(main_prompts, max_length=self.tokenizer.model_max_length, padding="max_length", return_tensors="pt")
        text_emb_6 = self.text_encoder(text_inputs_6.input_ids.to(device))[0]
        
        uncond_inputs_6 = self.tokenizer([""] * 6, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt")
        uncond_emb_6 = self.text_encoder(uncond_inputs_6.input_ids.to(device))[0]
        
        # Latents for 6 views
        latents_6 = torch.randn(
            (6, 4, self.unet.config.sample_size, self.unet.config.sample_size),
            generator=generator, device=device, dtype=self.unet.dtype
        ) * self.scheduler.init_noise_sigma

        # Denoising loop for T=6
        for t in tqdm(self.scheduler.timesteps, desc="Stage 1 (Main Views)"):
            latents_6[0] = ref_lat  # Fix anchor
            
            latents_input = torch.cat([self.scheduler.scale_model_input(latents_6, t), static_extra_6], dim=1)
            
            # Predict noise
            noise_pred = self.unet(latents_input, t, encoder_hidden_states=text_emb_6).sample
            
            # Unconditional with front face drop
            noise_pred_uncond = self.unet(latents_input, t, encoder_hidden_states=uncond_emb_6, cross_attention_kwargs={"front_face_drop": True}).sample
            
            combined = noise_pred_uncond + cfg_scale * (noise_pred - noise_pred_uncond)
            latents_6[1:] = self.scheduler.step(combined[1:], t, latents_6[1:]).prev_sample
        
        # Use result as anchor for next stage
        main_view_latents = latents_6
        
        # =========================================================
        # STAGE 2: Generate 12 Seam Views (Inpainting)
        # =========================================================
        if two_stage: 
            print("\n[Stage 2] Inpainting 12 Seam Views...")
            
            # Encode full 18 prompts
            text_inputs_18 = self.tokenizer(full_prompts, max_length=self.tokenizer.model_max_length, padding="max_length", return_tensors="pt")
            text_emb_18 = self.text_encoder(text_inputs_18.input_ids.to(device))[0]
            
            uncond_inputs_18 = self.tokenizer([""] * 18, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt")
            uncond_emb_18 = self.text_encoder(uncond_inputs_18.input_ids.to(device))[0]

            # Initialize 18 latents
            latents_18 = torch.randn(
                (18, 4, self.unet.config.sample_size, self.unet.config.sample_size),
                generator=generator, device=device, dtype=self.unet.dtype
            ) * self.scheduler.init_noise_sigma
            
            # Denoising loop for T=18
            for t in tqdm(self.scheduler.timesteps, desc="Stage 2 (Seam Inpainting)"):
                # FORCE FIX Main Views (0-5) to the result of Stage 1
                # We add noise to Stage 1 result to match current timestep t
                if t > 0: # Avoid adding noise at step 0 (fully denoised)
                    noise = torch.randn_like(main_view_latents)
                    noisy_main_latents = self.scheduler.add_noise(main_view_latents, noise, t.unsqueeze(0))
                    latents_18[:6] = noisy_main_latents
                else:
                    latents_18[:6] = main_view_latents

                latents_18[0] = ref_lat # Ensure Front is anchor (double insurance)

                latents_input = torch.cat([self.scheduler.scale_model_input(latents_18, t), static_extra_18], dim=1)
                
                # Predict noise
                noise_pred = self.unet(latents_input, t, encoder_hidden_states=text_emb_18).sample
                
                # Uncond
                noise_pred_uncond = self.unet(latents_input, t, encoder_hidden_states=uncond_emb_18, cross_attention_kwargs={"front_face_drop": True}).sample
                
                combined = noise_pred_uncond + cfg_scale * (noise_pred - noise_pred_uncond)
                
                # Update only seam views (6-17)
                # Note: We compute update for all, but only keep seams. 
                # Main views will be overwritten in next iter anyway.
                latents_18[1:] = self.scheduler.step(combined[1:], t, latents_18[1:]).prev_sample

            final_latents = latents_18
            # Restore clean main latents one last time
            final_latents[:6] = main_view_latents
            
        else:
            final_latents = latents_6 # Fallback if two_stage=False (should handle logic better)
            # Actually if two_stage=False we should run old logic. 
            # For now let's assume always True or modify logic to support one-pass 18 view.

        # --- decode ---------------------------------------------------------
        print("\n[INFO] Decoding 18 views...")
        imgs = self.vae.decode(final_latents / self.vae.config.scaling_factor).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        # Convert to numpy (18, H, W, 3)
        views_list = []
        for i in range(T):
            img = imgs[i].detach().cpu().permute(1, 2, 0).float().numpy()
            img = (img * 255).astype(np.uint8)
            views_list.append(img)
        
        views_np = np.array(views_list)

        # --- Fusion ---
        print(f"[INFO] Fusing views into ERP panorama (mode={fusion_mode})...")
        
        if fusion_mode == "tiered":
            # Tiered fusion: main views form backbone, seam views only blend at edges
            # Key: Only use center 90° of each 95° view (discard outer 5°)
            equirec = tiered_fusion(
                views_list, 
                VIEW_CONFIG_18, 
                fov_deg=fov_deg,
                erp_height=erp_height,
                erp_width=erp_width,
                valid_fov_deg=90.0,   # Only use center 90° (discard edges)
                core_fov_deg=80.0,    # Main view dominates in core
                sigma_factor=0.3,
            )
        else:
            # Original modes: wta or gaussian
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
