"""
SD1.5 Conv-out Only Training Script

This script uses a minimal fine-tuning strategy:
- Uses original SD1.5 UNet (4-channel input, no position encoding or mask)
- Only trains conv_out layer (~1,280 parameters)
- All other layers frozen
- Generates 6 faces independently with IP-Adapter conditioning

Usage:
    python train_convout.py --config convout_only
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.utils.torch_utils import is_compiled_module

from training.dataset import (
    PreExtractedCubemapDataset,
    preextracted_cubemap_collate_fn
)


def main(cfg: DictConfig):

    # ---------------------- Accelerator, SwanLab Setup --------------------------
    num_gpus = cfg.training.num_gpus
    per_gpu_batch_size = cfg.training.per_gpu_batch_size
    global_batch_size = cfg.training.batch_size
    acc_steps = global_batch_size // (per_gpu_batch_size * num_gpus)

    accelerator = Accelerator(
        mixed_precision=cfg.training.mixed_precision,
        gradient_accumulation_steps=acc_steps,
    )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if accelerator.is_main_process:
        print(f"[INFO] Accelerator detected {accelerator.num_processes} processes (GPUs)")
        print(f"[INFO] Mixed precision: {cfg.training.mixed_precision}")
        print(f"[INFO] Training mode: conv_out only (4-channel input)")

        import swanlab
        swanlab.init(
            project=cfg.swanlab.project_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            experiment_name=cfg.name,
            mode="cloud" if cfg.swanlab.use_cloud else "local"
        )
    
    # ---------------------- Seeding ----------------------------
    rank = accelerator.process_index
    base_seed = cfg.training.seed
    local_seed = base_seed + rank
    torch.manual_seed(local_seed)
    np.random.seed(local_seed)
    set_seed(base_seed, device_specific=True)
    
    # ---------------------- Load Original SD1.5 Pipeline ----------------------
    if accelerator.is_main_process:
        print("[DEBUG] Loading original SD1.5 pipeline (4-channel input)...")

    cache_dir = os.path.expanduser(cfg.directories.cache_dir)
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            cfg.model.id, 
            cache_dir=cache_dir, 
            local_files_only=True
        )
    except Exception as e:
        if accelerator.is_main_process:
            print(f"[WARNING] Local load failed: {e}")
            print("[INFO] Trying to download from HuggingFace...")
        pipe = StableDiffusionPipeline.from_pretrained(
            cfg.model.id, 
            cache_dir=cache_dir, 
            local_files_only=False
        )
    
    if accelerator.is_main_process:
        print("[DEBUG] Pipeline loaded.")
        print(f"[DEBUG] UNet conv_in channels: {pipe.unet.conv_in.in_channels}")

    # Configure scheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.config.prediction_type = "epsilon"

    # ---------------------- Load IP-Adapter (if enabled) ----------------------
    use_ip_adapter = getattr(cfg.training, 'use_ip_adapter', False)
    
    if use_ip_adapter:
        if accelerator.is_main_process:
            print("[INFO] Loading IP-Adapter...")
        pipe.load_ip_adapter(
            cfg.training.ip_adapter_repo,
            subfolder=cfg.training.ip_adapter_subfolder,
            weight_name=cfg.training.ip_adapter_weight_name
        )
        if accelerator.is_main_process:
            print("[INFO] IP-Adapter loaded successfully")

    # ---------------------- Freeze All, Unfreeze conv_out Only ----------------------
    # 策略：只训练 conv_out，其他全部冻结
    
    for param in pipe.vae.parameters():
        param.requires_grad = False

    for param in pipe.unet.parameters():
        param.requires_grad = False

    for param in pipe.text_encoder.parameters():
        param.requires_grad = False
    
    # Only unfreeze conv_out
    for param in pipe.unet.conv_out.parameters():
        param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)
    if accelerator.is_main_process:
        print(f"[INFO] Trainable parameters (conv_out only): {trainable_params:,}")
    
    # ---------------------- Dataset and DataLoader ----------------------
    if accelerator.is_main_process:
        print(f"[INFO] Using PreExtractedCubemapDataset")
        print(f"[INFO] Cubemap dir: {cfg.directories.cubemap_dir}")
        print(f"[INFO] Prompt dir: {cfg.directories.prompt_dir}")
    
    dataset = PreExtractedCubemapDataset(
        cubemap_dir=cfg.directories.cubemap_dir,
        prompt_dir=cfg.directories.prompt_dir,
        face_size=cfg.model.image_size,
        augment=getattr(cfg.training, 'augment', True),
        return_ref_images=use_ip_adapter
    )
    collate_fn = preextracted_cubemap_collate_fn
                
    if accelerator.is_main_process:
        print(f"[DEBUG] Dataset size: {len(dataset)}")

    # ---------------------- Fixed Validation Sample ----------------------
    val_dataset = PreExtractedCubemapDataset(
        cubemap_dir=cfg.directories.cubemap_dir,
        prompt_dir=cfg.directories.prompt_dir,
        face_size=cfg.model.image_size,
        augment=False,
        return_ref_images=False
    )
    
    val_sample_fixed = val_dataset[1]
    val_conditioning_image_fixed = val_sample_fixed[0][0]  # Front face
    val_prompts_fixed = val_sample_fixed[1]
    
    if accelerator.is_main_process:
        print(f"[INFO] Validation sample: scene {val_sample_fixed[3] if len(val_sample_fixed) > 3 else '0'}")
        print(f"[INFO] Validation prompts (Front): {val_prompts_fixed[0][:50]}...")

    # ---------------------- Dataloader ----------------------
    dataloader = DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    
    # ---------------------- Optimizer & LR Scheduler ----------------------
    params = [p for p in pipe.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, 
        lr=cfg.training.learning_rate, 
        betas=tuple(cfg.training.betas), 
        eps=cfg.training.eps
    )
    
    def lr_lambda(current_step):
        return current_step / cfg.training.warmup_steps if current_step < cfg.training.warmup_steps else 1.0
    
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # --------------------- Prepare components -----------------------
    unet, optimizer, dataloader = accelerator.prepare(
        pipe.unet, optimizer, dataloader
    )
    
    if accelerator.is_main_process:
        eff_bs = per_gpu_batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps
        print(f"[INFO] Per-GPU batch size: {per_gpu_batch_size}")
        print(f"[INFO] Effective global batch size: {eff_bs}")

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipe.vae.to(accelerator.device, dtype=weight_dtype)
    pipe.text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # IP-Adapter image_encoder
    if use_ip_adapter and hasattr(pipe, 'image_encoder') and pipe.image_encoder is not None:
        pipe.image_encoder.to(accelerator.device, dtype=weight_dtype)
        if accelerator.is_main_process:
            print(f"[INFO] Image encoder moved to {accelerator.device} with dtype {weight_dtype}")

    # ---------------------- Training Loop ----------------------
    epochs = cfg.training.epochs
    T = 6  # Number of faces
    checkpoint_dir = os.path.join(cfg.directories.checkpoint_dir, f"{cfg.name}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    step_start_time = time.time()
    global_step = 0
    debug_printed = False
    
    if accelerator.is_main_process:
        print("[DEBUG] Starting training...")
        print(f"[INFO] IP-Adapter conditioning: {'ENABLED' if use_ip_adapter else 'DISABLED'}")
    
    for epoch in range(epochs):
        if accelerator.is_main_process:
            print(f"Epoch {epoch + 1}/{epochs}")

        unet.train()
        optimizer.zero_grad()
        total_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", disable=not accelerator.is_main_process)
        for _, batch in enumerate(progress_bar):
            if batch is None:
                continue
            
            cubemap_batch, prompts, ref_images, scene_ids, single_captions = batch

            with accelerator.accumulate(unet):
                with accelerator.autocast():
                    with torch.no_grad():
                        # Encode cubemaps to latents [B*6, 4, H, W]
                        latents = pipe.vae.encode(cubemap_batch.to(accelerator.device)).latent_dist.mean
                        latents = latents * pipe.vae.config.scaling_factor

                        B = latents.shape[0] // T
                        _, _, H, W = latents.shape

                        # Prompt dropout (10% chance)
                        mask = torch.rand(B, device=accelerator.device) < 0.1
                        full_mask = mask.repeat_interleave(T)
                        prompts = [p if not m else "" for p, m in zip(prompts, full_mask)]

                        # Encode text
                        text_inputs = pipe.tokenizer(
                            prompts, padding="max_length", truncation=True, max_length=77, return_tensors="pt"
                        )
                        encoder_hidden_states = pipe.text_encoder(text_inputs.input_ids.to(accelerator.device))[0]
                        
                        # Generate timesteps
                        timesteps = torch.randint(
                            0, pipe.scheduler.config.num_train_timesteps, (B * T,),
                            device=latents.device, dtype=torch.long
                        )

                        # Create noise for ALL faces (no front mask logic)
                        noise = torch.randn_like(latents)
                        orig_latents = latents.clone()

                        # Prepare IP-Adapter embeddings if enabled
                        ip_embeds = None
                        
                        if use_ip_adapter and ref_images is not None:
                            flat_ref_images = []
                            for batch_refs in ref_images:
                                if batch_refs is not None:
                                    flat_ref_images.extend(batch_refs)
                            
                            if flat_ref_images and hasattr(pipe, 'feature_extractor') and pipe.feature_extractor is not None:
                                clip_image = pipe.feature_extractor(
                                    images=flat_ref_images, 
                                    return_tensors="pt"
                                ).pixel_values.to(accelerator.device, dtype=weight_dtype)
                                
                                ip_embeds = pipe.image_encoder(clip_image).image_embeds
                                
                                if accelerator.is_main_process and not debug_printed:
                                    print(f"[DEBUG] ip_embeds shape: {ip_embeds.shape}")
                                    print(f"[DEBUG] B={B}, T={T}, B*T={B*T}")
                                    debug_printed = True

                    # Add noise to ALL faces
                    noisy_latents = pipe.scheduler.add_noise(
                        latents, noise, timesteps
                    ).to(latents.dtype)

                    # NO extra channels - use original 4-channel latent directly
                    latent_input = noisy_latents  # [B*6, 4, H, W]

                    # Prepare kwargs
                    added_cond_kwargs = {}
                    if use_ip_adapter and ip_embeds is not None:
                        added_cond_kwargs["image_embeds"] = ip_embeds

                    # Forward pass
                    model_pred = unet(
                        latent_input,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        added_cond_kwargs=added_cond_kwargs if added_cond_kwargs else None,
                    ).sample

                    # Compute loss on ALL faces
                    loss = nn.functional.mse_loss(model_pred, noise)
                
                    total_loss += loss.detach().float()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    trainable_params = [p for p in unet.parameters() if p.requires_grad]
                    grad_norm = accelerator.clip_grad_norm_(trainable_params, max_norm=1.0)

                optimizer.step()

                if accelerator.sync_gradients:
                    lr_scheduler.step()

                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1
                    time_elapsed = time.time() - step_start_time
                    step_start_time = time.time()

                    avg_loss_this_process = total_loss / accelerator.gradient_accumulation_steps
                    gathered_losses = accelerator.gather_for_metrics(avg_loss_this_process.unsqueeze(0))
                    avg_loss = torch.mean(gathered_losses)
                    total_loss = 0.0

                    if accelerator.is_main_process:
                        import swanlab
                        swanlab.log({
                            "loss": avg_loss.item(),
                            "time_per_step": time_elapsed,
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                            "grad_norm": grad_norm.item()
                        })
                        progress_bar.set_postfix({"loss": f"{avg_loss.item():.4f}", "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"})

                    # Step-based checkpointing
                    if global_step % cfg.training.checkpoint_interval == 0:
                        ckpt_folder = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}_step_{global_step}")
                        accelerator.save_state(ckpt_folder)
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            print(f"[INFO] Checkpoint saved: {ckpt_folder}")
                        accelerator.wait_for_everyone()
    
    # Save final model
    ckpt_filename = os.path.join(checkpoint_dir, f"epoch_{epochs}_step_{global_step}_final")
    accelerator.save_state(ckpt_filename)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f"[INFO] Final checkpoint saved: {ckpt_filename}")
        import swanlab
        swanlab.finish()

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    os.environ["NCCL_TIMEOUT"] = "3600"
    os.environ["NCCL_DEBUG"] = "INFO"

    parser = argparse.ArgumentParser(description="Train SD1.5 conv_out only")
    parser.add_argument("--config", type=str, default="convout_only",
                        help="Name of the Hydra config to use")
    
    args, overrides = parser.parse_known_args()

    from hydra import initialize, compose
    with initialize(config_path="training/configs", version_base=None):
        cfg = compose(config_name=args.config, overrides=overrides)
        main(cfg)
