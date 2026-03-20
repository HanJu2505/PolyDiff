"""
CubeDiff Training Script with IP-Adapter Support

This script supports training with:
- Pre-extracted cubemap faces (no ERP conversion at runtime)
- Discrete face rotation augmentation
- IP-Adapter conditioning (frozen, not fine-tuned)

Usage:
    python train_ipadapter.py --config multitext_ipadapter
"""

import math
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
from diffusers import DDIMScheduler
from diffusers.utils.torch_utils import is_compiled_module

from training.dataset import (
    CubemapDataset, 
    cubemap_collate_fn,
    PreExtractedCubemapDataset,
    preextracted_cubemap_collate_fn
)
from cubediff.pipelines.pipeline import CubeDiffPipeline
from cubediff.modules.extra_channels import get_uv_tensors
from cubediff.modules.utils import load_sliced_unet_weights


def main(cfg: DictConfig):

    # ---------------------- Accelerator, wandb Setup --------------------------
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
        print(f"[INFO] Checkpoint interval type: {cfg.training.checkpoint_interval_type}")

        import swanlab
        swanlab.init(
            project=cfg.swanlab.project_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            experiment_name=cfg.name,
            mode="cloud" if cfg.swanlab.use_cloud else "local"  # 支持本地模式
        )
    
    # ---------------------- Seeding ----------------------------
    rank = accelerator.process_index
    base_seed = cfg.training.seed
    local_seed = base_seed + rank
    torch.manual_seed(local_seed)
    np.random.seed(local_seed)
    set_seed(base_seed, device_specific=True)
    
    # ---------------------- Load Pipeline & Patch ----------------------
    if accelerator.is_main_process:
        print("[DEBUG] Loading pipeline...")

    # 展开 ~ 为完整路径
    cache_dir = os.path.expanduser(cfg.directories.cache_dir)
    
    if accelerator.is_main_process:
        print(f"[DEBUG] Cache dir: {cache_dir}")
    
    try:
        pipe = CubeDiffPipeline.from_pretrained(
            cfg.model.id, 
            cache_dir=cache_dir, 
            local_files_only=True
        )
    except Exception as e:
        if accelerator.is_main_process:
            print(f"[WARNING] Local load failed: {e}")
            print("[INFO] Trying to download from HuggingFace...")
        pipe = CubeDiffPipeline.from_pretrained(
            cfg.model.id, 
            cache_dir=cache_dir, 
            local_files_only=False
        )
    
    if accelerator.is_main_process:
        print("[DEBUG] Pipeline loaded.")

    # Load CubeDiff pretrained weights (7→4 channel slicing) if specified
    cubediff_weights = getattr(cfg.training, 'cubediff_weights', None)
    if cubediff_weights is not None:
        cubediff_weights = os.path.expanduser(cubediff_weights)
        if os.path.exists(cubediff_weights):
            if cubediff_weights.endswith('.safetensors'):
                from safetensors.torch import load_file
                sd = load_file(cubediff_weights, device="cpu")
            else:
                sd = torch.load(cubediff_weights, map_location="cpu")
            load_sliced_unet_weights(pipe.unet, sd)
            if accelerator.is_main_process:
                print(f"[INFO] Loaded CubeDiff weights from {cubediff_weights}")
        else:
            if accelerator.is_main_process:
                print(f"[WARNING] CubeDiff weights not found: {cubediff_weights}")

    # Configure scheduler
    if cfg.training.prediction_type == "v_prediction":
        pipe.scheduler = DDIMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-1", 
            subfolder="scheduler", 
            cache_dir=cfg.directories.cache_dir, 
            local_files_only=True
        )
        pipe.scheduler.config.prediction_type = "v_prediction"
    elif cfg.training.prediction_type == "epsilon":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.config.prediction_type = "epsilon"

    # Set latent size
    latent_H = 64 if cfg.model.image_size == 512 else 96

    # ---------------------- Load IP-Adapter (if enabled) ----------------------
    use_ip_adapter = getattr(cfg.training, 'use_ip_adapter', False)
    
    # 必须在所有进程加载，不能只在主进程
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

    # ---------------------- Freeze and Unfreeze Parameters ----------------------
    for param in pipe.vae.parameters():
        param.requires_grad = False

    for param in pipe.unet.parameters():
        param.requires_grad = False

    for param in pipe.text_encoder.parameters():
        param.requires_grad = False
    
    # conv_in is NOT unfrozen — we use standard SD1.5 4-channel weights directly

    # Unfreeze attention layers (but NOT IP-Adapter layers)
    for name, param in pipe.unet.named_parameters():
        if "attn" in name:
            # Keep IP-Adapter projection layers frozen
            if "to_k_ip" not in name and "to_v_ip" not in name:
                param.requires_grad = True
    
    # ---------------------- Dataset and DataLoader ----------------------
    use_preextracted = getattr(cfg.training, 'use_preextracted', False)
    
    if use_preextracted:
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
    else:
        if accelerator.is_main_process:
            print(f"[INFO] Using CubemapDataset (ERP conversion)")
            print(f"[INFO] Data dir: {cfg.directories.data_dir}")
        
        dataset = CubemapDataset(
            root_dir=cfg.directories.data_dir,
            face_size=cfg.model.image_size,
            fov=95,
            checkpoint_dir=cfg.directories.checkpoint_dir,
            use_cached_data=False
        )
        collate_fn = cubemap_collate_fn
                    
    if accelerator.is_main_process:
        print(f"[DEBUG] Dataset size: {len(dataset)}")

    # ---------------------- Fixed Validation Sample (no augmentation) ----------------------
    # Create a separate dataset instance with augmentation disabled for validation
    if use_preextracted:
        val_dataset = PreExtractedCubemapDataset(
            cubemap_dir=cfg.directories.cubemap_dir,
            prompt_dir=cfg.directories.prompt_dir,
            face_size=cfg.model.image_size,
            augment=False,  # ⭐ 关闭增强，使用原始顺序
            return_ref_images=use_ip_adapter  # IP-Adapter 开启时加载参考图
        )
    else:
        val_dataset = dataset  # ERP 模式下使用相同 dataset
    
    # 获取固定的验证样本（第一个场景，原始顺序）
    val_sample_fixed = val_dataset[0]
    val_conditioning_image_fixed = val_sample_fixed[0][0]  # Front face (原始)
    val_prompts_fixed = val_sample_fixed[1]  # 原始顺序的 prompts
    # 提取验证集的参考图（用于 IP-Adapter）
    val_ref_images_fixed = val_sample_fixed[2] if (use_ip_adapter and len(val_sample_fixed) > 2) else None
    
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
    
    # 计算总训练步数（用于余弦退火终点）
    steps_per_epoch = len(dataloader) // cfg.training.gradient_accumulation_steps
    total_steps = cfg.training.epochs * steps_per_epoch
    min_lr = 1e-5  # 余弦退火最低学习率

    def lr_lambda_cosine(current_step):
        warmup = cfg.training.warmup_steps
        if current_step < warmup:
            # 线性 warmup
            return current_step / max(1, warmup)
        # 余弦退火：从 peak_lr 衰减到 min_lr
        progress = (current_step - warmup) / max(1, total_steps - warmup)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # 缩放到 [min_lr/peak_lr, 1.0] 区间
        min_ratio = min_lr / cfg.training.learning_rate
        return min_ratio + (1.0 - min_ratio) * cosine_decay

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_cosine)
    
    # --------------------- Prepare components -----------------------
    unet, optimizer, dataloader = accelerator.prepare(
        pipe.unet, optimizer, dataloader
    )
    
    if accelerator.is_main_process:
        eff_bs = per_gpu_batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps
        print(f"[INFO] Per-GPU batch size: {per_gpu_batch_size}")
        print(f"[INFO] Effective global batch size: {eff_bs}")
        print(f"[INFO] Trainable parameters: {sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad):,}")

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipe.vae.to(accelerator.device, dtype=weight_dtype)
    pipe.text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # IP-Adapter 的 image_encoder 也需要移到正确的设备和精度
    if use_ip_adapter and hasattr(pipe, 'image_encoder') and pipe.image_encoder is not None:
        pipe.image_encoder.to(accelerator.device, dtype=weight_dtype)
        if accelerator.is_main_process:
            print(f"[INFO] Image encoder moved to {accelerator.device} with dtype {weight_dtype}")
    
    # IP-Adapter 的 to_k_ip/to_v_ip 层也需要移到正确的设备
    # 这些层在 CubeDiffIPAdapterAttnProcessor 中
    if use_ip_adapter:
        from cubediff.modules.attention import CubeDiffIPAdapterAttnProcessor
        
        def move_ip_adapter_to_device(module):
            for child in module.children():
                if hasattr(child, 'attn2') and child.attn2 is not None:
                    processor = child.attn2.processor
                    if isinstance(processor, CubeDiffIPAdapterAttnProcessor):
                        if hasattr(processor, 'to_k_ip'):
                            processor.to_k_ip = processor.to_k_ip.to(accelerator.device, dtype=weight_dtype)
                        if hasattr(processor, 'to_v_ip'):
                            processor.to_v_ip = processor.to_v_ip.to(accelerator.device, dtype=weight_dtype)
                else:
                    move_ip_adapter_to_device(child)
        
        move_ip_adapter_to_device(pipe.unet)
        if accelerator.is_main_process:
            print(f"[INFO] IP-Adapter attention layers moved to {accelerator.device}")

    # ---------------------- Resume Training ----------------------
    global_step = 0
    start_epoch = 0
    if cfg.training.resume_checkpoint:
        resume_ckpt = os.path.join(cfg.directories.checkpoint_dir, cfg.training.resume_checkpoint_filename)
        if os.path.exists(resume_ckpt):
            start_epoch = int(os.path.basename(resume_ckpt).split("epoch_")[-1].split("_")[0])
            global_step = int(os.path.basename(resume_ckpt).split("step_")[-1].split("_")[0])
            accelerator.load_state(resume_ckpt)
            if accelerator.is_main_process:
                print(f"[INFO] Resumed from {resume_ckpt}, epoch {start_epoch}, step {global_step}")
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_cosine, last_epoch=global_step - 1)

    # ---------------------- Training Loop ----------------------
    epochs = cfg.training.epochs
    T = 6  # Number of faces
    checkpoint_dir = cfg.directories.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    step_start_time = time.time()
    
    # Flag to ensure debug info is printed only once
    debug_printed = False
    
    if accelerator.is_main_process:
        print("[DEBUG] Starting training...")
        print(f"[INFO] IP-Adapter conditioning: {'ENABLED' if use_ip_adapter else 'DISABLED'}")
    
    for epoch in range(start_epoch, epochs):
        if accelerator.is_main_process:
            print(f"Epoch {epoch + 1}/{epochs}")

        unet.train()
        optimizer.zero_grad()
        total_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", disable=not accelerator.is_main_process)
        for _, batch in enumerate(progress_bar):
            if batch is None:
                continue
            
            if use_preextracted:
                cubemap_batch, prompts, ref_images, scene_ids, single_captions = batch
            else:
                cubemap_batch, prompts, _, _, single_captions = batch
                ref_images = None

            with accelerator.accumulate(unet):
                with accelerator.autocast():
                    with torch.no_grad():
                        # Encode cubemaps to latents
                        latents = pipe.vae.encode(cubemap_batch.to(accelerator.device)).latent_dist.mean
                        latents = latents * pipe.vae.config.scaling_factor

                        B = latents.shape[0] // T
                        _, _, H, W = latents.shape

                        # Prompt dropout (10% chance)
                        mask = torch.rand(B, device=accelerator.device) < 0.1
                        full_mask = mask.repeat_interleave(T)
                        prompts = [p if not m else "" for p, m in zip(prompts, full_mask)]

                        # Handle training type
                        if cfg.training.type == "image_only":
                            prompts = [""] * len(prompts)
                        elif cfg.training.type == "single_caption":
                            prompts = []
                            for i, sc in enumerate(single_captions):
                                if full_mask[i * T]:
                                    prompts.extend([""] * T)
                                else:
                                    prompts.extend([sc] * T)

                        # Encode text
                        text_inputs = pipe.tokenizer(
                            prompts, padding="max_length", truncation=True, max_length=77, return_tensors="pt"
                        )
                        encoder_hidden_states = pipe.text_encoder(text_inputs.input_ids.to(accelerator.device))[0]
                        
                        # Generate timesteps
                        timesteps = torch.randint(
                            0, pipe.scheduler.config.num_train_timesteps, (B,),
                            device=latents.device, dtype=torch.long
                        )
                        timesteps = timesteps.repeat_interleave(T)

                        # Identify non-front faces
                        face_indices = torch.arange(B * T, device=latents.device)
                        face_ids = face_indices % T
                        non_front_mask = face_ids != 0

                        # Create noise for non-front faces only
                        noise = torch.randn_like(latents[non_front_mask])
                        orig_latents = latents.clone()

                        # Prepare IP-Adapter embeddings if enabled
                        # 完全参照 pipeline.py 的处理方式
                        ip_embeds_for_attn = None  # 投影后的嵌入，用于 cross_attention_kwargs
                        ip_embeds_raw = None       # 原始嵌入，用于 added_cond_kwargs
                        
                        if use_ip_adapter and ref_images is not None:
                            # Flatten ref_images: list of B lists -> list of B*6 images  
                            flat_ref_images = []
                            for batch_refs in ref_images:
                                if batch_refs is not None:
                                    flat_ref_images.extend(batch_refs)
                            
                            if flat_ref_images and hasattr(pipe, 'feature_extractor') and pipe.feature_extractor is not None:
                                # Step 1: 预处理图像，与推理代码相同
                                clip_image = pipe.feature_extractor(
                                    images=flat_ref_images, 
                                    return_tensors="pt"
                                ).pixel_values.to(accelerator.device, dtype=weight_dtype)
                                
                                # Step 2: 获取原始 CLIP 嵌入
                                # raw_image_embeds 形状: [num_images, 1024]
                                raw_image_embeds = pipe.image_encoder(clip_image).image_embeds
                                
                                # Step 3: 保存原始嵌入用于 added_cond_kwargs
                                # 这是 UNet 要求的，形状为 [num_images, 1024]
                                ip_embeds_raw = raw_image_embeds
                                
                                # Step 4: 投影嵌入用于 cross_attention_kwargs
                                # 参照 pipeline.py 第 279 行: cond_projected = self.unet.encoder_hid_proj(cond_raw)
                                # 注意：直接传 tensor，不要用列表包裹！
                                if hasattr(pipe.unet, 'encoder_hid_proj') and pipe.unet.encoder_hid_proj is not None:
                                    ip_embeds_projected = pipe.unet.encoder_hid_proj(raw_image_embeds)
                                    
                                    # encoder_hid_proj 可能返回列表
                                    if isinstance(ip_embeds_projected, list):
                                        ip_embeds_projected = ip_embeds_projected[0]
                                    
                                    # 去除多余维度 [N,1,4,768] -> [N,4,768]
                                    if ip_embeds_projected.ndim == 4:
                                        ip_embeds_projected = ip_embeds_projected.squeeze(1)
                                    
                                    # DEBUG: 打印形状（只打印一次，避免干扰进度条）
                                    if accelerator.is_main_process and not debug_printed:
                                        print(f"[DEBUG] raw_image_embeds shape: {raw_image_embeds.shape}")
                                        print(f"[DEBUG] ip_embeds_projected shape: {ip_embeds_projected.shape}")
                                        print(f"[DEBUG] B={B}, T={T}, B*T={B*T}")
                                        debug_printed = True
                                    
                                    # 包裹成列表，对应 attn processor 的格式
                                    ip_embeds_for_attn = [ip_embeds_projected]
                                else:
                                    ip_embeds_for_attn = [raw_image_embeds]

                    # Add noise to non-front faces
                    latents[non_front_mask] = pipe.scheduler.add_noise(
                        latents[non_front_mask], noise, timesteps[non_front_mask]
                    ).to(latents.dtype)

                    # Pure 4-channel latent input (no extra channels concatenation)
                    latent_input = latents

                    # Generate UV coordinates for PE injection
                    uv_coords = get_uv_tensors(B, H, W).to(latents.device, dtype=weight_dtype)

                    # Front face drop (10% chance)
                    front_face_drop = torch.rand(1, device=accelerator.device) < 0.1

                    # Prepare cross_attention_kwargs and added_cond_kwargs
                    cross_attn_kwargs = {
                        "front_face_drop": front_face_drop,
                        "uv_coords": uv_coords,
                    }
                    added_cond_kwargs = {}
                    
                    if use_ip_adapter and ip_embeds_for_attn is not None:
                        # 投影后的嵌入给 attention processor (对应 pipeline.py 第 349 行)
                        cross_attn_kwargs["ip_adapter_image_embeds"] = ip_embeds_for_attn
                        # 原始 CLIP 嵌入给 UNet 内部路径 (对应 pipeline.py 第 350 行)
                        added_cond_kwargs["image_embeds"] = ip_embeds_raw

                    # Forward pass
                    model_pred = unet(
                        latent_input,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attn_kwargs,
                        added_cond_kwargs=added_cond_kwargs if added_cond_kwargs else {},
                    ).sample

                    # Compute loss
                    if cfg.training.prediction_type == "v_prediction":
                        v_target = pipe.scheduler.get_velocity(
                            orig_latents[non_front_mask], noise, timesteps[non_front_mask]
                        )
                        loss = nn.functional.mse_loss(model_pred[non_front_mask], v_target)
                    else:
                        loss = nn.functional.mse_loss(model_pred[non_front_mask], noise)
                
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
                        # 更新进度条显示 loss
                        progress_bar.set_postfix({"loss": f"{avg_loss.item():.4f}", "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"})

                    # Step-based checkpointing
                    if cfg.training.checkpoint_interval_type == "steps" and global_step % cfg.training.checkpoint_interval == 0:
                        ckpt_folder = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}_step_{global_step}")
                        accelerator.save_state(ckpt_folder)
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            print(f"[INFO] Checkpoint saved: {ckpt_folder}")
                        accelerator.wait_for_everyone()
                                        # Step-based validation
                    if cfg.training.checkpoint_interval_type == "steps" and global_step % cfg.training.validation_interval == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            print(f"[INFO] Generating validation images at step {global_step}...")
                            
                            # Set model to eval mode
                            pipe.unet = unwrap_model(unet)
                            pipe.unet.eval()
                            
                            # 使用预加载的固定验证样本（无增强）
                            val_conditioning_image = val_conditioning_image_fixed.to(accelerator.device, dtype=weight_dtype)
                            val_prompts = val_prompts_fixed
                            
                            if cfg.training.type == "image_only":
                                val_prompts_for_gen = ""
                            elif cfg.training.type == "single_caption":
                                val_prompts_for_gen = val_prompts[0]
                            else:  # multitext
                                val_prompts_for_gen = val_prompts
                            
                            try:
                                # ⚠️ 不要调用 .to(dtype=weight_dtype)！
                                # .to(dtype) 会创建新 Tensor 对象，断开优化器与参数的引用关系，
                                # 导致第一次验证后权重永远不再被更新。
                                # 直接用 autocast 即可——它只改变计算精度，不改变存储 dtype。
                                with torch.no_grad(), torch.amp.autocast('cuda', dtype=weight_dtype):
                                    pipeline_output = pipe(
                                        prompts=val_prompts_for_gen,
                                        conditioning_image=val_conditioning_image,
                                        num_inference_steps=30,
                                        cfg_scale=3.5,
                                        ip_adapter_image=val_ref_images_fixed if (use_ip_adapter and val_ref_images_fixed is not None) else None,
                                    )
                                
                                # Convert to PIL images
                                pil_equirec = Image.fromarray(pipeline_output.equirectangular)
                                pil_faces = [Image.fromarray(face) for face in pipeline_output.faces]
                                
                                # Save validation images locally
                                val_save_dir = os.path.join(checkpoint_dir, "validation_samples")
                                os.makedirs(val_save_dir, exist_ok=True)
                                
                                equirec_path = os.path.join(val_save_dir, f"step_{global_step}_equirec.png")
                                pil_equirec.save(equirec_path)
                                
                                for i, face in enumerate(pil_faces):
                                    face_path = os.path.join(val_save_dir, f"step_{global_step}_face_{i}.png")
                                    face.save(face_path)
                                
                                # Log to SwanLab
                                import swanlab
                                swanlab.log({
                                    "validation/equirectangular": swanlab.Image(equirec_path, caption=f"Step {global_step}"),
                                    "validation/faces": [swanlab.Image(os.path.join(val_save_dir, f"step_{global_step}_face_{i}.png"), 
                                                                       caption=val_prompts[i][:50] if i < len(val_prompts) else "") 
                                                        for i in range(6)]
                                })
                                
                                print(f"[INFO] Validation images saved to {val_save_dir}")
                                
                                del pipeline_output, pil_equirec, pil_faces
                                torch.cuda.empty_cache()
                                
                            except Exception as e:
                                print(f"[WARNING] Validation failed: {e}")
                            finally:
                                # 只需切回 train 模式，不改变 dtype
                                pipe.unet.train()

                            
                        accelerator.wait_for_everyone()


        # Epoch-based checkpointing
        if cfg.training.checkpoint_interval_type == "epochs" and (epoch + 1) % cfg.training.checkpoint_interval == 0:
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

    parser = argparse.ArgumentParser(description="Train CubeDiff with IP-Adapter")
    parser.add_argument("--config", type=str, default="multitext_ipadapter",
                        help="Name of the Hydra config to use")
    
    args, overrides = parser.parse_known_args()

    from hydra import initialize, compose
    with initialize(config_path="training/configs", version_base=None):
        cfg = compose(config_name=args.config, overrides=overrides)
        main(cfg)
