"""
CubeDiff training script with front-only global style conditioning.

Usage:
    python train_appearance.py --config multitext_appearance
"""

import argparse
import math
import os
import time
from contextlib import nullcontext
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import set_seed

from cubediff.compat import ensure_torch_xpu_compat

ensure_torch_xpu_compat()

from diffusers import DDIMScheduler
from diffusers.utils.torch_utils import is_compiled_module
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file as safe_load_file
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm

from cubediff.modules.appearance import (
    FACE_ORDER,
    NON_FRONT_FACE_ORDER,
    FrontGlobalStyleConditioner,
    compute_style_loss_terms,
    expand_style_cond,
    flatten_style_cond,
    gaussian_blur_2d,
    make_style_scale_tensor,
)
from cubediff.modules.attention import install_global_style_processors, install_trainable_attn1_processors
from cubediff.modules.extra_channels import get_uv_tensors, make_extra_channels_tensor
from cubediff.pipelines.pipeline import CubeDiffPipeline
from training.dataset import (
    CubemapDataset,
    PreExtractedCubemapDataset,
    cubemap_collate_fn,
    preextracted_cubemap_collate_fn,
)


DEFAULT_FACE_SCALES = {
    "front": 0.90,
    "back": 0.15,
    "left": 0.45,
    "right": 0.45,
    "top": 0.10,
    "bottom": 0.10,
}


def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        return safe_load_file(path, device="cpu")
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload["state_dict"]
    return payload


def load_full_cubediff_unet_weights(unet: nn.Module, ckpt_path: str) -> None:
    state_dict = load_state_dict(ckpt_path)
    if "conv_in.weight" not in state_dict:
        unet_prefixed = {k[len("unet.") :]: v for k, v in state_dict.items() if k.startswith("unet.")}
        if "conv_in.weight" in unet_prefixed:
            state_dict = unet_prefixed

    conv_weight = state_dict.get("conv_in.weight")
    if conv_weight is None or conv_weight.shape[1] != 7:
        raise ValueError(
            f"Expected a 7-channel CubeDiff checkpoint, got conv_in.weight={None if conv_weight is None else tuple(conv_weight.shape)}"
        )

    if unet.conv_in.in_channels != conv_weight.shape[1]:
        old_conv = unet.conv_in
        unet.conv_in = nn.Conv2d(
            in_channels=conv_weight.shape[1],
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

    missing, unexpected = unet.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[CubeDiff] Missing keys when loading UNet: {len(missing)}")
    if unexpected:
        print(f"[CubeDiff] Unexpected keys when loading UNet: {len(unexpected)}")


def reconstruct_x0_from_prediction(
    scheduler: DDIMScheduler,
    prediction: torch.Tensor,
    latents_noisy: torch.Tensor,
    timesteps: torch.Tensor,
    prediction_type: str,
) -> torch.Tensor:
    alpha_t = scheduler.alphas_cumprod.to(device=latents_noisy.device, dtype=latents_noisy.dtype)[timesteps]
    alpha_t = alpha_t.view(-1, 1, 1, 1)
    sigma_t = (1.0 - alpha_t).clamp_min(0).sqrt()
    sqrt_alpha = alpha_t.sqrt()

    if prediction_type == "epsilon":
        return (latents_noisy - sigma_t * prediction) / sqrt_alpha.clamp_min(1e-6)
    if prediction_type == "v_prediction":
        return sqrt_alpha * latents_noisy - sigma_t * prediction
    raise ValueError(f"Unsupported prediction_type: {prediction_type}")


def get_lowpass_params(image_size: int):
    if image_size == 512:
        return 21, 5.0
    if image_size == 768:
        return 31, 7.5
    raise ValueError(f"Unsupported image_size for style conditioning: {image_size}")


def decode_latents_with_checkpoint(vae, latents: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    scaled_latents = latents / scaling_factor

    def _decode(z):
        return vae.decode(z).sample

    return checkpoint(_decode, scaled_latents, use_reentrant=False)


def get_front_prompts(prompts: list[str], batch_size: int, num_faces: int) -> list[str]:
    if len(prompts) != batch_size * num_faces:
        raise ValueError(f"Expected {batch_size * num_faces} prompts, got {len(prompts)}")
    return [prompts[i * num_faces] for i in range(batch_size)]


def build_sd_prompts(
    *,
    prompts: list[str],
    single_captions: list[str],
    drop_mask: torch.Tensor,
    training_type: str,
    num_faces: int,
) -> list[str]:
    full_mask = drop_mask.repeat_interleave(num_faces).tolist()
    if training_type == "image_only":
        return [""] * len(prompts)
    if training_type == "single_caption":
        output = []
        for index, caption in enumerate(single_captions):
            value = "" if bool(drop_mask[index]) else caption
            output.extend([value] * num_faces)
        return output
    return [prompt if not dropped else "" for prompt, dropped in zip(prompts, full_mask)]


def resolve_time_scale(timesteps_b: torch.Tensor, scheduler: DDIMScheduler, split: float, early: float, late: float) -> torch.Tensor:
    max_t = max(1, scheduler.config.num_train_timesteps - 1)
    progress = 1.0 - timesteps_b.float() / max_t
    return torch.where(progress < split, progress.new_full((), early), progress.new_full((), late))


def freeze_and_select_trainable_params(unet: nn.Module) -> None:
    for param in unet.parameters():
        param.requires_grad = False

    for name, param in unet.named_parameters():
        if ".attn1.processor." in name or ".attn1.to_q." in name or ".attn1.to_k." in name:
            param.requires_grad = True
        if name.startswith("up_blocks.1") and ".attn2.processor." in name:
            param.requires_grad = True

    for param in unet.conv_in.parameters():
        param.requires_grad = False


def main(cfg: DictConfig):
    num_gpus = cfg.training.num_gpus
    per_gpu_batch_size = cfg.training.per_gpu_batch_size
    global_batch_size = cfg.training.batch_size
    acc_steps = global_batch_size // (per_gpu_batch_size * num_gpus)
    swanlab_enabled = bool(getattr(cfg.swanlab, "enabled", True))

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
        if swanlab_enabled:
            import swanlab

            swanlab.init(
                project=cfg.swanlab.project_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                experiment_name=cfg.name,
                mode="cloud" if cfg.swanlab.use_cloud else "local",
            )

    rank = accelerator.process_index
    base_seed = cfg.training.seed
    torch.manual_seed(base_seed + rank)
    np.random.seed(base_seed + rank)
    set_seed(base_seed, device_specific=True)

    cache_dir = os.path.expanduser(cfg.directories.cache_dir)
    pipe = CubeDiffPipeline.from_pretrained(cfg.model.id, cache_dir=cache_dir, local_files_only=False)
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    load_full_cubediff_unet_weights(pipe.unet, os.path.expanduser(cfg.training.cubediff_weights))
    install_trainable_attn1_processors(pipe.unet)
    install_global_style_processors(pipe.unet, module_prefix=str(cfg.appearance.style_block))

    if cfg.training.prediction_type == "v_prediction":
        pipe.scheduler = DDIMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            subfolder="scheduler",
            cache_dir=cache_dir,
            local_files_only=True,
        )
        pipe.scheduler.config.prediction_type = "v_prediction"
    elif cfg.training.prediction_type == "epsilon":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.config.prediction_type = "epsilon"
    else:
        raise ValueError(f"Invalid prediction_type: {cfg.training.prediction_type}")

    if getattr(cfg.training, "use_checkpointing", False):
        pipe.unet.enable_gradient_checkpointing()

    for param in pipe.vae.parameters():
        param.requires_grad = False
    for param in pipe.text_encoder.parameters():
        param.requires_grad = False
    freeze_and_select_trainable_params(pipe.unet)

    lowpass_kernel, lowpass_sigma = get_lowpass_params(cfg.model.image_size)
    appearance_conditioner = FrontGlobalStyleConditioner(
        clip_model_id=str(cfg.appearance.clip_model_id),
        style_dim=int(cfg.appearance.style_dim),
        beta=float(cfg.appearance.beta),
        cache_dir=cache_dir,
        local_files_only=bool(getattr(cfg.appearance, "local_files_only", False)),
    )

    use_preextracted = getattr(cfg.training, "use_preextracted", False)
    if use_preextracted:
        dataset = PreExtractedCubemapDataset(
            cubemap_dir=cfg.directories.cubemap_dir,
            prompt_dir=cfg.directories.prompt_dir,
            face_size=cfg.model.image_size,
            augment=getattr(cfg.training, "augment", True),
            return_ref_images=False,
        )
        collate_fn = preextracted_cubemap_collate_fn
        val_dataset = PreExtractedCubemapDataset(
            cubemap_dir=cfg.directories.cubemap_dir,
            prompt_dir=cfg.directories.prompt_dir,
            face_size=cfg.model.image_size,
            augment=False,
            return_ref_images=False,
        )
    else:
        dataset = CubemapDataset(
            root_dir=cfg.directories.data_dir,
            face_size=cfg.model.image_size,
            fov=95,
            checkpoint_dir=cfg.directories.checkpoint_dir,
            use_cached_data=False,
        )
        collate_fn = cubemap_collate_fn
        val_dataset = dataset

    val_sample_fixed = val_dataset[0]
    val_conditioning_image_fixed = val_sample_fixed[0][0]
    val_prompts_fixed = val_sample_fixed[1]

    dataloader = DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        persistent_workers=cfg.training.num_workers > 0,
        drop_last=True,
    )

    params = [p for p in pipe.unet.parameters() if p.requires_grad]
    params.extend(list(appearance_conditioner.parameters()))
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg.training.learning_rate,
        betas=tuple(cfg.training.betas),
        eps=cfg.training.eps,
    )

    steps_per_epoch = max(1, len(dataloader) // acc_steps)
    total_steps = cfg.training.epochs * steps_per_epoch
    min_lr = 1e-5

    def lr_lambda_cosine(current_step):
        warmup = cfg.training.warmup_steps
        if current_step < warmup:
            return current_step / max(1, warmup)
        progress = (current_step - warmup) / max(1, total_steps - warmup)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = min_lr / cfg.training.learning_rate
        return min_ratio + (1.0 - min_ratio) * cosine_decay

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_cosine)
    unet, appearance_conditioner, optimizer, dataloader = accelerator.prepare(
        pipe.unet, appearance_conditioner, optimizer, dataloader
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipe.vae.to(accelerator.device, dtype=weight_dtype)
    pipe.text_encoder.to(accelerator.device, dtype=weight_dtype)
    unwrap_model(appearance_conditioner).move_backbone(device=accelerator.device, dtype=weight_dtype)

    global_step = 0
    start_epoch = 0
    if cfg.training.resume_checkpoint:
        resume_ckpt = os.path.join(cfg.directories.checkpoint_dir, cfg.training.resume_checkpoint_filename)
        if os.path.exists(resume_ckpt):
            start_epoch = int(os.path.basename(resume_ckpt).split("epoch_")[-1].split("_")[0])
            global_step = int(os.path.basename(resume_ckpt).split("step_")[-1].split("_")[0])
            accelerator.load_state(resume_ckpt)
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_cosine, last_epoch=global_step - 1)

    checkpoint_dir = cfg.directories.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    epochs = cfg.training.epochs
    step_start_time = time.time()
    num_faces = len(FACE_ORDER)
    max_train_steps = int(getattr(cfg.training, "max_train_steps", 0) or 0)
    lambda_style = float(cfg.appearance.loss.weight)
    lambda_color = float(cfg.appearance.loss.lambda_color)
    lambda_luma = float(cfg.appearance.loss.lambda_luma)
    time_split = float(cfg.appearance.time_schedule.split)
    time_early = float(cfg.appearance.time_schedule.early)
    time_late = float(cfg.appearance.time_schedule.late)
    face_scales = {
        name: float(getattr(cfg.appearance.face_scales, name, DEFAULT_FACE_SCALES[name]))
        for name in FACE_ORDER
    }
    style_loss_weights = {name: face_scales[name] for name in NON_FRONT_FACE_ORDER}

    if accelerator.is_main_process:
        trainable_unet = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        trainable_style = sum(p.numel() for p in appearance_conditioner.parameters() if p.requires_grad)
        print(f"[INFO] Trainable UNet params: {trainable_unet:,}")
        print(f"[INFO] Trainable appearance params: {trainable_style:,}")

    for epoch in range(start_epoch, epochs):
        unet.train()
        appearance_conditioner.train()
        optimizer.zero_grad()
        total_loss = 0.0
        total_denoise_loss = 0.0
        total_style_loss = 0.0
        total_color_loss = 0.0
        total_luma_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", disable=not accelerator.is_main_process)
        for _, batch in enumerate(progress_bar):
            if max_train_steps > 0 and global_step >= max_train_steps:
                break
            if batch is None:
                continue

            cubemap_batch, prompts, _, scene_ids, single_captions = batch
            cubemap_batch = cubemap_batch.to(accelerator.device)

            with accelerator.accumulate(unet):
                autocast_context = (
                    accelerator.autocast()
                    if accelerator.device.type == "cuda"
                    else nullcontext()
                )
                with autocast_context:
                    with torch.no_grad():
                        latents = pipe.vae.encode(cubemap_batch).latent_dist.mean
                        latents = latents * pipe.vae.config.scaling_factor

                        batch_size = latents.shape[0] // num_faces
                        _, _, height, width = latents.shape
                        cubemap_faces = cubemap_batch.view(batch_size, num_faces, 3, cubemap_batch.shape[-2], cubemap_batch.shape[-1])
                        front_images = cubemap_faces[:, 0]
                        front_prompts = get_front_prompts(prompts, batch_size, num_faces)

                        drop_mask = torch.rand(batch_size, device=accelerator.device) < 0.1
                        sd_prompts = build_sd_prompts(
                            prompts=prompts,
                            single_captions=single_captions,
                            drop_mask=drop_mask,
                            training_type=str(cfg.training.type),
                            num_faces=num_faces,
                        )

                        text_inputs = pipe.tokenizer(
                            sd_prompts,
                            padding="max_length",
                            truncation=True,
                            max_length=77,
                            return_tensors="pt",
                        )
                        encoder_hidden_states = pipe.text_encoder(text_inputs.input_ids.to(accelerator.device))[0]

                        timesteps_b = torch.randint(
                            0,
                            pipe.scheduler.config.num_train_timesteps,
                            (batch_size,),
                            device=latents.device,
                            dtype=torch.long,
                        )
                        timesteps = timesteps_b.repeat_interleave(num_faces)

                        face_indices = torch.arange(batch_size * num_faces, device=latents.device)
                        face_ids = face_indices % num_faces
                        non_front_mask = face_ids != 0

                        noise = torch.randn_like(latents[non_front_mask])
                        orig_latents = latents.clone()
                        front_low = gaussian_blur_2d(front_images, lowpass_kernel, lowpass_sigma)

                    style_out = appearance_conditioner(front_images, front_prompts)
                    style_faces = expand_style_cond(style_out.style_cond)
                    style_cond = flatten_style_cond(style_faces)
                    keep_mask = (~drop_mask).repeat_interleave(num_faces).view(batch_size * num_faces, 1, 1).to(style_cond.dtype)
                    style_cond = style_cond * keep_mask

                    time_scale = resolve_time_scale(
                        timesteps_b,
                        pipe.scheduler,
                        split=time_split,
                        early=time_early,
                        late=time_late,
                    )
                    style_scale = make_style_scale_tensor(
                        batch_size,
                        face_scales=face_scales,
                        time_scale=time_scale,
                        device=latents.device,
                        dtype=latents.dtype,
                    )
                    style_scale = style_scale * (~drop_mask).repeat_interleave(num_faces).to(style_scale.dtype)

                    latents[non_front_mask] = pipe.scheduler.add_noise(
                        latents[non_front_mask], noise, timesteps[non_front_mask]
                    ).to(latents.dtype)

                    extra_channels = make_extra_channels_tensor(batch_size, height, width).to(
                        latents.device, dtype=latents.dtype
                    )
                    latent_input = torch.cat([latents, extra_channels], dim=1)
                    uv_coords = get_uv_tensors(batch_size, height, width).to(latents.device, dtype=weight_dtype)
                    front_face_drop = bool(torch.rand(1, device=accelerator.device).item() < 0.1)

                    model_pred = unet(
                        latent_input,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs={
                            "front_face_drop": front_face_drop,
                            "uv_coords": uv_coords,
                            "style_cond": style_cond,
                            "style_scale": style_scale,
                        },
                    ).sample

                    if cfg.training.prediction_type == "v_prediction":
                        denoise_target = pipe.scheduler.get_velocity(
                            orig_latents[non_front_mask], noise, timesteps[non_front_mask]
                        )
                    else:
                        denoise_target = noise
                    denoise_loss = nn.functional.mse_loss(model_pred[non_front_mask], denoise_target)

                    pred_x0 = reconstruct_x0_from_prediction(
                        pipe.scheduler,
                        model_pred[non_front_mask],
                        latents[non_front_mask],
                        timesteps[non_front_mask],
                        cfg.training.prediction_type,
                    )
                    pred_x0_faces = pred_x0.view(batch_size, num_faces - 1, pred_x0.shape[1], pred_x0.shape[2], pred_x0.shape[3])
                    style_loss = denoise_loss.new_tensor(0.0)
                    style_metrics = {
                        "color": denoise_loss.new_tensor(0.0),
                        "luma": denoise_loss.new_tensor(0.0),
                    }
                    for face_idx, face_name in enumerate(NON_FRONT_FACE_ORDER):
                        face_weight = float(style_loss_weights[face_name])
                        face_latents = pred_x0_faces[:, face_idx]
                        face_rgb = decode_latents_with_checkpoint(
                            pipe.vae, face_latents, pipe.vae.config.scaling_factor
                        )
                        terms = compute_style_loss_terms(
                            pred_image=face_rgb,
                            front_low=front_low,
                            kernel_size=lowpass_kernel,
                            sigma=lowpass_sigma,
                        )
                        face_style = face_weight * (
                            lambda_color * terms["color"] + lambda_luma * terms["luma"]
                        )
                        style_loss = style_loss + face_style
                        style_metrics["color"] = style_metrics["color"] + lambda_style * face_weight * lambda_color * terms["color"]
                        style_metrics["luma"] = style_metrics["luma"] + lambda_style * face_weight * lambda_luma * terms["luma"]

                    loss = denoise_loss + lambda_style * style_loss

                    total_loss += loss.detach().float()
                    total_denoise_loss += denoise_loss.detach().float()
                    total_style_loss += style_loss.detach().float()
                    total_color_loss += style_metrics["color"].detach().float()
                    total_luma_loss += style_metrics["luma"].detach().float()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    trainable_params = [p for p in unet.parameters() if p.requires_grad]
                    trainable_params.extend([p for p in appearance_conditioner.parameters() if p.requires_grad])
                    grad_norm = accelerator.clip_grad_norm_(trainable_params, max_norm=1.0)

                optimizer.step()
                if accelerator.sync_gradients:
                    lr_scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1
                    time_elapsed = time.time() - step_start_time
                    step_start_time = time.time()

                    def gather_metric(value: torch.Tensor):
                        value = value / accelerator.gradient_accumulation_steps
                        gathered = accelerator.gather_for_metrics(value.unsqueeze(0))
                        return torch.mean(gathered)

                    avg_loss = gather_metric(total_loss)
                    avg_denoise = gather_metric(total_denoise_loss)
                    avg_style = gather_metric(total_style_loss)
                    avg_color = gather_metric(total_color_loss)
                    avg_luma = gather_metric(total_luma_loss)

                    total_loss = 0.0
                    total_denoise_loss = 0.0
                    total_style_loss = 0.0
                    total_color_loss = 0.0
                    total_luma_loss = 0.0

                    if accelerator.is_main_process:
                        style_norm = style_out.style_cond.detach().norm(dim=-1).mean().item()
                        if swanlab_enabled:
                            import swanlab

                            swanlab.log(
                                {
                                    "loss": avg_loss.item(),
                                    "loss_denoise": avg_denoise.item(),
                                    "loss_style": avg_style.item(),
                                    "loss_style/color": avg_color.item(),
                                    "loss_style/luma": avg_luma.item(),
                                    "style_cond_norm": style_norm,
                                    "time_per_step": time_elapsed,
                                    "learning_rate": lr_scheduler.get_last_lr()[0],
                                    "grad_norm": grad_norm.item(),
                                }
                            )
                        progress_bar.set_postfix(
                            {"loss": f"{avg_loss.item():.4f}", "style": f"{avg_style.item():.4f}", "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"}
                        )

                    if cfg.training.checkpoint_interval_type == "steps" and global_step % cfg.training.checkpoint_interval == 0:
                        ckpt_folder = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}_step_{global_step}")
                        accelerator.save_state(ckpt_folder)
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            print(f"[INFO] Checkpoint saved: {ckpt_folder}")

                    if cfg.training.checkpoint_interval_type == "steps" and global_step % cfg.training.validation_interval == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            pipe.unet = unwrap_model(unet)
                            pipe.unet.eval()
                            conditioner_eval = unwrap_model(appearance_conditioner)
                            conditioner_eval.eval()

                            val_conditioning_image = val_conditioning_image_fixed.to(accelerator.device, dtype=weight_dtype)
                            val_prompts = val_prompts_fixed
                            if cfg.training.type == "image_only":
                                val_prompts_for_gen = ""
                            elif cfg.training.type == "single_caption":
                                val_prompts_for_gen = val_prompts[0]
                            else:
                                val_prompts_for_gen = val_prompts

                            try:
                                amp_context = (
                                    torch.amp.autocast(accelerator.device.type, dtype=weight_dtype)
                                    if accelerator.device.type == "cuda"
                                    else nullcontext()
                                )
                                with torch.no_grad(), amp_context:
                                    val_front = val_conditioning_image.unsqueeze(0)
                                    val_style = conditioner_eval(val_front, [val_prompts[0]])
                                    val_style_cond = flatten_style_cond(expand_style_cond(val_style.style_cond))
                                    val_style_scale = make_style_scale_tensor(
                                        1,
                                        face_scales=face_scales,
                                        time_scale=1.0,
                                        device=accelerator.device,
                                        dtype=weight_dtype,
                                    )
                                    pipeline_output = pipe(
                                        prompts=val_prompts_for_gen,
                                        conditioning_image=val_conditioning_image,
                                        num_inference_steps=30,
                                        cfg_scale=3.5,
                                        cross_attention_kwargs={
                                            "style_cond": val_style_cond,
                                            "style_scale": val_style_scale,
                                        },
                                    )

                                pil_equirec = Image.fromarray(pipeline_output.equirectangular)
                                pil_faces = [Image.fromarray(face) for face in pipeline_output.faces]
                                val_save_dir = os.path.join(checkpoint_dir, "validation_samples")
                                os.makedirs(val_save_dir, exist_ok=True)

                                equirec_path = os.path.join(val_save_dir, f"step_{global_step}_equirec.png")
                                pil_equirec.save(equirec_path)
                                for i, face in enumerate(pil_faces):
                                    face.save(os.path.join(val_save_dir, f"step_{global_step}_face_{i}.png"))

                                if swanlab_enabled:
                                    import swanlab

                                    swanlab.log(
                                        {
                                            "validation/equirectangular": swanlab.Image(equirec_path, caption=f"Step {global_step}"),
                                            "validation/faces": [
                                                swanlab.Image(
                                                    os.path.join(val_save_dir, f"step_{global_step}_face_{i}.png"),
                                                    caption=val_prompts[i][:50] if i < len(val_prompts) else "",
                                                )
                                                for i in range(num_faces)
                                            ],
                                        }
                                    )
                            except Exception as exc:
                                print(f"[WARNING] Validation failed: {exc}")
                            finally:
                                pipe.unet.train()
                                conditioner_eval.train()

                        accelerator.wait_for_everyone()

        if max_train_steps > 0 and global_step >= max_train_steps:
            break

        if cfg.training.checkpoint_interval_type == "epochs" and (epoch + 1) % cfg.training.checkpoint_interval == 0:
            ckpt_folder = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}_step_{global_step}")
            accelerator.save_state(ckpt_folder)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                print(f"[INFO] Checkpoint saved: {ckpt_folder}")

    final_ckpt = os.path.join(checkpoint_dir, f"epoch_{epochs}_step_{global_step}_final")
    accelerator.save_state(final_ckpt)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"[INFO] Final checkpoint saved: {final_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="multitext_appearance")
    args = parser.parse_args()

    config_name = args.config.replace(".yaml", "")
    config_path = os.path.join("training", "configs", f"{config_name}.yaml")
    cfg = OmegaConf.load(config_path)
    main(cfg)
