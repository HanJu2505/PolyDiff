"""
CubeDiff training script with appearance-only conditioning.

Usage:
    python train_appearance.py --config multitext_appearance
"""

import argparse
import math
import os
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.checkpoint import checkpoint

from cubediff.compat import ensure_torch_xpu_compat

ensure_torch_xpu_compat()

from diffusers import DDIMScheduler
from diffusers.utils.torch_utils import is_compiled_module
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file as safe_load_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from cubediff.modules.appearance import (
    AppearanceConditioner,
    appearance_loss_terms,
    flatten_face_tokens,
)
from cubediff.modules.attention import install_appearance_processors
from cubediff.modules.extra_channels import make_extra_channels_tensor, get_uv_tensors
from cubediff.modules.utils import expand_unet_conv_in
from cubediff.pipelines.pipeline import CubeDiffPipeline
from training.dataset import (
    CubemapDataset,
    PreExtractedCubemapDataset,
    cubemap_collate_fn,
    preextracted_cubemap_collate_fn,
)


FACE_WEIGHTS = {"left": 1.0, "right": 1.0, "top": 0.6, "bottom": 0.6, "back": 0.2}
FACE_SCALES = {"front": 0.0, "back": 0.25, "left": 1.0, "right": 1.0, "top": 0.5, "bottom": 0.5}
NON_FRONT_FACE_ORDER = ["back", "left", "right", "top", "bottom"]


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
    raise ValueError(f"Unsupported image_size for appearance conditioning: {image_size}")


def decode_latents_with_checkpoint(vae, latents: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    scaled_latents = latents / scaling_factor

    def _decode(z):
        return vae.decode(z).sample

    return checkpoint(_decode, scaled_latents, use_reentrant=False)


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
    expand_unet_conv_in(pipe.unet, in_channels=7)
    load_full_cubediff_unet_weights(pipe.unet, os.path.expanduser(cfg.training.cubediff_weights))
    install_appearance_processors(
        pipe.unet,
        base_scale=float(cfg.appearance.base_scale),
        layer_scales={
            "shallow": float(cfg.appearance.layer_scales.shallow),
            "mid": float(cfg.appearance.layer_scales.mid),
            "deep": float(cfg.appearance.layer_scales.deep),
        },
    )

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
    for param in pipe.unet.parameters():
        param.requires_grad = False
    for name, param in pipe.unet.named_parameters():
        if "attn" in name:
            param.requires_grad = True
    for param in pipe.unet.conv_in.parameters():
        param.requires_grad = False

    lowpass_kernel, lowpass_sigma = get_lowpass_params(cfg.model.image_size)
    appearance_conditioner = AppearanceConditioner(
        lowpass_kernel=lowpass_kernel,
        lowpass_sigma=lowpass_sigma,
        global_dim=int(cfg.appearance.global_dim),
        texture_dim=int(cfg.appearance.texture_dim),
        fusion_dim=int(cfg.appearance.fusion_dim),
        num_tokens=int(cfg.appearance.num_tokens),
        token_dim=int(cfg.appearance.token_dim),
        face_scales=FACE_SCALES,
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
        persistent_workers=True,
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
    step_start_time = time.time()
    epochs = cfg.training.epochs
    T = 6
    max_train_steps = int(getattr(cfg.training, "max_train_steps", 0) or 0)

    if accelerator.is_main_process:
        trainable_count = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        appearance_count = sum(p.numel() for p in appearance_conditioner.parameters() if p.requires_grad)
        print(f"[INFO] Trainable UNet params: {trainable_count:,}")
        print(f"[INFO] Trainable appearance params: {appearance_count:,}")

    for epoch in range(start_epoch, epochs):
        unet.train()
        appearance_conditioner.train()
        optimizer.zero_grad()
        total_loss = 0.0
        total_denoise_loss = 0.0
        total_app_loss = 0.0
        total_color_loss = 0.0
        total_luma_loss = 0.0
        total_feat_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", disable=not accelerator.is_main_process)
        for _, batch in enumerate(progress_bar):
            if max_train_steps > 0 and global_step >= max_train_steps:
                break
            if batch is None:
                continue

            if use_preextracted:
                cubemap_batch, prompts, _, scene_ids, single_captions = batch
            else:
                cubemap_batch, prompts, _, scene_ids, single_captions = batch

            cubemap_batch = cubemap_batch.to(accelerator.device)
            with accelerator.accumulate(unet):
                with accelerator.autocast():
                    with torch.no_grad():
                        latents = pipe.vae.encode(cubemap_batch).latent_dist.mean
                        latents = latents * pipe.vae.config.scaling_factor

                        B = latents.shape[0] // T
                        _, _, H, W = latents.shape
                        cubemap_faces = cubemap_batch.view(B, T, 3, cubemap_batch.shape[-2], cubemap_batch.shape[-1])
                        front_images = cubemap_faces[:, 0]

                        mask = torch.rand(B, device=accelerator.device) < 0.1
                        full_mask = mask.repeat_interleave(T)
                        full_mask_list = full_mask.tolist()
                        prompts = [p if not dropped else "" for p, dropped in zip(prompts, full_mask_list)]

                        if cfg.training.type == "image_only":
                            prompts = [""] * len(prompts)
                        elif cfg.training.type == "single_caption":
                            prompts = []
                            for i, sc in enumerate(single_captions):
                                prompts.extend(["" if full_mask_list[i * T] else sc] * T)

                        text_inputs = pipe.tokenizer(
                            prompts, padding="max_length", truncation=True, max_length=77, return_tensors="pt"
                        )
                        encoder_hidden_states = pipe.text_encoder(text_inputs.input_ids.to(accelerator.device))[0]

                        timesteps = torch.randint(
                            0,
                            pipe.scheduler.config.num_train_timesteps,
                            (B,),
                            device=latents.device,
                            dtype=torch.long,
                        )
                        timesteps = timesteps.repeat_interleave(T)

                        face_indices = torch.arange(B * T, device=latents.device)
                        face_ids = face_indices % T
                        non_front_mask = face_ids != 0

                        noise = torch.randn_like(latents[non_front_mask])
                        orig_latents = latents.clone()

                    appearance_out = appearance_conditioner(front_images)
                    appearance_tokens = flatten_face_tokens(appearance_out.face_tokens)

                    latents[non_front_mask] = pipe.scheduler.add_noise(
                        latents[non_front_mask], noise, timesteps[non_front_mask]
                    ).to(latents.dtype)

                    extra_channels = make_extra_channels_tensor(B, H, W).to(latents.device, dtype=latents.dtype)
                    latent_input = torch.cat([latents, extra_channels], dim=1)
                    uv_coords = get_uv_tensors(B, H, W).to(latents.device, dtype=weight_dtype)
                    front_face_drop = torch.rand(1, device=accelerator.device) < 0.1

                    model_pred = unet(
                        latent_input,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs={
                            "front_face_drop": front_face_drop,
                            "uv_coords": uv_coords,
                            "appearance_tokens": appearance_tokens,
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
                    pred_x0_faces = pred_x0.view(B, T - 1, pred_x0.shape[1], pred_x0.shape[2], pred_x0.shape[3])
                    app_loss = denoise_loss.new_tensor(0.0)
                    app_metrics = {
                        "color": denoise_loss.new_tensor(0.0),
                        "luma": denoise_loss.new_tensor(0.0),
                        "feat": denoise_loss.new_tensor(0.0),
                    }
                    for face_idx, face_name in enumerate(NON_FRONT_FACE_ORDER):
                        face_weight = float(FACE_WEIGHTS[face_name])
                        face_latents = pred_x0_faces[:, face_idx]
                        face_rgb = decode_latents_with_checkpoint(
                            pipe.vae, face_latents, pipe.vae.config.scaling_factor
                        )
                        terms = appearance_loss_terms(
                            pred_image=face_rgb,
                            front_low=appearance_out.front_low,
                            vae=pipe.vae,
                            kernel_size=lowpass_kernel,
                            sigma=lowpass_sigma,
                        )
                        weighted_face_loss = face_weight * (
                            float(cfg.appearance.loss.lambda_color) * terms["color"]
                            + float(cfg.appearance.loss.lambda_luma) * terms["luma"]
                            + float(cfg.appearance.loss.lambda_feat) * terms["feat"]
                        )
                        app_loss = app_loss + weighted_face_loss
                        app_metrics["color"] = app_metrics["color"] + face_weight * terms["color"]
                        app_metrics["luma"] = app_metrics["luma"] + face_weight * terms["luma"]
                        app_metrics["feat"] = app_metrics["feat"] + face_weight * terms["feat"]
                    loss = denoise_loss + float(cfg.appearance.loss.weight) * app_loss

                    total_loss += loss.detach().float()
                    total_denoise_loss += denoise_loss.detach().float()
                    total_app_loss += app_loss.detach().float()
                    total_color_loss += app_metrics["color"].detach().float()
                    total_luma_loss += app_metrics["luma"].detach().float()
                    total_feat_loss += app_metrics["feat"].detach().float()

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
                    avg_app = gather_metric(total_app_loss)
                    avg_color = gather_metric(total_color_loss)
                    avg_luma = gather_metric(total_luma_loss)
                    avg_feat = gather_metric(total_feat_loss)

                    total_loss = 0.0
                    total_denoise_loss = 0.0
                    total_app_loss = 0.0
                    total_color_loss = 0.0
                    total_luma_loss = 0.0
                    total_feat_loss = 0.0

                    if accelerator.is_main_process:
                        token_norm = appearance_out.shared_tokens.detach().norm(dim=-1).mean().item()
                        if swanlab_enabled:
                            import swanlab

                            swanlab.log(
                                {
                                    "loss": avg_loss.item(),
                                    "loss_denoise": avg_denoise.item(),
                                    "loss_app": avg_app.item(),
                                    "loss_app/color": avg_color.item(),
                                    "loss_app/luma": avg_luma.item(),
                                    "loss_app/feat": avg_feat.item(),
                                    "appearance_token_norm": token_norm,
                                    "z_global_norm": appearance_out.z_global.detach().norm(dim=-1).mean().item(),
                                    "z_texture_norm": appearance_out.z_texture.detach().norm(dim=-1).mean().item(),
                                    "time_per_step": time_elapsed,
                                    "learning_rate": lr_scheduler.get_last_lr()[0],
                                    "grad_norm": grad_norm.item(),
                                }
                            )
                        progress_bar.set_postfix(
                            {"loss": f"{avg_loss.item():.4f}", "app": f"{avg_app.item():.4f}", "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"}
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
                                with torch.no_grad(), torch.amp.autocast("cuda", dtype=weight_dtype):
                                    val_front = val_conditioning_image.unsqueeze(0)
                                    val_tokens = flatten_face_tokens(conditioner_eval(val_front).face_tokens)
                                    pipeline_output = pipe(
                                        prompts=val_prompts_for_gen,
                                        conditioning_image=val_conditioning_image,
                                        num_inference_steps=30,
                                        cfg_scale=3.5,
                                        cross_attention_kwargs={"appearance_tokens": val_tokens},
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
                                                for i in range(6)
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
        if swanlab_enabled:
            import swanlab

            swanlab.finish()


if __name__ == "__main__":
    os.environ["NCCL_TIMEOUT"] = "3600"
    os.environ["NCCL_DEBUG"] = "INFO"

    parser = argparse.ArgumentParser(description="Train CubeDiff with appearance-only conditioning")
    parser.add_argument("--config", type=str, default="multitext_appearance", help="Name of the Hydra config to use")
    args, overrides = parser.parse_known_args()

    from hydra import compose, initialize

    with initialize(config_path="training/configs", version_base=None):
        cfg = compose(config_name=args.config, overrides=overrides)
        main(cfg)
    swanlab_enabled = bool(getattr(cfg.swanlab, "enabled", True))
