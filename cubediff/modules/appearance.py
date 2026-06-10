from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


FACE_ORDER = ["front", "back", "left", "right", "top", "bottom"]
NON_FRONT_FACE_ORDER = ["back", "left", "right", "top", "bottom"]
FACE_INDEX = {name: idx for idx, name in enumerate(FACE_ORDER)}


def _make_gaussian_kernel(kernel_size: int, sigma: float, device, dtype):
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-6)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum().clamp_min(1e-6)
    return kernel_2d


def gaussian_blur_2d(images: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")

    kernel = _make_gaussian_kernel(kernel_size, sigma, images.device, images.dtype)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(images.shape[1], 1, 1, 1)
    padding = kernel_size // 2
    padded = F.pad(images, (padding, padding, padding, padding), mode="reflect")
    return F.conv2d(padded, kernel, groups=images.shape[1])


def per_channel_mean_std(images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = images.mean(dim=(-2, -1))
    std = images.std(dim=(-2, -1), unbiased=False)
    return mean, std


def luminance_mean_std(images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    weights = images.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    luminance = (images * weights).sum(dim=1, keepdim=True)
    mean = luminance.mean(dim=(-2, -1))
    std = luminance.std(dim=(-2, -1), unbiased=False)
    return mean, std


def stats_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    src_mean, src_std = per_channel_mean_std(source)
    tgt_mean, tgt_std = per_channel_mean_std(target)
    return F.l1_loss(src_mean, tgt_mean) + F.l1_loss(src_std, tgt_std)


def luminance_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    src_mean, src_std = luminance_mean_std(source)
    tgt_mean, tgt_std = luminance_mean_std(target)
    return F.l1_loss(src_mean, tgt_mean) + F.l1_loss(src_std, tgt_std)


def compute_style_loss_terms(
    *,
    pred_image: torch.Tensor,
    front_low: torch.Tensor,
    kernel_size: int,
    sigma: float,
) -> Dict[str, torch.Tensor]:
    pred_low = gaussian_blur_2d(pred_image, kernel_size, sigma)
    return {
        "color": stats_loss(pred_low, front_low),
        "luma": luminance_loss(pred_low, front_low),
    }


def expand_style_cond(style_cond: torch.Tensor) -> torch.Tensor:
    if style_cond.ndim != 2:
        raise ValueError(f"Expected style_cond [B, D], got {tuple(style_cond.shape)}")
    bsz, dim = style_cond.shape
    return style_cond.view(bsz, 1, 1, dim).expand(bsz, len(FACE_ORDER), 1, dim)


def flatten_style_cond(style_cond_faces: torch.Tensor) -> torch.Tensor:
    if style_cond_faces.ndim != 4:
        raise ValueError(f"Expected style_cond_faces [B, 6, 1, D], got {tuple(style_cond_faces.shape)}")
    bsz, num_faces, num_tokens, dim = style_cond_faces.shape
    if num_faces != len(FACE_ORDER) or num_tokens != 1:
        raise ValueError(f"Expected [B, 6, 1, D], got {tuple(style_cond_faces.shape)}")
    return style_cond_faces.reshape(bsz * num_faces, num_tokens, dim)


def make_style_scale_tensor(
    batch_size: int,
    *,
    face_scales: Dict[str, float],
    time_scale: torch.Tensor | float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    face_values = torch.tensor([face_scales[name] for name in FACE_ORDER], device=device, dtype=dtype).view(1, -1)
    if not torch.is_tensor(time_scale):
        time_scale = torch.full((batch_size, 1), float(time_scale), device=device, dtype=dtype)
    else:
        time_scale = time_scale.to(device=device, dtype=dtype).view(batch_size, 1)
    return (time_scale * face_values).reshape(batch_size * len(FACE_ORDER))


@dataclass
class FrontGlobalStyleOutput:
    e_img_global: torch.Tensor
    e_txt_global: torch.Tensor
    s0: torch.Tensor
    style_raw: torch.Tensor
    style_cond: torch.Tensor
    style_gain: torch.Tensor


class FrontGlobalStyleConditioner(nn.Module):
    def __init__(
        self,
        *,
        clip_model_id: str = "openai/clip-vit-large-patch14",
        style_dim: int = 768,
        beta: float = 0.6,
        style_gain_init: float = 4.0,
        style_gain_max: float = 8.0,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.clip_model_id = clip_model_id
        self.beta = float(beta)
        self.style_dim = int(style_dim)
        self.style_gain_max = float(style_gain_max)
        if self.style_gain_max <= 0:
            raise ValueError(f"style_gain_max must be positive, got {self.style_gain_max}")
        self.mlp = nn.Sequential(
            nn.Linear(self.style_dim, self.style_dim),
            nn.LayerNorm(self.style_dim),
            nn.SiLU(),
            nn.Linear(self.style_dim, self.style_dim),
        )
        init_ratio = min(max(float(style_gain_init) / self.style_gain_max, 1e-4), 1.0 - 1e-4)
        init_logit = torch.logit(torch.tensor(init_ratio, dtype=torch.float32))
        self.style_gain_logit = nn.Parameter(init_logit)

        clip_processor = CLIPProcessor.from_pretrained(
            clip_model_id,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        clip_model = CLIPModel.from_pretrained(
            clip_model_id,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        clip_model.eval()
        for param in clip_model.parameters():
            param.requires_grad = False

        self.__dict__["clip_processor"] = clip_processor
        self.__dict__["clip_model"] = clip_model
        self.__dict__["_clip_device"] = torch.device("cpu")
        self.__dict__["_clip_dtype"] = torch.float32

    @property
    def clip_model(self) -> CLIPModel:
        return self.__dict__["clip_model"]

    @property
    def clip_processor(self) -> CLIPProcessor:
        return self.__dict__["clip_processor"]

    def move_backbone(self, *, device: torch.device, dtype: torch.dtype) -> None:
        target_dtype = dtype if device.type != "cpu" else torch.float32
        self.__dict__["clip_model"] = self.clip_model.to(device=device, dtype=target_dtype)
        self.clip_model.eval()
        self.__dict__["_clip_device"] = device
        self.__dict__["_clip_dtype"] = target_dtype

    def _prepare_images(self, front_image: torch.Tensor) -> torch.Tensor:
        image_processor = self.clip_processor.image_processor
        crop_size = image_processor.crop_size
        target_h = crop_size["height"] if isinstance(crop_size, dict) else int(crop_size)
        target_w = crop_size["width"] if isinstance(crop_size, dict) else int(crop_size)
        images = ((front_image.float() / 2) + 0.5).clamp(0, 1)
        images = F.interpolate(images, size=(target_h, target_w), mode="bicubic", align_corners=False)
        mean = torch.tensor(image_processor.image_mean, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
        std = torch.tensor(image_processor.image_std, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
        return (images - mean) / std

    def _encode_image(self, front_image: torch.Tensor) -> torch.Tensor:
        pixel_values = self._prepare_images(front_image).to(
            device=self.__dict__["_clip_device"],
            dtype=self.__dict__["_clip_dtype"],
        )
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
        return F.normalize(image_features.float(), dim=-1)

    def _encode_text(self, prompts: List[str]) -> torch.Tensor:
        text_inputs = self.clip_processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        text_inputs = {k: v.to(self.__dict__["_clip_device"]) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**text_inputs)
        return F.normalize(text_features.float(), dim=-1)

    def get_style_gain(self, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
        style_gain = self.style_gain_max * torch.sigmoid(self.style_gain_logit)
        if device is not None or dtype is not None:
            style_gain = style_gain.to(
                device=device if device is not None else style_gain.device,
                dtype=dtype if dtype is not None else style_gain.dtype,
            )
        return style_gain

    def forward(self, front_image: torch.Tensor, prompts: List[str]) -> FrontGlobalStyleOutput:
        if front_image.ndim != 4:
            raise ValueError(f"Expected front_image [B, 3, H, W], got {tuple(front_image.shape)}")
        if len(prompts) != front_image.shape[0]:
            raise ValueError(f"Expected {front_image.shape[0]} front prompts, got {len(prompts)}")

        e_img_global = self._encode_image(front_image)
        e_txt_global = self._encode_text(prompts)
        s_raw = e_img_global - self.beta * e_txt_global
        s0 = F.normalize(s_raw, dim=-1)
        style_raw = self.mlp(s0.to(device=front_image.device, dtype=front_image.dtype))
        style_unit = F.normalize(style_raw.float(), dim=-1).to(device=front_image.device, dtype=front_image.dtype)
        style_gain = self.get_style_gain(device=front_image.device, dtype=front_image.dtype)
        style_cond = style_gain * style_unit

        return FrontGlobalStyleOutput(
            e_img_global=e_img_global.to(device=front_image.device, dtype=front_image.dtype),
            e_txt_global=e_txt_global.to(device=front_image.device, dtype=front_image.dtype),
            s0=s0.to(device=front_image.device, dtype=front_image.dtype),
            style_raw=style_raw,
            style_cond=style_cond,
            style_gain=style_gain,
        )
