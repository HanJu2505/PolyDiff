from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(images.shape[1], 1, 1, 1)
    padding = kernel_size // 2
    padded = F.pad(images, (padding, padding, padding, padding), mode="reflect")
    return F.conv2d(padded, kernel, groups=images.shape[1])


def normalize_high_frequency(hf: torch.Tensor) -> torch.Tensor:
    mean = hf.mean(dim=(-2, -1), keepdim=True)
    std = hf.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-6)
    standardized = (hf - mean) / std
    return standardized.clamp(-3.0, 3.0) / 3.0


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GlobalAppearanceEncoder(nn.Module):
    def __init__(self, output_dim: int = 512):
        super().__init__()
        self.stem = ConvBlock(3, 32, stride=1)
        self.stage2 = ConvBlock(32, 64, stride=2)
        self.stage3 = ConvBlock(64, 128, stride=2)
        stats_dim = 2 * (32 + 64 + 128)
        self.proj = nn.Sequential(
            nn.Linear(stats_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    @staticmethod
    def _stats_pool(feat: torch.Tensor) -> torch.Tensor:
        mean = feat.mean(dim=(-2, -1))
        std = feat.std(dim=(-2, -1), unbiased=False)
        return torch.cat([mean, std], dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        f1 = self.stem(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        pooled = torch.cat([self._stats_pool(f) for f in (f1, f2, f3)], dim=-1)
        return self.proj(pooled), [f1, f2, f3]


class LocalTextureEncoder(nn.Module):
    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(3, 16, stride=1),
            ConvBlock(16, 32, stride=2),
            ConvBlock(32, 48, stride=2),
            ConvBlock(48, 64, stride=2),
        )
        self.proj = nn.Sequential(
            nn.Linear(64, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        pooled = feat.mean(dim=(-2, -1))
        return self.proj(pooled)


class AppearanceFusion(nn.Module):
    def __init__(self, global_dim: int = 512, texture_dim: int = 128, output_dim: int = 512, texture_weight: float = 0.4):
        super().__init__()
        self.texture_weight = texture_weight
        self.mlp = nn.Sequential(
            nn.Linear(global_dim + texture_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, z_global: torch.Tensor, z_texture: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([z_global, self.texture_weight * z_texture], dim=-1)
        return self.mlp(fused)


class AppearanceTokenGenerator(nn.Module):
    def __init__(self, input_dim: int = 512, num_tokens: int = 4, token_dim: int = 768):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, num_tokens * token_dim),
        )

    def forward(self, z_app: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(z_app)
        return tokens.view(z_app.shape[0], self.num_tokens, self.token_dim)


class FaceAwareModulation(nn.Module):
    def __init__(
        self,
        num_tokens: int = 4,
        token_dim: int = 768,
        face_scales: Dict[str, float] | None = None,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.face_bias = nn.Parameter(torch.zeros(len(FACE_ORDER), num_tokens, token_dim))
        self.face_gate = nn.Parameter(torch.zeros(len(FACE_ORDER), num_tokens, 1))
        if face_scales is None:
            face_scales = {
                "front": 0.0,
                "back": 0.25,
                "left": 1.0,
                "right": 1.0,
                "top": 0.5,
                "bottom": 0.5,
            }
        scales = [face_scales[name] for name in FACE_ORDER]
        self.register_buffer("face_scales", torch.tensor(scales, dtype=torch.float32).view(1, len(FACE_ORDER), 1, 1))

    def forward(self, shared_tokens: torch.Tensor) -> torch.Tensor:
        tokens = shared_tokens.unsqueeze(1) + self.face_bias.unsqueeze(0)
        gates = torch.sigmoid(self.face_gate).unsqueeze(0)
        tokens = tokens * gates * self.face_scales.to(tokens.dtype)
        return tokens


class ContentSuppressionFrontEnd(nn.Module):
    def __init__(self, kernel_size: int, sigma: float):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)

    def forward(self, front_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        low = gaussian_blur_2d(front_image, self.kernel_size, self.sigma)
        high = front_image - low
        high_norm = normalize_high_frequency(high)
        return low, high_norm


@dataclass
class AppearanceConditioningOutput:
    front_low: torch.Tensor
    high_freq_norm: torch.Tensor
    z_global: torch.Tensor
    z_texture: torch.Tensor
    z_app: torch.Tensor
    shared_tokens: torch.Tensor
    face_tokens: torch.Tensor


class AppearanceConditioner(nn.Module):
    def __init__(
        self,
        *,
        lowpass_kernel: int = 21,
        lowpass_sigma: float = 5.0,
        global_dim: int = 512,
        texture_dim: int = 128,
        fusion_dim: int = 512,
        num_tokens: int = 4,
        token_dim: int = 768,
        face_scales: Dict[str, float] | None = None,
    ):
        super().__init__()
        self.frontend = ContentSuppressionFrontEnd(lowpass_kernel, lowpass_sigma)
        self.global_encoder = GlobalAppearanceEncoder(global_dim)
        self.texture_encoder = LocalTextureEncoder(texture_dim)
        self.fusion = AppearanceFusion(global_dim, texture_dim, fusion_dim, texture_weight=0.4)
        self.token_generator = AppearanceTokenGenerator(fusion_dim, num_tokens, token_dim)
        self.face_modulation = FaceAwareModulation(num_tokens, token_dim, face_scales=face_scales)

    def forward(self, front_image: torch.Tensor) -> AppearanceConditioningOutput:
        front_low, high_freq_norm = self.frontend(front_image)
        z_global, _ = self.global_encoder(front_low)
        z_texture = self.texture_encoder(high_freq_norm)
        z_app = self.fusion(z_global, z_texture)
        shared_tokens = self.token_generator(z_app)
        face_tokens = self.face_modulation(shared_tokens)
        return AppearanceConditioningOutput(
            front_low=front_low,
            high_freq_norm=high_freq_norm,
            z_global=z_global,
            z_texture=z_texture,
            z_app=z_app,
            shared_tokens=shared_tokens,
            face_tokens=face_tokens,
        )


def flatten_face_tokens(face_tokens: torch.Tensor) -> torch.Tensor:
    bsz, num_faces, num_tokens, token_dim = face_tokens.shape
    return face_tokens.reshape(bsz * num_faces, num_tokens, token_dim)


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


def shallow_vae_features(vae: nn.Module, images: torch.Tensor) -> List[torch.Tensor]:
    encoder = vae.encoder
    feats = []
    hidden = encoder.conv_in(images)
    feats.append(hidden)
    if hasattr(encoder, "down_blocks"):
        for block in list(encoder.down_blocks)[:2]:
            hidden = block(hidden)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            feats.append(hidden)
    return feats[:2]


def stats_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    src_mean, src_std = per_channel_mean_std(source)
    tgt_mean, tgt_std = per_channel_mean_std(target)
    return F.l1_loss(src_mean, tgt_mean) + F.l1_loss(src_std, tgt_std)


def luminance_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    src_mean, src_std = luminance_mean_std(source)
    tgt_mean, tgt_std = luminance_mean_std(target)
    return F.l1_loss(src_mean, tgt_mean) + F.l1_loss(src_std, tgt_std)


def feature_stats_loss(vae: nn.Module, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    src_feats = shallow_vae_features(vae, source)
    tgt_feats = shallow_vae_features(vae, target)
    loss = source.new_tensor(0.0)
    for src_feat, tgt_feat in zip(src_feats, tgt_feats):
        src_mean, src_std = per_channel_mean_std(src_feat)
        tgt_mean, tgt_std = per_channel_mean_std(tgt_feat)
        loss = loss + F.l1_loss(src_mean, tgt_mean) + F.l1_loss(src_std, tgt_std)
    return loss


def appearance_loss_terms(
    *,
    pred_image: torch.Tensor,
    front_low: torch.Tensor,
    vae: nn.Module,
    kernel_size: int,
    sigma: float,
) -> Dict[str, torch.Tensor]:
    pred_low = gaussian_blur_2d(pred_image, kernel_size, sigma)
    pred_feat_input = F.interpolate(pred_low, size=(256, 256), mode="bilinear", align_corners=False)
    front_feat_input = F.interpolate(front_low, size=(256, 256), mode="bilinear", align_corners=False)
    return {
        "color": stats_loss(pred_low, front_low),
        "luma": luminance_loss(pred_low, front_low),
        "feat": feature_stats_loss(vae, pred_feat_input, front_feat_input),
    }


def facewise_appearance_loss(
    *,
    pred_images: torch.Tensor,
    front_low: torch.Tensor,
    vae: nn.Module,
    kernel_size: int,
    sigma: float,
    face_weights: Dict[str, float],
    lambda_color: float = 1.0,
    lambda_luma: float = 0.2,
    lambda_feat: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    pred_images: (B, 5, 3, H, W) ordered as NON_FRONT_FACE_ORDER
    front_low:  (B, 3, H, W)
    """
    total = pred_images.new_tensor(0.0)
    metrics = {
        "color": pred_images.new_tensor(0.0),
        "luma": pred_images.new_tensor(0.0),
        "feat": pred_images.new_tensor(0.0),
    }

    for face_idx, face_name in enumerate(NON_FRONT_FACE_ORDER):
        weight = float(face_weights.get(face_name, 0.0))
        if weight <= 0.0:
            continue

        pred_low = gaussian_blur_2d(pred_images[:, face_idx], kernel_size, sigma)
        color = stats_loss(pred_low, front_low)
        luma = luminance_loss(pred_low, front_low)
        feat = feature_stats_loss(vae, pred_low, front_low)
        face_loss = weight * (lambda_color * color + lambda_luma * luma + lambda_feat * feat)

        total = total + face_loss
        metrics["color"] = metrics["color"] + weight * color
        metrics["luma"] = metrics["luma"] + weight * luma
        metrics["feat"] = metrics["feat"] + weight * feat

    return total, metrics
