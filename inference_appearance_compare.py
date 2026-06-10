"""
Compare front-global-style inference variants for the appearance training pipeline.

Outputs five runs from the same checkpoint and front conditioning image:
1. with_style_dynamic
2. with_style_t05
3. with_style_t03
4. without_style
5. higher_cfg_scale
python inference_appearance_compare.py \
  --config training/configs/multitext_appearance.yaml \
  --image /home/dell/Datasets/Sun360/MiniVal_CubeMap/030003_front.png \
  --checkpoint-dir /home/dell/code/PolyDiff/checkpoints/polydiff-multitext-appearance/epoch_1_step_188_final \
  --output-dir /home/dell/code/PolyDiff/output/appearance_compare_030003_front \
  --prompts-json /home/dell/Datasets/Sun360/MiniVal_json/030003.json

"""

import argparse
import json
import os
import re
from typing import Dict, List

import torch
import torchvision.transforms as T
from PIL import Image
from omegaconf import OmegaConf
from safetensors.torch import load_file as safe_load_file

from cubediff.compat import ensure_torch_xpu_compat

ensure_torch_xpu_compat()

from diffusers import DDIMScheduler

from cubediff.modules.appearance import FrontGlobalStyleConditioner, expand_style_cond, flatten_style_cond, make_style_scale_tensor
from cubediff.modules.attention import install_global_style_processors, install_trainable_attn1_processors
from cubediff.pipelines.pipeline import CubeDiffPipeline


FACE_ORDER = ["front", "back", "left", "right", "top", "bottom"]
DEFAULT_FACE_SCALES = {
    "front": 0.90,
    "back": 0.15,
    "left": 0.45,
    "right": 0.45,
    "top": 0.10,
    "bottom": 0.10,
}
DEFAULT_PROMPTS = {
    "front": "A neoclassical stone building facade faces a wide street.",
    "back": "A street-level rear view of the same urban location.",
    "left": "A side view with nearby storefronts and pavement.",
    "right": "A side street view with building frontage and sidewalk.",
    "top": "Sky above the city street.",
    "bottom": "Road and pavement under the camera.",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare front-global-style inference variants.")
    parser.add_argument("--config", default="training/configs/multitext_appearance.yaml")
    parser.add_argument("--image", required=True, help="Front conditioning image path.")
    parser.add_argument("--checkpoint-dir", required=True, help="Accelerate checkpoint directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs.")
    parser.add_argument("--prompts-json", default="", help="Optional per-face prompt json.")
    parser.add_argument("--cfg-scale", type=float, default=3.5, help="Base CFG scale.")
    parser.add_argument("--higher-cfg-scale", type=float, default=7.0, help="Stronger CFG scale.")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".safetensors"):
        return safe_load_file(path, device="cpu")
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload["state_dict"]
    return payload


def load_full_cubediff_unet_weights(unet: torch.nn.Module, ckpt_path: str) -> None:
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
        unet.conv_in = torch.nn.Conv2d(
            in_channels=conv_weight.shape[1],
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

    missing, unexpected = unet.load_state_dict(state_dict, strict=False)
    missing = [key for key in missing if ".processor." not in key]
    unexpected = [key for key in unexpected if ".processor." not in key]
    if missing:
        print(f"[CubeDiff] Missing keys when loading UNet: {len(missing)}")
    if unexpected:
        print(f"[CubeDiff] Unexpected keys when loading UNet: {len(unexpected)}")


def build_prompt_list(prompts_dict: Dict[str, str]) -> List[str]:
    return [prompts_dict.get(face, "") for face in FACE_ORDER]


def load_prompts(prompts_json: str) -> Dict[str, str]:
    if not prompts_json:
        return DEFAULT_PROMPTS.copy()
    with open(prompts_json, "r") as f:
        prompt_data = json.load(f)
    return {face: prompt_data.get(face, "") for face in FACE_ORDER}


def build_prompt_filter_patterns(words: List[str]) -> List[re.Pattern[str]]:
    patterns = []
    for word in words:
        phrase = str(word).strip()
        if not phrase:
            continue
        patterns.append(re.compile(re.escape(phrase), flags=re.IGNORECASE))
    return patterns


def sanitize_content_prompt(prompt: str, patterns: List[re.Pattern[str]]) -> str:
    cleaned = prompt
    for pattern in patterns:
        cleaned = pattern.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,.;:!?]){2,}", r"\1", cleaned)
    return cleaned.strip(" ,.;:!?") or prompt


def resolve_front_content_prompt(cfg, prompts_json: str, prompts_dict: Dict[str, str]) -> str:
    if prompts_json and os.path.exists(prompts_json):
        with open(prompts_json, "r") as f:
            prompt_data = json.load(f)
        content_prompt = str(prompt_data.get("front_content_prompt", "")).strip()
        if content_prompt:
            return content_prompt

    content_filter_cfg = getattr(cfg.appearance, "content_prompt_filter", {})
    enabled = bool(getattr(content_filter_cfg, "enabled", False))
    front_prompt = prompts_dict.get("front", "")
    if not enabled:
        return front_prompt

    patterns = build_prompt_filter_patterns(list(getattr(content_filter_cfg, "words", [])))
    return sanitize_content_prompt(front_prompt, patterns)


def load_style_conditioner(cfg, checkpoint_dir: str, device: torch.device, dtype: torch.dtype) -> FrontGlobalStyleConditioner:
    conditioner = FrontGlobalStyleConditioner(
        clip_model_id=str(cfg.appearance.clip_model_id),
        style_dim=int(cfg.appearance.style_dim),
        beta=float(cfg.appearance.beta),
        cache_dir=os.path.expanduser(cfg.directories.cache_dir),
        local_files_only=bool(getattr(cfg.appearance, "local_files_only", False)),
    )
    appearance_ckpt = os.path.join(checkpoint_dir, "model_1.safetensors")
    state_dict = load_state_dict(appearance_ckpt)
    missing, unexpected = conditioner.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Style] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[Style] Unexpected keys: {len(unexpected)}")
    conditioner.to(device=device, dtype=dtype)
    conditioner.move_backbone(device=device, dtype=dtype)
    conditioner.eval()
    return conditioner


def load_pipeline(cfg, checkpoint_dir: str, device: torch.device, dtype: torch.dtype) -> CubeDiffPipeline:
    cache_dir = os.path.expanduser(cfg.directories.cache_dir)
    pipe = CubeDiffPipeline.from_pretrained(cfg.model.id, cache_dir=cache_dir, local_files_only=False)
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    install_trainable_attn1_processors(pipe.unet)
    install_global_style_processors(pipe.unet, module_prefix=str(cfg.appearance.style_block))
    load_full_cubediff_unet_weights(pipe.unet, os.path.join(checkpoint_dir, "model.safetensors"))

    if cfg.training.prediction_type == "v_prediction":
        pipe.scheduler = DDIMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            subfolder="scheduler",
            cache_dir=cache_dir,
            local_files_only=True,
        )
        pipe.scheduler.config.prediction_type = "v_prediction"
    else:
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.config.prediction_type = "epsilon"

    pipe = pipe.to(device)
    pipe.unet.to(device=device, dtype=dtype)
    pipe.vae.to(device=device, dtype=dtype)
    pipe.text_encoder.to(device=device, dtype=dtype)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    return pipe


def make_conditioning_tensor(image_path: str, image_size: int, device: torch.device, dtype: torch.dtype):
    transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    conditioning_image = transform(image).to(device=device, dtype=dtype)
    return conditioning_image, image


def save_output(output, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for face_name, face_img in zip(FACE_ORDER, output.faces_cropped):
        Image.fromarray(face_img).save(os.path.join(output_dir, f"{face_name}.png"))
    Image.fromarray(output.equirectangular).save(os.path.join(output_dir, "equirectangular.png"))


def run_variant(
    *,
    pipe: CubeDiffPipeline,
    prompts: List[str],
    conditioning_image: torch.Tensor,
    style_cond: torch.Tensor | None,
    style_scale: torch.Tensor | None,
    style_time_scale_override: float | None,
    cfg_scale: float,
    num_inference_steps: int,
    generator: torch.Generator,
):
    cross_attention_kwargs = {}
    if style_cond is not None:
        cross_attention_kwargs["style_cond"] = style_cond
        cross_attention_kwargs["style_scale"] = style_scale
        if style_time_scale_override is not None:
            cross_attention_kwargs["style_time_scale_override"] = style_time_scale_override

    with torch.no_grad():
        return pipe(
            prompts=prompts,
            conditioning_image=conditioning_image,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            generator=generator,
            cross_attention_kwargs=cross_attention_kwargs,
        )


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    os.makedirs(args.output_dir, exist_ok=True)

    prompts_dict = load_prompts(args.prompts_json)
    prompts = build_prompt_list(prompts_dict)
    front_content_prompt = resolve_front_content_prompt(cfg, args.prompts_json, prompts_dict)
    face_scales = {
        name: float(getattr(cfg.appearance.face_scales, name, DEFAULT_FACE_SCALES[name]))
        for name in FACE_ORDER
    }
    compare_time_scales = list(getattr(cfg.appearance, "compare_time_scales", [0.5, 0.3]))
    if len(compare_time_scales) < 2:
        raise ValueError("appearance.compare_time_scales must contain at least two values")
    validation_time_scale = float(getattr(cfg.appearance, "validation_time_scale", compare_time_scales[0]))

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading checkpoint from {args.checkpoint_dir}")
    pipe = load_pipeline(cfg, args.checkpoint_dir, device, dtype)
    conditioner = load_style_conditioner(cfg, args.checkpoint_dir, device, dtype)

    conditioning_image, original_pil = make_conditioning_tensor(args.image, int(cfg.model.image_size), device, dtype)
    original_pil.save(os.path.join(args.output_dir, "conditioning_front.png"))

    with torch.no_grad():
        style_out = conditioner(conditioning_image.unsqueeze(0), [front_content_prompt])
        style_cond = flatten_style_cond(expand_style_cond(style_out.style_cond))
        style_scale = make_style_scale_tensor(
            1,
            face_scales=face_scales,
            time_scale=1.0,
            device=device,
            dtype=dtype,
        )

    runs = [
        ("with_style_dynamic", style_cond, style_scale, None, args.cfg_scale),
        ("with_style_t05", style_cond, style_scale, validation_time_scale, args.cfg_scale),
        ("with_style_t03", style_cond, style_scale, float(compare_time_scales[1]), args.cfg_scale),
        ("without_style", None, None, None, args.cfg_scale),
        ("higher_cfg_scale", style_cond, style_scale, None, args.higher_cfg_scale),
    ]

    for index, (name, run_style_cond, run_style_scale, run_time_scale, cfg_scale) in enumerate(runs):
        print(f"[INFO] Running {name} with cfg_scale={cfg_scale}")
        generator = torch.Generator(device=device).manual_seed(args.seed + index)
        output = run_variant(
            pipe=pipe,
            prompts=prompts,
            conditioning_image=conditioning_image,
            style_cond=run_style_cond,
            style_scale=run_style_scale,
            style_time_scale_override=run_time_scale,
            cfg_scale=cfg_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        )
        save_output(output, os.path.join(args.output_dir, name))

    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(
            {
                "image": args.image,
                "checkpoint_dir": args.checkpoint_dir,
                "cfg_scale": args.cfg_scale,
                "higher_cfg_scale": args.higher_cfg_scale,
                "num_inference_steps": args.num_inference_steps,
                "seed": args.seed,
                "front_content_prompt": front_content_prompt,
                "validation_time_scale": validation_time_scale,
                "compare_time_scales": compare_time_scales,
                "prompts": prompts_dict,
            },
            f,
            indent=2,
        )

    print(f"[INFO] Saved comparison outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
