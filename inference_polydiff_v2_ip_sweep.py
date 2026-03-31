"""
Batch inference script for sweeping IP-Adapter scales.

This script reuses the full PolyDiff v2 + IP-Adapter inference flow and runs
multiple `IP_ADAPTER_SCALE` values in one execution.

Usage example:
    python inference_polydiff_v2_ip_sweep.py \
        --image /home/dell/Datasets/Sun360/MiniVal_CubeMap/030003_front.png \
        --weights ./checkpoints/polydiff-multitext-ipadapter-3000/epoch_40_step_7500/model.safetensors \
        --output-base output/030003_front_polydiff_v2_ip-3000 \
        --scales 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99
"""

import argparse
import os

import numpy as np
import py360convert
import torch
import torchvision.transforms as T
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image

from cubediff.pipelines.pipeline import CubeDiffPipeline
from cubediff.pipelines.seam_repair import FACE_NAMES, repair_all_seams


# ============== DEFAULT USER CONFIGURATION ==============


DEFAULT_IMAGE_FILENAME = ""

DEFAULT_PROMPTS = {
    "front": "A large stone cathedral facade with arched entrances and rose window faces a street with double yellow lines.",
    "right": "A white sedan parks on a sloped street lined with Victorian houses and trees.",
    "back": "A paved walkway leads into a tree-lined park with benches and lampposts.",
    "left": "A red car parks beside a tree-lined street with buildings ahead.",
    "top": "Church spires and rose window visible against overcast sky.",
    "bottom": "Concrete floor with scattered dark specks and faint stains.",
}

USE_IP_ADAPTER = True
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "models"
IP_ADAPTER_WEIGHT_NAME = "ip-adapter_sd15.bin"

FACE_REF_IMAGES = {
    "front": None,
    "back": "/home/dell/Datasets/Sun360/MiniVal_CubeMap/030003_back.png",
    "left": "/home/dell/Datasets/Sun360/MiniVal_CubeMap/030003_left.png",
    "right": "/home/dell/Datasets/Sun360/MiniVal_CubeMap/030003_right.png",
    "top": "/home/dell/Datasets/Sun360/MiniVal_CubeMap/030003_top.png",
    "bottom": "/home/dell/Datasets/Sun360/MiniVal_CubeMap/030003_bottom.png",
}
GLOBAL_REF_IMAGE = None

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
DEFAULT_CUBEDIFF_WEIGHTS = "./checkpoints/polydiff-multitext-ipadapter-3000/epoch_40_step_7500/model.safetensors"
DEFAULT_SCALES = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99"

CFG_SCALE = 3.5
NUM_INFERENCE_STEPS = 50
ERP_HEIGHT = 1024
ERP_WIDTH = 2048

SEAM_WIDTH = 70
FEATHER = 50
INPAINT_STEPS = 20
INPAINT_STRENGTH = 0.7
DEBUG_SEAMS = True


def faces_to_erp(faces, erp_height=1024, erp_width=2048):
    """Convert 6 faces to ERP panorama."""
    cube_dict = {
        "F": faces[0],
        "B": faces[1],
        "L": faces[2],
        "R": faces[3],
        "U": faces[4],
        "D": faces[5],
    }
    return py360convert.c2e(cube_dict, h=erp_height, w=erp_width, cube_format="dict")


def create_inpaint_fn(
    device="cuda",
    prompt="smooth blending, natural continuation, matching colors and textures, coherent scene, photorealistic",
    negative_prompt="abrupt change, color mismatch, inconsistent lighting, blurry, artificial, seam line, hard edge",
    num_inference_steps=20,
    strength=0.55,
):
    """Create SD Inpainting function for seam repair."""
    print("[Inpaint] Loading SD Inpainting model...")
    pipe_dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        dtype=pipe_dtype,
        safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    print("[Inpaint] Model loaded.")

    def inpaint_fn(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray(image)
        mask_uint8 = (mask * 255).astype(np.uint8)
        pil_mask = Image.fromarray(mask_uint8)

        original_size = pil_image.size
        if original_size != (512, 512):
            pil_image = pil_image.resize((512, 512), Image.LANCZOS)
            pil_mask = pil_mask.resize((512, 512), Image.NEAREST)

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pil_image,
            mask_image=pil_mask,
            num_inference_steps=num_inference_steps,
            strength=strength,
            guidance_scale=3.0,
        ).images[0]

        if result.size != original_size:
            result = result.resize(original_size, Image.LANCZOS)

        return np.array(result)

    return inpaint_fn


def parse_args():
    image_name = os.path.splitext(os.path.basename(DEFAULT_IMAGE_FILENAME))[0]
    default_output_base = f"output/{image_name}_polydiff_v2_ip-3000"

    parser = argparse.ArgumentParser(
        description="Run PolyDiff v2 IP-Adapter inference for multiple IP-Adapter scales."
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE_FILENAME, help="Front conditioning image path.")
    parser.add_argument("--weights", default=DEFAULT_CUBEDIFF_WEIGHTS, help="Path to trained model weights.")
    parser.add_argument(
        "--output-base",
        default=default_output_base,
        help="Output directory prefix. Each scale is saved as <output-base>_scale-<value>/",
    )
    parser.add_argument(
        "--scales",
        default=DEFAULT_SCALES,
        help="Comma-separated IP-Adapter scales. Default: 0.1,0.2,...,0.9,0.99",
    )
    return parser.parse_args()


def parse_scales(scales_arg):
    scales = []
    for raw in scales_arg.split(","):
        value = raw.strip()
        if not value:
            continue
        try:
            scale = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid scale value: {value}") from exc
        if not (0.0 < scale <= 1.0):
            raise ValueError(f"Scale must be in (0, 1], got {scale}")
        scales.append(scale)

    if not scales:
        raise ValueError("No valid scales were provided.")

    return scales


def format_scale(scale):
    return f"{scale:.2f}"


def build_prompt_list(prompts):
    if isinstance(prompts, dict):
        return [
            prompts.get("front", ""),
            prompts.get("back", ""),
            prompts.get("left", ""),
            prompts.get("right", ""),
            prompts.get("top", ""),
            prompts.get("bottom", ""),
        ]
    return [prompts] * 6


def load_conditioning_image(image_path):
    transform = T.Compose(
        [
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return transform(image), image


def prepare_ip_adapter_images(conditioning_pil):
    if not USE_IP_ADAPTER:
        return None

    if GLOBAL_REF_IMAGE is not None:
        print(f"[INFO] Using global reference image: {GLOBAL_REF_IMAGE}")
        return load_image(GLOBAL_REF_IMAGE)

    ref_images_list = []
    face_order = ["front", "back", "left", "right", "top", "bottom"]

    print("[INFO] Preparing per-face IP-Adapter references...")
    for face_name in face_order:
        ref_path = FACE_REF_IMAGES.get(face_name)
        if ref_path is not None and os.path.exists(ref_path):
            print(f"  - {face_name}: {ref_path}")
            ref_images_list.append(load_image(ref_path))
        else:
            print(f"  - {face_name}: using conditioning image (default)")
            ref_images_list.append(conditioning_pil)

    return ref_images_list


def save_scale_outputs(output_dir, face_arrays, repaired_faces):
    print("\n[INFO] Saving 6 face images...")
    for face_img, name in zip(face_arrays, FACE_NAMES):
        face_path = os.path.join(output_dir, f"{name}.png")
        Image.fromarray(face_img).save(face_path)
        print(f"  ✓ Saved {name}.png")

    print("\n[INFO] Creating ERP panorama (before repair)...")
    erp_before = faces_to_erp(face_arrays, ERP_HEIGHT, ERP_WIDTH)
    Image.fromarray(erp_before).save(os.path.join(output_dir, "erp_before.png"))
    print("  ✓ Saved erp_before.png")

    print("\n[INFO] Saving repaired faces...")
    for face_arr, name in zip(repaired_faces, FACE_NAMES):
        face_path = os.path.join(output_dir, f"{name}_repaired.png")
        Image.fromarray(face_arr).save(face_path)
        print(f"  ✓ Saved {name}_repaired.png")

    print("\n[INFO] Creating ERP panorama (after repair)...")
    erp_after = faces_to_erp(repaired_faces, ERP_HEIGHT, ERP_WIDTH)
    Image.fromarray(erp_after).save(os.path.join(output_dir, "erp_after.png"))
    print("  ✓ Saved erp_after.png")

    Image.fromarray(erp_after).save(os.path.join(output_dir, "equirectangular.png"))
    print("  ✓ Saved equirectangular.png (final output)")


def run_single_scale(
    scale,
    scale_index,
    total_scales,
    cubediff_pipe,
    inpaint_fn,
    conditioning_image,
    prompt_list,
    ip_adapter_images,
    output_base,
    device,
):
    scale_str = format_scale(scale)
    output_dir = f"{output_base}_scale-{scale_str}"
    os.makedirs(output_dir, exist_ok=True)

    debug_dir = os.path.join(output_dir, "debug") if DEBUG_SEAMS else None
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    print("\n" + "=" * 72)
    print(f"[Sweep] Running scale {scale_str} ({scale_index}/{total_scales})")
    print(f"[Sweep] Output directory: {output_dir}")
    print("=" * 72)

    cubediff_pipe.set_ip_adapter_scale(scale)

    print("\n" + "=" * 60)
    print("[Stage 1] Generating 6 cubemap faces with CubeDiff...")
    if USE_IP_ADAPTER:
        print(f"         IP-Adapter: ENABLED (scale={scale_str})")
    print("=" * 60)

    output = cubediff_pipe(
        prompts=prompt_list,
        conditioning_image=conditioning_image.unsqueeze(0).to(device),
        ip_adapter_image=ip_adapter_images if ip_adapter_images is not None else None,
        num_inference_steps=NUM_INFERENCE_STEPS,
        cfg_scale=CFG_SCALE,
    )

    face_arrays = [face_img.copy() for face_img in output.faces_cropped]

    print("\n" + "=" * 60)
    print("[Stage 2] Repairing 12 seams with edge-by-edge SD Inpainting...")
    print(f"         (seam_width={SEAM_WIDTH}, feather={FEATHER})")
    print("=" * 60)

    repaired_faces = repair_all_seams(
        [face.copy() for face in face_arrays],
        inpaint_fn,
        seam_width=SEAM_WIDTH,
        feather=FEATHER,
        debug_dir=debug_dir,
    )

    save_scale_outputs(output_dir, face_arrays, repaired_faces)

    print("\n" + "=" * 60)
    print("[INFO] Scale run completed successfully.")
    print(f"[INFO] Scale: {scale_str}")
    print(f"[INFO] Output saved to: {output_dir}")
    print("=" * 60)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()

    image_path = os.path.abspath(os.path.expanduser(args.image))
    weights_path = os.path.abspath(os.path.expanduser(args.weights))
    output_base = os.path.abspath(os.path.expanduser(args.output_base))
    scales = parse_scales(args.scales)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Conditioning image not found: {image_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 72)
    print("[Sweep] PolyDiff v2 IP-Adapter scale sweep")
    print(f"[Sweep] Image: {image_path}")
    print(f"[Sweep] Weights: {weights_path}")
    print(f"[Sweep] Output base: {output_base}")
    print(f"[Sweep] Total scales: {len(scales)}")
    print(f"[Sweep] Scales: {', '.join(format_scale(scale) for scale in scales)}")
    print("=" * 72)

    prompt_list = build_prompt_list(DEFAULT_PROMPTS)
    conditioning_image, conditioning_pil = load_conditioning_image(image_path)
    ip_adapter_images = prepare_ip_adapter_images(conditioning_pil)

    print("\n" + "=" * 60)
    print("[Init] Loading CubeDiff Pipeline...")
    print(f"       Base: {BASE_MODEL}")
    print(f"       Weights: {weights_path}")
    print("=" * 60)
    cubediff_pipe = CubeDiffPipeline.from_pretrained(
        BASE_MODEL,
        cubediff_weights_path=weights_path,
    ).to(device)

    if USE_IP_ADAPTER:
        print(f"[Init] Loading IP-Adapter from {IP_ADAPTER_REPO}...")
        cubediff_pipe.load_ip_adapter(
            IP_ADAPTER_REPO,
            subfolder=IP_ADAPTER_SUBFOLDER,
            weight_name=IP_ADAPTER_WEIGHT_NAME,
            local_files_only=True,
        )
        print("[Init] IP-Adapter loaded.")

    inpaint_fn = create_inpaint_fn(
        device=device,
        num_inference_steps=INPAINT_STEPS,
        strength=INPAINT_STRENGTH,
    )

    for idx, scale in enumerate(scales, start=1):
        run_single_scale(
            scale=scale,
            scale_index=idx,
            total_scales=len(scales),
            cubediff_pipe=cubediff_pipe,
            inpaint_fn=inpaint_fn,
            conditioning_image=conditioning_image,
            prompt_list=prompt_list,
            ip_adapter_images=ip_adapter_images,
            output_base=output_base,
            device=device,
        )

    print("\n" + "=" * 72)
    print("[Sweep] All scale tests finished.")
    print("=" * 72)


if __name__ == "__main__":
    main()
