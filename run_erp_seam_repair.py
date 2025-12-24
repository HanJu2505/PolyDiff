"""
ERP-Level Seam Repair Test

This script tests a simpler seam repair approach:
1. Create edge masks for each face (white on edges, black in center)
2. Project face masks to ERP mask using py360convert
3. Use SD Inpainting on the full ERP image

Usage:
    python run_erp_seam_repair.py --input_dir output/030002_front_up_polydiff
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import py360convert
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FACE_NAMES = ["front", "back", "left", "right", "top", "bottom"]


def create_face_edge_mask(size: int = 512, 
                          edge_width: int = 24, 
                          feather: int = 12) -> np.ndarray:
    """
    Create a face mask with white edges and black center.
    
    Args:
        size: Face size in pixels
        edge_width: Width of the edge region (white)
        feather: Width of feather/gradient transition
    
    Returns:
        Mask array [size, size] with values 0-1
    """
    mask = np.zeros((size, size), dtype=np.float32)
    
    # Create edge mask (1 at edges, 0 in center)
    for i in range(size):
        for j in range(size):
            # Distance from nearest edge
            dist_from_edge = min(i, j, size - 1 - i, size - 1 - j)
            
            if dist_from_edge < edge_width:
                # Core edge area
                mask[i, j] = 1.0
            elif dist_from_edge < edge_width + feather:
                # Feather region
                t = (dist_from_edge - edge_width) / feather
                mask[i, j] = 1.0 - t
    
    return mask


def load_faces(input_dir: str):
    """Load 6 cubemap faces from a directory."""
    faces = []
    for name in FACE_NAMES:
        path = os.path.join(input_dir, f"{name}.png")
        if os.path.exists(path):
            img = np.array(Image.open(path).convert("RGB"))
            faces.append(img)
        else:
            raise FileNotFoundError(f"Face not found: {path}")
    return faces


def faces_to_erp(faces, erp_height=1024, erp_width=2048):
    """Convert 6 faces to ERP panorama using py360convert."""
    cube_dict = {
        "F": faces[0],  # front
        "B": faces[1],  # back
        "L": faces[2],  # left
        "R": faces[3],  # right
        "U": faces[4],  # top
        "D": faces[5],  # bottom
    }
    return py360convert.c2e(cube_dict, h=erp_height, w=erp_width, cube_format='dict')


def masks_to_erp(masks, erp_height=1024, erp_width=2048):
    """Convert 6 face masks to ERP mask."""
    # Convert float masks (0-1) to uint8 (0-255) for py360convert
    masks_uint8 = [(m * 255).astype(np.uint8) for m in masks]
    
    # Convert to 3-channel for py360convert
    masks_3ch = [np.stack([m, m, m], axis=-1) for m in masks_uint8]
    
    erp_mask_3ch = faces_to_erp(masks_3ch, erp_height, erp_width)
    
    # Take single channel and convert back to float
    erp_mask = erp_mask_3ch[:, :, 0].astype(np.float32) / 255.0
    
    return erp_mask


def create_inpaint_pipeline(device="cuda"):
    """Create SD Inpainting Pipeline."""
    from diffusers import StableDiffusionInpaintPipeline
    
    print("[Inpaint] Loading SD Inpainting model...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    print("[Inpaint] Model loaded.")
    
    return pipe


def inpaint_erp(pipe, erp_image: np.ndarray, erp_mask: np.ndarray,
                prompt: str = "seamless panorama, continuous scene, high quality",
                negative_prompt: str = "visible seam, dividing line, artifacts",
                num_inference_steps: int = 20,
                strength: float = 0.6) -> np.ndarray:
    """
    Inpaint ERP image using SD Inpainting with smart blending.
    
    Strategy:
    1. Downsample: 2048×1024 → 1024×512 (SD1.5 golden ratio for panorama)
    2. Inpaint at low-res
    3. Upsample back to 2048×1024
    4. Smart blend: use original high-res mask to blend only seam areas
       - Original clear areas remain at full 2048 resolution
       - Only seam areas use upsampled result (but seams are blurred anyway)
    """
    from PIL import Image as PILImage
    
    original_h, original_w = erp_image.shape[:2]  # 1024, 2048
    
    # Step 1: Downsample to SD-compatible size (1024x512)
    sd_size = (1024, 512)  # W, H
    
    pil_image_lowres = PILImage.fromarray(erp_image).resize(sd_size, PILImage.LANCZOS)
    
    # Downsample mask too
    mask_uint8 = (erp_mask * 255).astype(np.uint8)
    pil_mask_lowres = PILImage.fromarray(mask_uint8).resize(sd_size, PILImage.LANCZOS)
    
    print(f"[Inpaint] Step 1: Downsampled to {sd_size}")
    
    # Step 2: Inpaint at low-res
    print(f"[Inpaint] Step 2: Running inpainting (steps={num_inference_steps}, strength={strength})")
    result_lowres = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=pil_image_lowres,
        mask_image=pil_mask_lowres,
        num_inference_steps=num_inference_steps,
        strength=strength,
        guidance_scale=3.0,
    ).images[0]
    
    # Step 3: Upsample back to original size
    result_upsampled = result_lowres.resize((original_w, original_h), PILImage.LANCZOS)
    result_upsampled = np.array(result_upsampled)
    print(f"[Inpaint] Step 3: Upsampled to {original_w}x{original_h}")
    
    # Step 4: Smart blend - use original high-res mask
    # Only replace seam areas (where mask > 0), keep original elsewhere
    print("[Inpaint] Step 4: Smart blending with high-res mask")
    
    # Expand mask to 3 channels
    mask_3ch = np.stack([erp_mask, erp_mask, erp_mask], axis=-1)
    
    # Blend: result = mask * upsampled_repair + (1-mask) * original
    result_blended = (mask_3ch * result_upsampled + 
                      (1 - mask_3ch) * erp_image).astype(np.uint8)
    
    return result_blended


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="output/030003_front_up_polydiff",
                        help="Directory containing 6 face images")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--edge_width", type=int, default=24,
                        help="Edge mask width in pixels")
    parser.add_argument("--feather", type=int, default=12,
                        help="Feather width for mask")
    parser.add_argument("--steps", type=int, default=20,
                        help="Inpainting steps")
    parser.add_argument("--strength", type=float, default=0.6,
                        help="Inpainting strength (0-1)")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, "erp_seam_repair")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Load faces and create ERP
    print(f"\n=== Loading faces from {args.input_dir} ===")
    faces = load_faces(args.input_dir)
    print(f"Loaded {len(faces)} faces ({faces[0].shape})")
    
    print("\n=== Creating ERP from faces ===")
    erp_image = faces_to_erp(faces)
    Image.fromarray(erp_image).save(os.path.join(args.output_dir, "erp_before.png"))
    print(f"Saved: erp_before.png ({erp_image.shape})")
    
    # Step 2: Create face edge masks and project to ERP
    print(f"\n=== Creating Edge Masks (edge_width={args.edge_width}, feather={args.feather}) ===")
    face_masks = [create_face_edge_mask(512, args.edge_width, args.feather) for _ in range(6)]
    
    # Save a sample face mask
    Image.fromarray((face_masks[0] * 255).astype(np.uint8)).save(
        os.path.join(args.output_dir, "face_mask_sample.png"))
    print("Saved: face_mask_sample.png")
    
    print("\n=== Projecting Masks to ERP ===")
    erp_mask = masks_to_erp(face_masks)
    Image.fromarray((erp_mask * 255).astype(np.uint8)).save(
        os.path.join(args.output_dir, "erp_mask.png"))
    print(f"Saved: erp_mask.png ({erp_mask.shape})")
    
    # Step 3: Inpaint ERP
    print(f"\n=== Inpainting ERP (steps={args.steps}, strength={args.strength}) ===")
    pipe = create_inpaint_pipeline()
    
    erp_repaired = inpaint_erp(
        pipe, erp_image, erp_mask,
        num_inference_steps=args.steps,
        strength=args.strength
    )
    
    Image.fromarray(erp_repaired).save(os.path.join(args.output_dir, "erp_after.png"))
    print("Saved: erp_after.png")
    
    print("\n✅ ERP seam repair complete!")
    print(f"Compare: {args.output_dir}/erp_before.png vs erp_after.png")


if __name__ == "__main__":
    main()
