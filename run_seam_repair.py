"""
Full Seam Repair Pipeline Test with SD Inpainting

This script:
1. Loads 6 faces from CubeDiff output
2. Repairs all 12 seams using SD Inpainting
3. Converts to ERP panorama
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import py360convert

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cubediff.pipelines.seam_repair import (
    repair_all_seams,
    repair_single_seam,
    FACE_NAMES,
    EDGE_DEFINITIONS
)


def create_inpaint_pipeline(device="cuda"):
    """
    Create SD Inpainting Pipeline.
    Uses the standard SD 1.5 Inpainting model.
    """
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


def make_inpaint_fn(pipe, prompt="", negative_prompt="", 
                    num_inference_steps=20, strength=0.5):
    """
    Create an inpainting function that matches the expected signature.
    
    The function takes (image, mask) and returns the repaired image.
    """
    def inpaint_fn(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint the masked region of the image.
        
        Args:
            image: RGB image [H, W, 3] uint8
            mask: Float mask [H, W] 0-1 (1 = inpaint region)
        
        Returns:
            Repaired image [H, W, 3] uint8
        """
        # Convert to PIL
        pil_image = Image.fromarray(image)
        
        # Convert mask to PIL (SD expects white = inpaint)
        mask_uint8 = (mask * 255).astype(np.uint8)
        pil_mask = Image.fromarray(mask_uint8)
        
        # Resize to 512x512 if not already
        original_size = pil_image.size
        if original_size != (512, 512):
            pil_image = pil_image.resize((512, 512), Image.LANCZOS)
            pil_mask = pil_mask.resize((512, 512), Image.NEAREST)
        
        # Run inpainting
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pil_image,
            mask_image=pil_mask,
            num_inference_steps=num_inference_steps,
            strength=strength,
            guidance_scale=3.0,  # Lower for better coherence
        ).images[0]
        
        # Resize back if needed
        if result.size != original_size:
            result = result.resize(original_size, Image.LANCZOS)
        
        return np.array(result)
    
    return inpaint_fn


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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="output/030002_front_up_polydiff",
                        help="Directory containing 6 face images")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: input_dir/seam_repaired)")
    parser.add_argument("--seam_width", type=int, default=64,
                        help="Width of seam region to repair")
    parser.add_argument("--feather", type=int, default=32,
                        help="Feather width for blending")
    parser.add_argument("--steps", type=int, default=20,
                        help="Inpainting steps")
    parser.add_argument("--strength", type=float, default=0.5,
                        help="Inpainting strength (0-1)")
    parser.add_argument("--debug", action="store_true",
                        help="Save debug images")
    parser.add_argument("--edge", type=int, default=None,
                        help="Only repair this edge (0-11)")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, "seam_repaired")
    os.makedirs(args.output_dir, exist_ok=True)
    
    debug_dir = os.path.join(args.output_dir, "debug") if args.debug else None
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
    
    # Load faces
    print(f"\n=== Loading faces from {args.input_dir} ===")
    faces = load_faces(args.input_dir)
    print(f"Loaded {len(faces)} faces")
    
    # Save original ERP for comparison
    print("\n=== Creating original ERP (before repair) ===")
    erp_before = faces_to_erp(faces)
    Image.fromarray(erp_before).save(os.path.join(args.output_dir, "erp_before.png"))
    print(f"Saved: {args.output_dir}/erp_before.png")
    
    # Create inpainting pipeline
    print("\n=== Setting up Inpainting Pipeline ===")
    pipe = create_inpaint_pipeline()
    inpaint_fn = make_inpaint_fn(
        pipe, 
        prompt="seamless natural texture, high quality",
        negative_prompt="seam, edge, discontinuity, artifact",
        num_inference_steps=args.steps,
        strength=args.strength
    )
    
    # Repair seams
    print(f"\n=== Repairing Seams (width={args.seam_width}, feather={args.feather}) ===")
    
    if args.edge is not None:
        # Repair single edge
        faces = repair_single_seam(
            faces, args.edge, inpaint_fn,
            seam_width=args.seam_width,
            feather=args.feather,
            debug_dir=debug_dir
        )
    else:
        # Repair all 12 edges
        faces = repair_all_seams(
            faces, inpaint_fn,
            seam_width=args.seam_width,
            feather=args.feather,
            debug_dir=debug_dir
        )
    
    # Save repaired faces
    print("\n=== Saving Repaired Faces ===")
    for i, name in enumerate(FACE_NAMES):
        path = os.path.join(args.output_dir, f"{name}_repaired.png")
        Image.fromarray(faces[i]).save(path)
        print(f"  Saved: {path}")
    
    # Create repaired ERP
    print("\n=== Creating Repaired ERP ===")
    erp_after = faces_to_erp(faces)
    Image.fromarray(erp_after).save(os.path.join(args.output_dir, "erp_after.png"))
    print(f"Saved: {args.output_dir}/erp_after.png")
    
    print("\n✅ Seam repair complete!")
    print(f"Compare: {args.output_dir}/erp_before.png vs erp_after.png")


if __name__ == "__main__":
    main()
