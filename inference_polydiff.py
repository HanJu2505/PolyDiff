"""
PolyDiff Inference Script (2-Stage Pipeline)

Generate 360° panoramas using:
  Stage 1: Generate 6 main cubemap faces using CubeDiff
  Stage 2: Repair seams using ERP-level SD Inpainting

Usage:
    python inference_polydiff.py
"""

import torch
import os
from PIL import Image
import torchvision.transforms as T
import numpy as np
import py360convert

# Use original CubeDiff pipeline for 6-view generation
from cubediff.pipelines.pipeline import CubeDiffPipeline

# 6 main view names
FACE_NAMES = ["front", "back", "left", "right", "top", "bottom"]


def faces_to_erp(faces, erp_height=1024, erp_width=2048):
    """Convert 6 faces to ERP panorama."""
    cube_dict = {
        "F": faces[0],  # front
        "B": faces[1],  # back
        "L": faces[2],  # left
        "R": faces[3],  # right
        "U": faces[4],  # top
        "D": faces[5],  # bottom
    }
    return py360convert.c2e(cube_dict, h=erp_height, w=erp_width, cube_format='dict')


def create_face_edge_mask(size: int = 512, 
                          edge_width: int = 24, 
                          feather: int = 12) -> np.ndarray:
    """Create a face mask with white edges and black center."""
    mask = np.zeros((size, size), dtype=np.float32)
    
    for i in range(size):
        for j in range(size):
            dist_from_edge = min(i, j, size - 1 - i, size - 1 - j)
            
            if dist_from_edge < edge_width:
                mask[i, j] = 1.0
            elif dist_from_edge < edge_width + feather:
                t = (dist_from_edge - edge_width) / feather
                mask[i, j] = 1.0 - t
    
    return mask


def masks_to_erp(masks, erp_height=1024, erp_width=2048):
    """Convert 6 face masks to ERP mask."""
    masks_uint8 = [(m * 255).astype(np.uint8) for m in masks]
    masks_3ch = [np.stack([m, m, m], axis=-1) for m in masks_uint8]
    erp_mask_3ch = faces_to_erp(masks_3ch, erp_height, erp_width)
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
                prompt: str = "seamless transition, continuous structure, smooth blending, unified texture, high quality",
                negative_prompt: str = "visible seam, dividing line, border, edge, frame, split, gap, distortion, artifacts",
                num_inference_steps: int = 20,
                strength: float = 0.6) -> np.ndarray:
    """Inpaint ERP image using SD Inpainting with smart blending."""
    from PIL import Image as PILImage
    
    original_h, original_w = erp_image.shape[:2]
    sd_size = (1024, 512)
    
    # Step 1: Downsample
    pil_image_lowres = PILImage.fromarray(erp_image).resize(sd_size, PILImage.LANCZOS)
    mask_uint8 = (erp_mask * 255).astype(np.uint8)
    pil_mask_lowres = PILImage.fromarray(mask_uint8).resize(sd_size, PILImage.LANCZOS)
    
    # Step 2: Inpaint
    print(f"[Inpaint] Running inpainting (steps={num_inference_steps}, strength={strength})")
    result_lowres = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=pil_image_lowres,
        mask_image=pil_mask_lowres,
        num_inference_steps=num_inference_steps,
        strength=strength,
        guidance_scale=3.0,
    ).images[0]
    
    # Step 3: Upsample
    result_upsampled = result_lowres.resize((original_w, original_h), PILImage.LANCZOS)
    result_upsampled = np.array(result_upsampled)
    
    # Step 4: Smart blend
    mask_3ch = np.stack([erp_mask, erp_mask, erp_mask], axis=-1)
    result_blended = (mask_3ch * result_upsampled + 
                      (1 - mask_3ch) * erp_image).astype(np.uint8)
    
    return result_blended


if __name__ == "__main__":
    # ============== USER CONFIGURATION ==============
    
    # Input image (front view anchor)
    IMAGE_FILENAME = "/home/dell/Datasets/Sun360/MiniVal_views/030003_front_up.png"
    
    # Prompts for each direction
    PROMPTS = {
        "Front": "Church stands between two buildings",
        "Right": "Car parked by road, sidewalk, and trees",
        "Back": "Cars parked along road with trees and sidewalk",
        "Left": "Car parked by road, tree, and street light",
        "Top": "sky",
        "Bottom": "street",
    }
    
    # Model checkpoint (CubeDiff)
    CHECKPOINT = "./models/cubediff-512-multitxt"
    
    # Output directory
    IMAGE_NAME = os.path.splitext(os.path.basename(IMAGE_FILENAME))[0]
    OUTPUT_DIR = f"output/{IMAGE_NAME}_polydiff/"
    
    # Generation parameters
    CFG_SCALE = 3.5
    NUM_INFERENCE_STEPS = 50
    ERP_HEIGHT = 1024
    ERP_WIDTH = 2048
    
    # Seam repair parameters
    EDGE_WIDTH = 24
    FEATHER = 12
    INPAINT_STEPS = 20
    INPAINT_STRENGTH = 0.55
    
    # ================================================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # =================== STAGE 1: Generate 6 Faces ===================
    print("\n" + "="*60)
    print("[Stage 1] Generating 6 cubemap faces with CubeDiff...")
    print("="*60)
    
    # Load CubeDiff pipeline
    print(f"[INFO] Loading CubeDiff Pipeline from {CHECKPOINT}...")
    cubediff_pipe = CubeDiffPipeline.from_pretrained(CHECKPOINT).to(device)
    
    # Load conditioning image
    print(f"[INFO] Loading conditioning image {IMAGE_FILENAME}...")
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    image = Image.open(IMAGE_FILENAME).convert("RGB")
    conditioning_image = transform(image)
    
    # Prepare prompts
    prompt_list = [
        PROMPTS.get("Front", ""),
        PROMPTS.get("Back", ""),
        PROMPTS.get("Left", ""),
        PROMPTS.get("Right", ""),
        PROMPTS.get("Top", ""),
        PROMPTS.get("Bottom", ""),
    ]
    
    # Generate 6 faces
    output = cubediff_pipe(
        prompts=prompt_list,
        conditioning_image=conditioning_image.unsqueeze(0).to(device),
        num_inference_steps=NUM_INFERENCE_STEPS,
        cfg_scale=CFG_SCALE,
    )
    
    # Save face images
    print("\n[INFO] Saving 6 face images...")
    faces = output.faces_cropped  # numpy arrays [6, H, W, 3]
    face_arrays = []
    for face_img, name in zip(faces, FACE_NAMES):
        face_path = os.path.join(OUTPUT_DIR, f"{name}.png")
        Image.fromarray(face_img).save(face_path)
        face_arrays.append(face_img)
        print(f"  ✓ Saved {name}.png")
    
    # Create ERP (before repair)
    print("\n[INFO] Creating ERP panorama (before repair)...")
    erp_before = faces_to_erp(face_arrays, ERP_HEIGHT, ERP_WIDTH)
    Image.fromarray(erp_before).save(os.path.join(OUTPUT_DIR, "erp_before.png"))
    print("  ✓ Saved erp_before.png")
    
    # Free CubeDiff memory
    del cubediff_pipe
    torch.cuda.empty_cache()
    
    # =================== STAGE 2: Seam Repair ===================
    print("\n" + "="*60)
    print("[Stage 2] Repairing seams with SD Inpainting...")
    print("="*60)
    
    # Create edge masks
    print(f"[INFO] Creating edge masks (edge_width={EDGE_WIDTH}, feather={FEATHER})...")
    face_masks = [create_face_edge_mask(512, EDGE_WIDTH, FEATHER) for _ in range(6)]
    
    # Project to ERP mask
    print("[INFO] Projecting masks to ERP...")
    erp_mask = masks_to_erp(face_masks, ERP_HEIGHT, ERP_WIDTH)
    Image.fromarray((erp_mask * 255).astype(np.uint8)).save(os.path.join(OUTPUT_DIR, "erp_mask.png"))
    print("  ✓ Saved erp_mask.png")
    
    # Load SD Inpainting
    inpaint_pipe = create_inpaint_pipeline(device)
    
    # Inpaint ERP
    erp_after = inpaint_erp(
        inpaint_pipe, erp_before, erp_mask,
        num_inference_steps=INPAINT_STEPS,
        strength=INPAINT_STRENGTH
    )
    
    # Save repaired ERP
    Image.fromarray(erp_after).save(os.path.join(OUTPUT_DIR, "erp_after.png"))
    print("  ✓ Saved erp_after.png")
    
    # Also save as main output
    Image.fromarray(erp_after).save(os.path.join(OUTPUT_DIR, "equirectangular.png"))
    print("  ✓ Saved equirectangular.png (final output)")
    
    print("\n" + "="*60)
    print("✅ Pipeline complete!")
    print(f"[INFO] Output saved to: {OUTPUT_DIR}")
    print("  - erp_before.png: Before seam repair")
    print("  - erp_after.png: After seam repair")
    print("  - equirectangular.png: Final output")
    print("="*60)
