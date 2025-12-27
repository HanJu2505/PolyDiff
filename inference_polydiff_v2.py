"""
PolyDiff Inference Script (2-Stage Pipeline with Edge-by-Edge Seam Repair)

Generate 360° panoramas using:
  Stage 1: Generate 6 main cubemap faces using CubeDiff
  Stage 2: Repair 12 seams individually using SD Inpainting

Usage:
    python inference_polydiff_v2.py 
"""

import torch
import os
from PIL import Image
import torchvision.transforms as T
import numpy as np
import py360convert

# Use original CubeDiff pipeline for 6-view generation
from cubediff.pipelines.pipeline import CubeDiffPipeline
from cubediff.pipelines.seam_repair import repair_all_seams, FACE_NAMES

# For SD Inpainting
from diffusers import StableDiffusionInpaintPipeline


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


def create_inpaint_fn(device="cuda", 
                      prompt="seamless transition, continuous structure, unified texture, high quality,4K",
                      negative_prompt="visible seam, dividing line, border, edge, frame, split, gap, distortion, artifacts",
                      num_inference_steps=20, 
                      strength=0.55):
    """Create SD Inpainting function for seam repair."""
    
    print("[Inpaint] Loading SD Inpainting model...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    print("[Inpaint] Model loaded.")
    
    def inpaint_fn(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint the masked region."""
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


if __name__ == "__main__":
    # ============== USER CONFIGURATION ==============
    
    # Input image (front view anchor)
    IMAGE_FILENAME = "/home/dell/Datasets/Sun360/MiniVal_views/030002_front_up.png"
    
    # Prompts for each direction
    PROMPTS = {
        "Front": "Person walks on cobblestone street",
        "Right": "Statue stands before building",
        "Back": "Statues stand before buildings across left and right rear views",
        "Left": "Statue stands before buildings",
        "Top": "sky with sun",
        "Bottom": "street with sidewalk and road",
    }
    
    # Model checkpoint (CubeDiff)
    CHECKPOINT = "./models/cubediff-512-multitxt"
    
    # Output directory
    IMAGE_NAME = os.path.splitext(os.path.basename(IMAGE_FILENAME))[0]
    OUTPUT_DIR = f"output/{IMAGE_NAME}_polydiff_v2/"
    
    # Generation parameters
    CFG_SCALE = 3.5
    NUM_INFERENCE_STEPS = 50
    ERP_HEIGHT = 1024
    ERP_WIDTH = 2048
    
    # Seam repair parameters (edge-by-edge)
    SEAM_WIDTH = 50      # Width of seam region
    FEATHER = 20          # Feather width for blending
    INPAINT_STEPS = 20    # Inpainting steps per edge
    INPAINT_STRENGTH = 0.55
    DEBUG_SEAMS = True    # Save debug images for each edge
    
    # ================================================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    debug_dir = os.path.join(OUTPUT_DIR, "debug") if DEBUG_SEAMS else None
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
    
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
    
    # Save face images and get arrays
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
    
    # =================== STAGE 2: Edge-by-Edge Seam Repair ===================
    print("\n" + "="*60)
    print("[Stage 2] Repairing 12 seams with edge-by-edge SD Inpainting...")
    print(f"         (seam_width={SEAM_WIDTH}, feather={FEATHER})")
    print("="*60)
    
    # Create inpainting function
    inpaint_fn = create_inpaint_fn(
        device=device,
        num_inference_steps=INPAINT_STEPS,
        strength=INPAINT_STRENGTH
    )
    
    # Repair all 12 seams
    repaired_faces = repair_all_seams(
        face_arrays,
        inpaint_fn,
        seam_width=SEAM_WIDTH,
        feather=FEATHER,
        debug_dir=debug_dir
    )
    
    # Save repaired faces
    print("\n[INFO] Saving repaired faces...")
    for face_arr, name in zip(repaired_faces, FACE_NAMES):
        face_path = os.path.join(OUTPUT_DIR, f"{name}_repaired.png")
        Image.fromarray(face_arr).save(face_path)
        print(f"  ✓ Saved {name}_repaired.png")
    
    # Create repaired ERP
    print("\n[INFO] Creating ERP panorama (after repair)...")
    erp_after = faces_to_erp(repaired_faces, ERP_HEIGHT, ERP_WIDTH)
    Image.fromarray(erp_after).save(os.path.join(OUTPUT_DIR, "erp_after.png"))
    print("  ✓ Saved erp_after.png")
    
    # Also save as main output
    Image.fromarray(erp_after).save(os.path.join(OUTPUT_DIR, "equirectangular.png"))
    print("  ✓ Saved equirectangular.png (final output)")
    
    print("\n" + "="*60)
    print("✅ Pipeline complete!")
    print(f"[INFO] Output saved to: {OUTPUT_DIR}")
    print("  - erp_before.png: Before seam repair")
    print("  - erp_after.png: After seam repair (12 edges)")
    print("  - equirectangular.png: Final output")
    if DEBUG_SEAMS:
        print(f"  - debug/: Debug images for each edge")
    print("="*60)
