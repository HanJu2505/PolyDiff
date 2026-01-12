"""
PolyDiff Inference Script with IP-Adapter (ERP-level Seam Repair)

Generate 360° panoramas using:
  Stage 1: Generate 6 main cubemap faces using CubeDiff + IP-Adapter
  Stage 2: Repair seams using ERP-level SD Inpainting

Features:
  - Per-face IP-Adapter reference images
  - ERP-level seam repair (faster than edge-by-edge)

Usage:
    python inference_polydiff_ip.py
"""

import torch
import os
from PIL import Image
import torchvision.transforms as T
import numpy as np
import py360convert

# Use original CubeDiff pipeline for 6-view generation
from cubediff.pipelines.pipeline import CubeDiffPipeline
from diffusers.utils import load_image

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
                prompt: str = "smooth blending, natural continuation, matching colors and textures, coherent scene, photorealistic",
                negative_prompt: str = "abrupt change, color mismatch, inconsistent lighting, blurry, artificial, seam line, hard edge",
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
    IMAGE_FILENAME = "/home/dell/Datasets/Sun360/MiniVal_CubeMap/030003_front.png"
    
    # Prompts for each direction
    PROMPTS = {
        "Front": "Church stands between two buildings",
        "Right": "Car parked by road, sidewalk, and trees",
        "Back": "Cars parked along road with trees and sidewalk",
        "Left": "Car parked by road, tree, and street light",
        "Top": "sky",
        "Bottom": "street ",
    }
    
    # ============== IP-ADAPTER CONFIGURATION ==============
    # Enable/disable IP-Adapter
    USE_IP_ADAPTER = True 
    
    # IP-Adapter model settings
    IP_ADAPTER_REPO = "h94/IP-Adapter"
    IP_ADAPTER_SUBFOLDER = "models"
    IP_ADAPTER_WEIGHT_NAME = "ip-adapter_sd15.bin" # "ip-adapter_sd15.bin" or "ip-adapter-plus_sd15.bin"
    IP_ADAPTER_SCALE = 0.45  # Weight for IP-Adapter influence (0.0 - 1.0)
    
    # Per-face reference images (order: Front, Back, Left, Right, Top, Bottom)
    # Set to None to use conditioning image as reference
    FACE_REF_IMAGES = {
        "Front": None,  # Use None to skip, or provide path like "assets/ref_front.jpg"
        "Back": "/home/dell/Datasets/Sun360/MiniVal_CubeMap/030003_back.png",
        "Left": "/home/dell/Datasets/Sun360/MiniVal_CubeMap/030003_left.png",
        "Right": "/home/dell/Datasets/Sun360/MiniVal_CubeMap/030003_right.png",
        "Top": "/home/dell/Datasets/Sun360/MiniVal_CubeMap/030003_top.png",
        "Bottom": "/home/dell/Datasets/Sun360/MiniVal_CubeMap/030003_bottom.png",
    }
    
    # Alternative: Use a single image for all faces (global style)
    GLOBAL_REF_IMAGE = None
    
    # ============== LORA CONFIGURATION ==============
    # Enable/disable LoRA
    USE_LORA = True
    
    # LoRA model settings
    LORA_PATH = "./models/UNDERWATER_SCENE_v2.safetensors"
    LORA_TRIGGER_WORD = "UNDERWATER_SCENE, deep sea, blue water"
    LORA_SCALE = 0.9  # Strength: 0.5-0.8 recommended (too high = too blue)
    
    # =================================================
    
    # Model checkpoint (CubeDiff)
    CHECKPOINT = "./models/cubediff-512-multitxt"
    
    # Output directory
    IMAGE_NAME = os.path.splitext(os.path.basename(IMAGE_FILENAME))[0]
    OUTPUT_DIR = f"output/{IMAGE_NAME}_polydiff_ip_LoRA/"
    
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
    if USE_IP_ADAPTER:
        print("         IP-Adapter: ENABLED")
    print("="*60)
    
    # Load CubeDiff pipeline
    print(f"[INFO] Loading CubeDiff Pipeline from {CHECKPOINT}...")
    cubediff_pipe = CubeDiffPipeline.from_pretrained(CHECKPOINT).to(device)
    
    # ================= Load Underwater LoRA =================
    if USE_LORA:
        if os.path.exists(LORA_PATH):
            print(f"\n[INFO] Loading Underwater LoRA from: {LORA_PATH}")
            try:
                # 1. Load LoRA weights
                cubediff_pipe.load_lora_weights(LORA_PATH, adapter_name="underwater")
                
                # 2. Fuse LoRA into UNet (critical for PolyDiff's Attention compatibility)
                cubediff_pipe.fuse_lora(lora_scale=LORA_SCALE)
                print(f"[INFO] LoRA fused successfully with scale {LORA_SCALE}")
                
                # 3. Append trigger words to prompts
                print(f"[INFO] Appending trigger words: '{LORA_TRIGGER_WORD}'")
                if isinstance(PROMPTS, dict):
                    for face_key in PROMPTS:
                        PROMPTS[face_key] = f"{PROMPTS[face_key]}, {LORA_TRIGGER_WORD}"
                elif isinstance(PROMPTS, str):
                    PROMPTS = f"{PROMPTS}, {LORA_TRIGGER_WORD}"
                    
            except Exception as e:
                print(f"[WARNING] Failed to load LoRA: {e}")
                print("Continuing without LoRA...")
        else:
            print(f"[WARNING] LoRA file not found at {LORA_PATH}")
            print("Please download 'underwater_v1.safetensors' to the models folder.")
    # =========================================================
    
    # Load IP-Adapter if enabled
    ip_adapter_images = None
    if USE_IP_ADAPTER:
        print(f"[INFO] Loading IP-Adapter from {IP_ADAPTER_REPO}...")
        cubediff_pipe.load_ip_adapter(
            IP_ADAPTER_REPO, 
            subfolder=IP_ADAPTER_SUBFOLDER, 
            weight_name=IP_ADAPTER_WEIGHT_NAME
        )
        cubediff_pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
        print(f"[INFO] IP-Adapter loaded with scale={IP_ADAPTER_SCALE}")
        
        # Prepare reference images
        conditioning_pil = Image.open(IMAGE_FILENAME).convert("RGB")
        
        if GLOBAL_REF_IMAGE is not None:
            print(f"[INFO] Using global reference image: {GLOBAL_REF_IMAGE}")
            ip_adapter_images = load_image(GLOBAL_REF_IMAGE)
        else:
            # Collect per-face reference images
            ref_images_list = []
            face_order = ["Front", "Back", "Left", "Right", "Top", "Bottom"]
            
            print("[INFO] Preparing per-face IP-Adapter references...")
            for face_name in face_order:
                ref_path = FACE_REF_IMAGES.get(face_name)
                if ref_path is not None and os.path.exists(ref_path):
                    print(f"  - {face_name}: {ref_path}")
                    ref_images_list.append(load_image(ref_path))
                else:
                    # Use conditioning image as default reference
                    print(f"  - {face_name}: using conditioning image (default)")
                    ref_images_list.append(conditioning_pil)
            
            ip_adapter_images = ref_images_list
    
    # Load conditioning image
    print(f"[INFO] Loading conditioning image {IMAGE_FILENAME}...")
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    image = Image.open(IMAGE_FILENAME).convert("RGB")
    conditioning_image = transform(image)
    
    # Prepare prompts - handle both dict and string formats
    if isinstance(PROMPTS, dict):
        prompt_list = [
            PROMPTS.get("Front", ""),
            PROMPTS.get("Back", ""),
            PROMPTS.get("Left", ""),
            PROMPTS.get("Right", ""),
            PROMPTS.get("Top", ""),
            PROMPTS.get("Bottom", ""),
        ]
    else:
        # Single string prompt for all faces
        prompt_list = [PROMPTS] * 6
    
    # Generate 6 faces
    output = cubediff_pipe(
        prompts=prompt_list,
        conditioning_image=conditioning_image.unsqueeze(0).to(device),
        ip_adapter_image=[ip_adapter_images] if ip_adapter_images is not None else None,
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
    
    # =================== STAGE 2: Seam Repair (ERP-level) ===================
    print("\n" + "="*60)
    print("[Stage 2] Repairing seams with ERP-level SD Inpainting...")
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
    if USE_IP_ADAPTER:
        print(f"  - IP-Adapter scale: {IP_ADAPTER_SCALE}")
    print("="*60)
