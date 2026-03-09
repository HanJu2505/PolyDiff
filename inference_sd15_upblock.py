"""
Inference script for SD1.5 up_block trained model.

This script uses the original SD1.5 pipeline (4-channel input)
and loads up_blocks[3] + conv_out weights trained with train_upblock.py.
"""

import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from safetensors.torch import load_file
import py360convert


def load_trained_upblock(pipe, checkpoint_dir):
    """Load trained up_blocks[3] + conv_out weights from checkpoint."""
    model_path = os.path.join(checkpoint_dir, "model.safetensors")
    
    if not os.path.exists(model_path):
        print(f"[WARNING] No model.safetensors found at {model_path}")
        return pipe
    
    print(f"[INFO] Loading trained up_block weights from {model_path}...")
    state_dict = load_file(model_path)
    
    # Filter up_block and conv_out weights
    upblock_weights = {k: v for k, v in state_dict.items() if "up_blocks.3" in k or "conv_out" in k}
    print(f"[INFO] Found {len(upblock_weights)} up_block + conv_out parameters")
    
    # Load with strict=False to only load matching keys
    missing, unexpected = pipe.unet.load_state_dict(state_dict, strict=False)
    print(f"[INFO] Loaded weights - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    return pipe


def faces_to_erp(faces, erp_height=1024, erp_width=2048):
    """Convert 6 faces to ERP panorama using py360convert.
    
    Args:
        faces: List of 6 numpy arrays [front, back, left, right, top, bottom]
        erp_height: Output height
        erp_width: Output width
    
    Returns:
        ERP image as numpy array
    """
    cube_dict = {
        "F": faces[0],  # front
        "B": faces[1],  # back
        "L": faces[2],  # left
        "R": faces[3],  # right
        "U": faces[4],  # top
        "D": faces[5],  # bottom
    }
    return py360convert.c2e(cube_dict, h=erp_height, w=erp_width, cube_format='dict')


if __name__ == "__main__":
    # ============== USER CONFIGURATION ==============
    
    # Trained checkpoint path
    TRAINED_CHECKPOINT = "./checkpoints/sd15-upblock/sd15-upblock-only/epoch_20_step_220_final"
    
    # Base model (must be SD1.5, same as training)
    BASE_MODEL = "runwayml/stable-diffusion-v1-5"
    
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
    
    # IP-Adapter configuration
    USE_IP_ADAPTER = True
    IP_ADAPTER_SCALE = 0.60
    
    # Per-face reference images
    FACE_REF_IMAGES = {
        "Front": "/home/dell/Datasets/Sun360/MiniVal_CubeMap/030003_front.png",  # Use None to skip, or provide path like "assets/ref_front.jpg"
        "Back": "/home/dell/Datasets/Sun360/MiniVal_CubeMap/030003_back.png",
        "Left": "/home/dell/Datasets/Sun360/MiniVal_CubeMap/030003_left.png",
        "Right": "/home/dell/Datasets/Sun360/MiniVal_CubeMap/030003_right.png",
        "Top": "/home/dell/Datasets/Sun360/MiniVal_CubeMap/030003_top.png",
        "Bottom": "/home/dell/Datasets/Sun360/MiniVal_CubeMap/030003_bottom.png",
    }
    
    # Output
    IMAGE_NAME = os.path.splitext(os.path.basename(IMAGE_FILENAME))[0]
    OUTPUT_DIR = f"output/{IMAGE_NAME}_sd15_upblock/"
    
    # Generation parameters
    CFG_SCALE = 7.5
    NUM_INFERENCE_STEPS = 30
    
    # ============== MAIN EXECUTION ==============
    
    print("=" * 60)
    print("[Stage 1] Generating 6 cubemap faces with SD1.5 + trained up_block")
    print(f"         Checkpoint: {TRAINED_CHECKPOINT}")
    print(f"         IP-Adapter: {'ENABLED' if USE_IP_ADAPTER else 'DISABLED'}")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load SD1.5 pipeline
    print(f"[INFO] Loading SD1.5 from {BASE_MODEL}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        local_files_only=True
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    
    # Load IP-Adapter
    if USE_IP_ADAPTER:
        print("[INFO] Loading IP-Adapter...")
        pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter_sd15.bin"
        )
        pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
        print(f"[INFO] IP-Adapter loaded with scale={IP_ADAPTER_SCALE}")
    
    # Load trained conv_out weights
    pipe = load_trained_upblock(pipe, TRAINED_CHECKPOINT)
    
    # Generate 6 faces
    face_order = ["Front", "Back", "Left", "Right", "Top", "Bottom"]
    generated_faces = []
    
    for face_name in face_order:
        print(f"\n[INFO] Generating {face_name} face...")
        
        prompt = PROMPTS[face_name]
        
        # Prepare IP-Adapter image
        ip_adapter_image = None
        if USE_IP_ADAPTER and FACE_REF_IMAGES.get(face_name):
            ref_path = FACE_REF_IMAGES[face_name]
            if os.path.exists(ref_path):
                ip_adapter_image = Image.open(ref_path).convert("RGB")
                print(f"  Using ref: {os.path.basename(ref_path)}")
        
        # Generate
        with torch.no_grad():
            output = pipe(
                prompt=prompt,
                ip_adapter_image=ip_adapter_image,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=CFG_SCALE,
                generator=torch.Generator("cuda").manual_seed(42),
            )
        
        face_image = output.images[0]
        generated_faces.append(np.array(face_image))
        
        # Save individual face
        face_path = os.path.join(OUTPUT_DIR, f"{face_name.lower()}.png")
        face_image.save(face_path)
        print(f"  Saved: {face_path}")
    
    # Convert to equirectangular
    print("\n[INFO] Converting to equirectangular...")
    equirect = faces_to_erp(generated_faces)
    equirect_path = os.path.join(OUTPUT_DIR, "equirectangular.png")
    Image.fromarray(equirect).save(equirect_path)
    print(f"[INFO] Saved: {equirect_path}")
    
    print("\n" + "=" * 60)
    print(f"[DONE] All outputs saved to {OUTPUT_DIR}")
    print("=" * 60)
