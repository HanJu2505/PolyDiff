"""
Inference script for SD1.5 conv_out trained model.

This script uses the original SD1.5 pipeline (4-channel input)
and loads conv_out weights trained with train_convout.py.
"""

import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from safetensors.torch import load_file


def load_trained_convout(pipe, checkpoint_dir):
    """Load trained conv_out weights from checkpoint."""
    model_path = os.path.join(checkpoint_dir, "model.safetensors")
    
    if not os.path.exists(model_path):
        print(f"[WARNING] No model.safetensors found at {model_path}")
        return pipe
    
    print(f"[INFO] Loading trained conv_out weights from {model_path}...")
    state_dict = load_file(model_path)
    
    # Filter only conv_out weights
    conv_out_weights = {k: v for k, v in state_dict.items() if "conv_out" in k}
    print(f"[INFO] Found {len(conv_out_weights)} conv_out parameters")
    
    # Load with strict=False to only load matching keys
    missing, unexpected = pipe.unet.load_state_dict(state_dict, strict=False)
    print(f"[INFO] Loaded weights - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    return pipe


def cubemap_to_equirectangular(faces, output_width=2048, output_height=1024):
    """Convert 6 cubemap faces to equirectangular projection."""
    
    # Face order: front, back, left, right, top, bottom
    front, back, left, right, top, bottom = faces
    face_size = front.shape[0]
    
    # Create output
    equirect = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # Generate spherical coordinates
    theta = np.linspace(0, 2 * np.pi, output_width, endpoint=False)  # longitude
    phi = np.linspace(0, np.pi, output_height)  # latitude
    theta, phi = np.meshgrid(theta, phi)
    
    # Convert to 3D unit sphere coordinates
    x = np.sin(phi) * np.sin(theta)
    y = np.cos(phi)
    z = np.sin(phi) * np.cos(theta)
    
    # For each pixel, determine which face and sample
    abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)
    
    # Determine dominant axis
    max_axis = np.maximum(np.maximum(abs_x, abs_y), abs_z)
    
    for i in range(output_height):
        for j in range(output_width):
            px, py, pz = x[i, j], y[i, j], z[i, j]
            ax, ay, az = abs_x[i, j], abs_y[i, j], abs_z[i, j]
            
            if ax >= ay and ax >= az:
                if px > 0:  # right
                    u = (-pz / ax + 1) / 2
                    v = (-py / ax + 1) / 2
                    face = right
                else:  # left
                    u = (pz / ax + 1) / 2
                    v = (-py / ax + 1) / 2
                    face = left
            elif ay >= ax and ay >= az:
                if py > 0:  # top
                    u = (px / ay + 1) / 2
                    v = (pz / ay + 1) / 2
                    face = top
                else:  # bottom
                    u = (px / ay + 1) / 2
                    v = (-pz / ay + 1) / 2
                    face = bottom
            else:
                if pz > 0:  # front
                    u = (px / az + 1) / 2
                    v = (-py / az + 1) / 2
                    face = front
                else:  # back
                    u = (-px / az + 1) / 2
                    v = (-py / az + 1) / 2
                    face = back
            
            # Sample from face
            u = min(max(int(u * face_size), 0), face_size - 1)
            v = min(max(int(v * face_size), 0), face_size - 1)
            equirect[i, j] = face[v, u]
    
    return equirect


if __name__ == "__main__":
    # ============== USER CONFIGURATION ==============
    
    # Trained checkpoint path
    TRAINED_CHECKPOINT = "./checkpoints/sd15-convout/sd15-convout-only/epoch_20_step_220_final"
    
    # Base model (must be SD1.5, same as training)
    BASE_MODEL = "runwayml/stable-diffusion-v1-5"
    
    # Input conditioning image (front view)
    IMAGE_FILENAME = "/home/dell/Datasets/Underwater360/cubemap/360underwater2_3_front.png"
    
    # Prompts for each direction
    PROMPTS = {
        "Front": "Fish swim near jellyfish and aquatic plants; cuttlefish is below.",
        "Right": "Fish swim in water; sea floor below.",
        "Back": "Fish are in water surrounded by aquatic plants.",
        "Left": "Diver is near reef; fish surround the area.",
        "Top": "Ocean surface ",
        "Bottom": "Ocean floor ",
    }
    
    # IP-Adapter configuration
    USE_IP_ADAPTER = True
    IP_ADAPTER_SCALE = 0.60
    
    # Per-face reference images
    FACE_REF_IMAGES = {
        "Front": "/home/dell/Datasets/Underwater360/cubemap/360underwater2_3_front.png",
        "Back": "/home/dell/Datasets/Underwater360/cubemap/360underwater2_3_back.png",
        "Left": "/home/dell/Datasets/Underwater360/cubemap/360underwater2_3_left.png",
        "Right": "/home/dell/Datasets/Underwater360/cubemap/360underwater2_3_right.png",
        "Top": "/home/dell/Datasets/Underwater360/cubemap/360underwater2_3_top.png",
        "Bottom": "/home/dell/Datasets/Underwater360/cubemap/360underwater2_3_bottom.png",
    }
    
    # Output
    IMAGE_NAME = os.path.splitext(os.path.basename(IMAGE_FILENAME))[0]
    OUTPUT_DIR = f"output/{IMAGE_NAME}_sd15_convout/"
    
    # Generation parameters
    CFG_SCALE = 7.5
    NUM_INFERENCE_STEPS = 30
    
    # ============== MAIN EXECUTION ==============
    
    print("=" * 60)
    print("[Stage 1] Generating 6 cubemap faces with SD1.5 + trained conv_out")
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
    pipe = load_trained_convout(pipe, TRAINED_CHECKPOINT)
    
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
    equirect = cubemap_to_equirectangular(generated_faces)
    equirect_path = os.path.join(OUTPUT_DIR, "equirectangular.png")
    Image.fromarray(equirect).save(equirect_path)
    print(f"[INFO] Saved: {equirect_path}")
    
    print("\n" + "=" * 60)
    print(f"[DONE] All outputs saved to {OUTPUT_DIR}")
    print("=" * 60)
