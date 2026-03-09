"""
Stable Diffusion 1.5 Inference with IP-Adapter

Generate images using SD1.5 with IP-Adapter for style/content transfer.

Usage:
    python inference_sd15_ipadapter.py
"""

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from diffusers.utils import load_image
import os

if __name__ == "__main__":
    # ============== USER CONFIGURATION ==============
    
    # Text prompt for image generation
    PROMPT = "Ocean surface underwater view with sunlight rays penetrating the water"
    
    # Negative prompt
    NEGATIVE_PROMPT = "blurry, low quality, distorted, ugly"
    
    # Model checkpoint
    MODEL_ID = "SG161222/Realistic_Vision_V5.1_noVAE"
    
    # Output directory
    OUTPUT_DIR = "output/sd15_ipadapter_generation/"
    
    # ============== IP-ADAPTER CONFIGURATION ==============
    
    # Enable/disable IP-Adapter
    USE_IP_ADAPTER = True
    
    # IP-Adapter model settings
    IP_ADAPTER_REPO = "h94/IP-Adapter"
    IP_ADAPTER_SUBFOLDER = "models"
    IP_ADAPTER_WEIGHT_NAME = "ip-adapter_sd15.bin"  # or "ip-adapter-plus_sd15.bin"
    IP_ADAPTER_SCALE = 0.45  # Weight for IP-Adapter influence (0.0 - 1.0)
    
    # Reference image for style/content transfer
    REFERENCE_IMAGE = "/home/dell/Datasets/UIEB/raw-90/263_img_.png"
    
    # ============== GENERATION PARAMETERS ==============
    
    NUM_INFERENCE_STEPS = 50
    GUIDANCE_SCALE = 7.5
    WIDTH = 512
    HEIGHT = 512
    SEED = 42  # Set to None for random
    
    # ================================================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the Stable Diffusion pipeline
    print(f"\n[INFO] Loading Stable Diffusion 1.5 from: {MODEL_ID}")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipe = pipe.to(device)
    print(f"[INFO] Pipeline loaded successfully")

    # Load IP-Adapter if enabled
    if USE_IP_ADAPTER:
        print(f"\n[INFO] Loading IP-Adapter...")
        pipe.load_ip_adapter(
            IP_ADAPTER_REPO,
            subfolder=IP_ADAPTER_SUBFOLDER,
            weight_name=IP_ADAPTER_WEIGHT_NAME,
            local_files_only=True,  # Use cached model to avoid network issues
        )
        pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
        print(f"[INFO] IP-Adapter loaded with scale={IP_ADAPTER_SCALE}")
        
        # Load reference image
        print(f"[INFO] Loading reference image: {REFERENCE_IMAGE}")
        ip_image = load_image(REFERENCE_IMAGE)
    else:
        ip_image = None

    # Set up generator for reproducibility
    generator = torch.Generator(device=device)
    if SEED is not None:
        generator.manual_seed(SEED)
        print(f"[INFO] Using seed: {SEED}")

    # Generate image
    print("\n" + "="*60)
    print(f"[INFO] Generating image...")
    print(f"[INFO] Prompt: {PROMPT}")
    if USE_IP_ADAPTER:
        print(f"[INFO] IP-Adapter scale: {IP_ADAPTER_SCALE}")
        print(f"[INFO] Reference: {REFERENCE_IMAGE}")
    print("="*60 + "\n")

    # Build generation kwargs
    gen_kwargs = {
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "width": WIDTH,
        "height": HEIGHT,
        "generator": generator,
    }
    
    # Add IP-Adapter image if enabled
    if USE_IP_ADAPTER and ip_image is not None:
        gen_kwargs["ip_adapter_image"] = ip_image

    image = pipe(**gen_kwargs).images[0]

    # Save the generated image
    output_path = os.path.join(OUTPUT_DIR, "generated_image.png")
    image.save(output_path)
    
    # Also save reference image for comparison
    if USE_IP_ADAPTER and ip_image is not None:
        ref_save_path = os.path.join(OUTPUT_DIR, "reference_image.png")
        ip_image.save(ref_save_path)
        print(f"[INFO] Reference saved to: {ref_save_path}")
    
    print("\n" + "="*60)
    print(f"[INFO] ✨ Image generated successfully!")
    print(f"[INFO] Saved to: {output_path}")
    print("="*60)
