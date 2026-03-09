import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
import os

if __name__ == "__main__":
    # ============== USER CONFIGURATION ==============
    
    # Text prompt for image generation
    PROMPT = "Ocean surface underwater view with sunlight rays penetrating the water"
    
    # Negative prompt (optional, helps avoid unwanted elements)
    NEGATIVE_PROMPT = "blurry, low quality, distorted, ugly"
    
    # Model checkpoint
    MODEL_ID = "SG161222/Realistic_Vision_V5.1_noVAE"
    
    # Output directory
    OUTPUT_DIR = "output/sd15_generation/"
    
    # Generation parameters
    NUM_INFERENCE_STEPS = 50  # More steps = higher quality but slower
    GUIDANCE_SCALE = 7.5      # Higher = more closely follows prompt (typical: 7-10)
    WIDTH = 512               # Image width (SD 1.5 works best at 512x512)
    HEIGHT = 512              # Image height
    SEED = 42                 # Set to None for random generation
    
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

    # Optional: Enable memory optimizations for lower VRAM usage
    # Uncomment these if you run out of GPU memory:
    # pipe.enable_attention_slicing()
    # pipe.enable_vae_slicing()

    # Set up generator for reproducibility
    generator = torch.Generator(device=device)
    if SEED is not None:
        generator.manual_seed(SEED)
        print(f"[INFO] Using seed: {SEED}")

    # Generate image
    print("\n" + "="*60)
    print(f"[INFO] Generating image...")
    print(f"[INFO] Prompt: {PROMPT}")
    print("="*60 + "\n")

    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        width=WIDTH,
        height=HEIGHT,
        generator=generator,
    ).images[0]

    # Save the generated image
    output_path = os.path.join(OUTPUT_DIR, "generated_image.png")
    image.save(output_path)
    
    print("\n" + "="*60)
    print(f"[INFO] ✨ Image generated successfully!")
    print(f"[INFO] Saved to: {output_path}")
    print("="*60)
