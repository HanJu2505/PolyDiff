"""
PolyDiff-18 Inference Script

Generate 360° panoramas using 18-view over-complete generation.
"""

import torch
import os
from PIL import Image
from torchvision import transforms
from cubediff.pipelines.polydiff_pipeline import PolyDiffPipeline
from cubediff.modules.geometry import VIEW_NAMES_18


if __name__ == "__main__":
    # ============== USER CONFIGURATION ==============
    # Modify these variables directly in the script
    
    # Image filename (anchor/front view)
    # IMAGE_FILENAME = "/home/dell/Datasets/UIEB/raw-90/202_img_.png"
    IMAGE_FILENAME = "/home/dell/Datasets/Sun360/MiniVal_views/030001_front_up.png"

    # Prompts: can be a single string, list of 18 strings, or dict with directional keys
    # Option 1: Single prompt for all views
    # PROMPTS = "Underwater seabed with coral reefs, fish, and marine plants."
    
    # Option 2: Dict with directional prompts (main views get specific, seams get overall)
    # PROMPTS = {
    #     "Front": "Seabed with aquatic plants and small rocks; two human divers in underwater scene.",
    #     "Right": "Echinus on seabed; green aquatic plant visible below.",
    #     "Back": "Seabed with holothurian, starfish, and echinus; aquatic plants present.",
    #     "Left": "Holothurian and starfish on seabed; rocky underwater environment.",
    #     "Top": "Ocean surface",
    #     "Bottom": "Underwater seabed",
    #     "Overall": "Underwater seabed spans all directions; divers, marine animals, and plants occupy space."
    # }

    PROMPTS = {
        "Front": "Pool is next to lifeguard tower; trees in background.",
        "Right": "Diving board is next to pool and fence.",
        "Back": "Pool and slide are near trees.",
        "Left": "Pool with slide is surrounded by trees.",
        "Top": "sky with cloud",
        "Bottom": "Pool with slide",
        "Overall": "Pool occupies central area; trees surround perimeter, diving board and lifeguard tower at edges."
    }
    
    # Model checkpoint path (CubeDiff checkpoint)
    CHECKPOINT = "./models/cubediff-512-multitxt"
    
    # Output directory
    IMAGE_NAME = os.path.splitext(os.path.basename(IMAGE_FILENAME))[0]
    OUTPUT_DIR = f"output/{IMAGE_NAME}_polydiff/"
    
    # Generation parameters
    CFG_SCALE = 3.5
    NUM_INFERENCE_STEPS = 50
    FOV_DEG = 95.0
    ERP_HEIGHT = 1024
    ERP_WIDTH = 2048
    
    # Fusion parameters
    # "wta" = Winner-Takes-All (for debugging coordinate alignment)
    # "gaussian" = Soft blending (for final output)
    FUSION_MODE = "wta"
    EFFECTIVE_FOV = 70.0  # Only use center 70° of each 95° view
    
    # ================================================

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- Load Pipeline ----------------
    print(f"Loading PolyDiff-18 pipeline from: {CHECKPOINT}")
    pipe = PolyDiffPipeline.from_pretrained(CHECKPOINT)
    pipe = pipe.to(device)
    print(f"Pipeline loaded successfully and moved to {device}")

    image_size = pipe.vae.config.sample_size

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n[INFO] Loading anchor image: {IMAGE_FILENAME}")
    image = Image.open(IMAGE_FILENAME).convert("RGB")
    conditioning_image = transform(image)

    print("\n" + "="*60)
    print("[INFO] Starting PolyDiff-18 panorama generation...")
    print("="*60)
    
    output = pipe(
        prompts=PROMPTS,
        conditioning_image=conditioning_image.unsqueeze(0).to(device),
        num_inference_steps=NUM_INFERENCE_STEPS,
        cfg_scale=CFG_SCALE,
        fov_deg=FOV_DEG,
        erp_height=ERP_HEIGHT,
        erp_width=ERP_WIDTH,
        fusion_mode=FUSION_MODE,
        effective_fov_deg=EFFECTIVE_FOV,
    )

    # ---------------- Save Results ----------------
    print("\n[INFO] Post-processing and saving results...")
    
    # Save all 18 view images
    for i, view_name in enumerate(VIEW_NAMES_18):
        view_img = Image.fromarray(output.views[i])
        view_img.save(os.path.join(OUTPUT_DIR, f"{view_name}.png"))
        print(f"  ✓ Saved {view_name} view to {OUTPUT_DIR}{view_name}.png")
    
    # Save the equirectangular image
    equirec_img = Image.fromarray(output.equirectangular)
    equirec_img.save(os.path.join(OUTPUT_DIR, "equirectangular.png"))
    print(f"  ✓ Saved equirectangular panorama to {OUTPUT_DIR}equirectangular.png")

    print("\n" + "="*60)
    print("[INFO] ✨ PolyDiff-18 inference completed successfully!")
    print(f"[INFO] Output saved to: {OUTPUT_DIR}")
    print("="*60)
