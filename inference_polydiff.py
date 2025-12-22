"""
PolyDiff-18 Inference Script

Generate 360° panoramas using 18-view over-complete generation.
"""

import torch
import os
from PIL import Image
import torchvision.transforms as T
from cubediff.pipelines.polydiff_pipeline import PolyDiffPipeline
from cubediff.modules.geometry import VIEW_NAMES_18


if __name__ == "__main__":
    # ============== USER CONFIGURATION ==============
    # Modify these variables directly in the script
    
    # Image filename (anchor/front view)
    # IMAGE_FILENAME = "/home/dell/Datasets/UIEB/raw-90/202_img_.png"
    IMAGE_FILENAME = "/home/dell/Datasets/Sun360/MiniVal_views/030002_front_up.png"

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
        "Front": "Person walks on cobblestone street",
        "Right": "Statue stands before building; chairs and umbrellas nearby.",
        "Back": "Statues stand before buildings across left and right rear views",
        "Left": "Statue stands before buildings",
        "Top": "sky with sun",
        "Bottom": "street area",
        "Overall": "Multiple statues, buildings, umbrellas, chairs, and people distributed around a central street area."
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
    # "gaussian" = Soft blending (all views equal weight)
    # "tiered" = Main views priority + seam views for edge blending (RECOMMENDED)
    FUSION_MODE = "tiered"
    # IMPORTANT: Must be >= FOV_DEG to ensure full coverage from 6 main views
    EFFECTIVE_FOV = 90.0  # Only used for wta/gaussian modes
    
    # New: Two-Stage Generation
    TWO_STAGE = True
    
    # ================================================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load pipeline
    print(f"[INFO] Loading PolyDiff Pipeline from {CHECKPOINT}...")
    pipe = PolyDiffPipeline.from_pretrained(
        CHECKPOINT,
        num_faces=18 
    ).to(device)
    
    # Load conditioning image (Front view anchor)
    print(f"[INFO] Loading conditioning image {IMAGE_FILENAME}...")
    
    image_size = pipe.vae.config.sample_size  # Should be 64 for latent, but 512 for input
    
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    image = Image.open(IMAGE_FILENAME).convert("RGB")
    conditioning_image = transform(image)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*60)
    print("[INFO] Starting PolyDiff-18 panorama generation...")
    print(f"       Mode: {'Two-Stage' if TWO_STAGE else 'Single-Stage'}")
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
        two_stage=TWO_STAGE,
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
