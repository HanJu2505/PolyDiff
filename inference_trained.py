"""
PolyDiff Inference Script with Trained Weights + IP-Adapter

Generate 360° panoramas using trained CubeDiff model:
  Stage 1: Generate 6 main cubemap faces using CubeDiff + IP-Adapter
  Stage 2: Repair 12 seams individually using SD Inpainting

Features:
  - Load trained UNet weights from checkpoint
  - Per-face IP-Adapter reference images
  - Edge-by-edge seam repair
  - Option to replace front face with original to preserve quality

Usage:
    python inference_trained.py
"""

import torch
import os
from PIL import Image
import torchvision.transforms as T
import numpy as np
import py360convert
from safetensors.torch import load_file

# Use original CubeDiff pipeline for 6-view generation
from cubediff.pipelines.pipeline import CubeDiffPipeline
from cubediff.pipelines.seam_repair import repair_all_seams, FACE_NAMES

# For SD Inpainting
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image


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


def load_trained_unet_weights(pipe, checkpoint_path):
    """
    Load trained UNet weights from Accelerator checkpoint.
    """
    model_path = os.path.join(checkpoint_path, "model.safetensors")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    print(f"[INFO] Loading trained weights from {model_path}...")
    
    # Load the safetensors file
    state_dict = load_file(model_path)
    
    # Remove potential prefixes from Accelerator
    unet_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("module."):
            new_key = key[7:]
        unet_state_dict[new_key] = value
    
    # Load state dict into UNet
    missing, unexpected = pipe.unet.load_state_dict(unet_state_dict, strict=False)
    
    print(f"[INFO] Loaded weights - Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    
    return pipe


def create_inpaint_fn(device="cuda", 
                      prompt="smooth blending, natural continuation, matching colors and textures, coherent scene, photorealistic",
                      negative_prompt="abrupt change, color mismatch, inconsistent lighting, blurry, artificial, seam line, hard edge",
                      num_inference_steps=20, 
                      strength=0.55):
    """Create SD Inpainting function for seam repair."""
    
    print("[Inpaint] Loading SD Inpainting model...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        safety_checker=None,
        local_files_only=True,  # Use cached model
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
    
    # 训练好的检查点路径
    TRAINED_CHECKPOINT = "./checkpoints/polydiff-multitext-ipadapter/polydiff-multitext-ipadapter/epoch_20_step_200_final"
    
    # 是否加载训练权重 (设为 False 可对比原始模型效果)
    LOAD_TRAINED_WEIGHTS = True
    
    # 基础模型路径 (用于加载完整 pipeline)
    BASE_MODEL = "./models/cubediff-512-multitxt"
    
    # Input image (front view anchor)
    IMAGE_FILENAME = "/home/dell/Datasets/UIEB/raw-90/268_img_.png"
    
    # Prompts for each direction
    PROMPTS = {
        "Front": "Fish swim near jellyfish and aquatic plants; cuttlefish is below.",
        "Right": "Fish swim in water; sea floor below.",
        "Back": "Fish are in water surrounded by aquatic plants.",
        "Left": "Diver is near reef; fish surround the area.",
        "Top": "Ocean surface ",
        "Bottom": "Ocean floor ",
    }
    
    # ============== IP-ADAPTER CONFIGURATION ==============
    USE_IP_ADAPTER = True 
    
    IP_ADAPTER_REPO = "h94/IP-Adapter"
    IP_ADAPTER_SUBFOLDER = "models"
    IP_ADAPTER_WEIGHT_NAME = "ip-adapter_sd15.bin"
    IP_ADAPTER_SCALE = 0.99
    
    # Per-face reference images
    FACE_REF_IMAGES = {
        "Front": None,  # Use None to skip, or provide path like "assets/ref_front.jpg"
        "Back": "/home/dell/Datasets/UIIS/UDW/extracted_objects/fish/XL_1176_ann697_0011.png",
        "Left": "/home/dell/Datasets/UIIS/UDW/extracted_objects/fish/XL_1176_ann697_0011.png",
        "Right": "/home/dell/Datasets/UIIS/UDW/extracted_objects/fish/XL_1176_ann697_0011.png",
        "Top": "/home/dell/Datasets/UIIS/UDW/extracted_objects/fish/XL_1176_ann697_0011.png",
        "Bottom": "/home/dell/Datasets/UIIS/UDW/extracted_objects/fish/XL_1176_ann697_0011.png",
    }
    
    GLOBAL_REF_IMAGE = None
    
    # ============== FRONT FACE REPLACEMENT ==============
    # Replace front face with original image to avoid VAE quality loss
    REPLACE_FRONT_WITH_ORIGINAL = False
    
    # =================================================
    
    # Output directory
    IMAGE_NAME = os.path.splitext(os.path.basename(IMAGE_FILENAME))[0]
    OUTPUT_DIR = f"output/{IMAGE_NAME}_trained/"
    
    # Generation parameters
    CFG_SCALE = 3.5
    NUM_INFERENCE_STEPS = 50
    ERP_HEIGHT = 1024
    ERP_WIDTH = 2048
    
    # Seam repair parameters
    SEAM_WIDTH = 50
    FEATHER = 30
    INPAINT_STEPS = 20
    INPAINT_STRENGTH = 0.55
    DEBUG_SEAMS = True
    
    # ================================================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    debug_dir = os.path.join(OUTPUT_DIR, "debug") if DEBUG_SEAMS else None
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
    
    # =================== STAGE 1: Generate 6 Faces ===================
    print("\n" + "="*60)
    print("[Stage 1] Generating 6 cubemap faces with Trained CubeDiff...")
    print(f"         Checkpoint: {TRAINED_CHECKPOINT}")
    if USE_IP_ADAPTER:
        print("         IP-Adapter: ENABLED")
    print("="*60)
    
    # Load CubeDiff pipeline
    print(f"[INFO] Loading base CubeDiff Pipeline from {BASE_MODEL}...")
    cubediff_pipe = CubeDiffPipeline.from_pretrained(BASE_MODEL).to(device)
    
    # Load IP-Adapter if enabled
    ip_adapter_images = None
    if USE_IP_ADAPTER:
        print(f"[INFO] Loading IP-Adapter from cache...")
        cubediff_pipe.load_ip_adapter(
            IP_ADAPTER_REPO, 
            subfolder=IP_ADAPTER_SUBFOLDER, 
            weight_name=IP_ADAPTER_WEIGHT_NAME,
            local_files_only=True
        )
        cubediff_pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
        print(f"[INFO] IP-Adapter loaded with scale={IP_ADAPTER_SCALE}")
        
        # Prepare reference images
        conditioning_pil = Image.open(IMAGE_FILENAME).convert("RGB")
        
        if GLOBAL_REF_IMAGE is not None:
            print(f"[INFO] Using global reference image: {GLOBAL_REF_IMAGE}")
            ip_adapter_images = load_image(GLOBAL_REF_IMAGE)
        else:
            ref_images_list = []
            face_order = ["Front", "Back", "Left", "Right", "Top", "Bottom"]
            
            print("[INFO] Preparing per-face IP-Adapter references...")
            for face_name in face_order:
                ref_path = FACE_REF_IMAGES.get(face_name)
                if ref_path is not None and os.path.exists(ref_path):
                    print(f"  - {face_name}: {ref_path}")
                    ref_images_list.append(load_image(ref_path))
                else:
                    print(f"  - {face_name}: using conditioning image (default)")
                    ref_images_list.append(conditioning_pil)
            
            ip_adapter_images = ref_images_list
    
    # Load trained weights (if enabled)
    if LOAD_TRAINED_WEIGHTS:
        print("\n[INFO] Loading trained UNet weights...")
        cubediff_pipe = load_trained_unet_weights(cubediff_pipe, TRAINED_CHECKPOINT)
        print("[INFO] Trained weights loaded successfully!")
    else:
        print("\n[INFO] Using original model weights (no fine-tuning)")
    
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
        prompt_list = [PROMPTS] * 6
    
    # Generate 6 faces
    output = cubediff_pipe(
        prompts=prompt_list,
        conditioning_image=conditioning_image.unsqueeze(0).to(device),
        ip_adapter_image=[ip_adapter_images] if ip_adapter_images is not None else None,
        num_inference_steps=NUM_INFERENCE_STEPS,
        cfg_scale=CFG_SCALE,
    )
    
    # Get face arrays
    faces = output.faces_cropped
    face_arrays = list(faces)  # Convert to list for modification
    
    # Replace front face with original if enabled
    if REPLACE_FRONT_WITH_ORIGINAL:
        print("\n[INFO] Replacing front face with original image to preserve quality...")
        face_size = face_arrays[0].shape[0]
        original_front = np.array(Image.open(IMAGE_FILENAME).convert("RGB").resize((face_size, face_size)))
        face_arrays[0] = original_front
    
    # Save face images
    print("\n[INFO] Saving 6 face images...")
    for face_img, name in zip(face_arrays, FACE_NAMES):
        face_path = os.path.join(OUTPUT_DIR, f"{name}.png")
        Image.fromarray(face_img).save(face_path)
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
    print(f"  - Trained checkpoint: {TRAINED_CHECKPOINT}")
    if USE_IP_ADAPTER:
        print(f"  - IP-Adapter scale: {IP_ADAPTER_SCALE}")
    if REPLACE_FRONT_WITH_ORIGINAL:
        print("  - Front face: Replaced with original")
    if DEBUG_SEAMS:
        print(f"  - debug/: Debug images for each edge")
    print("="*60)
