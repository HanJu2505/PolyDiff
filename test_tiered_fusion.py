"""
Test script for tiered fusion - loads existing views and applies new fusion.
"""

import os
import numpy as np
from PIL import Image
from cubediff.pipelines.fusion import tiered_fusion, inverse_mapping_fusion
from cubediff.modules.geometry import VIEW_CONFIG_18, VIEW_NAMES_18

if __name__ == "__main__":
    output_dir = "output/030002_front_up_polydiff"
    
    # Load all 18 views
    print("Loading views...")
    views = []
    for i, name in enumerate(VIEW_NAMES_18):
        path = f"{output_dir}/{name}.png"
        if os.path.exists(path):
            img = np.array(Image.open(path).convert("RGB"))
            print(f"  View {i}: {name}, shape={img.shape}")
            views.append(img)
        else:
            print(f"  View {i}: {name} NOT FOUND!")
            views.append(np.zeros((512, 512, 3), dtype=np.uint8))
    
    print(f"\nTotal views: {len(views)}")
    
    # Test Tiered Fusion (new)
    print("\n===== Testing Tiered Fusion (Main View Priority) =====")
    erp_tiered = tiered_fusion(
        views, VIEW_CONFIG_18, 
        fov_deg=95.0,
        valid_fov_deg=90.0,   # Only use center 90° (discard outer 5°)
        core_fov_deg=80.0,    # Main view dominates in core 80°
        sigma_factor=0.3
    )
    Image.fromarray(erp_tiered).save(f"{output_dir}/erp_tiered.png")
    print(f"Saved to {output_dir}/erp_tiered.png")
    
    # Compare with Gaussian (old)
    print("\n===== Testing Gaussian (For Comparison) =====")
    erp_gaussian = inverse_mapping_fusion(
        views, VIEW_CONFIG_18, 
        fov_deg=95.0,
        mode="gaussian",
        effective_fov_deg=95.0,
        sigma_factor=0.25
    )
    Image.fromarray(erp_gaussian).save(f"{output_dir}/erp_gaussian_compare.png")
    print(f"Saved to {output_dir}/erp_gaussian_compare.png")
    
    print("\n✅ Done! Compare erp_tiered.png vs erp_gaussian_compare.png")
    print("   Tiered should be sharper in main view regions.")
