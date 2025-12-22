"""
Test script for tiered_fusion_v2 - uses py360convert for perfect main view coverage.
"""

import os
import numpy as np
from PIL import Image
from cubediff.pipelines.fusion import tiered_fusion_v2
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
    
    # Test 1: Main views only (no seam blending)
    print("\n===== Test 1: Main Views Only (py360convert) =====")
    erp_main_only = tiered_fusion_v2(
        views, VIEW_CONFIG_18, 
        fov_deg=95.0,
        crop_to_90=True,
        blend_seam_views=False
    )
    Image.fromarray(erp_main_only).save(f"{output_dir}/erp_v2_main_only.png")
    print(f"Saved to {output_dir}/erp_v2_main_only.png")
    
    # Test 2: Main views + seam blending ONLY at seam lines
    print("\n===== Test 2: Main Views + Seam-Line-Only Blending =====")
    erp_with_seams = tiered_fusion_v2(
        views, VIEW_CONFIG_18, 
        fov_deg=95.0,
        crop_to_90=True,
        blend_seam_views=True,
        seam_start_deg=85.0,  # Seam blend starts at 85° from center
        seam_end_deg=90.0     # Seam blend ends at 90° (edge)
    )
    Image.fromarray(erp_with_seams).save(f"{output_dir}/erp_v2_seam_blend.png")
    print(f"Saved to {output_dir}/erp_v2_seam_blend.png")
    
    print("\n✅ Done! Compare:")
    print("   - erp_v2_main_only.png: Pure cubemap (may have hard seams)")
    print("   - erp_v2_seam_blend.png: Seams blended (core untouched, only edges smoothed)")
