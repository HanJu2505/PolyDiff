"""
Test the new precise seam mask on a polar edge.
"""

import os
import numpy as np
from PIL import Image
from cubediff.pipelines.seam_repair import (
    create_precise_seam_mask,
    project_faces_to_seam_view,
    EDGE_DEFINITIONS,
    FACE_NAMES
)

if __name__ == "__main__":
    output_dir = "output/mask_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test on edge 4 (top-front) - should show curved mask following actual boundary
    edge_id = 4
    edge_def = EDGE_DEFINITIONS[edge_id]
    _, face_a_idx, face_b_idx, yaw_deg, pitch_deg = edge_def
    
    print(f"Testing Edge {edge_id}: {FACE_NAMES[face_a_idx]}-{FACE_NAMES[face_b_idx]}")
    print(f"  yaw={yaw_deg}°, pitch={pitch_deg}°")
    
    # Create precise mask with wider coverage
    mask = create_precise_seam_mask(
        face_a_idx, face_b_idx,
        yaw_deg, pitch_deg,
        size=512,
        seam_width=64,   # Match run_seam_repair.py default
        feather=32
    )
    
    # Save mask
    mask_img = (mask * 255).astype(np.uint8)
    Image.fromarray(mask_img).save(f"{output_dir}/edge{edge_id}_precise_mask.png")
    print(f"  Saved: {output_dir}/edge{edge_id}_precise_mask.png")
    
    # Test all 12 edges
    print("\nGenerating precise masks for all edges...")
    for edge_def in EDGE_DEFINITIONS:
        edge_id, face_a_idx, face_b_idx, yaw_deg, pitch_deg = edge_def
        
        mask = create_precise_seam_mask(
            face_a_idx, face_b_idx,
            yaw_deg, pitch_deg,
            size=512,
            seam_width=64,
            feather=32
        )
        
        mask_img = (mask * 255).astype(np.uint8)
        Image.fromarray(mask_img).save(f"{output_dir}/edge{edge_id:02d}_mask.png")
        print(f"  Edge {edge_id:2d}: {FACE_NAMES[face_a_idx]:6s}-{FACE_NAMES[face_b_idx]:6s}")
    
    print(f"\n✅ All masks saved to {output_dir}/")
