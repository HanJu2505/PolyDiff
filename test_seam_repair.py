"""
Test seam repair with real cubemap images from CubeDiff output.
"""

import os
import numpy as np
from PIL import Image
from cubediff.pipelines.seam_repair import (
    project_faces_to_seam_view,
    create_seam_mask,
    paste_back_to_face,
    EDGE_DEFINITIONS,
    FACE_NAMES
)

# Dummy inpaint function for testing (just returns the image as-is)
def dummy_inpaint(image, mask):
    """Just return image as-is (no actual inpainting)"""
    return image

# Simple blur inpaint for testing
def blur_inpaint(image, mask):
    """Simple blur-based inpaint for testing"""
    from scipy.ndimage import gaussian_filter
    
    result = image.copy().astype(float)
    
    # Blur the masked region
    for c in range(3):
        blurred = gaussian_filter(result[:, :, c], sigma=5)
        result[:, :, c] = result[:, :, c] * (1 - mask) + blurred * mask
    
    return result.astype(np.uint8)


if __name__ == "__main__":
    input_dir = "output/030002_front_up_polydiff"
    output_dir = f"{input_dir}/seam_repair_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load 6 faces
    print("Loading 6 faces...")
    faces = []
    for i, name in enumerate(FACE_NAMES):
        path = f"{input_dir}/{name}.png"
        if os.path.exists(path):
            img = np.array(Image.open(path).convert("RGB"))
            print(f"  {name}: {img.shape}")
            faces.append(img)
        else:
            print(f"  {name}: NOT FOUND")
            faces.append(np.zeros((512, 512, 3), dtype=np.uint8))
    
    # Test Edge 0: Front-Right
    edge_id = 0
    _, face_a_idx, face_b_idx, yaw_deg, pitch_deg = EDGE_DEFINITIONS[edge_id]
    
    print(f"\nTesting Edge {edge_id}: {FACE_NAMES[face_a_idx]}-{FACE_NAMES[face_b_idx]}")
    print(f"  Yaw: {yaw_deg}°, Pitch: {pitch_deg}°")
    
    # Project
    seam_view = project_faces_to_seam_view(
        faces[face_a_idx], faces[face_b_idx],
        face_a_idx, face_b_idx,
        yaw_deg, pitch_deg
    )
    Image.fromarray(seam_view).save(f"{output_dir}/edge0_projected.png")
    print(f"  Saved: {output_dir}/edge0_projected.png")
    
    # Create mask
    mask = create_seam_mask(512, seam_width=64, feather=32)
    mask_vis = (mask * 255).astype(np.uint8)
    Image.fromarray(mask_vis).save(f"{output_dir}/edge0_mask.png")
    print(f"  Saved: {output_dir}/edge0_mask.png")
    
    # Visualize mask on seam view
    seam_with_mask = seam_view.copy()
    seam_with_mask[mask > 0.5, 0] = np.clip(seam_with_mask[mask > 0.5, 0].astype(int) + 100, 0, 255)
    Image.fromarray(seam_with_mask).save(f"{output_dir}/edge0_with_mask.png")
    print(f"  Saved: {output_dir}/edge0_with_mask.png")
    
    # Test blur inpaint
    print("\n  Testing blur inpaint...")
    repaired = blur_inpaint(seam_view, mask)
    Image.fromarray(repaired).save(f"{output_dir}/edge0_blur_inpainted.png")
    print(f"  Saved: {output_dir}/edge0_blur_inpainted.png")
    
    # Paste back
    print("\n  Testing paste back...")
    front_repaired = paste_back_to_face(
        repaired, faces[0], 0, yaw_deg, pitch_deg, mask
    )
    right_repaired = paste_back_to_face(
        repaired, faces[3], 3, yaw_deg, pitch_deg, mask
    )
    
    Image.fromarray(front_repaired).save(f"{output_dir}/front_repaired.png")
    Image.fromarray(right_repaired).save(f"{output_dir}/right_repaired.png")
    print(f"  Saved: {output_dir}/front_repaired.png")
    print(f"  Saved: {output_dir}/right_repaired.png")
    
    print("\n✅ Test complete! Check the output images.")
