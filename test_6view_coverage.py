"""
Test script for 6-view only fusion - debugging coverage issues.
"""

import os
import numpy as np
from PIL import Image
from cubediff.pipelines.fusion import spherical_to_perspective, erp_to_spherical
from cubediff.modules.geometry import VIEW_CONFIG_18, VIEW_NAMES_18

if __name__ == "__main__":
    output_dir = "output/030002_front_up_polydiff"
    
    # Load only the 6 main views
    print("Loading 6 main views...")
    views = []
    for i in range(6):
        name = VIEW_NAMES_18[i]
        path = f"{output_dir}/{name}.png"
        if os.path.exists(path):
            img = np.array(Image.open(path).convert("RGB"))
            print(f"  View {i}: {name}, shape={img.shape}")
            views.append(img)
        else:
            print(f"  View {i}: {name} NOT FOUND!")
            views.append(np.zeros((512, 512, 3), dtype=np.uint8))
    
    print(f"\nTotal views: {len(views)}")
    
    # Fusion parameters
    fov_deg = 95.0    # Generation FOV
    valid_fov_deg = 90.0  # Only use center 90°
    erp_height, erp_width = 1024, 2048
    
    # Create ERP coordinate grid
    erp_u, erp_v = np.meshgrid(np.arange(erp_width), np.arange(erp_height))
    theta, phi = erp_to_spherical(erp_u, erp_v, erp_width, erp_height)
    
    # Calculate valid distance
    img_size = 512
    f = img_size / (2 * np.tan(np.radians(fov_deg) / 2))
    valid_max_dist = f * np.tan(np.radians(valid_fov_deg / 2))
    
    print(f"\nFOV: {fov_deg}° (generation), {valid_fov_deg}° (valid)")
    print(f"Valid max distance from center: {valid_max_dist:.1f} pixels")
    print(f"Image size: {img_size}, center: {img_size/2}")
    
    # Test coverage from 6 main views
    erp_image = np.zeros((erp_height, erp_width, 3), dtype=np.float64)
    weight_sum = np.zeros((erp_height, erp_width))
    
    print("\n===== Processing 6 Main Views =====")
    for view_idx in range(6):
        view_img = views[view_idx].astype(np.float64)
        name, yaw_deg, pitch_deg = VIEW_CONFIG_18[view_idx]
        
        # Project ERP to this view
        px, py, valid = spherical_to_perspective(
            theta, phi, yaw_deg, pitch_deg, fov_deg, img_size
        )
        
        # Calculate distance from view center
        center = img_size / 2.0
        dist = np.sqrt((px - center)**2 + (py - center)**2)
        
        # Only use content within 90°
        valid = valid & (dist <= valid_max_dist)
        
        # Count coverage
        coverage_count = np.sum(valid)
        coverage_pct = coverage_count / (erp_height * erp_width) * 100
        print(f"  {name}: yaw={yaw_deg}°, pitch={pitch_deg}°, coverage={coverage_pct:.2f}%")
        
        # Accumulate
        weight = np.where(valid, 1.0, 0.0)
        px_int = np.clip(px.astype(np.int32), 0, img_size - 1)
        py_int = np.clip(py.astype(np.int32), 0, img_size - 1)
        sampled = view_img[py_int, px_int]
        
        erp_image += sampled * weight[..., None]
        weight_sum += weight
    
    # Final coverage
    total_coverage = np.sum(weight_sum > 0) / (erp_height * erp_width) * 100
    print(f"\n===== Total Coverage from 6 views: {total_coverage:.2f}% =====")
    
    # Find uncovered pixels
    uncovered = weight_sum == 0
    uncovered_count = np.sum(uncovered)
    print(f"Uncovered pixels: {uncovered_count} ({100-total_coverage:.2f}%)")
    
    # Visualize uncovered regions
    erp_image = erp_image / (weight_sum[..., None] + 1e-8)
    erp_image = np.clip(erp_image, 0, 255).astype(np.uint8)
    
    # Mark uncovered pixels in RED
    erp_with_gaps = erp_image.copy()
    erp_with_gaps[uncovered] = [255, 0, 0]  # Red for gaps
    
    Image.fromarray(erp_image).save(f"{output_dir}/erp_6views_only.png")
    Image.fromarray(erp_with_gaps).save(f"{output_dir}/erp_6views_gaps_marked.png")
    print(f"\nSaved: {output_dir}/erp_6views_only.png")
    print(f"Saved: {output_dir}/erp_6views_gaps_marked.png (gaps in RED)")
