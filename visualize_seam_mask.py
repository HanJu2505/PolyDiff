"""
Draw seam lines on the 12 seam view images.
Shows where each seam view should contribute (the center line of the seam).
"""

import os
import numpy as np
from PIL import Image, ImageDraw
from cubediff.modules.geometry import VIEW_CONFIG_18, VIEW_NAMES_18

if __name__ == "__main__":
    output_dir = "output/030002_front_up_polydiff"
    vis_dir = f"{output_dir}/seam_visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    img_size = 512
    center = img_size // 2
    
    # For each seam view, the "seam line" is where it's positioned
    # Seam views are at 45° yaw or 45° pitch, pointing at cube edges
    # The seam line on the image is at the CENTER (the view is looking directly at the seam)
    
    print("Drawing seam lines on seam views...")
    
    for i in range(6, 18):
        name = VIEW_NAMES_18[i]
        path = f"{output_dir}/{name}.png"
        
        if not os.path.exists(path):
            print(f"  {name}: NOT FOUND")
            continue
        
        img = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Get the yaw/pitch of this seam view
        _, yaw, pitch = VIEW_CONFIG_18[i]
        
        # Draw the seam line based on orientation
        # The seam line is where this view contributes most
        # - For equator seams (pitch=0): vertical line at center
        # - For polar seams (pitch=±45): horizontal line at center
        
        if abs(pitch) < 10:  # Equator seam (pitch ≈ 0)
            # These look at horizontal seams between L-R views
            # Draw vertical center line
            draw.line([(center, 0), (center, img_size-1)], fill='red', width=3)
            orientation = "horizontal seam (vertical line)"
        else:  # Polar seam (pitch = ±45)
            # These look at seams between equator and poles
            # Draw horizontal center line
            draw.line([(0, center), (img_size-1, center)], fill='red', width=3)
            orientation = "polar seam (horizontal line)"
        
        # Add text label
        label = f"{name}\nyaw={yaw}° pitch={pitch}°"
        draw.text((10, 10), label, fill='red')
        
        # Also draw the valid 90° FOV boundary (circle)
        # At 90° FOV, the valid region is a square inscribed in a circle
        fov_deg = 95.0
        valid_fov = 90.0
        f = img_size / (2 * np.tan(np.radians(fov_deg) / 2))
        valid_radius = f * np.tan(np.radians(valid_fov / 2))
        
        # Draw circle showing 90° FOV boundary
        draw.ellipse([center - valid_radius, center - valid_radius,
                      center + valid_radius, center + valid_radius],
                     outline='yellow', width=2)
        
        # Save
        save_path = f"{vis_dir}/{name}_seam_line.png"
        img.save(save_path)
        print(f"  {name}: yaw={yaw}°, pitch={pitch}° → {orientation}")
    
    print(f"\n✅ Saved to {vis_dir}/")
    print("Red line = center seam line")
    print("Yellow circle = 90° valid FOV boundary")
