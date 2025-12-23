"""
Debug script: Test all 12 edge projections with colored faces.
This will generate 12 images, one for each edge.
Expected: clear split between two face colors with seam in middle.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from cubediff.pipelines.seam_repair import (
    project_faces_to_seam_view,
    EDGE_DEFINITIONS,
    FACE_NAMES
)

# Face colors - easily distinguishable
FACE_COLORS = {
    0: (255, 0, 0),      # Front: Red
    1: (0, 255, 0),      # Back: Green
    2: (0, 0, 255),      # Left: Blue
    3: (255, 255, 0),    # Right: Yellow
    4: (255, 0, 255),    # Top: Magenta
    5: (0, 255, 255),    # Bottom: Cyan
}

def create_labeled_face(face_idx, size=512):
    """Create a face with solid color and label in center."""
    color = FACE_COLORS[face_idx]
    name = FACE_NAMES[face_idx].upper()
    
    img = Image.new('RGB', (size, size), color)
    draw = ImageDraw.Draw(img)
    
    # Draw face name in center
    text = name
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
    except:
        font = ImageFont.load_default()
    
    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    x = (size - text_w) // 2
    y = (size - text_h) // 2
    
    # Draw black text for contrast
    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    
    return np.array(img)

def main():
    output_dir = "output/edge_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create labeled faces
    print("Creating labeled faces...")
    faces = [create_labeled_face(i) for i in range(6)]
    
    # Save faces for reference
    for i, face in enumerate(faces):
        Image.fromarray(face).save(f"{output_dir}/face_{i}_{FACE_NAMES[i]}.png")
    
    print("\nTesting all 12 edges...")
    print("="*60)
    
    for edge_def in EDGE_DEFINITIONS:
        edge_id, face_a_idx, face_b_idx, yaw_deg, pitch_deg = edge_def
        
        face_a_name = FACE_NAMES[face_a_idx]
        face_b_name = FACE_NAMES[face_b_idx]
        
        # Project
        seam_view = project_faces_to_seam_view(
            faces[face_a_idx], faces[face_b_idx],
            face_a_idx, face_b_idx,
            yaw_deg, pitch_deg
        )
        
        # Add label
        img = Image.fromarray(seam_view)
        draw = ImageDraw.Draw(img)
        label = f"Edge {edge_id}: {face_a_name}-{face_b_name} (yaw={yaw_deg}°, pitch={pitch_deg}°)"
        draw.text((10, 10), label, fill=(255, 255, 255))
        
        # Expected split
        expected = f"Expected: {face_a_name.upper()} left, {face_b_name.upper()} right"
        draw.text((10, 30), expected, fill=(255, 255, 255))
        
        # Save
        filename = f"edge{edge_id:02d}_{face_a_name}_{face_b_name}.png"
        img.save(f"{output_dir}/{filename}")
        
        print(f"Edge {edge_id:2d}: {face_a_name:6s}-{face_b_name:6s}  yaw={yaw_deg:6.1f}°  pitch={pitch_deg:6.1f}°  → {filename}")
    
    print("="*60)
    print(f"\n✅ Saved to {output_dir}/")
    print("\nCheck each image:")
    print("  - Left half should be Face A color")
    print("  - Right half should be Face B color")
    print("  - Clear vertical seam in middle")

if __name__ == "__main__":
    main()
