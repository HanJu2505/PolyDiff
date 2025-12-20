"""
Quick test of new inverse mapping fusion with existing views.
"""
import numpy as np
from PIL import Image
from cubediff.modules.geometry import VIEW_CONFIG_18, VIEW_NAMES_18
from cubediff.pipelines.fusion import inverse_mapping_fusion

# Load existing views
output_dir = "output/030001_front_up_polydiff"
views = []

print("Loading views...")
for i, name in enumerate(VIEW_NAMES_18):
    img = Image.open(f"{output_dir}/{name}.png")
    views.append(np.array(img))
    print(f"  View {i}: {name}, shape={np.array(img).shape}")

print(f"\nTotal views: {len(views)}")

# Test WTA fusion
print("\n===== Testing WTA (Winner-Takes-All) =====")
erp_wta = inverse_mapping_fusion(
    views, VIEW_CONFIG_18, 
    fov_deg=95.0,
    mode="wta",
    effective_fov_deg=70.0
)
Image.fromarray(erp_wta).save(f"{output_dir}/erp_wta.png")
print(f"Saved to {output_dir}/erp_wta.png")

# Test Gaussian fusion
print("\n===== Testing Gaussian (Soft Blending) =====")
erp_gaussian = inverse_mapping_fusion(
    views, VIEW_CONFIG_18, 
    fov_deg=95.0,
    mode="gaussian",
    effective_fov_deg=75.0,
    sigma_factor=0.2
)
Image.fromarray(erp_gaussian).save(f"{output_dir}/erp_gaussian.png")
print(f"Saved to {output_dir}/erp_gaussian.png")

print("\n✅ Done! Compare the two outputs to check coordinate alignment.")
