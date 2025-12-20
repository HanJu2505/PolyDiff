import numpy as np
from PIL import Image
from cubediff.modules.geometry import VIEW_CONFIG_18
from cubediff.pipelines.fusion import gaussian_spherical_fusion_fast

# Load the generated views
views = []
view_names = [VIEW_CONFIG_18[i][0] for i in range(18)]

for i, name in enumerate(view_names):
    img_path = f'output/202_img__polydiff/{name}.png'
    img = Image.open(img_path)
    views.append(np.array(img))
    print(f'Loaded view {i}: {name}, shape={np.array(img).shape}, mean={np.array(img).mean():.1f}')

print(f'\nTotal views loaded: {len(views)}')
print(f'View config has {len(VIEW_CONFIG_18)} entries')

# Check if VIEW_CONFIG_18 is being used correctly
for i in range(18):
    name, yaw, pitch = VIEW_CONFIG_18[i]
    print(f'{i}: {name:12s} yaw={yaw:6.1f}° pitch={pitch:6.1f}°')

# Try fusion with debug
print('\n--- Running fusion ---')
erp = gaussian_spherical_fusion_fast(views, VIEW_CONFIG_18, fov_deg=95.0)
print(f'ERP shape: {erp.shape}, min: {erp.min()}, max: {erp.max()}, mean: {erp.mean():.1f}')

# Save debug ERP
Image.fromarray(erp).save('output/debug_erp.png')
print('Saved to output/debug_erp.png')
