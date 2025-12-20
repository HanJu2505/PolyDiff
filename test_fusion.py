import numpy as np
from cubediff.modules.geometry import VIEW_CONFIG_18
from cubediff.pipelines.fusion import create_view_projection_lut

# Test with front view
name, yaw, pitch = VIEW_CONFIG_18[0]
print(f'Testing view 0: {name}, yaw={yaw}, pitch={pitch}')

erp_u, erp_v, weights = create_view_projection_lut(512, yaw, pitch, 95.0, 2048, 1024)
print(f'ERP u range: [{erp_u.min():.1f}, {erp_u.max():.1f}]')
print(f'ERP v range: [{erp_v.min():.1f}, {erp_v.max():.1f}]')
print(f'Weights range: [{weights.min():.4f}, {weights.max():.4f}]')
print(f'Weights mean: {weights.mean():.4f}')

# Check if coordinates are valid
erp_u_int = np.clip(np.round(erp_u).astype(np.int32), 0, 2047)
erp_v_int = np.clip(np.round(erp_v).astype(np.int32), 0, 1023)
print(f'\nAfter clipping:')
print(f'U unique values: {len(np.unique(erp_u_int))}')
print(f'V unique values: {len(np.unique(erp_v_int))}')
