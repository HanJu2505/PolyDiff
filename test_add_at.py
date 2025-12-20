import numpy as np

# Test np.add.at behavior
canvas = np.zeros((10, 10, 3), dtype=np.float64)
weight_canvas = np.zeros((10, 10), dtype=np.float64)

# Simulate adding some values
flat_idx = np.array([0, 1, 2, 11, 12])  # Some indices
flat_weights = np.array([1.0, 0.8, 0.6, 0.9, 0.7])
flat_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 0], [0, 128, 128]], dtype=np.float64)

print('Testing np.add.at...')
for c in range(3):
    weighted_colors = flat_colors[:, c] * flat_weights
    print(f'Channel {c}: weighted_colors shape={weighted_colors.shape}, values={weighted_colors[:3]}')
    np.add.at(canvas[:, :, c].ravel(), flat_idx, weighted_colors)

np.add.at(weight_canvas.ravel(), flat_idx, flat_weights)

print(f'\nCanvas sum: {canvas.sum()}')
print(f'Weight canvas sum: {weight_canvas.sum()}')
print(f'Canvas at [0,0]: {canvas[0,0]}')
print(f'Canvas at [1,1]: {canvas[1,1]}')
print(f'Weight at [0,0]: {weight_canvas[0,0]}')
