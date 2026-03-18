import torch
import math


def calculate_positional_encoding(resolution=(128, 128), fov_deg=95.0):
    """
    Computes (u,v) positional encodings per CubeDiff Eq.(1) for all six cubemap faces
    using unit cube formulation and consistent global normalization.
    """
    extent = math.tan(math.radians(fov_deg / 2))  # Extent of cube face per FOV
    faces = ["front", "back", "left", "right", "top", "bottom"]
    encodings = {}

    for face in faces:
        u_range = torch.linspace(-extent, extent, resolution[0])
        v_range = torch.linspace(extent, -extent, resolution[1])
        u_grid, v_grid = torch.meshgrid(u_range, v_range, indexing='xy')

        if face == "front":
            x, y, z = u_grid, v_grid, torch.ones_like(u_grid)
        elif face == "back":
            x, y, z = -u_grid, v_grid, -torch.ones_like(u_grid)
        elif face == "left":
            x, y, z = -torch.ones_like(u_grid), v_grid, u_grid
        elif face == "right":
            x, y, z = torch.ones_like(u_grid), v_grid, -u_grid
        elif face == "top":
            x, y, z = u_grid, torch.ones_like(u_grid), -v_grid
        elif face == "bottom":
            x, y, z = u_grid, -torch.ones_like(u_grid), v_grid

        # Positional encoding via CubeDiff Equation (1)
        u_enc = torch.atan2(x, z)
        v_enc = torch.atan2(y, torch.sqrt(x ** 2 + z ** 2))

        # Normalize to [0, 1] using global angle range
        u_enc = (u_enc / math.pi + 1.0) / 2.0
        v_enc = (v_enc / math.pi + 1.0) / 2.0

        encodings[face] = torch.stack([u_enc, v_enc], dim=0)  # Shape: (2, H, W)

    return encodings


def get_uv_tensors(batch_size, latent_height, latent_width, face_order=None, encodings=None):
    """
    Generate stacked (u, v) coordinate tensors for each face in a CubeDiff batch.
    
    Args:
        batch_size: Number of samples in the batch (B).
        latent_height: Height of the latent feature map.
        latent_width: Width of the latent feature map.
        face_order: Optional list of face names. Defaults to 
                     ["front", "back", "left", "right", "top", "bottom"].
        encodings: Optional pre-computed encodings dict from calculate_positional_encoding.
    
    Returns:
        Tensor of shape (B*T, 2, H, W) where T=6, containing (u, v) coordinates
        for each face at the given latent resolution.
    """
    if face_order is None:
        face_order = ["front", "back", "left", "right", "top", "bottom"]

    if encodings is None:
        encodings = calculate_positional_encoding((latent_height, latent_width))

    # Stack all faces: (T, 2, H, W)
    per_face_tensor = torch.stack([encodings[face] for face in face_order], dim=0)
    
    # Expand for batch: (B, T, 2, H, W) -> (B*T, 2, H, W)
    expanded = per_face_tensor.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
    stacked = expanded.reshape(batch_size * len(face_order), 2, latent_height, latent_width)
    
    return stacked  # float32, to be moved to device/dtype by caller
