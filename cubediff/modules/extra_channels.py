import torch.nn as nn
import torch
import math
import numpy as np

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


def calculate_positional_encoding_18(resolution=(128, 128), fov_deg=95.0, view_config=None):
    """
    Computes (u,v) positional encodings for all 18 views based on yaw/pitch rotation.
    
    Each view is a perspective projection looking in a direction defined by (yaw, pitch).
    
    Args:
        resolution: (H, W) resolution of the encoding
        fov_deg: Field of view in degrees
        view_config: Dict {idx: (name, yaw_deg, pitch_deg)} or None to use default
    
    Returns:
        Dict {view_name: encoding} where encoding is shape (2, H, W)
    """
    if view_config is None:
        from .geometry import VIEW_CONFIG_18
        view_config = VIEW_CONFIG_18
    
    extent = math.tan(math.radians(fov_deg / 2))
    encodings = {}
    
    for idx, (name, yaw_deg, pitch_deg) in view_config.items():
        # Create pixel grid on the image plane (looking along +Z initially)
        u_range = torch.linspace(-extent, extent, resolution[0])
        v_range = torch.linspace(extent, -extent, resolution[1])
        u_grid, v_grid = torch.meshgrid(u_range, v_range, indexing='xy')
        
        # Initial ray directions (looking along +Z)
        x = u_grid
        y = v_grid
        z = torch.ones_like(u_grid)
        
        # Normalize to unit vectors
        norm = torch.sqrt(x**2 + y**2 + z**2)
        x, y, z = x/norm, y/norm, z/norm
        
        # Apply rotation: first yaw (around Y), then pitch (around X)
        yaw_rad = math.radians(yaw_deg)
        pitch_rad = math.radians(pitch_deg)
        
        # Rotation around Y-axis (yaw)
        x_yaw = x * math.cos(yaw_rad) + z * math.sin(yaw_rad)
        y_yaw = y
        z_yaw = -x * math.sin(yaw_rad) + z * math.cos(yaw_rad)
        
        # Rotation around X-axis (pitch)
        x_rot = x_yaw
        y_rot = y_yaw * math.cos(pitch_rad) - z_yaw * math.sin(pitch_rad)
        z_rot = y_yaw * math.sin(pitch_rad) + z_yaw * math.cos(pitch_rad)
        
        # Convert to spherical coordinates for positional encoding
        # u_enc: azimuth angle (around Y-axis), v_enc: elevation
        u_enc = torch.atan2(x_rot, z_rot)
        v_enc = torch.atan2(y_rot, torch.sqrt(x_rot**2 + z_rot**2))
        
        # Normalize to [0, 1]
        u_enc = (u_enc / math.pi + 1.0) / 2.0
        v_enc = (v_enc / math.pi + 1.0) / 2.0
        
        encodings[name] = torch.stack([u_enc, v_enc], dim=0)  # (2, H, W)
    
    return encodings


def mask_tensors(batch_size, latent_height, latent_width, num_faces=6):
    mask = torch.zeros((batch_size * num_faces, 1, latent_height, latent_width), dtype=torch.float16) # Shape: (B*T, 1, H, W)
    front_indices = torch.arange(0, batch_size * num_faces, num_faces) # Front face indices (0, 6, 12, ...)
    mask[front_indices] = 1.0
    return mask


def mask_tensors_18(batch_size, latent_height, latent_width, num_views=18):
    """
    Generate mask tensors for 18 views.
    View 0 (front) is the anchor and gets mask=1, all others get mask=0.
    """
    mask = torch.zeros((batch_size * num_views, 1, latent_height, latent_width), dtype=torch.float16)
    front_indices = torch.arange(0, batch_size * num_views, num_views)  # [0, 18, 36, ...]
    mask[front_indices] = 1.0
    return mask


def encoding_tensors(batch_size, latent_height, latent_width, face_order=None, encodings=None):
    """
    Generate stacked u_enc and v_enc channels for each face in a CubeDiff batch.
    Returns tensor of shape (B*T, 2, H, W)
    """
    if face_order is None:
        face_order = ["front", "back", "left", "right", "top", "bottom"]

    if encodings is None:
        encodings = calculate_positional_encoding((latent_height, latent_width))

    per_face_tensor = torch.stack([encodings[face] for face in face_order], dim=0)  # (T, 2, H, W)
    expanded_tensor = per_face_tensor.repeat(batch_size, 1, 1, 1, 1)  # (B, T, 2, H, W)
    stacked = expanded_tensor.reshape(batch_size * len(face_order), 2, latent_height, latent_width).to(dtype=torch.float16)  # (B*T, 2, H, W)
    return stacked


def encoding_tensors_18(batch_size, latent_height, latent_width, view_config=None, encodings=None):
    """
    Generate stacked u_enc and v_enc channels for 18 views.
    Returns tensor of shape (B*18, 2, H, W)
    """
    if view_config is None:
        from .geometry import VIEW_CONFIG_18, VIEW_NAMES_18
        view_config = VIEW_CONFIG_18
        view_names = VIEW_NAMES_18
    else:
        view_names = [view_config[i][0] for i in range(len(view_config))]
    
    if encodings is None:
        encodings = calculate_positional_encoding_18((latent_height, latent_width), view_config=view_config)
    
    num_views = len(view_names)
    per_view_tensor = torch.stack([encodings[name] for name in view_names], dim=0)  # (18, 2, H, W)
    expanded_tensor = per_view_tensor.repeat(batch_size, 1, 1, 1, 1)  # (B, 18, 2, H, W)
    stacked = expanded_tensor.reshape(batch_size * num_views, 2, latent_height, latent_width).to(dtype=torch.float16)
    return stacked


def make_extra_channels_tensor(batch_size, latent_height, latent_width, face_order=None, encodings=None):
    """
    Combine encoding tensors and mask tensors into a single (B*T, 3, H, W) tensor.
    Channel 0-1: u_enc, v_enc, Channel 2: binary mask
    """
    enc_tensor = encoding_tensors(batch_size, latent_height, latent_width, face_order, encodings)  # (B*T, 2, H, W)
    mask_tensor = mask_tensors(batch_size, latent_height, latent_width)  # (B*T, 1, H, W)
    return torch.cat([enc_tensor, mask_tensor], dim=1).to(dtype=torch.float16)  # (B*T, 3, H, W)


def make_extra_channels_tensor_18(batch_size, latent_height, latent_width, view_config=None, encodings=None):
    """
    Combine encoding tensors and mask tensors for 18 views into a single (B*18, 3, H, W) tensor.
    Channel 0-1: u_enc, v_enc, Channel 2: binary mask (1 for front view, 0 otherwise)
    """
    enc_tensor = encoding_tensors_18(batch_size, latent_height, latent_width, view_config, encodings)  # (B*18, 2, H, W)
    mask_tensor = mask_tensors_18(batch_size, latent_height, latent_width, num_views=18)  # (B*18, 1, H, W)
    return torch.cat([enc_tensor, mask_tensor], dim=1).to(dtype=torch.float16)  # (B*18, 3, H, W)

