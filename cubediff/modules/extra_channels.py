import math

import torch


DEFAULT_FACE_ORDER = ["front", "back", "left", "right", "top", "bottom"]


def _build_face_axes(u_grid: torch.Tensor, v_grid: torch.Tensor, face: str):
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
    else:
        raise ValueError(f"Unsupported face: {face}")
    return x, y, z


def calculate_positional_encoding(resolution=(128, 128), fov_deg=95.0):
    """
    Computes (u,v) positional encodings per CubeDiff Eq.(1) for all six cubemap faces
    using unit cube formulation and consistent global normalization.
    """
    extent = math.tan(math.radians(fov_deg / 2))
    encodings = {}

    for face in DEFAULT_FACE_ORDER:
        u_range = torch.linspace(-extent, extent, resolution[0])
        v_range = torch.linspace(extent, -extent, resolution[1])
        u_grid, v_grid = torch.meshgrid(u_range, v_range, indexing="xy")
        x, y, z = _build_face_axes(u_grid, v_grid, face)

        u_enc = torch.atan2(x, z)
        v_enc = torch.atan2(y, torch.sqrt(x**2 + z**2))
        u_enc = (u_enc / math.pi + 1.0) / 2.0
        v_enc = (v_enc / math.pi + 1.0) / 2.0

        encodings[face] = torch.stack([u_enc, v_enc], dim=0)

    return encodings


def calculate_directional_encoding(resolution=(128, 128), fov_deg=95.0):
    """
    Computes normalized direction vectors (x, y, z) for each cubemap face.

    These are the three extra channels used by the original 7-channel CubeDiff UNet.
    """
    extent = math.tan(math.radians(fov_deg / 2))
    encodings = {}

    for face in DEFAULT_FACE_ORDER:
        u_range = torch.linspace(-extent, extent, resolution[0])
        v_range = torch.linspace(extent, -extent, resolution[1])
        u_grid, v_grid = torch.meshgrid(u_range, v_range, indexing="xy")
        x, y, z = _build_face_axes(u_grid, v_grid, face)
        dirs = torch.stack([x, y, z], dim=0)
        dirs = dirs / torch.linalg.norm(dirs, dim=0, keepdim=True).clamp_min(1e-6)
        encodings[face] = dirs

    return encodings


def get_uv_tensors(batch_size, latent_height, latent_width, face_order=None, encodings=None):
    """
    Generate stacked (u, v) coordinate tensors for each face in a CubeDiff batch.
    """
    if face_order is None:
        face_order = DEFAULT_FACE_ORDER

    if encodings is None:
        encodings = calculate_positional_encoding((latent_height, latent_width))

    per_face_tensor = torch.stack([encodings[face] for face in face_order], dim=0)
    expanded = per_face_tensor.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
    stacked = expanded.reshape(batch_size * len(face_order), 2, latent_height, latent_width)
    return stacked


def make_extra_channels_tensor(batch_size, latent_height, latent_width, face_order=None, encodings=None):
    """
    Generate the original CubeDiff 3-channel directional encoding used for 7-channel UNet input.
    """
    if face_order is None:
        face_order = DEFAULT_FACE_ORDER

    if encodings is None:
        encodings = calculate_directional_encoding((latent_height, latent_width))

    per_face_tensor = torch.stack([encodings[face] for face in face_order], dim=0)
    expanded = per_face_tensor.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
    stacked = expanded.reshape(batch_size * len(face_order), 3, latent_height, latent_width)
    return stacked
