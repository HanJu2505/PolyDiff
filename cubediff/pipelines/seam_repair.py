"""
Seam Repair Pipeline for 360° Panorama Generation

This module implements a 3-stage seam repair process:
1. Project adjacent cubemap faces to a seam view camera
2. Use SD Inpainting to fix the hard seam
3. Paste repaired pixels back to the original faces

The 12 edges of a cubemap are processed sequentially.
"""

import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
import math


# Edge definitions: (edge_id, face_a, face_b, yaw, pitch)
# Face indices: 0=front, 1=back, 2=left, 3=right, 4=top, 5=bottom
EDGE_DEFINITIONS = [
    # Equator edges (pitch = 0)
    (0, 0, 3, 45.0, 0.0),    # Front-Right
    (1, 3, 1, 135.0, 0.0),   # Right-Back
    (2, 1, 2, 225.0, 0.0),   # Back-Left
    (3, 2, 0, 315.0, 0.0),   # Left-Front
    # Top edges (pitch = 45)
    (4, 4, 0, 0.0, 45.0),    # Top-Front
    (5, 4, 3, 90.0, 45.0),   # Top-Right
    (6, 4, 1, 180.0, 45.0),  # Top-Back
    (7, 4, 2, 270.0, 45.0),  # Top-Left
    # Bottom edges (pitch = -45)
    (8, 5, 0, 0.0, -45.0),   # Bottom-Front
    (9, 5, 3, 90.0, -45.0),  # Bottom-Right
    (10, 5, 1, 180.0, -45.0), # Bottom-Back
    (11, 5, 2, 270.0, -45.0), # Bottom-Left
]

FACE_NAMES = ["front", "back", "left", "right", "top", "bottom"]


def spherical_to_cartesian(theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert spherical coordinates to 3D Cartesian unit vectors.
    
    Args:
        theta: Azimuth angle in radians (0 = +Z, π/2 = +X)
        phi: Elevation angle in radians (+π/2 = +Y up, -π/2 = -Y down)
    
    Returns:
        x, y, z: 3D unit vector components
    """
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi)
    z = np.cos(phi) * np.cos(theta)
    return x, y, z


def cartesian_to_spherical(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert 3D Cartesian to spherical coordinates.
    
    Returns:
        theta: Azimuth in radians
        phi: Elevation in radians
    """
    theta = np.arctan2(x, z)
    phi = np.arcsin(np.clip(y, -1, 1))
    return theta, phi


def rotate_vector(x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                  yaw_rad: float, pitch_rad: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply intrinsic yaw-pitch rotation to a 3D vector.
    
    For intrinsic rotations (camera-centric):
    - Yaw: rotation around world Y-axis
    - Pitch: rotation around camera's local X-axis (after yaw)
    
    The correct order for R = R_yaw * R_pitch is to apply pitch FIRST, then yaw.
    
    Convention:
    - +yaw = rotate right (clockwise when viewed from above)
    - +pitch = look up
    """
    # Step 1: Apply pitch first (around X-axis)
    # Negate pitch so +pitch = look UP
    pitch_rad = -pitch_rad
    cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
    x_pitch = x
    y_pitch = y * cos_p - z * sin_p
    z_pitch = y * sin_p + z * cos_p
    
    # Step 2: Apply yaw (around Y-axis)
    cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
    x_rot = x_pitch * cos_y + z_pitch * sin_y
    z_rot = -x_pitch * sin_y + z_pitch * cos_y
    y_rot = y_pitch
    
    return x_rot, y_rot, z_rot


def unrotate_vector(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    yaw_rad: float, pitch_rad: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply inverse rotation (undo the rotation from rotate_vector).
    
    Since rotate_vector does: pitch first, then yaw
    Inverse is: undo yaw first, then undo pitch
    """
    # Step 1: Undo yaw (around Y-axis, negative angle)
    cos_y, sin_y = np.cos(-yaw_rad), np.sin(-yaw_rad)
    x_unyaw = x * cos_y + z * sin_y
    z_unyaw = -x * sin_y + z * cos_y
    y_unyaw = y
    
    # Step 2: Undo pitch (around X-axis, positive angle since we negated in forward)
    # In rotate_vector we used -pitch_rad, so to undo we use +pitch_rad
    cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
    x_unrot = x_unyaw
    y_unrot = y_unyaw * cos_p - z_unyaw * sin_p
    z_unrot = y_unyaw * sin_p + z_unyaw * cos_p
    
    return x_unrot, y_unrot, z_unrot


def project_to_cubemap_face(x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                            face_idx: int, face_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D direction vectors to pixel coordinates on a cubemap face.
    
    A point belongs to a face only if:
    1. It faces that direction (correct sign)
    2. That axis is DOMINANT (has largest absolute value)
    
    Args:
        x, y, z: 3D direction vectors
        face_idx: 0=front, 1=back, 2=left, 3=right, 4=top, 5=bottom
        face_size: Size of each face in pixels (e.g., 512)
    
    Returns:
        px, py: Pixel coordinates on the face
        valid: Boolean mask for valid projections
    """
    half_size = face_size / 2.0
    abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)
    
    if face_idx == 0:  # Front (+Z)
        # Valid if z > 0 AND z is the dominant axis
        valid = (z > 0) & (abs_z >= abs_x) & (abs_z >= abs_y)
        px = (x / (z + 1e-10)) * half_size + half_size
        py = (-y / (z + 1e-10)) * half_size + half_size
    elif face_idx == 1:  # Back (-Z)
        valid = (z < 0) & (abs_z >= abs_x) & (abs_z >= abs_y)
        px = (-x / (-z + 1e-10)) * half_size + half_size
        py = (-y / (-z + 1e-10)) * half_size + half_size
    elif face_idx == 2:  # Left (-X)
        valid = (x < 0) & (abs_x >= abs_y) & (abs_x >= abs_z)
        # Looking at -X: +Z is to the RIGHT, +Y is UP
        px = (z / (-x + 1e-10)) * half_size + half_size  # Fixed: was -z
        py = (-y / (-x + 1e-10)) * half_size + half_size
    elif face_idx == 3:  # Right (+X)
        valid = (x > 0) & (abs_x >= abs_y) & (abs_x >= abs_z)
        px = (-z / (x + 1e-10)) * half_size + half_size
        py = (-y / (x + 1e-10)) * half_size + half_size
    elif face_idx == 4:  # Top (+Y)
        valid = (y > 0) & (abs_y >= abs_x) & (abs_y >= abs_z)
        px = (x / (y + 1e-10)) * half_size + half_size
        py = (z / (y + 1e-10)) * half_size + half_size
    elif face_idx == 5:  # Bottom (-Y)
        valid = (y < 0) & (abs_y >= abs_x) & (abs_y >= abs_z)
        px = (x / (-y + 1e-10)) * half_size + half_size
        py = (-z / (-y + 1e-10)) * half_size + half_size
    else:
        raise ValueError(f"Invalid face_idx: {face_idx}")
    
    # Check bounds
    valid = valid & (px >= 0) & (px < face_size) & (py >= 0) & (py < face_size)
    
    return px, py, valid


def project_faces_to_seam_view(face_a: np.ndarray, face_b: np.ndarray,
                                face_a_idx: int, face_b_idx: int,
                                yaw_deg: float, pitch_deg: float,
                                output_size: int = 512,
                                fov_deg: float = 90.0) -> np.ndarray:
    """
    Project two adjacent cubemap faces to a seam view camera.
    
    The seam camera is positioned to look directly at the edge between the two faces.
    The result shows face_a content on one side and face_b content on the other,
    with a visible hard seam in the middle.
    
    Args:
        face_a, face_b: Two adjacent face images [H, W, 3]
        face_a_idx, face_b_idx: Face indices (0-5)
        yaw_deg, pitch_deg: Seam camera orientation
        output_size: Output image size
        fov_deg: Field of view of seam camera
    
    Returns:
        Projected seam view image [output_size, output_size, 3]
    """
    face_size = face_a.shape[0]
    yaw_rad = np.radians(yaw_deg)
    pitch_rad = np.radians(pitch_deg)
    
    # Create seam view pixel grid
    u, v = np.meshgrid(np.arange(output_size), np.arange(output_size))
    
    # Convert pixel coords to normalized camera coords
    half_fov = np.tan(np.radians(fov_deg / 2))
    nx = (u - output_size / 2) / (output_size / 2) * half_fov
    ny = -(v - output_size / 2) / (output_size / 2) * half_fov
    nz = np.ones_like(nx)
    
    # Normalize to unit vectors
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    
    # Rotate to world coordinates (seam camera orientation)
    wx, wy, wz = rotate_vector(nx, ny, nz, yaw_rad, pitch_rad)
    
    # Project to each face
    output = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    
    for face_idx, face_img in [(face_a_idx, face_a), (face_b_idx, face_b)]:
        px, py, valid = project_to_cubemap_face(wx, wy, wz, face_idx, face_size)
        
        px_int = np.clip(px.astype(np.int32), 0, face_size - 1)
        py_int = np.clip(py.astype(np.int32), 0, face_size - 1)
        
        # Sample from face
        sampled = face_img[py_int, px_int]
        
        # Only write valid pixels (first face takes priority in overlap)
        mask = valid & (output.sum(axis=-1) == 0)  # Only write if not already filled
        output = np.where(mask[..., None], sampled, output)
    
    return output


def paste_back_to_face(seam_view: np.ndarray, 
                       original_face: np.ndarray,
                       face_idx: int,
                       yaw_deg: float, pitch_deg: float,
                       mask: np.ndarray,
                       fov_deg: float = 90.0) -> np.ndarray:
    """
    Project pixels from seam view back to a cubemap face.
    
    Only pastes pixels that are covered by the mask and fall within the face.
    
    Args:
        seam_view: Repaired seam view image [H, W, 3]
        original_face: Original face image to modify [H, W, 3]
        face_idx: Which face to paste to
        yaw_deg, pitch_deg: Seam camera orientation
        mask: Alpha mask [H, W] from seam view (0-1, what to paste)
        fov_deg: FOV of seam camera
    
    Returns:
        Modified face image [H, W, 3]
    """
    seam_size = seam_view.shape[0]
    face_size = original_face.shape[0]
    yaw_rad = np.radians(yaw_deg)
    pitch_rad = np.radians(pitch_deg)
    
    # Create face pixel grid
    face_u, face_v = np.meshgrid(np.arange(face_size), np.arange(face_size))
    
    # Convert face pixels to 3D direction vectors
    half_size = face_size / 2.0
    
    if face_idx == 0:  # Front (+Z)
        # project: px = x/z, py = -y/z
        # inverse: x = (face_u - half)/half, y = -(face_v - half)/half, z = 1
        fx = (face_u - half_size) / half_size
        fy = -(face_v - half_size) / half_size
        fz = np.ones_like(fx)
    elif face_idx == 1:  # Back (-Z)
        # project: px = -x/-z, py = -y/-z
        # inverse: x = -(face_u - half)/half, y = -(face_v - half)/half, z = -1
        fx = -(face_u - half_size) / half_size
        fy = -(face_v - half_size) / half_size
        fz = -np.ones_like(fx)
    elif face_idx == 2:  # Left (-X)
        # project: px = z/-x, py = -y/-x
        # inverse: z = (face_u - half)/half * (-x), y = -(face_v - half)/half * (-x)
        fx = -np.ones_like(face_u, dtype=float)
        fy = -(face_v - half_size) / half_size
        fz = (face_u - half_size) / half_size  # Fixed: was negative
    elif face_idx == 3:  # Right (+X)
        # project: px = -z/x, py = -y/x
        # inverse: z = -(face_u - half)/half, y = -(face_v - half)/half, x = 1
        fx = np.ones_like(face_u, dtype=float)
        fy = -(face_v - half_size) / half_size
        fz = -(face_u - half_size) / half_size  # Fixed: was positive
    elif face_idx == 4:  # Top (+Y)
        # project: px = x/y, py = z/y
        # inverse: x = (face_u - half)/half, z = (face_v - half)/half, y = 1
        fx = (face_u - half_size) / half_size
        fy = np.ones_like(face_u, dtype=float)
        fz = (face_v - half_size) / half_size
    elif face_idx == 5:  # Bottom (-Y)
        # project: px = x/-y, py = -z/-y
        # inverse: x = (face_u - half)/half, z = -(face_v - half)/half, y = -1
        fx = (face_u - half_size) / half_size
        fy = -np.ones_like(face_u, dtype=float)
        fz = -(face_v - half_size) / half_size
    else:
        raise ValueError(f"Invalid face_idx: {face_idx}")
    
    # Normalize
    norm = np.sqrt(fx**2 + fy**2 + fz**2)
    fx, fy, fz = fx / norm, fy / norm, fz / norm
    
    # Unrotate to seam camera coordinates
    cx, cy, cz = unrotate_vector(fx, fy, fz, yaw_rad, pitch_rad)
    
    # Project to seam camera image plane
    valid = cz > 0
    half_fov = np.tan(np.radians(fov_deg / 2))
    
    seam_u = (cx / cz / half_fov) * (seam_size / 2) + seam_size / 2
    seam_v = (-cy / cz / half_fov) * (seam_size / 2) + seam_size / 2
    
    valid = valid & (seam_u >= 0) & (seam_u < seam_size) & (seam_v >= 0) & (seam_v < seam_size)
    
    seam_u_int = np.clip(seam_u.astype(np.int32), 0, seam_size - 1)
    seam_v_int = np.clip(seam_v.astype(np.int32), 0, seam_size - 1)
    
    # Sample from seam view
    sampled = seam_view[seam_v_int, seam_u_int]
    sampled_mask = mask[seam_v_int, seam_u_int]
    
    # Blend with original
    result = original_face.copy().astype(np.float32)
    alpha = (sampled_mask * valid)[..., None]
    result = result * (1 - alpha) + sampled.astype(np.float32) * alpha
    
    return result.astype(np.uint8)


def create_seam_mask(size: int = 512, 
                     seam_width: int = 64,
                     feather: int = 32,
                     horizontal: bool = False) -> np.ndarray:
    """
    Create a simple strip mask for seam inpainting (legacy function).
    """
    mask = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    
    for i in range(size):
        dist_from_center = abs(i - center)
        
        if dist_from_center <= seam_width // 2:
            value = 1.0
        elif dist_from_center <= seam_width // 2 + feather:
            t = (dist_from_center - seam_width // 2) / feather
            value = 1.0 - t
        else:
            value = 0.0
        
        if horizontal:
            mask[i, :] = value
        else:
            mask[:, i] = value
    
    return mask


def create_precise_seam_mask(face_a_idx: int, face_b_idx: int,
                             yaw_deg: float, pitch_deg: float,
                             size: int = 512,
                             fov_deg: float = 90.0,
                             seam_width: int = 64,
                             feather: int = 32,
                             corner_margin: float = 0.15) -> np.ndarray:
    """
    Create a seam mask: simple strip intersected with rectangular region.
    
    For equator edges (0-3): vertical strip with top/bottom margins
    For polar edges (4-11): horizontal strip with left/right margins
    
    Args:
        face_a_idx, face_b_idx: Indices of the two adjacent faces
        yaw_deg, pitch_deg: Seam camera orientation
        size: Output mask size
        fov_deg: Field of view of seam camera
        seam_width: Width of the core seam area (full mask value)
        feather: Width of gradient transition on each side
        corner_margin: Fraction of image to exclude from edges (0-0.5)
                       Default 0.30 (30%) excludes corner triangles for 90° FOV
    
    Returns:
        Mask array [size, size] with values 0-1
    """
    yaw_rad = np.radians(yaw_deg)
    pitch_rad = np.radians(pitch_deg)
    
    # Create seam view pixel grid
    u, v = np.meshgrid(np.arange(size), np.arange(size))
    
    # Convert pixel coords to normalized camera coords
    half_fov = np.tan(np.radians(fov_deg / 2))
    nx = (u - size / 2) / (size / 2) * half_fov
    ny = -(v - size / 2) / (size / 2) * half_fov
    nz = np.ones_like(nx)
    
    # Normalize to unit vectors
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    
    # Rotate to world coordinates
    wx, wy, wz = rotate_vector(nx, ny, nz, yaw_rad, pitch_rad)
    
    # Determine which face each pixel belongs to (based on dominant axis)
    abs_x, abs_y, abs_z = np.abs(wx), np.abs(wy), np.abs(wz)
    
    # Face assignment
    face_map = np.full((size, size), -1, dtype=np.int32)
    
    # Front (+Z)
    mask_front = (wz > 0) & (abs_z >= abs_x) & (abs_z >= abs_y)
    face_map = np.where(mask_front, 0, face_map)
    # Back (-Z)
    mask_back = (wz < 0) & (abs_z >= abs_x) & (abs_z >= abs_y)
    face_map = np.where(mask_back, 1, face_map)
    # Left (-X)
    mask_left = (wx < 0) & (abs_x >= abs_y) & (abs_x >= abs_z)
    face_map = np.where(mask_left, 2, face_map)
    # Right (+X)
    mask_right = (wx > 0) & (abs_x >= abs_y) & (abs_x >= abs_z)
    face_map = np.where(mask_right, 3, face_map)
    # Top (+Y)
    mask_top = (wy > 0) & (abs_y >= abs_x) & (abs_y >= abs_z)
    face_map = np.where(mask_top, 4, face_map)
    # Bottom (-Y)
    mask_bottom = (wy < 0) & (abs_y >= abs_x) & (abs_y >= abs_z)
    face_map = np.where(mask_bottom, 5, face_map)
    
    # Determine if this is a polar edge (horizontal seam) or equator edge (vertical seam)
    is_polar_edge = (abs(pitch_deg) > 1.0)
    
    # Use fixed rectangular region with configurable margin
    # corner_margin = 0.30 means exclude 30% from edges where corner triangles appear
    margin = int(size * corner_margin)
    
    # Create rectangular mask with margins
    rect_mask = np.zeros((size, size), dtype=np.float32)
    if is_polar_edge:
        # Horizontal seam: exclude left/right corners
        rect_mask[:, margin:size-margin] = 1.0
    else:
        # Vertical seam: exclude top/bottom corners
        rect_mask[margin:size-margin, :] = 1.0
    
    # Create simple strip mask
    strip_mask = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    
    for i in range(size):
        dist_from_center = abs(i - center)
        
        if dist_from_center <= seam_width // 2:
            value = 1.0
        elif dist_from_center <= seam_width // 2 + feather:
            t = (dist_from_center - seam_width // 2) / feather
            value = 1.0 - t
        else:
            value = 0.0
        
        if is_polar_edge:
            strip_mask[i, :] = value  # Horizontal strip
        else:
            strip_mask[:, i] = value  # Vertical strip
    
    # Final mask = strip mask × rectangular bounding box
    mask = strip_mask * rect_mask
    
    return mask


def repair_single_seam(faces: List[np.ndarray],
                       edge_id: int,
                       inpaint_fn,
                       seam_width: int = 100,
                       feather: int = 50,
                       debug_dir: Optional[str] = None) -> List[np.ndarray]:
    """
    Repair a single seam between two adjacent faces.
    
    Args:
        faces: List of 6 face images [H, W, 3]
        edge_id: Which edge to repair (0-11)
        inpaint_fn: Inpainting function(image, mask) -> repaired_image
        seam_width: Width of seam to repair
        feather: Feather width for blending
        debug_dir: If set, save intermediate images for debugging
    
    Returns:
        Modified faces list
    """
    edge_def = EDGE_DEFINITIONS[edge_id]
    _, face_a_idx, face_b_idx, yaw_deg, pitch_deg = edge_def
    
    face_a = faces[face_a_idx]
    face_b = faces[face_b_idx]
    
    print(f"[SeamRepair] Edge {edge_id}: {FACE_NAMES[face_a_idx]}-{FACE_NAMES[face_b_idx]} (yaw={yaw_deg}°, pitch={pitch_deg}°)")
    
    # Step 1: Project to seam view
    seam_view = project_faces_to_seam_view(
        face_a, face_b, face_a_idx, face_b_idx,
        yaw_deg, pitch_deg
    )
    
    # Step 2: Create precise seam mask that follows actual face boundary
    size = seam_view.shape[0]
    mask = create_precise_seam_mask(
        face_a_idx, face_b_idx,
        yaw_deg, pitch_deg,
        size=size,
        seam_width=seam_width,
        feather=feather
    )
    
    if debug_dir:
        Image.fromarray(seam_view).save(f"{debug_dir}/edge{edge_id}_before.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(f"{debug_dir}/edge{edge_id}_mask.png")
    
    # Step 3: Inpaint
    repaired = inpaint_fn(seam_view, mask)
    
    if debug_dir:
        Image.fromarray(repaired).save(f"{debug_dir}/edge{edge_id}_after.png")
    
    # Step 4: Paste back to both faces
    faces[face_a_idx] = paste_back_to_face(
        repaired, faces[face_a_idx], face_a_idx,
        yaw_deg, pitch_deg, mask
    )
    faces[face_b_idx] = paste_back_to_face(
        repaired, faces[face_b_idx], face_b_idx,
        yaw_deg, pitch_deg, mask
    )
    
    return faces


def repair_all_seams(faces: List[np.ndarray],
                     inpaint_fn,
                     seam_width: int = 100,
                     feather: int = 50,
                     debug_dir: Optional[str] = None) -> List[np.ndarray]:
    """
    Repair all 12 seams of a cubemap.
    
    Args:
        faces: List of 6 face images [H, W, 3]
        inpaint_fn: Inpainting function
        seam_width: Width of seam to repair
        feather: Feather width for blending
        debug_dir: If set, save debug images
    
    Returns:
        Modified faces list with all seams repaired
    """
    import os
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
    
    print(f"[SeamRepair] Repairing 12 seams...")
    
    for edge_id in range(12):
        faces = repair_single_seam(
            faces, edge_id, inpaint_fn,
            seam_width=seam_width,
            feather=feather,
            debug_dir=debug_dir
        )
    
    print(f"[SeamRepair] All seams repaired.")
    return faces


# Simple test
if __name__ == "__main__":
    print("Seam Repair Module - Testing projections")
    
    # Create test faces (solid colors)
    size = 512
    faces = [
        np.full((size, size, 3), [255, 0, 0], dtype=np.uint8),    # Front: Red
        np.full((size, size, 3), [0, 255, 0], dtype=np.uint8),    # Back: Green
        np.full((size, size, 3), [0, 0, 255], dtype=np.uint8),    # Left: Blue
        np.full((size, size, 3), [255, 255, 0], dtype=np.uint8),  # Right: Yellow
        np.full((size, size, 3), [255, 0, 255], dtype=np.uint8),  # Top: Magenta
        np.full((size, size, 3), [0, 255, 255], dtype=np.uint8),  # Bottom: Cyan
    ]
    
    # Test projection for Edge 0 (Front-Right)
    seam_view = project_faces_to_seam_view(
        faces[0], faces[3], 0, 3,
        yaw_deg=45.0, pitch_deg=0.0
    )
    
    Image.fromarray(seam_view).save("test_seam_projection.png")
    print("Saved test_seam_projection.png")
    print("Left half should be RED (Front), right half should be YELLOW (Right)")
