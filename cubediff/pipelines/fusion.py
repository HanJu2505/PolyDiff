"""
PolyDiff-18 Gaussian Weighted Spherical Fusion Module

Fuses 18 perspective views into a single Equirectangular Panorama (ERP) image
using Gaussian-weighted blending to eliminate seams.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import math


def perspective_to_spherical(px: np.ndarray, py: np.ndarray, 
                              yaw_deg: float, pitch_deg: float, 
                              fov_deg: float, img_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert pixel coordinates in a perspective image to spherical coordinates.
    
    Args:
        px, py: Pixel coordinates in the perspective image (0 to img_size-1)
        yaw_deg: Yaw angle of the camera (degrees)
        pitch_deg: Pitch angle of the camera (degrees)
        fov_deg: Field of view (degrees)
        img_size: Size of the square perspective image
    
    Returns:
        theta: Azimuth angle (-pi to pi)
        phi: Elevation angle (-pi/2 to pi/2)
    """
    # Convert pixel coords to normalized image plane coords (-1 to 1)
    cx = img_size / 2.0
    cy = img_size / 2.0
    
    # Focal length based on FOV
    f = img_size / (2 * np.tan(np.radians(fov_deg) / 2))
    
    # Ray direction in camera space (looking along +Z)
    x = (px - cx)
    y = (cy - py)  # Flip y for image coordinates
    z = np.full_like(x, f)
    
    # Normalize to unit vectors
    norm = np.sqrt(x**2 + y**2 + z**2)
    x, y, z = x/norm, y/norm, z/norm
    
    # Apply rotation: first yaw (around Y), then pitch (around X)
    yaw_rad = np.radians(yaw_deg)
    pitch_rad = np.radians(pitch_deg)
    
    # Rotation around Y-axis (yaw)
    x_yaw = x * np.cos(yaw_rad) + z * np.sin(yaw_rad)
    y_yaw = y
    z_yaw = -x * np.sin(yaw_rad) + z * np.cos(yaw_rad)
    
    # Rotation around X-axis (pitch)
    x_rot = x_yaw
    y_rot = y_yaw * np.cos(pitch_rad) - z_yaw * np.sin(pitch_rad)
    z_rot = y_yaw * np.sin(pitch_rad) + z_yaw * np.cos(pitch_rad)
    
    # Convert to spherical coordinates
    theta = np.arctan2(x_rot, z_rot)  # Azimuth: -pi to pi
    phi = np.arcsin(np.clip(y_rot, -1, 1))  # Elevation: -pi/2 to pi/2
    
    return theta, phi


def spherical_to_erp(theta: np.ndarray, phi: np.ndarray, 
                     erp_width: int, erp_height: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert spherical coordinates to ERP pixel coordinates.
    
    Args:
        theta: Azimuth angle (-pi to pi)
        phi: Elevation angle (-pi/2 to pi/2)
        erp_width, erp_height: Dimensions of the ERP image
    
    Returns:
        u, v: Pixel coordinates in the ERP image
    """
    # theta: -pi -> 0, pi -> erp_width
    u = (theta / np.pi + 1) / 2 * erp_width
    # phi: pi/2 -> 0, -pi/2 -> erp_height
    v = (0.5 - phi / np.pi) * erp_height
    
    return u, v


def gaussian_weight(px: np.ndarray, py: np.ndarray, 
                    center_x: float, center_y: float, 
                    sigma: float) -> np.ndarray:
    """
    Calculate Gaussian weight based on distance from image center.
    
    Args:
        px, py: Pixel coordinates
        center_x, center_y: Center of the image
        sigma: Standard deviation of the Gaussian
    
    Returns:
        weight: Gaussian weight (0 to 1)
    """
    dist_sq = (px - center_x)**2 + (py - center_y)**2
    weight = np.exp(-dist_sq / (2 * sigma**2))
    return weight


def create_view_projection_lut(img_size: int, yaw_deg: float, pitch_deg: float, 
                                fov_deg: float, erp_width: int, erp_height: int,
                                sigma_factor: float = 0.4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create lookup tables for projecting a perspective view to ERP.
    
    Args:
        img_size: Size of the square perspective image
        yaw_deg, pitch_deg: Camera orientation
        fov_deg: Field of view
        erp_width, erp_height: ERP dimensions
        sigma_factor: Gaussian sigma as fraction of image size
    
    Returns:
        erp_u: ERP u coordinates for each pixel [img_size, img_size]
        erp_v: ERP v coordinates for each pixel [img_size, img_size]
        weights: Gaussian weights for each pixel [img_size, img_size]
    """
    # Create pixel coordinate grids
    px, py = np.meshgrid(np.arange(img_size), np.arange(img_size))
    
    # Convert to spherical coordinates
    theta, phi = perspective_to_spherical(px, py, yaw_deg, pitch_deg, fov_deg, img_size)
    
    # Convert to ERP coordinates
    erp_u, erp_v = spherical_to_erp(theta, phi, erp_width, erp_height)
    
    # Calculate Gaussian weights
    center = img_size / 2.0
    sigma = sigma_factor * img_size
    weights = gaussian_weight(px, py, center, center, sigma)
    
    return erp_u, erp_v, weights


def gaussian_spherical_fusion(views: List[np.ndarray], 
                               view_config: Dict[int, Tuple[str, float, float]],
                               fov_deg: float = 95.0,
                               erp_height: int = 1024,
                               erp_width: int = 2048,
                               sigma_factor: float = 0.4) -> np.ndarray:
    """
    Fuse 18 perspective views into an ERP panorama using Gaussian-weighted blending.
    
    Args:
        views: List of 18 perspective images [H, W, 3] in RGB, values 0-255
        view_config: Dict {idx: (name, yaw_deg, pitch_deg)} for each view
        fov_deg: Field of view of each perspective image
        erp_height, erp_width: Output ERP dimensions
        sigma_factor: Gaussian sigma as fraction of image size (controls blending sharpness)
    
    Returns:
        erp_image: Fused ERP image [erp_height, erp_width, 3] as uint8
    """
    if len(views) == 0:
        raise ValueError("No views provided for fusion")
    
    img_size = views[0].shape[0]
    num_views = len(views)
    
    # Initialize accumulators
    erp_canvas = np.zeros((erp_height, erp_width, 3), dtype=np.float64)
    weight_canvas = np.zeros((erp_height, erp_width), dtype=np.float64)
    
    print(f"[Fusion] Fusing {num_views} views to {erp_width}x{erp_height} ERP...")
    
    for view_idx in range(num_views):
        view_img = views[view_idx].astype(np.float64)
        name, yaw_deg, pitch_deg = view_config[view_idx]
        
        # Create projection lookup tables
        erp_u, erp_v, weights = create_view_projection_lut(
            img_size, yaw_deg, pitch_deg, fov_deg, erp_width, erp_height, sigma_factor
        )
        
        # Clamp coordinates to valid range
        erp_u_int = np.clip(np.round(erp_u).astype(np.int32), 0, erp_width - 1)
        erp_v_int = np.clip(np.round(erp_v).astype(np.int32), 0, erp_height - 1)
        
        # Flatten for scatter-add operation
        flat_u = erp_u_int.ravel()
        flat_v = erp_v_int.ravel()
        flat_weights = weights.ravel()
        flat_colors = view_img.reshape(-1, 3)
        
        # Accumulate weighted colors
        for i in range(len(flat_u)):
            u, v, w = flat_u[i], flat_v[i], flat_weights[i]
            erp_canvas[v, u] += flat_colors[i] * w
            weight_canvas[v, u] += w
    
    # Normalize by total weight
    eps = 1e-8
    erp_canvas = erp_canvas / (weight_canvas[..., None] + eps)
    
    # Handle any remaining zero-weight pixels (shouldn't happen with 18 views)
    zero_mask = weight_canvas < eps
    if np.any(zero_mask):
        print(f"[Fusion] Warning: {np.sum(zero_mask)} pixels have zero coverage")
    
    # Clip and convert to uint8
    erp_image = np.clip(erp_canvas, 0, 255).astype(np.uint8)
    
    print(f"[Fusion] Fusion complete.")
    return erp_image


def histogram_match_channel(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Match histogram of source channel to reference channel.
    """
    # Get histograms
    src_values, bin_idx, src_counts = np.unique(source.ravel(), return_inverse=True, return_counts=True)
    ref_values, ref_counts = np.unique(reference.ravel(), return_counts=True)
    
    # Compute CDFs
    src_cdf = np.cumsum(src_counts).astype(np.float64)
    src_cdf /= src_cdf[-1]
    ref_cdf = np.cumsum(ref_counts).astype(np.float64)
    ref_cdf /= ref_cdf[-1]
    
    # Map source to reference via CDF matching
    matched_values = np.interp(src_cdf, ref_cdf, ref_values)
    return matched_values[bin_idx].reshape(source.shape)


def histogram_match_to_reference(views: List[np.ndarray], reference_idx: int = 0) -> List[np.ndarray]:
    """
    Match histogram of all views to the reference view (Front view by default).
    
    This corrects brightness/color differences between views before fusion.
    
    Args:
        views: List of perspective images [H, W, 3]
        reference_idx: Index of reference view (0 = Front)
    
    Returns:
        List of color-corrected views
    """
    reference = views[reference_idx]
    corrected = []
    
    for i, view in enumerate(views):
        if i == reference_idx:
            corrected.append(view.copy())
        else:
            # Match each RGB channel separately
            matched = np.zeros_like(view, dtype=np.float64)
            for c in range(3):
                matched[:, :, c] = histogram_match_channel(
                    view[:, :, c].astype(np.float64),
                    reference[:, :, c].astype(np.float64)
                )
            corrected.append(np.clip(matched, 0, 255).astype(np.uint8))
    
    return corrected


def gaussian_spherical_fusion_fast(views: List[np.ndarray], 
                                    view_config: Dict[int, Tuple[str, float, float]],
                                    fov_deg: float = 95.0,
                                    erp_height: int = 1024,
                                    erp_width: int = 2048,
                                    sigma_factor: float = 0.25,
                                    main_view_boost: float = 1.5,
                                    color_correct: bool = True) -> np.ndarray:
    """
    Fast version of Gaussian spherical fusion with color correction and main view priority.
    
    Args:
        views: List of 18 perspective images
        view_config: Dict mapping view index to (name, yaw, pitch)
        fov_deg: Field of view
        erp_height, erp_width: Output dimensions
        sigma_factor: Gaussian sigma as fraction of image size (smaller = sharper center focus)
        main_view_boost: Weight multiplier for main views (0-5), default 1.5x
        color_correct: Whether to apply histogram matching to reference view
    """
    if len(views) == 0:
        raise ValueError("No views provided for fusion")
    
    img_size = views[0].shape[0]
    num_views = len(views)
    
    # Step 1: Color correction - match all views to Front view
    if color_correct:
        print(f"[Fusion] Applying histogram matching to reference (Front view)...")
        views = histogram_match_to_reference(views, reference_idx=0)
    
    # Use separate 2D arrays for each channel to enable proper np.add.at
    erp_r = np.zeros((erp_height, erp_width), dtype=np.float64)
    erp_g = np.zeros((erp_height, erp_width), dtype=np.float64)
    erp_b = np.zeros((erp_height, erp_width), dtype=np.float64)
    weight_canvas = np.zeros((erp_height, erp_width), dtype=np.float64)
    
    print(f"[Fusion] Fusing {num_views} views (sigma={sigma_factor}, main_boost={main_view_boost})...")
    
    for view_idx in range(num_views):
        view_img = views[view_idx].astype(np.float64)
        name, yaw_deg, pitch_deg = view_config[view_idx]
        
        # Create projection lookup tables
        erp_u, erp_v, weights = create_view_projection_lut(
            img_size, yaw_deg, pitch_deg, fov_deg, erp_width, erp_height, sigma_factor
        )
        
        # Apply main view boost (views 0-5 are main views)
        if view_idx < 6:
            weights = weights * main_view_boost
        
        # Clamp coordinates to valid range
        erp_u_int = np.clip(np.round(erp_u).astype(np.int32), 0, erp_width - 1)
        erp_v_int = np.clip(np.round(erp_v).astype(np.int32), 0, erp_height - 1)
        
        # Flatten for vectorized operations
        flat_v = erp_v_int.ravel()
        flat_u = erp_u_int.ravel()
        flat_weights = weights.ravel()
        
        # Create 1D linear indices
        linear_idx = flat_v * erp_width + flat_u
        
        # Weighted colors
        weighted_r = view_img[:, :, 0].ravel() * flat_weights
        weighted_g = view_img[:, :, 1].ravel() * flat_weights
        weighted_b = view_img[:, :, 2].ravel() * flat_weights
        
        # Vectorized add.at on 1D views (these work because erp_r.ravel() returns a view)
        np.add.at(erp_r.ravel(), linear_idx, weighted_r)
        np.add.at(erp_g.ravel(), linear_idx, weighted_g)
        np.add.at(erp_b.ravel(), linear_idx, weighted_b)
        np.add.at(weight_canvas.ravel(), linear_idx, flat_weights)
    
    # Normalize by total weight
    eps = 1e-8
    erp_r = erp_r / (weight_canvas + eps)
    erp_g = erp_g / (weight_canvas + eps)
    erp_b = erp_b / (weight_canvas + eps)
    
    # Stack channels
    erp_canvas = np.stack([erp_r, erp_g, erp_b], axis=-1)
    
    # Clip and convert to uint8
    erp_image = np.clip(erp_canvas, 0, 255).astype(np.uint8)
    
    print(f"[Fusion] Fusion complete.")
    return erp_image


# ============================================================================
# INVERSE MAPPING FUSION (Recommended - No Moiré patterns)
# ============================================================================

def erp_to_spherical(u: np.ndarray, v: np.ndarray, 
                     erp_width: int, erp_height: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert ERP pixel coordinates to spherical coordinates.
    
    Args:
        u, v: ERP pixel coordinates
        erp_width, erp_height: ERP dimensions
    
    Returns:
        theta: Azimuth angle (-pi to pi)
        phi: Elevation angle (-pi/2 to pi/2)
    """
    theta = (u / erp_width * 2 - 1) * np.pi  # -pi to pi
    phi = (0.5 - v / erp_height) * np.pi      # pi/2 to -pi/2
    return theta, phi


def spherical_to_perspective(theta: np.ndarray, phi: np.ndarray,
                              yaw_deg: float, pitch_deg: float,
                              fov_deg: float, img_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert spherical coordinates to perspective image pixel coordinates.
    
    This is the INVERSE of perspective_to_spherical.
    
    Args:
        theta, phi: Spherical coordinates
        yaw_deg, pitch_deg: Camera orientation
        fov_deg: Field of view
        img_size: Size of perspective image
    
    Returns:
        px, py: Pixel coordinates in perspective image
        valid: Boolean mask for valid projections (in front of camera and within FOV)
    """
    yaw_rad = np.radians(yaw_deg)
    pitch_rad = np.radians(pitch_deg)
    
    # Convert spherical to 3D unit vector (world space)
    x_world = np.cos(phi) * np.sin(theta)
    y_world = np.sin(phi)
    z_world = np.cos(phi) * np.cos(theta)
    
    # Inverse rotation: first -pitch (around X), then -yaw (around Y)
    # Undo pitch
    x_unpitch = x_world
    y_unpitch = y_world * np.cos(-pitch_rad) - z_world * np.sin(-pitch_rad)
    z_unpitch = y_world * np.sin(-pitch_rad) + z_world * np.cos(-pitch_rad)
    
    # Undo yaw
    x_cam = x_unpitch * np.cos(-yaw_rad) + z_unpitch * np.sin(-yaw_rad)
    y_cam = y_unpitch
    z_cam = -x_unpitch * np.sin(-yaw_rad) + z_unpitch * np.cos(-yaw_rad)
    
    # Check if point is in front of camera (z > 0)
    valid = z_cam > 0.01
    
    # Project to image plane
    f = img_size / (2 * np.tan(np.radians(fov_deg) / 2))
    
    # Avoid division by zero
    z_safe = np.where(valid, z_cam, 1.0)
    
    px = x_cam / z_safe * f + img_size / 2.0
    py = img_size / 2.0 - y_cam / z_safe * f  # Flip y for image coords
    
    # Check if within image bounds
    valid = valid & (px >= 0) & (px < img_size) & (py >= 0) & (py < img_size)
    
    return px, py, valid


def inverse_mapping_fusion(views: List[np.ndarray],
                           view_config: Dict[int, Tuple[str, float, float]],
                           fov_deg: float = 95.0,
                           erp_height: int = 1024,
                           erp_width: int = 2048,
                           mode: str = "wta",
                           effective_fov_deg: float = 70.0,
                           sigma_factor: float = 0.15,
                           debug_views: Optional[List[int]] = None) -> np.ndarray:
    """
    Inverse mapping fusion: for each ERP pixel, sample from perspective views.
    
    This avoids Moiré patterns caused by forward projection.
    
    Args:
        views: List of perspective images [H, W, 3]
        view_config: Dict {idx: (name, yaw, pitch)}
        fov_deg: Actual FOV of perspective images
        erp_height, erp_width: Output ERP dimensions
        mode: Fusion mode
            - "wta": Winner-Takes-All (for debugging coordinate alignment)
            - "gaussian": Soft blending with Gaussian weights
        effective_fov_deg: Only use center portion of each view (ignore edges)
        sigma_factor: Gaussian sigma for soft blending
        debug_views: If set, only use these view indices (for debugging)
    
    Returns:
        ERP image [erp_height, erp_width, 3]
    """
    if len(views) == 0:
        raise ValueError("No views provided")
    
    img_size = views[0].shape[0]
    num_views = len(views) if debug_views is None else len(debug_views)
    view_indices = list(range(len(views))) if debug_views is None else debug_views
    
    print(f"[Fusion] Inverse mapping fusion (mode={mode}, views={len(view_indices)})...")
    
    # Create ERP coordinate grid
    erp_u, erp_v = np.meshgrid(np.arange(erp_width), np.arange(erp_height))
    
    # Convert to spherical
    theta, phi = erp_to_spherical(erp_u, erp_v, erp_width, erp_height)
    
    # Initialize output
    erp_image = np.zeros((erp_height, erp_width, 3), dtype=np.float64)
    
    if mode == "wta":
        # Winner-Takes-All: only use closest view center
        best_weight = np.full((erp_height, erp_width), -np.inf)
        
        for view_idx in view_indices:
            view_img = views[view_idx].astype(np.float64)
            name, yaw_deg_v, pitch_deg_v = view_config[view_idx]
            
            # Project ERP to this view
            px, py, valid = spherical_to_perspective(
                theta, phi, yaw_deg_v, pitch_deg_v, fov_deg, img_size
            )
            
            # Calculate distance from view center
            center = img_size / 2.0
            dist_sq = (px - center)**2 + (py - center)**2
            
            # Effective FOV mask
            max_dist = img_size / 2.0 * np.tan(np.radians(effective_fov_deg/2)) / np.tan(np.radians(fov_deg/2))
            valid = valid & (np.sqrt(dist_sq) < max_dist)
            
            # Weight = negative distance (closer = higher)
            weight = -dist_sq
            weight = np.where(valid, weight, -np.inf)
            
            # Update where this view is better
            better = weight > best_weight
            
            # Sample from view (bilinear interpolation)
            px_int = np.clip(px.astype(np.int32), 0, img_size - 1)
            py_int = np.clip(py.astype(np.int32), 0, img_size - 1)
            
            sampled = view_img[py_int, px_int]
            
            erp_image = np.where(better[..., None], sampled, erp_image)
            best_weight = np.where(better, weight, best_weight)
        
        print(f"[Fusion] WTA complete. Coverage: {np.sum(best_weight > -np.inf) / (erp_height * erp_width) * 100:.1f}%")
    
    elif mode == "gaussian":
        # Gaussian weighted blending
        weight_sum = np.zeros((erp_height, erp_width))
        
        for view_idx in view_indices:
            view_img = views[view_idx].astype(np.float64)
            name, yaw_deg_v, pitch_deg_v = view_config[view_idx]
            
            # Project ERP to this view
            px, py, valid = spherical_to_perspective(
                theta, phi, yaw_deg_v, pitch_deg_v, fov_deg, img_size
            )
            
            # Calculate distance from view center
            center = img_size / 2.0
            dist_sq = (px - center)**2 + (py - center)**2
            
            # Effective FOV mask - zero weight outside
            max_dist = img_size / 2.0 * np.tan(np.radians(effective_fov_deg/2)) / np.tan(np.radians(fov_deg/2))
            valid = valid & (np.sqrt(dist_sq) < max_dist)
            
            # Gaussian weight
            sigma = sigma_factor * img_size
            weight = np.exp(-dist_sq / (2 * sigma**2))
            weight = np.where(valid, weight, 0)
            
            # Sample from view
            px_int = np.clip(px.astype(np.int32), 0, img_size - 1)
            py_int = np.clip(py.astype(np.int32), 0, img_size - 1)
            
            sampled = view_img[py_int, px_int]
            
            erp_image += sampled * weight[..., None]
            weight_sum += weight
        
        # Normalize
        erp_image = erp_image / (weight_sum[..., None] + 1e-8)
        print(f"[Fusion] Gaussian complete. Coverage: {np.sum(weight_sum > 0.01) / (erp_height * erp_width) * 100:.1f}%")
    
    erp_image = np.clip(erp_image, 0, 255).astype(np.uint8)
    print(f"[Fusion] Done.")
    return erp_image


if __name__ == "__main__":
    # Simple test with dummy data
    from .geometry import VIEW_CONFIG_18
    
    # Create dummy views (all black)
    img_size = 64
    views = [np.zeros((img_size, img_size, 3), dtype=np.uint8) for _ in range(18)]
    
    # Run fusion
    erp = inverse_mapping_fusion(views, VIEW_CONFIG_18, fov_deg=95.0, mode="wta")
    print(f"Output ERP shape: {erp.shape}")
