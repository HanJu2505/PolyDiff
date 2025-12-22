"""
PolyDiff-18 Geometry Configuration Module

Defines the 18-view camera configurations and adjacency matrix for sparse self-attention.
"""

import torch
import numpy as np
from typing import Dict, Tuple, List

# ==============================================================================
# 18-View Camera Configurations
# ==============================================================================

# View configuration: {index: (name, yaw_deg, pitch_deg)}
# Group A (Main Views - 6): Standard cubemap faces
# Group B (Equator Seams - 4): Cover front/back/left/right seams
# Group C (Top-Polar Seams - 4): Cover top-side seams
# Group D (Bottom-Polar Seams - 4): Cover bottom-side seams

VIEW_CONFIG_18: Dict[int, Tuple[str, float, float]] = {
    # Group A: Main Views (Matching CubeDiff order: front, back, left, right, top, bottom)
    0: ("front", 0.0, 0.0),       # Anchor view - looking at +Z
    1: ("back", 180.0, 0.0),      # Looking at -Z
    2: ("left", 270.0, 0.0),      # Looking at -X
    3: ("right", 90.0, 0.0),      # Looking at +X
    4: ("top", 0.0, 90.0),        # Looking at +Y (up/sky)
    5: ("bottom", 0.0, -90.0),    # Looking at -Y (down/ground)
    # Group B: Equator Seams (yaw 45°, 135°, 225°, 315°; pitch 0°)
    6: ("seam_fr", 45.0, 0.0),    # Front-Right seam
    7: ("seam_rb", 135.0, 0.0),   # Right-Back seam
    8: ("seam_bl", 225.0, 0.0),   # Back-Left seam
    9: ("seam_lf", 315.0, 0.0),   # Left-Front seam
    # Group C: Top-Polar Seams (yaw 0°, 90°, 180°, 270°; pitch 45°)
    10: ("seam_tf", 0.0, 45.0),   # Top-Front seam
    11: ("seam_tr", 90.0, 45.0),  # Top-Right seam
    12: ("seam_tb", 180.0, 45.0), # Top-Back seam
    13: ("seam_tl", 270.0, 45.0), # Top-Left seam
    # Group D: Bottom-Polar Seams (yaw 0°, 90°, 180°, 270°; pitch -45°)
    14: ("seam_bf", 0.0, -45.0),  # Bottom-Front seam
    15: ("seam_br", 90.0, -45.0), # Bottom-Right seam
    16: ("seam_bb", 180.0, -45.0),# Bottom-Back seam
    17: ("seam_bl2", 270.0, -45.0),# Bottom-Left seam
}

# View names list for easy access
VIEW_NAMES_18 = [VIEW_CONFIG_18[i][0] for i in range(18)]

# ==============================================================================
# Adjacency Matrix for Sparse Self-Attention
# ==============================================================================

def get_angular_distance(view1_idx: int, view2_idx: int) -> float:
    """
    Calculate angular distance between two views on the sphere.
    Returns angle in degrees.
    """
    _, yaw1, pitch1 = VIEW_CONFIG_18[view1_idx]
    _, yaw2, pitch2 = VIEW_CONFIG_18[view2_idx]
    
    # Convert to radians
    yaw1_rad = np.radians(yaw1)
    pitch1_rad = np.radians(pitch1)
    yaw2_rad = np.radians(yaw2)
    pitch2_rad = np.radians(pitch2)
    
    # Convert to 3D unit vectors
    x1 = np.cos(pitch1_rad) * np.sin(yaw1_rad)
    y1 = np.sin(pitch1_rad)
    z1 = np.cos(pitch1_rad) * np.cos(yaw1_rad)
    
    x2 = np.cos(pitch2_rad) * np.sin(yaw2_rad)
    y2 = np.sin(pitch2_rad)
    z2 = np.cos(pitch2_rad) * np.cos(yaw2_rad)
    
    # Dot product gives cos of angle
    dot = x1*x2 + y1*y2 + z1*z2
    dot = np.clip(dot, -1.0, 1.0)
    
    return np.degrees(np.arccos(dot))


def build_adjacency_mask_18(threshold_deg: float = 90.0) -> torch.Tensor:
    """
    Build a [18, 18] adjacency mask for sparse self-attention.
    
    Views are considered adjacent if their angular distance is less than threshold_deg.
    
    Args:
        threshold_deg: Maximum angular distance for two views to be considered adjacent.
                      Default 90° means views within 90° of each other can attend to each other.
    
    Returns:
        adjacency: [18, 18] tensor where 1 = can attend, 0 = cannot attend
    """
    adjacency = torch.zeros(18, 18, dtype=torch.float32)
    
    for i in range(18):
        for j in range(18):
            if i == j:
                # Always attend to self
                adjacency[i, j] = 1.0
            else:
                dist = get_angular_distance(i, j)
                if dist <= threshold_deg:
                    adjacency[i, j] = 1.0
    
    return adjacency


def get_adjacency_list_18(threshold_deg: float = 90.0) -> Dict[int, List[int]]:
    """
    Get adjacency list for each view (for debugging/visualization).
    """
    adjacency = build_adjacency_mask_18(threshold_deg)
    result = {}
    for i in range(18):
        neighbors = [j for j in range(18) if adjacency[i, j] == 1.0]
        result[i] = neighbors
    return result


# ==============================================================================
# Rotation Matrices
# ==============================================================================

def yaw_pitch_to_rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """
    Convert yaw and pitch angles to a 3x3 rotation matrix.
    
    Convention:
    - Yaw: rotation around Y-axis (left/right look)
    - Pitch: rotation around X-axis (up/down look)
    - Default (yaw=0, pitch=0): looking along +Z axis
    
    Returns:
        R: [3, 3] rotation matrix
    """
    yaw_rad = np.radians(yaw_deg)
    pitch_rad = np.radians(pitch_deg)
    
    # Rotation around Y-axis (yaw)
    Ry = np.array([
        [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [0, 1, 0],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ])
    
    # Rotation around X-axis (pitch)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    
    # Combined rotation: first yaw, then pitch
    R = Rx @ Ry
    return R


def get_rotation_matrices_18() -> Dict[int, np.ndarray]:
    """
    Get rotation matrices for all 18 views.
    
    Returns:
        Dict mapping view index to [3, 3] rotation matrix
    """
    matrices = {}
    for idx, (name, yaw, pitch) in VIEW_CONFIG_18.items():
        matrices[idx] = yaw_pitch_to_rotation_matrix(yaw, pitch)
    return matrices


# ==============================================================================
# View Index Utilities
# ==============================================================================

def get_main_view_indices() -> List[int]:
    """Return indices of the 6 main cubemap views."""
    return list(range(6))


def get_seam_view_indices() -> List[int]:
    """Return indices of the 12 seam views."""
    return list(range(6, 18))


def get_equator_seam_indices() -> List[int]:
    """Return indices of equator seam views (6-9)."""
    return list(range(6, 10))


def get_top_seam_indices() -> List[int]:
    """Return indices of top-polar seam views (10-13)."""
    return list(range(10, 14))


def get_bottom_seam_indices() -> List[int]:
    """Return indices of bottom-polar seam views (14-17)."""
    return list(range(14, 18))


# ==============================================================================
# Prompt Assignment Strategy
# ==============================================================================

def assign_prompts_18(layout: Dict[str, str]) -> List[str]:
    """
    Assign prompts to 18 views based on layout description.
    
    Main views (0-5) get specific directional prompts.
    Seam views (6-17) get EMPTY prompts to rely on self-attention from main views.
    
    Args:
        layout: Dict with keys 'Front', 'Right', 'Back', 'Left', 'Top', 'Bottom', 'Overall'
    
    Returns:
        List of 18 prompts, one for each view
    """
    prompts = [""] * 18
    
    # Main views - use specific descriptions (matching CubeDiff face order)
    prompts[0] = layout.get('Front', layout.get('Overall', ''))
    prompts[1] = layout.get('Back', layout.get('Overall', ''))
    prompts[2] = layout.get('Left', layout.get('Overall', ''))
    prompts[3] = layout.get('Right', layout.get('Overall', ''))
    prompts[4] = layout.get('Top', layout.get('Overall', ''))
    prompts[5] = layout.get('Bottom', layout.get('Overall', ''))
    
    # Seam views - use EMPTY prompts to rely on self-attention
    # This makes seam views "copy" content from adjacent main views
    # instead of generating their own content based on 'Overall' description
    for i in range(6, 18):
        prompts[i] = ""  # Empty prompt
    
    return prompts


if __name__ == "__main__":
    # Test adjacency matrix
    adj = build_adjacency_mask_18(90.0)
    print("Adjacency Matrix (90° threshold):")
    print(adj)
    print(f"\nTotal connections per view: {adj.sum(dim=1).tolist()}")
    
    # Show adjacency list
    adj_list = get_adjacency_list_18(90.0)
    print("\nAdjacency List:")
    for i, neighbors in adj_list.items():
        name = VIEW_CONFIG_18[i][0]
        neighbor_names = [VIEW_CONFIG_18[j][0] for j in neighbors]
        print(f"  {i:2d} ({name:10s}): {neighbor_names}")
