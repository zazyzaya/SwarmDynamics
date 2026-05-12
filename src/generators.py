import math

import torch

from src.phys_globals import CEILING

def generate_random_triangles(num_columns, device, batch_size=None):
    """
    Generates T random triangular columns.
    If batch_size is provided, generates a unique map for every batch!
    Returns:
        vertices: (B, T, 3, 2) or (T, 3, 2)
        heights: (B, T) or (T,)
    """
    shape = (batch_size, num_columns) if batch_size else (num_columns,)

    # 1. Random Centers and Heights
    centers = (torch.rand(*shape, 2, device=device) * 0.8) + 0.1
    heights = (torch.rand(*shape, device=device) * (4*CEILING / 5)) + CEILING / 5
    heights = heights.sort().values # For viz reasons

    # 2. Spin Vertices around the Centers
    base_angles = torch.tensor([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0], device=device)

    # Expand base angles to match our target shape
    angles = base_angles.expand(*shape, 3)

    # Add +/- 0.5 radians of noise
    angles = angles + (torch.rand(*shape, 3, device=device) - 0.5)

    # Convert polar to cartesian offsets
    radius = 0.05
    x_offsets = torch.cos(angles) * radius
    y_offsets = torch.sin(angles) * radius
    offsets = torch.stack([x_offsets, y_offsets], dim=-1)

    # 3. Add offsets to centers
    # centers is (..., 2), offsets is (..., 3, 2). Unsqueeze centers to (..., 1, 2)!
    vertices = centers.unsqueeze(-2) + offsets

    return vertices, heights

def generate_random_columns(num_columns, device, batch_size=None):
    """
    Generates T random circular columns.
    Returns:
        centers: (B, T, 2) or (T, 2)
        heights: (B, T) or (T,)
    """
    shape = (batch_size, num_columns) if batch_size else (num_columns,)
    centers = (torch.rand(*shape, 2, device=device) * 0.8) + 0.1
    heights = (torch.rand(*shape, device=device) * (4*CEILING / 5)) + CEILING / 5
    radii = (torch.rand(*shape, device=device) * 0.03) + 0.02

    return centers, heights, radii

def generate_games(n, device, pct_self_play=1.):
    blue = torch.randperm(n, device=device)
    red = torch.randperm(n, device=device)
    use_baseline = torch.rand(red.size(0), device=device)
    red = torch.where(use_baseline > pct_self_play, -1, red)

    return blue, red