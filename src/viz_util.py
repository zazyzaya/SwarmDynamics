import math

import dearpygui.dearpygui as dpg
import torch

from src.phys_globals import CEILING

SIZE = 1000
X_TILT = -0.0015
Y_TILT = 0.025

def project_topdown_single(x, y, z, screen_w, screen_h):
    """
    Projects a top-down view where Z-height extrudes in a specific 2D direction.
    """
    x_centered = x - 0.5
    y_centered = y - 0.5

    scale = min(screen_w, screen_h) * 0.8

    # Z now pushes the X coordinate to the right, and the Y coordinate UP (negative)
    screen_x = (x_centered * scale) + (screen_w * 0.5) + (z * scale * X_TILT)
    screen_y = (y_centered * scale) + (screen_h * 0.5) - (z * scale * Y_TILT)

    return screen_x, screen_y

def _3d_columns(obs_centers, obs_heights, obs_r):
    # 1. Sort the columns by their Y-coordinate (back-to-front)
    # obs_centers[:, 1] grabs all the Y values. argsort defaults to ascending (smallest Y first)
    sort_idx = torch.argsort(obs_centers[:, 1])

    # Reorder all three tensors using the sorted indices
    sorted_centers = obs_centers[sort_idx]
    sorted_heights = obs_heights[sort_idx]
    sorted_r = obs_r[sort_idx]

    scale = SIZE * 0.8

    # 2. Run the loop on the sorted data
    for i, center in enumerate(sorted_centers.tolist()):
        height = sorted_heights[i].item()
        screen_radius = sorted_r[i].item() * scale

        # Base and Roof Projections (using our shared function defaults)
        base_x, base_y = project_topdown_single(center[0], center[1], 0.0, SIZE, SIZE)
        top_x, top_y = project_topdown_single(center[0], center[1], height, SIZE, SIZE)

        wall_color = int(200 * (height / CEILING))
        roof_color = int(255 * (height / CEILING))

        # Tangent Math for perfect walls
        dx = top_x - base_x
        dy = top_y - base_y
        L = math.hypot(dx, dy) + 1e-5

        nx = -(dy / L) * screen_radius
        ny = (dx / L) * screen_radius

        # A. Base Circle
        dpg.draw_circle(
            center=[base_x, base_y], radius=screen_radius,
            color=(wall_color, wall_color, wall_color, 255),
            fill=(wall_color, wall_color, wall_color, 255), parent="main_drawlist"
        )

        # B. The Walls
        dpg.draw_polygon(
            points=[
                [base_x + nx, base_y + ny],
                [base_x - nx, base_y - ny],
                [top_x - nx, top_y - ny],
                [top_x + nx, top_y + ny],
            ],
            color=(wall_color, wall_color, wall_color, 255),
            fill=(wall_color, wall_color, wall_color, 255), parent="main_drawlist"
        )

        # C. The Roof
        dpg.draw_circle(
            center=[top_x, top_y], radius=screen_radius,
            color=(0, 0, 0, 255),
            fill=(roof_color, roof_color, roof_color, 255), parent="main_drawlist"
        )

def get_triangles_3d(swarm, base_scale=3.0, z_factor=1.5):
    # --- 1. Project Positions (Using the shared function!) ---
    screen_x, screen_y = project_topdown_single(
        swarm.s[:, 0], swarm.s[:, 1], swarm.s[:, 2], SIZE, SIZE
    )

    screen_centers = torch.stack([screen_x, screen_y], dim=-1)

    # --- 2. Standard 2D Velocity Heading ---
    v_2d = swarm.v[:, :2]
    speed = torch.norm(v_2d, dim=1, keepdim=True).clamp(min=1e-5)
    heading = v_2d / speed

    # --- 3. Draw Triangles ---
    tri_scale = base_scale + (swarm.s[:, 2].unsqueeze(-1) * z_factor)

    perp = torch.empty_like(heading)
    perp[:, 0] = -heading[:, 1]
    perp[:, 1] = heading[:, 0]

    p1 = screen_centers + heading * tri_scale
    back_center = screen_centers - heading * (tri_scale * 0.6)
    p2 = back_center + perp * (tri_scale * 0.5)
    p3 = back_center - perp * (tri_scale * 0.5)

    return torch.stack((p1, p2, p3), dim=1).detach().tolist()

def get_triangles(swarm, base_scale=3.0, z_factor=1.5):
    # 1. 2D positions and dynamically scaled sizes based on Z-height
    pos_2d = swarm.s[:, :2] * SIZE
    z = swarm.s[:, 2:3]
    scale = base_scale + (z * z_factor)

    # 2. Get normalized 2D velocity heading
    v_2d = swarm.v[:, :2]
    speed = torch.norm(v_2d, dim=1, keepdim=True).clamp(min=1e-5)
    heading = v_2d / speed

    # 3. Get the perpendicular vector to the heading (-y, x)
    perp = torch.empty_like(heading)
    perp[:, 0] = -heading[:, 1]
    # Invert Y for perpendicular to match screen coordinates correctly
    perp[:, 1] = heading[:, 0]

    # 4. Calculate the three vertices
    p1 = pos_2d + heading * scale                         # Tip
    back_center = pos_2d - heading * (scale * 0.6)        # Base center
    p2 = back_center + perp * (scale * 0.5)               # Left wing
    p3 = back_center - perp * (scale * 0.5)               # Right wing

    # 5. Stack and convert to a Python list for Dear PyGui
    # Output shape becomes (N, 3, 2) -> list of [ [x1,y1], [x2,y2], [x3,y3] ]
    return torch.stack((p1, p2, p3), dim=1).detach().tolist()