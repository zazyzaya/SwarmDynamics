import math

import dearpygui.dearpygui as dpg
import torch

from src.phys_globals import CEILING

SIZE = 1000

# Isometric view
X_TILT = -0.00
Y_TILT = 0.125 / CEILING

# Planes perspective warp
Z_FACTOR = 7.5 / CEILING
TIP_WARP = 0.3
TAIL_WARP = 0.5
WING_WARP = 0.1

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
    to_draw =[]
    coords =[]

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
        to_draw.append((
            dpg.draw_circle,
            dict(
                center=[base_x, base_y], radius=screen_radius,
                color=(wall_color, wall_color, wall_color, 255),
                fill=(wall_color, wall_color, wall_color, 255), parent="main_drawlist"
            )
        ))
        coords.append(center[1])

        # B. The Walls
        to_draw.append((
            dpg.draw_polygon,
            dict(
                points=[
                    [base_x + nx, base_y + ny],
                    [base_x - nx, base_y - ny],
                    [top_x - nx, top_y - ny],
                    [top_x + nx, top_y + ny],
                ],
                color=(wall_color, wall_color, wall_color, 255),
                fill=(wall_color, wall_color, wall_color, 255), parent="main_drawlist"
            )
        ))
        coords.append(center[1] + 1e-5)

        # C. The Roof
        to_draw.append((
            dpg.draw_circle,
            dict(
                center=[top_x, top_y], radius=screen_radius,
                color=(0, 0, 0, 255),
                fill=(roof_color, roof_color, roof_color, 255), parent="main_drawlist"
            )
        ))
        coords.append(center[1] + 2e-5)

    return to_draw, coords

def get_triangles_3d(swarm, color, base_scale=3.0):
    # --- 1. Project Positions ---
    screen_x, screen_y = project_topdown_single(
        swarm.s[:, 0], swarm.s[:, 1], swarm.s[:, 2], SIZE, SIZE
    )
    shadow_x, shadow_y = project_topdown_single(
        swarm.s[:, 0], swarm.s[:, 1], 0, SIZE, SIZE
    )

    screen_centers = torch.stack([screen_x, screen_y], dim=-1)
    shadow_centers = torch.stack([shadow_x, shadow_y], dim=-1)

    # --- 2. Calculate Velocity & Pitch ---
    v_3d = swarm.v
    speed_3d = torch.norm(v_3d, dim=1, keepdim=True).clamp(min=1e-5)

    # 2D Heading for the screen
    v_2d = swarm.v[:, :2]
    speed_2d = torch.norm(v_2d, dim=1, keepdim=True).clamp(min=1e-5)
    heading = v_2d / speed_2d

    # Pitch ranges from -1.0 (straight down) to 1.0 (straight up)
    pitch = v_3d[:, 2:3] / speed_3d

    # --- 3. Dynamic Foreshortening (The 3D Illusion) ---
    tri_scale = base_scale + (swarm.s[:, 2].unsqueeze(-1) * Z_FACTOR)

    # Tip stretches forward if pointing UP
    tip_len = tri_scale * (1.0 + pitch * TIP_WARP)

    # Tail tucks underneath if pointing UP, stretches back if pointing DOWN
    tail_len = tri_scale * 0.6 * (1.0 - pitch * TAIL_WARP)

    # Wings get slightly narrower if pointing UP (tail is further away)
    wing_width = tri_scale * 0.5 * (1.0 - pitch * WING_WARP)

    perp = torch.empty_like(heading)
    perp[:, 0] = -heading[:, 1]
    perp[:, 1] = heading[:, 0]

    # Apply the dynamic lengths to the Drone
    p1 = screen_centers + heading * tip_len
    back_center = screen_centers - heading * tail_len
    p2 = back_center + perp * wing_width
    p3 = back_center - perp * wing_width

    # Apply the exact same dynamic lengths to the Shadows!
    s1 = shadow_centers + heading * tip_len
    shadow_back_center = shadow_centers - heading * tail_len
    s2 = shadow_back_center + perp * wing_width
    s3 = shadow_back_center - perp * wing_width

    to_draw, coords = [], []
    points = torch.stack((p1, p2, p3), dim=1).detach().tolist()
    shadows = torch.stack((s1, s2, s3), dim=1).detach().tolist()

    for i, (p1, p2, p3) in enumerate(points):
        # 1. Draw the Drone
        to_draw.append((
            dpg.draw_triangle,
            dict(
                p1=p1, p2=p2, p3=p3,
                color=(*color, 255),
                fill=(*color, 150),
                parent="main_drawlist"
            )
        ))
        coords.append(swarm.s[i, 1].item())

        # 2. Draw the Shadow (Fake Blur via 3-Pass Stacking)
        s1, s2, s3 = shadows[i]
        c_x, c_y = shadow_centers[i].tolist()
        base_depth = shadow_centers[i, 1].item()

        blur_passes = [
            (1.60, 15, 0.0),
            (1.30, 25, 1e-6),
            (1.00, 50, 2e-6)
        ]

        for scale, alpha, z_offset in blur_passes:
            b1 = [c_x + (s1[0] - c_x) * scale, c_y + (s1[1] - c_y) * scale]
            b2 = [c_x + (s2[0] - c_x) * scale, c_y + (s2[1] - c_y) * scale]
            b3 = [c_x + (s3[0] - c_x) * scale, c_y + (s3[1] - c_y) * scale]

            to_draw.append((
                dpg.draw_triangle,
                dict(
                    p1=b1, p2=b2, p3=b3,
                    color=(0, 0, 0, 0),
                    fill=(0, 0, 0, alpha),
                    parent='main_drawlist'
                )
            ))
            coords.append(base_depth + z_offset)

    return to_draw, coords


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


def painters_alg(commands, depths):
    depth_order = torch.tensor(depths).argsort().tolist()

    for i in depth_order:
        fn,kwarg = commands[i]
        fn(**kwarg)