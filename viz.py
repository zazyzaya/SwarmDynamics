import dearpygui.dearpygui as dpg
import torch

from sim import Env

SIZE = 1000
df = Env(100, 100)

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

dpg.create_context()
dpg.create_viewport(title='PyTorch + Dear PyGui', width=SIZE, height=SIZE)
dpg.setup_dearpygui()

df = Env(100, 100)

# Define the UI structure ONCE outside the loop
with dpg.window(label="Pixel Renderer", width=SIZE, height=SIZE):
    # We give the drawlist a 'tag' so we can refer to it later
    with dpg.drawlist(width=SIZE, height=SIZE, tag="main_drawlist"):
        pass

dpg.show_viewport()

game_over = torch.zeros(2)
first_game_over = True
active_explosions = []
steps = 0
while dpg.is_dearpygui_running():
    # 1. Clear the previous frame's pixels
    dpg.delete_item("main_drawlist", children_only=True)

    # Draw the boids
    blue_tris = get_triangles(df.blue)
    red_tris = get_triangles(df.red)

    # Draw Blue Team
    for p1, p2, p3 in blue_tris:
        dpg.draw_triangle(
            p1=p1, p2=p2, p3=p3,
            color=(0, 255, 255, 255),  # Outline
            fill=(0, 255, 255, 150),   # Slightly transparent fill looks cool when they stack
            parent="main_drawlist"
        )

    # Draw Red Team
    for p1, p2, p3 in red_tris:
        dpg.draw_triangle(
            p1=p1, p2=p2, p3=p3,
            color=(255, 0, 0, 255),
            fill=(255, 0, 0, 150),
            parent="main_drawlist"
        )

    # 2. Draw and update active explosions
    for exp in active_explosions[:]:  # Iterate over a slice [:] so we can safely remove items
        # Draw the explosion
        dpg.draw_circle(
            center=exp['pos'],
            radius=exp['radius'],
            color=exp['color'],  # Orange-red outline
            fill=exp['fill'],    # Transparent red fill
            parent="main_drawlist"
        )

        # Update the radius for the next frame
        if exp['expanding']:
            exp['radius'] += 1.5  # Growth speed
            if exp['radius'] >= 15.0:  # Max size
                exp['expanding'] = False
        else:
            exp['radius'] -= 1.5  # Shrink speed
            if exp['radius'] <= 0.0:
                active_explosions.remove(exp) # Animation finished, remove it!

    if not game_over.any():
        # Update physics
        game_over, new_explosions, b_col, r_col = df.update()
        steps += 1

        # Add drones shot down
        if new_explosions is not None and len(new_explosions) > 0:
            # Convert PyTorch tensor to scaled pixel coordinates
            pixel_exps = (new_explosions[:, :2] * SIZE).detach().tolist()

            for p in pixel_exps:
                active_explosions.append({
                    'pos': p,
                    'radius': 1.0,
                    'expanding': True,
                    'color': (255, 255, 0, 255),
                    'fill': (255, 150, 0, 150)
                })

        # Add collisions
        if b_col is not None and len(b_col):
            pixel_exps = (b_col[:, :2] * SIZE).detach().tolist()

            for p in pixel_exps:
                active_explosions.append({
                    'pos': p,
                    'radius': 1.0,
                    'expanding': True,
                    'color': (0, 100, 255, 255),
                    'fill': (0, 50, 255, 150)
                })
        if r_col is not None and len(r_col):
            pixel_exps = (r_col[:, :2] * SIZE).detach().tolist()

            for p in pixel_exps:
                active_explosions.append({
                    'pos': p,
                    'radius': 1.0,
                    'expanding': True,
                    'color': (255, 0, 0, 255),
                    'fill': (255, 25, 0, 150)
                })

    else:
        center_x = SIZE // 2 - 100
        center_y = SIZE // 2

        dpg.draw_text(
            pos=(center_x, center_y - 40),
            text="Game over", size=40,
            color=(255, 255, 255, 255),
            parent='main_drawlist'
        )

        dpg.draw_text(
            pos=(center_x, center_y + 40),
            text=f"   {steps} Steps", size=20,
            color=(255, 255, 255, 255),
            parent='main_drawlist'
        )

        if game_over[0] and not game_over[1]:
            dpg.draw_text(
                pos=(center_x, center_y),
                text="  Red wins", size=30,
                color=(255, 0, 0, 255),
                parent='main_drawlist'
            )
        elif game_over[1] and not game_over[0]:
            dpg.draw_text(
                pos=(center_x, center_y),
                text="  Blue wins", size=30,
                color=(0, 255, 255, 255),
                parent='main_drawlist'
            )
        else:
            dpg.draw_text(
                pos=(center_x, center_y),
                text="    Draw!",
                size=30, color=(0, 255, 0, 255),
                parent='main_drawlist'
            )

        if first_game_over:
            first_game_over = False
            b_s, b_k, r_s, r_k = df.get_stats(top_k=10)

            print("Longest Livers:")
            print("Blue")
            print(b_s)
            print("Red")
            print(r_s)
            print('\n')
            print("Harshest Killers:")
            print("Blue")
            print(b_k)
            print("Red")
            print(r_k)


    # 3. Render the frame
    dpg.render_dearpygui_frame()


dpg.destroy_context()