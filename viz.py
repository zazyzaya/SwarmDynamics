from argparse import ArgumentParser

import dearpygui.dearpygui as dpg
import torch

from src.cpu.env import EnvCPU as Env
from src.dna import GenePool, MAX_GAME_LEN
from src.generators import generate_random_columns
from src.phys_globals import CEILING
from src.viz_util import SIZE, _3d_columns, get_triangles_3d as get_triangles, project_topdown_single, painters_alg

N_OBSTACLES = 15

ap = ArgumentParser()
ap.add_argument('--self-play', action='store_true')
ap.add_argument('--genes', default='baseline')
args = ap.parse_args()

if args.genes == 'baseline':
    gp = GenePool(100, use_baseline=True)
else:
    gp = GenePool.load(f'genes/{args.genes}.pt', device='cpu')

default = GenePool(100, use_baseline=True)

if args.self_play:
    bg, bs = gp.create_swarm(100, torch.randint(0,gp.population, (1,)))
    rg, rs = gp.create_swarm(100, torch.randint(0,gp.population, (1,)))
else:
    bg, bs = gp.create_swarm(100, torch.randint(0,gp.population, (1,)))
    rg, rs = default.create_swarm(100, torch.tensor([0]))

obs_pos, obs_z, obs_r = generate_random_columns(N_OBSTACLES, 'cpu')
env = Env(
    bg.squeeze(0), bs.squeeze(0),
    rg.squeeze(0), rs.squeeze(0),
    obstacles=(obs_pos, obs_z, obs_r)
)

dpg.create_context()
dpg.create_viewport(title='PyTorch + Dear PyGui', width=SIZE, height=SIZE)
dpg.setup_dearpygui()

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

    # Get draw instructions
    col_f, col_c = _3d_columns(obs_pos, obs_z, obs_r)
    b_f, b_c = get_triangles(env.blue, (0,255,255))
    r_f, r_c = get_triangles(env.red,(255,0,0))

    # Draw in depth order for 3d effect
    painters_alg(
        col_f + b_f + r_f,
        col_c + b_c + r_c
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
                active_explosions.remove(exp) # Animation finished

    # Track what step we're at
    dpg.draw_text(
            pos=(10, 10),
            text=f"Step: {steps:04d}", size=20,
            color=(255, 255, 255, 255),
            parent='main_drawlist'
        )

    if steps >= MAX_GAME_LEN:
        game_over = torch.tensor([1,1])

    if not game_over.any():
        # Update physics
        game_over, new_explosions, b_col, r_col = env.update(viz=True)
        steps += 1

        # Add drones shot down
        if new_explosions is not None and len(new_explosions) > 0:
            screen_x, screen_y = project_topdown_single(
                new_explosions[:, 0],
                new_explosions[:, 1],
                new_explosions[:, 2],
                SIZE, SIZE
            )

            # 2. Stack them back together and convert to a Python list
            pixel_exps = torch.stack([screen_x, screen_y], dim=-1).detach().tolist()

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
            screen_x, screen_y = project_topdown_single(
                b_col[:, 0],
                b_col[:, 1],
                b_col[:, 2],
                SIZE, SIZE
            )

            # 2. Stack them back together and convert to a Python list
            pixel_exps = torch.stack([screen_x, screen_y], dim=-1).detach().tolist()

            for p in pixel_exps:
                active_explosions.append({
                    'pos': p,
                    'radius': 1.0,
                    'expanding': True,
                    'color': (0, 100, 255, 255),
                    'fill': (0, 50, 255, 150)
                })
        if r_col is not None and len(r_col):
            screen_x, screen_y = project_topdown_single(
                r_col[:, 0],
                r_col[:, 1],
                r_col[:, 2],
                SIZE, SIZE
            )

            # 2. Stack them back together and convert to a Python list
            pixel_exps = torch.stack([screen_x, screen_y], dim=-1).detach().tolist()

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
            b_s, b_k, r_s, r_k = env.get_stats(top_k=10)

            print("Harshest Killers:")
            print("Blue")
            print(b_k)
            print("Red")
            print(r_k)


    # 3. Render the frame
    dpg.render_dearpygui_frame()


dpg.destroy_context()