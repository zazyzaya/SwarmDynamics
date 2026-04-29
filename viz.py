import dearpygui.dearpygui as dpg

from sim import Env

df = Env(100, 100)

dpg.create_context()
dpg.create_viewport(title='PyTorch + Dear PyGui', width=600, height=600)
dpg.setup_dearpygui()

df = Env(100, 100)

# Define the UI structure ONCE outside the loop
with dpg.window(label="Pixel Renderer", width=600, height=600):
    # We give the drawlist a 'tag' so we can refer to it later
    with dpg.drawlist(width=500, height=500, tag="main_drawlist"):
        pass

dpg.show_viewport()

# The manual render loop
while dpg.is_dearpygui_running():
    # 1. Clear the previous frame's pixels
    dpg.delete_item("main_drawlist", children_only=True)

    blue_pixels = (df.blue.s[df.blue.alive, :2] * 500 + 250).detach().tolist()
    red_pixels = (df.red.s[df.red.alive, :2] * 500 + 250).detach().tolist()

    # 2. Draw the new positions
    for p in blue_pixels:
        dpg.draw_circle(center=p, radius=1.5, color=(0, 0, 255, 255), parent="main_drawlist")

    for p in red_pixels:
        dpg.draw_circle(center=p, radius=1.5, color=(255, 0, 0, 255), parent="main_drawlist")

    # 3. Update physics and render
    df.update()
    dpg.render_dearpygui_frame()

dpg.destroy_context()