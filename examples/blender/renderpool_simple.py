from pathlib import Path

from visionsim.simulate.blender import BlenderClients


def render(client, blend_file):
    root = Path("renders") / Path(blend_file).stem
    client.initialize(blend_file, root)
    client.set_resolution((512, 512))
    client.move_keyframes(scale=0.5)
    client.render_animation()


if __name__ == "__main__":
    with BlenderClients.pool(2, log_dir="logs", timeout=30) as pool:
        # Note: The client will be automagically passed to `render` here.
        pool.map(render, ["assets/monkey.blend", "assets/cube.blend", "assets/metaballs.blend"])
