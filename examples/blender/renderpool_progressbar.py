from pathlib import Path

from visionsim.simulate.blender import BlenderClients
from visionsim.utils.progress import PoolProgress


def render(client, blend_file, tick):
    root = Path("renders") / Path(blend_file).stem
    client.initialize(blend_file, root)
    client.set_resolution((512, 512))
    client.move_keyframes(scale=0.5)
    client.render_animation(update_fn=tick)


if __name__ == "__main__":
    with (
        BlenderClients.pool(2, log_dir="logs", timeout=30) as pool,
        PoolProgress() as progress,
    ):
        for blend_file in ["assets/monkey.blend", "assets/cube.blend", "assets/metaballs.blend"]:
            tick = progress.add_task(f"Rendering {blend_file}...")

            # Note: The client will be automagically passed to `render` here.
            pool.apply_async(render, args=(blend_file, tick))
        progress.wait()
        pool.close()
        pool.join()
