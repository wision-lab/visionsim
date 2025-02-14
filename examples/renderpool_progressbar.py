from pathlib import Path

from spsim.simulate.blender import BlenderClients
from spsim.utils.progress import PoolProgress


def render(client, blend_file, tick):
    root = Path("renders").resolve() / Path(blend_file).stem
    client.initialize(blend_file, root)

    client.set_resolution((512, 512))
    client.move_keyframes(scale=0.5)
    client.render_animation(update_fn=tick)


if __name__ == "__main__":
    with BlenderClients.pool(
        2, log_dir="logs", timeout=30, executable="flatpak run --die-with-parent org.blender.Blender"
    ) as pool, PoolProgress() as progress:
        for blend_file in ["monkey.blend", "cube.blend", "metaballs.blend"]:
            tick = progress.add_task(f"Rendering {blend_file}...")
            pool.apply_async(render, (Path(blend_file).resolve(), tick))
        progress.wait()
