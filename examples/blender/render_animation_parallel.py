from functools import partial
from pathlib import Path

from rich.progress import Progress

from visionsim.simulate.blender import BlenderClients

if __name__ == "__main__":
    with BlenderClients.spawn(jobs=2, timeout=30, log="logs") as clients, Progress() as progress:
        clients.initialize(Path("assets/monkey.blend").resolve(), Path("renders/monkey").resolve())
        clients.include_frames()
        task = progress.add_task("Rendering monkey.blend...")
        clients.render_animation(update_fn=partial(progress.update, task))
