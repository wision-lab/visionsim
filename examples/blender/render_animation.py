from functools import partial
from pathlib import Path

from rich.progress import Progress

from visionsim.simulate.blender import BlenderClient

with BlenderClient.spawn(timeout=30) as client, Progress() as progress:
    client.initialize(Path("assets/monkey.blend").resolve(), Path("renders/monkey").resolve())
    task = progress.add_task("Rendering monkey.blend...")
    client.render_animation(update_fn=partial(progress.update, task))
