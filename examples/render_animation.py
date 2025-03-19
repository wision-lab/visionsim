from functools import partial

from rich.progress import Progress

from spsim.simulate.blender import BlenderClient

with BlenderClient.spawn(timeout=30) as client, Progress() as progress:
    client.initialize("monkey.blend", "renders/monkey")
    task = progress.add_task("Rendering monkey.blend...", total=len(client.animation_range()))
    client.render_animation(update_fn=partial(progress.update, task, advance=1))
