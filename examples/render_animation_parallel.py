from functools import partial

from rich.progress import Progress

from spsim.simulate.blender import BlenderClients

with BlenderClients.spawn(jobs=2, timeout=30) as clients, Progress() as progress:
    clients.initialize("monkey.blend", "renders/monkey")
    task = progress.add_task("Rendering monkey.blend...", total=len(clients.common_animation_range()))
    clients.render_animation(update_fn=partial(progress.update, task, advance=1))
