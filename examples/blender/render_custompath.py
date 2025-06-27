from functools import partial
from pathlib import Path

import numpy as np
from rich.progress import Progress

from visionsim.simulate.blender import BlenderClient

with BlenderClient.spawn(timeout=30) as client, Progress() as progress:
    client.initialize(Path("assets/monkey.blend").resolve(), Path("renders/monkey").resolve())

    # Unbind camera from any parents, otherwise position will be relative to parent
    client.unbind_camera()

    # Define camera trajectory as keyframes
    for frame, theta in enumerate(np.linspace(0, 2 * np.pi, 100, endpoint=False)):
        client.position_camera(location=[5 * np.cos(theta), 5 * np.sin(theta), 1], look_at=[0, 0, 0])
        client.rotate_camera(2 * np.pi / 100 * frame)
        client.set_camera_keyframe(frame)
    client.set_animation_range(start=0, stop=100)

    task = progress.add_task("Rendering monkey.blend...")
    client.render_animation(update_fn=partial(progress.update, task))
