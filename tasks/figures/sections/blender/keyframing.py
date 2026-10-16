"""Keyframing figure: ``docs/source/sections/blender/keyframing.rst``.

The camera work is ``examples/blender/render_custompath.py``, which the page shows: orbit
the origin at the example's radius and height, always looking at it, while turning about
the camera's own optical axis once per revolution. It renders the example's own
``monkey.blend``, so the figure is what that script produces.
"""

from __future__ import annotations

import os

import numpy as np

from ..._page import CACHE, Node, cached, dataset_check, page_task, static, video_recipe

# The example's trajectory: orbit at this radius and height, one turn per revolution.
RADIUS = 5.0
HEIGHT = 1.0
FRAMES = 100
# The page shows this clip at 80% of the content column, which is wider than the
# scene's own 512px, so render above that to keep it sharp when it is enlarged.
RENDER_SIZE = 1024


def _render(executable: str | None, force: bool = False) -> None:
    """Keyframe the camera around the scene's origin, then render the pass.

    Args:
        executable: Blender to render with, ``None`` for the one on ``$PATH``.
        force: Ignored; the render overwrites unconditionally.
    """
    root = cached("keyframing")
    from visionsim.simulate.blender import BlenderClient

    with BlenderClient.spawn(
        executable=executable or os.environ.get("VSIM_BLENDER"),
        log=CACHE / "logs",
        factory_startup=True,
    ) as client:
        client.initialize(cached("monkey.blend"), root)
        client.set_resolution(height=RENDER_SIZE, width=RENDER_SIZE)
        client.include_frames()

        # Unbinds the camera from what the scene had it on, its own keyframes included:
        # every scene in the cache animates its camera, and those f-curves would
        # otherwise keep driving it over the keyframes set below.
        client.unbind_camera()
        for frame, theta in enumerate(np.linspace(0, 2 * np.pi, FRAMES, endpoint=False)):
            client.position_camera(
                location=[RADIUS * np.cos(theta), RADIUS * np.sin(theta), HEIGHT],
                look_at=[0, 0, 0],
            )
            client.rotate_camera(2 * np.pi / FRAMES * frame)
            client.set_camera_keyframe(frame)
        client.set_animation_range(start=0, stop=FRAMES)

        client.render_animation()


NODES = (
    Node(
        name="keyframing-frames",
        files=(cached("keyframing", "frames"),),
        check=dataset_check(cached("keyframing", "frames"), FRAMES),
        recipe=_render,
    ),
    Node(
        name="keyframing-preview",
        files=(static("blender/keyframing-api.mp4"),),
        is_figure=True,
        requires=("keyframing-frames",),
        recipe=video_recipe("keyframing/frames/*/*.png", 1, "blender/keyframing-api", size=RENDER_SIZE),
    ),
)

build = page_task(NODES, "keyframing", "Regenerate the camera keyframing figure.")
