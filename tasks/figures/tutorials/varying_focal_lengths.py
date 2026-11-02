"""Focal length figures: ``docs/source/tutorials/varying-focal-lengths.rst``.

``contrazoom.blend`` animates the lens from 200 mm to 35 mm over frames 0 through
49 at 30 fps, counter-zooming while the camera dollies toward the subject. Only that
range is rendered, and the preview plays it back at half the render's rate so the
zoom reads at a comfortable pace.
"""

from __future__ import annotations

from .._page import Node, _command_recipe, cached, dataset_check, page_task, static, video_recipe

# The scene renders at 30 fps; the lens animation spans frames 0..49, so 50 frames.
SOURCE_FPS = 30
FRAME_COUNT = 50
# Keep every frame and play at half the render's rate: the zoom crosses the whole room
# in under two seconds at full speed.
PREVIEW_STEP = 1
PREVIEW_FPS = SOURCE_FPS // 2


NODES = (
    Node(
        name="contrazoom",
        files=(cached("contrazoom", "frames"),),
        check=dataset_check(cached("contrazoom", "frames"), FRAME_COUNT),
        recipe=_command_recipe(
            "contrazoom",
            "visionsim blender.render-animation contrazoom.blend contrazoom/"
            " --frame-start=0 --frame-end=49{executable}{force}",
        ),
    ),
    Node(
        name="contrazoom-preview",
        files=(static("varying-focal-lengths/contrazoom-preview.mp4"),),
        is_figure=True,
        requires=("contrazoom",),
        recipe=video_recipe(
            "contrazoom/frames/*/*.png", PREVIEW_STEP, "varying-focal-lengths/contrazoom-preview", fps=PREVIEW_FPS
        ),
    ),
)

build = page_task(NODES, "varying-focal-lengths", "Regenerate the contrazoom preview figure.")
