"""IMU trajectory preview figure: ``docs/source/sections/sensors/imu.rst``.

The overview is rendered by ``examples/blender/imu_preview.py``: it integrates a set
of trajectories from emulated IMU readings, draws each one as a frustum-and-trail
marker inside the ultramodern scene alongside the ground truth, and renders the result
from the scene's own overview camera. The room is rendered once and the markers are
rendered over that still, so the figure only costs the scene one render.

The page also illustrates the markers with an interactive graph of their geometry node
tree, which it loads straight from the asset the driver imports into the scene
(``examples/blender/mesh_camera.json``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

from ..._page import CACHE, ROOT, Node, cached, dataset_check, page_task, static, video_recipe

# The driver keys one frame per ground-truth pose, 201 of them at the quick-start's
# 50fps range.  Keeping every other frame plays those four seconds back at 25fps.
FRAME_COUNT = 201
PREVIEW_STEP = 2
# The frames are rendered at this size, and the gif keeps it: the preview is the figure
# both the sensor page and the tutorial embed, and at the default 320px it reads too
# small to tell the trajectories apart.
PREVIEW_SIZE = 512


def _render(executable: str | None, force: bool = False) -> None:
    """Render the overview by running the example driver from the checkout.

    The driver runs in the ordinary interpreter rather than inside Blender, since
    the integration needs scipy, and starts the service subclass in its own Blender
    process.  It reads the scene, the poses and the output directory from its own
    defaults, relative to the working directory below, and takes the Blender binary
    from ``$VSIM_BLENDER`` rather than from the ``--config.executable`` the CLI
    commands use, so ``executable`` is passed on through the environment.
    """
    env = os.environ.copy()
    if executable:
        env["VSIM_BLENDER"] = executable
    cmd = [sys.executable, str(ROOT / "examples" / "blender" / "imu_preview.py")]
    # The driver composites the overlay pass into ``frames/`` in place and skips
    # frames already on disk, so a previous run's output has to go first or it
    # would be composited a second time.
    shutil.rmtree(cached("imu-preview", "frames"), ignore_errors=True)
    subprocess.run(cmd, check=True, cwd=CACHE, env=env)


NODES = (
    Node(
        name="imu-preview",
        files=(cached("imu-preview", "frames"),),
        check=dataset_check(cached("imu-preview", "frames"), FRAME_COUNT),
        requires=("ultramodern-gt",),
        recipe=_render,
    ),
    Node(
        name="imu-preview-video",
        files=(static("sensors/imu-preview.mp4"),),
        is_figure=True,
        requires=("imu-preview",),
        recipe=video_recipe("imu-preview/frames/*/*.png", PREVIEW_STEP, "sensors/imu-preview", size=PREVIEW_SIZE),
    ),
)

build = page_task(NODES, "imu", "Regenerate the IMU trajectory preview figures.")
