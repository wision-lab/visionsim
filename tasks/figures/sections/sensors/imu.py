"""IMU trajectory preview figure: ``docs/source/sections/sensors/imu.rst``.

The preview integrates a set of trajectories from emulated IMU readings, draws each one
as a frustum-and-trail marker inside the ultramodern scene alongside the ground truth,
and renders the result from the scene's own overview camera. The room is rendered once
and the markers are rendered over that still, so the figure only costs the scene one
render.

The integration mirrors ``examples/blender/imu_preview.py`` rather than running or
importing it, so the figure and the example do not entangle. They do share the drawing
service (``examples/blender/imu_preview_service.py``), which this task spawns.

The page also illustrates the markers with an interactive graph of their geometry node
tree, which it loads straight from the asset the service imports into the scene
(``examples/blender/mesh_camera.json``).
"""

from __future__ import annotations

import shutil

import numpy as np

from visionsim.dataset import Metadata
from visionsim.emulate.imu import emulate_imu, imu_integration
from visionsim.interpolate.pose import pose_interp
from visionsim.simulate.blender import BlenderClient
from visionsim.utils.pose import camcoord_blender_to_opengl, camcoord_opengl_to_blender

from ..._page import ROOT, Node, cached, dataset_check, page_task, static, video_recipe

SERVICE = f"{ROOT / 'examples' / 'blender' / 'imu_preview_service.py'}:ImuPreviewService"
"""File target of the example's drawing service, spawned inside blender."""

GRAVITY = np.array([0.0, 0.0, -9.8])
"""Gravity vector in the world frame, in m/s^2."""

# The service keys one frame per ground-truth pose, 201 of them at the quick-start's
# 50fps range.  Keeping every other frame plays those four seconds back at 25fps.
FRAME_COUNT = 201
PREVIEW_STEP = 2
# The frames are rendered at this size, and the gif keeps it: the preview is the figure
# both the sensor page and the tutorial embed, and at the default 320px it reads too
# small to tell the trajectories apart.
PREVIEW_SIZE = 512


def _integrate(
    poses: np.ndarray, dt: float = 0.02, count: int = 5, seed: int = 0, noise_mult: float = 10.0
) -> list[np.ndarray]:
    """Integrate emulated IMU readings into ``count`` trajectories.

    Copied from ``examples/blender/imu_preview.py``; see it for the reasoning behind the
    noise scaling and the frame conversions.
    """
    # The emulator reads the pose matrices as camera-frame quantities, so it is given a
    # copy converted to blender's camera frame; the estimates come back in that frame
    # and are converted back, which is what makes them keyframable.
    emulated = np.array([camcoord_opengl_to_blender(pose) for pose in poses])
    times = np.arange(len(poses)) * dt
    velocity_init = pose_interp(emulated, times)(times, order=1)[1][0]

    noise = {
        "std_acc": 8e-3 * noise_mult,
        "std_bias_acc": 5.5e-5 * noise_mult,
        "std_gyro": 1.2e-3 * noise_mult,
        "std_bias_gyro": 2e-5 * noise_mult,
    }

    trajectories = []
    for i in range(count):
        readings = list(emulate_imu(emulated, dt=dt, gravity=GRAVITY, rng=np.random.default_rng(seed + i), **noise))
        estimate = np.array(
            list(
                imu_integration(
                    acc_pos=np.array([reading["acc_reading"] for reading in readings]),
                    vel_ang=np.array([reading["gyro_reading"] for reading in readings]),
                    pose_init=emulated[0],
                    vel_init=velocity_init,
                    gravity=GRAVITY,
                    dt=dt,
                )
            )
        )
        trajectories.append(np.array([camcoord_blender_to_opengl(pose) for pose in estimate]))

    return trajectories


def _render(executable: str | None, force: bool = False) -> None:
    """Compute the trajectories and render the preview from the scene's overview camera.

    The integration runs in the ordinary interpreter rather than inside blender, since
    it needs scipy, and the service subclass is spawned in its own blender process on
    the ``--executable`` the figures are run with. The scene and poses are read from
    ``cache/``.
    """
    root = cached("imu-preview")
    # The service composites the overlay pass into ``frames/`` in place and skips frames
    # already on disk, so a previous run's output has to go first or it would be
    # composited a second time.
    shutil.rmtree(root / "frames", ignore_errors=True)

    poses = cached("quickstart", "ultramodern-gt", "frames", "transforms.db")
    ground_truth = np.array(Metadata.from_path(poses).poses)
    trajectories = _integrate(ground_truth)

    with BlenderClient.spawn(service=SERVICE, executable=executable, log=root / "logs") as client:
        client.initialize(cached("ultramodern.blend"), root)
        client.set_resolution(height=PREVIEW_SIZE, width=PREVIEW_SIZE)
        client.cycles_settings(
            device_type="optix",
            use_cpu=True,
            max_samples=128,
            adaptive_threshold=0.05,
            use_denoising=True,
        )
        client.add_trajectories(
            [trajectory.tolist() for trajectory in trajectories],
            ground_truth.tolist(),
            start_frame=1,
        )
        backdrop = root / "backdrop.png"
        client.render_backdrop(str(backdrop))
        client.render_over_backdrop(str(backdrop))
        client.include_frames()
        client.render_animation()


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
