"""Render the IMU trajectory preview embedded in the sensor documentation.

The preview shows the ultramodern ground-truth camera trajectory together with five
trajectories integrated from emulated IMU readings, drawn as frustum-and-trail
markers, seen from the scene's own overview camera.

This script runs in a normal python environment rather than inside blender: the
integration needs scipy, which blender's interpreter does not ship. It computes the
trajectories here, spawns ``examples/blender/imu_preview_service.py`` in blender (as the
``/path/to/imu_preview_service.py:ImuPreviewService`` file target) and hands the
trajectories over RPC.

The room is rendered once, then the markers and their trails are rendered over that
still, which the renderer uses as the world background. Only the markers move, so this
costs one render of the scene instead of one per frame.

:func:`render` reads the ultramodern scene and the quick-start poses from the working
directory, spawns a ``blender`` on ``$PATH``, and writes to ``imu-preview/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from visionsim.dataset import Metadata
from visionsim.emulate.imu import emulate_imu, imu_integration
from visionsim.interpolate.pose import pose_interp
from visionsim.simulate.blender import BlenderClient
from visionsim.utils.pose import camcoord_blender_to_opengl, camcoord_opengl_to_blender

GRAVITY = np.array([0.0, 0.0, -9.8])
"""Gravity vector in the world frame, in m/s^2."""


def integrate(
    poses: np.ndarray,
    dt: float = 0.02,
    count: int = 5,
    seed: int = 0,
    noise_mult: float = 10.0,
) -> list[np.ndarray]:
    """Integrate emulated IMU readings into ``count`` trajectories.

    Every trajectory starts on the ground truth with its true initial velocity, then
    drifts away from it as the emulated noise and bias random walk accumulate. The
    noise is scaled by ``noise_mult`` because the sensor parameters are quoted for
    trajectories minutes long: over the four seconds of the quick-start dataset the
    default drift stays under half a metre, which leaves the estimates lying on top of
    each other. ``examples/sensors/emulate_imu/sim_and_integrate.py`` scales the same
    parameters for the same reason.

    Args:
        poses: ``(N, 4, 4)`` ground-truth poses, sampled every ``dt`` seconds.
        dt: Time between consecutive poses, in seconds.
        count: Number of trajectories to integrate.
        seed: Seed of the first trajectory's noise; the others follow on from it.
        noise_mult: Factor applied to the accelerometer and gyroscope noise.

    Returns:
        One ``(N, 4, 4)`` trajectory per sample, in the same convention as ``poses``.
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


def render() -> None:
    """Compute the trajectories, build the preview and render it."""
    output_dir = Path("imu-preview").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # The stored matrices map the camera frame straight into blender's world axes, which
    # is exactly what a blender camera's ``matrix_world`` expects, so they are used as
    # they are.
    ground_truth = np.array(Metadata.from_path(Path("quickstart/ultramodern-gt/frames/transforms.db").resolve()).poses)
    print(f"Loaded {len(ground_truth)} ground truth poses")
    trajectories = integrate(ground_truth)

    with BlenderClient.spawn(
        service=f"{Path(__file__).resolve().with_name('imu_preview_service.py')}:ImuPreviewService",
        log=output_dir / "logs",
    ) as client:
        client.initialize(Path("ultramodern.blend").resolve(), output_dir)
        # Rendered square, at the size the figure's gif is built for.
        client.set_resolution(height=512, width=512)
        client.cycles_settings(
            device_type="optix",
            use_cpu=True,
            max_samples=128,
            adaptive_threshold=0.05,
            use_denoising=True,
        )
        preview = client.add_trajectories(
            [trajectory.tolist() for trajectory in trajectories],
            ground_truth.tolist(),
            start_frame=1,
        )
        start, stop = preview["frame_range"]
        backdrop = output_dir / "backdrop.png"
        print("Rendering the room once, for the markers to be rendered over")
        client.render_backdrop(str(backdrop))

        print(f"Drew {len(preview['markers'])} markers, rendering frames {start} to {stop}")
        client.render_over_backdrop(str(backdrop))
        client.include_frames()
        client.render_animation()

    print(f"Rendered to {output_dir / 'frames'}")


if __name__ == "__main__":
    render()
