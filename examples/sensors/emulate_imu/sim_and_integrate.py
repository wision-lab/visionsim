from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visionsim.dataset import Metadata
from visionsim.emulate.imu import emulate_imu, imu_integration
from visionsim.interpolate.pose import pose_interp
from visionsim.utils.pose import camcoord_opengl_to_blender

DB_PATH = Path(__file__).parent.parent.parent.parent / "transforms.db"
PLOT_PATH = Path("integrated_trajectories.pdf")
INTERP_FACTOR = 16  # pose spline is sampled at 16x the dataset's rate, since an IMU runs faster than a camera
NUM_SAMPLES = 10
# Sensor noise is quoted for trajectories minutes long, x25 makes the drift visible over the four seconds here
NOISE_MULT = 25.0
GRAVITY = np.array([0, 0, -9.8])  # m/s^2, in the world frame; +Z up, +X right, +Y out of the image plane
PLANE = "xy"  # plane the trajectories are plotted on


def sim_and_integrate(db_path: Path = DB_PATH, plot_path: Path = PLOT_PATH):
    dataset = Metadata.from_path(db_path)
    dt = 1.0 / (dataset.fps * dataset.keyframe_scale)
    poses = np.array(dataset.poses)
    num_poses = len(poses)

    # Convert to Blender's coordinate convention (+Z up) for both world and camera frames
    poses = np.array([camcoord_opengl_to_blender(T) for T in poses])

    # Interpolate poses to increase temporal resolution, the render might be 100fps, but an IMU can be 1000Hz
    interp_dt = dt / INTERP_FACTOR
    times = np.arange(num_poses) * dt
    interp_times = np.arange(num_poses * INTERP_FACTOR) * interp_dt
    interp_poses = pose_interp(poses, times)(interp_times)
    _, interp_velocities_w = pose_interp(poses, times)(interp_times, order=1)

    init_vel_w = interp_velocities_w[0]
    print(
        f"Found {num_poses} poses, which were interpolated by a factor of {INTERP_FACTOR} (dt: {dt:.4f} -> {interp_dt:.4f})"
    )
    print(f"Initial velocity in world frame: {init_vel_w}")

    std_acc = 8e-3 * NOISE_MULT
    std_bias_acc = 5.5e-5 * NOISE_MULT
    std_gyro = 1.2e-3 * NOISE_MULT
    std_bias_gyro = 2e-5 * NOISE_MULT
    integrated_trajectories = []

    # Emulate IMU data and integrate trajectories
    for _ in range(NUM_SAMPLES):
        imu_data = list(
            emulate_imu(
                interp_poses,
                dt=interp_dt,
                gravity=GRAVITY,
                std_acc=std_acc,
                std_bias_acc=std_bias_acc,
                std_gyro=std_gyro,
                std_bias_gyro=std_bias_gyro,
            )
        )

        integrated_trajectories.append(
            np.array(
                list(
                    imu_integration(
                        acc_pos=np.array([d["acc_reading"] for d in imu_data]),
                        vel_ang=np.array([d["gyro_reading"] for d in imu_data]),
                        pose_init=interp_poses[0],
                        vel_init=init_vel_w,  # world frame; imu_integration expresses it in the camera frame
                        gravity=GRAVITY,
                        dt=interp_dt,
                    )
                )
            )
        )

    # Down-interpolate to same number of poses as original
    trajectories = []
    for tr in integrated_trajectories:
        tr_interp = pose_interp(tr, interp_times)(times)
        trajectories.append(tr_interp)

    # Plot the true and estimated trajectories
    i, j = "xyz".index(PLANE[0]), "xyz".index(PLANE[1])
    gt_positions = poses[:, :3, -1]  # use original (down-sampled) ground truth
    fig, ax = plt.subplots(1, 1)
    ax.plot(gt_positions[:, i], gt_positions[:, j], color="k", label="Ground Truth")
    for tr in trajectories:
        positions = tr[:, :3, -1]
        ax.plot(positions[:, i], positions[:, j])
    ax.set_xlabel(PLANE[0].upper())
    ax.set_ylabel(PLANE[1].upper())
    ax.set_title(f"IMU-Integrated vs. Ground Truth Trajectories ({PLANE.upper()} Plane)")
    ax.legend()
    ax.set_aspect("equal")
    fig.savefig(plot_path)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    sim_and_integrate()
