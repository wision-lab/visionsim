from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visionsim.dataset import Metadata
from visionsim.emulate.imu import emulate_imu, imu_integration
from visionsim.interpolate.pose import pose_interp
from visionsim.utils.pose import tform_camcoord_gl2bl

""" Coordinate convention for both the 3D world and the image coordinate system:
        +X: right
        +Z: up
        +Y: out of the image plane
"""
gravity_vector = np.array([0, 0, -9.8])  # m/s^2

db_path = Path(__file__).parent / "transforms_800Hz.db"
if not db_path.exists():
    raise RuntimeError(f"transforms file {db_path} does not exist!")

dataset = Metadata.from_path(db_path)
true_poses_orig = np.array(dataset.poses)
true_poses = np.array([tform_camcoord_gl2bl(T) for T in dataset.poses])
init_pose = true_poses[0]

dt = 1.0 / (dataset.fps * dataset.keyframe_scale)
print(f"#poses: {len(true_poses)}, dt: {dt}")
times = np.arange(len(true_poses)) * dt
pose_spline = pose_interp(true_poses, times)
_, vel_w = pose_spline(times, order=1)
init_vel_c = true_poses[0][:3, :3].T @ vel_w[0]
print(f"initial velocity in camera frame: {str(init_vel_c)}")

num_trajectory_samples = 10
acc_data, gyro_data = [], []
noise_mult = 1
std_acc = 8e-3 * noise_mult
std_bias_acc = 5.5e-5 * noise_mult
std_gyro = 1.2e-3 * noise_mult
std_bias_gyro = 2e-5 * noise_mult
integrated_trajectories = []
for _ in range(num_trajectory_samples):
    imu_data = list(
        emulate_imu(
            true_poses_orig,
            dt=dt,
            gravity=gravity_vector,
            std_acc=std_acc,
            std_bias_acc=std_bias_acc,
            std_gyro=std_gyro,
            std_bias_gyro=std_bias_gyro,
        )
    )
    acc_data.append([d["acc_reading"] for d in imu_data])
    gyro_data.append([d["gyro_reading"] for d in imu_data])
    integrated_trajectories.append(
        np.array(
            list(
                imu_integration(
                    acc_pos=acc_data[-1],
                    vel_ang=gyro_data[-1],
                    pose_init=init_pose,
                    vel_init=init_vel_c,
                    gravity=gravity_vector,
                    dt=dt,
                )
            )
        )
    )

# Plot the true and estimated trajectories (on XY plane)
ax = plt.subplot(1, 1, 1)
ax.plot(*true_poses[:, :2, -1].T, color="k", label="Ground Truth Trajectory")
for tr in integrated_trajectories:
    ax.plot(*tr[:, :2, -1].T)
ax.legend()
plt.savefig("integrated_trajectories.pdf")
