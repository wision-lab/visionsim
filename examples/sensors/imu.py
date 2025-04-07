import json

import matplotlib.pyplot as plt
import numpy as np

from visionsim.emulate.imu import emulate_imu, imu_integration

# Load ground truth poses
with open("transforms.json", "r") as f:
    gt = json.load(f)
    gt_poses = np.array([f["transform_matrix"] for f in gt["frames"]])
    dt = 1 / 50

# Load pre-generated CSV
data = np.loadtxt("imu.csv", skiprows=1, delimiter=",")
acc_reading, gyro_reading = data[:, 1:4], data[:, 4:7]

# Or use the emulation API directly
data = list(emulate_imu(gt_poses, dt=dt))
acc_reading = [d["acc_reading"] for d in data]
gyro_reading = [d["gyro_reading"] for d in data]

# Estimate the trajectory from measurements
estimated = np.array(
    list(
        imu_integration(
            acc_pos=acc_reading,
            vel_ang=gyro_reading,
            pose_init=gt_poses[0],
            dt=dt,
        )
    )
)

# Plot the true and estimated trajectories (on XY plane)
plt.plot(*gt_poses[:, :2, -1].T, label="Ground Truth Trajectory")
plt.plot(*estimated[:, :2, -1].T, label="Estimated Trajectory")
plt.legend()
plt.show()
