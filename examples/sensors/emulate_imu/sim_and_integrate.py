import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visionsim.dataset import Metadata
from visionsim.emulate.imu import emulate_imu, imu_integration
from visionsim.interpolate.pose import pose_interp
from visionsim.utils.pose import tform_camcoord_gl2bl

# Coordinate convention for both the 3D world and the image coordinate system:
#   +X: right
#   +Z: up
#   +Y: out of the image plane
GRAVITY = np.array([0, 0, -9.8])  # m/s^2


AXIS_IDX = {"x": 0, "y": 1, "z": 2}
AXIS_LABELS = {0: "X", 1: "Y", 2: "Z"}
VALID_PLANES = ["".join(p) for p in itertools.permutations("xyz", 2)]


def sim_and_integrate(
    db_path: Path,
    start: int = 0,
    end: int | None = None,
    interp_factor: int = 16,
    num_samples: int = 10,
    noise_mult: float = 25.0,
    plot_path: Path | None = Path("integrated_trajectories.pdf"),
    plane: str = "xy",
    save_path: Path | None = None,
):
    # Load dataset and get poses
    dataset = Metadata.from_path(db_path)
    dt = 1.0 / (dataset.fps * dataset.keyframe_scale)
    poses_c = np.array(dataset.poses)[start:end]
    num_poses = len(poses_c)

    # Convert to world coordinate system
    poses_w = np.array([tform_camcoord_gl2bl(T) for T in poses_c])

    # Interpolate poses to increase temporal resolution, the render might be 100fps, but an IMU can be 1000Hz
    interp_dt = dt / interp_factor
    times = np.arange(num_poses) * dt
    interp_times = np.arange(num_poses * interp_factor) * interp_dt
    interp_poses_c = pose_interp(poses_c, times)(interp_times)
    interp_poses_w = pose_interp(poses_w, times)(interp_times)
    _, interp_velocities_w = pose_interp(poses_w, times)(interp_times, order=1)

    # Get initial velocity in camera frame
    init_vel_w = interp_velocities_w[0]
    print(
        f"Found {num_poses} poses, which were interpolated by a factor of {interp_factor} (dt: {dt:.4f} -> {interp_dt:.4f})"
    )
    print(f"Initial velocity in world frame: {init_vel_w}")

    # IMU noise parameters, scaled by noise_mult from defaults for simplicity
    std_acc = 8e-3 * noise_mult
    std_bias_acc = 5.5e-5 * noise_mult
    std_gyro = 1.2e-3 * noise_mult
    std_bias_gyro = 2e-5 * noise_mult
    integrated_trajectories = []

    # Emulate IMU data and integrate trajectories
    for _ in range(num_samples):
        imu_data = list(
            emulate_imu(
                interp_poses_c,  #  <------- Poses in camera frame!!
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
                        pose_init=interp_poses_w[0],  #  <------- Pose in world frame!!
                        vel_init=init_vel_w,  #  <------- Velocity in world frame!!
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
    if save_path is not None:
        np.save(save_path, np.array(trajectories))
        print(f"Trajectories saved to {save_path}")

    # Plot the true and estimated trajectories
    if plot_path is not None:
        i, j = tuple(AXIS_IDX[i] for i in plane.lower())
        xlabel, ylabel = AXIS_LABELS[i], AXIS_LABELS[j]
        gt_positions = poses_w[:, :3, -1]  # use original (down-sampled) ground truth
        fig, ax = plt.subplots(1, 1)
        ax.plot(gt_positions[:, i], gt_positions[:, j], color="k", label="Ground Truth")
        for tr in trajectories:
            positions = tr[:, :3, -1]
            ax.plot(positions[:, i], positions[:, j])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"IMU-Integrated vs. Ground Truth Trajectories ({plane.upper()} Plane)")
        ax.legend()
        ax.set_aspect("equal")
        fig.savefig(plot_path)
        print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate and integrate IMU data.")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path(__file__).parent.parent.parent.parent / "transforms.db",
        help="Path to transforms.db or json",
    )
    parser.add_argument("--start", type=int, default=0, help="Start pose index")
    parser.add_argument("--end", type=int, default=None, help="End pose index (exclusive)")
    parser.add_argument("--interp-factor", type=int, default=16, help="Interpolation factor")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of trajectories to emulate")
    parser.add_argument("--noise-mult", type=float, default=5.0, help="Noise multiplier for all noise parameters")
    parser.add_argument(
        "--plot-path", type=Path, default=Path("integrated_trajectories.pdf"), help="Save plot to this path"
    )
    parser.add_argument(
        "--plane", default="xy", choices=VALID_PLANES, help="Plane to plot trajectories on (e.g. xy, xz, zx)"
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Save down-interpolated trajectories as .npy to this path (omit to skip)",
    )
    args = parser.parse_args()

    sim_and_integrate(
        db_path=args.db_path,
        start=args.start,
        end=args.end,
        interp_factor=args.interp_factor,
        num_samples=args.num_samples,
        noise_mult=args.noise_mult,
        plot_path=args.plot_path,
        plane=args.plane,
        save_path=args.save_path,
    )
