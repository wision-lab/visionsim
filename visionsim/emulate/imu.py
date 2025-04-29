from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

from visionsim.interpolate.pose import pose_interp


def imu_integration(
    acc_pos: Iterable[npt.ArrayLike],
    vel_ang: Iterable[npt.ArrayLike],
    dt: float,
    gravity: npt.NDArray | None = None,
    pose_init: npt.NDArray | None = None,
    vel_init: npt.NDArray | None = None,
) -> Iterator[npt.NDArray]:
    """Integrate IMU measurements and estimate trajectory using forward Euler integration [1]_.

    Args:
        acc_pos (Iterable[npt.ArrayLike]): Positional acceleration as measured by the IMU. Expects an iterable
            of [ax, ay, az] vectors in m/s^2 (in camera-coordinates).
        vel_ang (Iterable[npt.ArrayLike]): Rotational velocity as measured by the gyro. Expects an iterable
            of [wx, wy, wz] vectors in Rad/s (in camera-coordinates).
        dt (float): Sampling period in seconds. Typically equal to 1/fps.
        gravity (npt.NDArray, optional): Gravity vector in m/s^2 (in world-coordinates). Defaults to -9.8 m/s^2 in Z.
        pose_init (npt.NDArray, optional): Initial pose. Defaults to identity.
        vel_init (npt.NDArray, optional): Initial positional velocity. Defaults to the zero vector.

    Yields:
        Iterator[npt.NDArray]: Estimated pose

    References:
        .. [1] `Forster et al. (2015), "IMU Preintegration on Manifold for Efficient Visual-Inertial
                Maximum-a-Posteriori Estimation". <https://www.roboticsproceedings.org/rss11/p06.pdf>`_
    """
    pose = np.eye(4) if pose_init is None else np.array(pose_init)
    vel_pos = np.zeros((3,)) if vel_init is None else np.array(vel_init)
    gravity = np.array([0, 0, -9.8]) if gravity is None else np.array(gravity)

    for ap, va in zip(acc_pos, vel_ang):
        yield pose

        acc_pos_c = np.array(ap) + (pose[:3, :3].T @ gravity)
        pose, vel_pos = imu_integration_step(pose=pose, vel_pos=vel_pos, vel_ang=np.array(va), acc_pos=acc_pos_c, dt=dt)


def imu_integration_step(
    pose: npt.NDArray,
    vel_pos: npt.NDArray,
    vel_ang: npt.NDArray,
    acc_pos: npt.NDArray,
    dt: float,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Computes single Euler integration step [1]_.

    While the integration is performed in world coordinates, this helper
    operates on velocities and accelerations in camera coordinates and perform
    the coordinate change internally. Poses remain in world coordinates.

    Note:
        Angular acceleration handling is not implemented!

    Args:
        pose (npt.NDArray): Current camera pose in world coordinates.
        vel_pos_c (npt.NDArray): Translational velocity in camera coordinates.
        vel_ang_c (npt.NDArray): Angular velocity in camera coordinates.
        acc_pos_c (npt.NDArray): Translational acceleration in camera coordinates.
        dt (float): Sampling period in seconds.

    Returns:
        tuple[npt.NDArray, npt.NDArray]:
            Camera pose at next time step (in world-coords),
            Translational velocity at next time step (in camera coords)
    """
    # Note: Here we use `_w` for world coordinate frame and `_c` for camera-centered coordinate frame
    # Extract rotation and positional components from pose
    R_wc = pose[:3, :3]
    p_w = pose[:3, 3]

    # Convert position velocity and acceleration from camera coordinates to world
    vel_pos_w = R_wc @ vel_pos
    acc_pos_w = R_wc @ acc_pos

    # Apply Euler integration step (Eq. 23 in Forster et al.)
    # post-multiply by dR_c rotates in camera coordinates
    dR_c = R.from_rotvec(vel_ang * dt).as_matrix()
    R_wc_next = R_wc @ dR_c
    vel_pos_w_next = vel_pos_w + acc_pos_w * dt
    p_w_next = p_w + (vel_pos_w * dt) + (0.5 * acc_pos_w * (dt**2))

    # Re-assemble pose from new rot/pos, map position velocity back to camera-coords
    bottom = np.array([0.0, 0.0, 0.0, 1.0])
    T_wc_next = np.vstack((np.hstack((R_wc_next, p_w_next[..., None])), bottom))
    vel_pos_c_next = R_wc_next.T @ vel_pos_w_next

    return T_wc_next, vel_pos_c_next


def emulate_imu(
    poses: list[npt.NDArray] | npt.NDArray,
    *,
    dt: float = 1 / 800,
    std_acc: float = 8e-3,
    std_gyro: float = 1.2e-3,
    std_bias_acc: float = 5.5e-5,
    std_bias_gyro: float = 2e-5,
    init_bias_acc: npt.NDArray | None = None,
    init_bias_gyro: npt.NDArray | None = None,
    gravity: npt.NDArray | None = None,
    rng: np.random.Generator | None = None,
) -> Iterator[dict[str, npt.ArrayLike]]:
    """Emulate IMU measurements from a sequence of ground-truth poses.

    Follows the Appendix in Crassidis (2006) [2]_, also see Sec. IV.B. in
    Leutenegger et al. (2015) [3]_, and Sec. IV in Forster et al. (2015) [1]_.

    The default parameter values are taken from Table I in Leutenegger et al [3]_.

    Args:
        poses (list[npt.NDArray] | npt.NDArray): Sequence of ground-truth poses to emulate IMU from.
        dt (float, optional): Sampling period in seconds. Defaults to 1/800.
        std_acc (float, optional): Standard deviation for positional acceleration in m/(s^2 sqrt(Hz)). Defaults to 8e-3.
        std_gyro (float, optional): Standard deviation for angular velocity in rad/(s sqrt(Hz)). Defaults to 1.2e-3.
        std_bias_acc (float, optional): Bias for positional acceleration in m/(s^3 sqrt(Hz)). Defaults to 5.5e-5.
        std_bias_gyro (float, optional): Bias for angular velocity in rad/(s^2 sqrt(Hz)). Defaults to 2e-5.
        init_bias_acc (npt.NDArray, optional): Initial positional acceleration. Defaults to the zero vector.
        init_bias_gyro (npt.NDArray, optional): Initial angular velocity. Defaults to the zero vector.
        gravity (npt.NDArray, optional): Gravity vector in m/s^2 (in world-coordinates). Defaults to -9.8 m/s^2 in Z.
        rng (np.random.Generator, optional): Random generator instance. Defaults to ``np.random.default_rng``.

    Yields:
        Iterator[dict[str, npt.ArrayLike]]: Return "acc_reading", "gyro_reading", "acc_bias", "gyro_bias", and "t".

    References:
        .. [2] `Crassidis (2006), "Sigma-Point Kalman Filtering for Integrated GPS and
                Inertial Navigation". <https://www.acsu.buffalo.edu/~johnc/gpsins_gnc05.pdf>`_
        .. [3] `Leutenegger et al. (2015), "Keyframe-based visual-inertial odometry using nonlinear
                optimization". <https://www.roboticsproceedings.org/rss09/p37.pdf>`_
    """
    gravity = np.array([0, 0, -9.8]) if gravity is None else np.array(gravity)
    init_bias_acc = np.array([0, 0, 0]) if init_bias_acc is None else np.array(init_bias_acc)
    init_bias_gyro = np.array([0, 0, 0]) if init_bias_gyro is None else np.array(init_bias_gyro)
    rng = np.random.default_rng() if rng is None else rng

    # discrete-time noise from continuous-time process parameters (Crassidis)
    std_acc_discrete = std_acc / (dt**0.5)
    std_gyro_discrete = std_gyro / (dt**0.5)
    std_bias_acc_discrete = std_bias_acc * (dt**0.5)
    std_bias_gyro_discrete = std_bias_gyro * (dt**0.5)

    # get angular velocity (in world coords) and positional acceleration (in camera space)
    times = np.arange(len(poses)) * dt
    pose_spline = pose_interp(poses, times)
    vel_ang_w, _ = pose_spline(times, order=1)
    _, acc_pos_w = pose_spline(times, order=2)
    acc_pos_c = np.array([T_wc[:3, :3].T @ a for T_wc, a in zip(poses, acc_pos_w)])

    # loop initialization
    bias_acc = init_bias_acc
    bias_gyro = init_bias_gyro
    t = 0.0

    for T_wc, a_pos_c, v_ang_w in zip(poses, acc_pos_c, vel_ang_w):
        a_pos_w = (T_wc[:3, :3] @ a_pos_c).flatten()
        # IMU is assumed collocated with the camera
        a_pos_IMU = T_wc[:3, :3].transpose() @ (a_pos_w - gravity)

        # Eq. 109a and 109b in Crassidis
        bias_acc_next = bias_acc + std_bias_acc_discrete * rng.standard_normal((3,))
        sim_a_pos = (
            a_pos_IMU
            + 0.5 * (bias_acc + bias_acc_next)
            + ((((std_acc_discrete**2) + (1 / 12) * (std_bias_acc_discrete**2)) ** 0.5) * rng.standard_normal((3,)))
        )

        bias_gyro_next = bias_gyro + std_bias_gyro_discrete * rng.standard_normal((3,))
        sim_v_ang = (
            v_ang_w
            + 0.5 * (bias_gyro + bias_gyro_next)
            + ((((std_gyro_discrete**2) + (1 / 12) * (std_bias_gyro_discrete**2)) ** 0.5) * rng.standard_normal((3,)))
        )

        data = {
            "acc_reading": sim_a_pos,
            "gyro_reading": sim_v_ang,
            "acc_bias": bias_acc,
            "gyro_bias": bias_gyro,
            "t": t,
        }
        yield data

        bias_acc = bias_acc_next
        bias_gyro = bias_gyro_next
        t = t + dt
