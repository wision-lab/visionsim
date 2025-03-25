from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


# Pose derivatives
# (translational and angular velocity, and translational acceleration)
def egomotion_numdiff(
    T: list[npt.NDArray],
    Dt: float = 1,
) -> Iterator[tuple[npt.NDArray, npt.NDArray, npt.NDArray]]:
    locs = np.array([Tn[:3, 3] for Tn in T])
    vels_tr_w = np.gradient(locs, Dt, axis=0)
    accs_tr_w = np.gradient(vels_tr_w, Dt, axis=0)
    for n in range(len(T)):
        # Angular acceleration is not computed.
        if n == 0:
            rot_c = R.from_matrix(T[n][:3, :3].transpose() @ T[n + 1][:3, :3])
        elif n == len(T) - 1:
            rot_c = R.from_matrix(T[n - 1][:3, :3].transpose() @ T[n][:3, :3])
        else:
            rot_c_x2 = T[n - 1][:3, :3].transpose() @ T[n + 1][:3, :3]
            slerp = Slerp(np.array([0.0, 1.0]), R.from_matrix(np.array([np.eye(3), rot_c_x2])))
            rot_c = slerp(0.5)
        vel_ang_c = rot_c.as_rotvec() * (1.0 / Dt)

        vel_tr_c = T[n][:3, :3].transpose() @ vels_tr_w[n, :]
        acc_tr_c = T[n][:3, :3].transpose() @ accs_tr_w[n, :]

        yield (vel_tr_c, acc_tr_c, vel_ang_c)


""" Forster et al., "IMU Preintegration on Manifold for Efficient Visual-Inertial
                    Maximum-a-Posteriori Estimation", 2015.

    _w: in world coordinate frame
    _c: in camera-centered coordinate frame
"""


def egomotion_int_step_fwd_Euler(
    T_wc: npt.NDArray = np.eye(4),
    vel_tr_c: npt.NDArray = np.zeros((3,)),
    vel_ang_c: npt.NDArray = np.zeros((3,)),
    acc_tr_c: npt.NDArray = np.zeros((3,)),
    acc_ang_c: npt.NDArray = np.zeros((3,)),
    Dt: float = 1,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    R_wc = T_wc[:3, :3]
    vel_tr_w = R_wc @ vel_tr_c
    acc_tr_w = R_wc @ acc_tr_c
    vel_tr_w_next = vel_tr_w + acc_tr_w * Dt
    p_w = T_wc[:3, 3]
    p_w_next = p_w + (vel_tr_w * Dt) + (0.5 * acc_tr_w * (Dt**2))
    p_w_next = p_w_next.reshape((3, 1))

    if np.any(acc_ang_c != 0):
        raise RuntimeError("Angular acceleration handling not implemented!")
    vel_ang_w = R_wc @ vel_ang_c
    vel_ang_w_next = vel_ang_w
    # post-multiply rotates in camera coordinates
    dR_c = R.from_rotvec(vel_ang_c * Dt).as_matrix()
    R_wc_next = R_wc @ dR_c

    T_wc_next = np.vstack((np.hstack((R_wc_next, p_w_next)), np.array([0, 0, 0, 1])))

    vel_tr_c_next = R_wc_next.transpose() @ vel_tr_w_next
    vel_ang_c_next = R_wc_next.transpose() @ vel_ang_w_next

    return (T_wc_next, vel_tr_w_next, vel_ang_w_next, vel_tr_c_next, vel_ang_c_next)


"""
Follows the Appendix in Crassidis (2006), "Sigma-Point Kalman Filtering for Integrated GPS and
Inertial Navigation".
Also see Sec. IV.B. in Leutenegger et al. (2015), 
    "Keyframe-based visual-inertial odometry using nonlinear optimization", 
and Sec. IV in Forster et al. (2015), 
    "IMU Preintegration on Manifold for Efficient Visual-Inertial 
    Maximum-a-Posteriori Estimation".

The default parameter values are taken from Table I in Leutenegger et al.
"""


def sim_IMU(
    T_wc_seq: list[npt.NDArray],
    Dt: float = 1.0 / 800,  # seconds
    std_bias_acc: float = 5.5e-5,  # m/(s^3 \sqrt{Hz})
    std_acc: float = 8e-3,  # m/(s^2 \sqrt{Hz})
    std_bias_gyro: float = 2e-5,  # rad/(s^2 \sqrt{Hz})
    std_gyro: float = 1.2e-3,  # rad/(s \sqrt{Hz})
    grav_w: npt.NDArray = np.array([0, 0, -9.8]),  # m/(s^2)
    init_bias_acc: npt.NDArray = np.array([0, 0, 0]),
    init_bias_gyro: npt.NDArray = np.array([0, 0, 0]),
    rng: Any = np.random.default_rng(),
) -> Iterator[dict[str, npt.ArrayLike]]:
    # discrete-time noise from continuous-time process parameters (Crassidis)
    std_acc_discrete = std_acc / (Dt**0.5)
    std_gyro_discrete = std_gyro / (Dt**0.5)
    std_bias_acc_discrete = std_bias_acc * (Dt**0.5)
    std_bias_gyro_discrete = std_bias_gyro * (Dt**0.5)
    bias_acc = init_bias_acc
    bias_gyro = init_bias_gyro

    t = 0.0
    for T_wc, (v_tr_c, a_tr_c, v_ang_c) in zip(T_wc_seq, egomotion_numdiff(T_wc_seq, Dt)):
        a_tr_w = (T_wc[:3, :3] @ a_tr_c).flatten()
        # IMU is assumed collocated with the camera
        a_tr_IMU = T_wc[:3, :3].transpose() @ (a_tr_w - grav_w)

        # Eq. 109a and 109b in Crassidis
        bias_acc_next = bias_acc + std_bias_acc_discrete * rng.standard_normal((3,))
        sim_a_tr = (
            a_tr_IMU
            + 0.5 * (bias_acc + bias_acc_next)
            + ((((std_acc_discrete**2) + (1 / 12) * (std_bias_acc_discrete**2)) ** 0.5) * rng.standard_normal((3,)))
        )

        bias_gyro_next = bias_gyro + std_bias_gyro_discrete * rng.standard_normal((3,))
        sim_v_ang = (
            v_ang_c
            + 0.5 * (bias_gyro + bias_gyro_next)
            + ((((std_gyro_discrete**2) + (1 / 12) * (std_bias_gyro_discrete**2)) ** 0.5) * rng.standard_normal((3,)))
        )

        data = {"acc_reading": sim_a_tr, "gyro_reading": sim_v_ang, "acc_bias": bias_acc, "gyro_bias": bias_gyro, "t": t}
        yield data

        bias_acc = bias_acc_next
        bias_gyro = bias_gyro_next
        t = t + Dt


def egomotion_int_IMUdata_fwd_Euler(
    acc_IMU: Iterable[npt.NDArray],
    vel_ang_gyro: Iterable[npt.NDArray],
    Dt: float = 1.0 / 800,
    grav_w: npt.NDArray = np.array([0, 0, -9.8]),  # m/(s^2)
    T_wc_init: npt.NDArray = np.eye(4),
    vel_tr_c_init: npt.NDArray = np.zeros((3,)),
) -> Iterator[tuple[npt.NDArray, npt.NDArray, npt.NDArray]]:
    T_wc = T_wc_init
    vel_tr_c = vel_tr_c_init
    for a, vel_ang_c in zip(acc_IMU, vel_ang_gyro):
        yield T_wc, vel_tr_c, vel_ang_c

        acc_tr_c = a + (T_wc[:3, :3].transpose() @ grav_w)
        r = egomotion_int_step_fwd_Euler(T_wc=T_wc, vel_tr_c=vel_tr_c, vel_ang_c=vel_ang_c, acc_tr_c=acc_tr_c, Dt=Dt)
        T_wc, vel_tr_c = r[0], r[3]
