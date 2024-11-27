# IMU.py

import numpy as np
import numpy.typing as npt
from typing import Any
from collections.abc import Iterable, Iterator

from .numdiff import egomotion_numdiff, egomotion_int_step_fwd_Euler

"""
Follows the Appendix in Crassidis (2006), "Sigma-Point Kalman Filtering for Integrated GPS and
Inertial Navigation".
Also see Sec. IV.B. in Leutenegger et al. (2015), 
    "Keyframe-based visual–inertial odometry using nonlinear optimization", 
and Sec. IV in Forster et al. (2015), 
    "IMU Preintegration on Manifold for Efficient Visual-Inertial 
    Maximum-a-Posteriori Estimation".

The default parameter values are taken from Table I in Leutenegger et al.
"""
def sim_IMU(T_wc_seq:       list[npt.NDArray[float]],
            Dt:             float = 1.0/800,    # seconds
            std_bias_acc:   float = 5.5e-5,     # m/(s^3 \sqrt{Hz})
            std_acc:        float = 8e-3,       # m/(s^2 \sqrt{Hz})
            std_bias_gyro:  float = 2e-5,       # rad/(s^2 \sqrt{Hz})
            std_gyro:       float = 1.2e-3,     # rad/(s \sqrt{Hz})
            grav_w:         npt.NDArray[float] = np.array([0,0,0]), # m/(s^2)
            init_bias_acc:  npt.NDArray[float] = np.array([0,0,0]),
            init_bias_gyro: npt.NDArray[float] = np.array([0,0,0]),
            rng:            Any = None) \
                    -> Iterator[dict[str, npt.ArrayLike]]:
    if rng is None:
        rng = np.random.default_rng()

    vel_tr_c, acc_tr_c, vel_ang_c = egomotion_numdiff(T_wc_seq, Dt)

    # discrete-time noise from continuous-time process parameters (Crassidis)
    std_acc_discrete = std_acc / (Dt**0.5)
    std_gyro_discrete = std_gyro / (Dt**0.5)
    std_bias_acc_discrete = std_bias_acc * (Dt**0.5)
    std_bias_gyro_discrete = std_bias_gyro * (Dt**0.5)
    bias_acc = init_bias_acc
    bias_gyro = init_bias_gyro

    t = 0
    for T_wc, a_tr_c, v_ang_c in zip(T_wc_seq, acc_tr_c, vel_ang_c):
        a_tr_w = (T_wc[:3,:3] @ a_tr_c).flatten()
        # IMU is assumed collocated with the camera
        a_tr_IMU = T_wc[:3,:3].transpose() @ (a_tr_w - grav_w)

        # Eq. 109a and 109b in Crassidis
        bias_acc_next = bias_acc + std_bias_acc_discrete*rng.standard_normal((3,))
        sim_a_tr = (a_tr_IMU + 0.5*(bias_acc + bias_acc_next)
                        + ((((std_acc_discrete**2) 
                                + (1/12)*(std_bias_acc_discrete**2))**0.5)
                            *rng.standard_normal((3,))))

        bias_gyro_next = bias_gyro + std_bias_gyro_discrete*rng.standard_normal((3,))
        sim_v_ang = (v_ang_c + 0.5*(bias_gyro + bias_gyro_next)
                        + ((((std_gyro_discrete**2) 
                                + (1/12)*(std_bias_gyro_discrete**2))**0.5)
                            *rng.standard_normal((3,))))

        data = {"acc_tr_c": sim_a_tr, "vel_ang_c": sim_v_ang,
                "bias_acc": bias_acc, "bias_gyro": bias_gyro,
                "t": t}
        yield data

        bias_acc = bias_acc_next
        bias_gyro = bias_gyro_next
        t = t + Dt
# sim_IMU(...)

def egomotion_int_IMUdata_fwd_Euler(
        acc_IMU:        Iterable[npt.NDArray[float]],
        vel_ang_gyro:   Iterable[npt.NDArray[float]],
        Dt:             float = 1.0/800,
        grav_w:         npt.NDArray[float] = np.array([0,0,0]), # m/(s^2)
        T_wc_init:      npt.NDArray[float] = np.eye(4),
        vel_tr_c_init:  npt.NDArray[float] = np.zeros((3,))) \
                -> Iterator[tuple[npt.NDArray[float], npt.NDArray[float]]]:
    T_wc = T_wc_init
    vel_tr_c = vel_tr_c_init
    for a, vel_ang_c in zip(acc_IMU, vel_ang_gyro):
        yield T_wc, vel_tr_c, vel_ang_c

        acc_tr_c = a + (T_wc[:3,:3].transpose() @ grav_w)
        r = egomotion_int_step_fwd_Euler(T_wc=T_wc, 
                                        vel_tr_c=vel_tr_c,
                                        vel_ang_c=vel_ang_c, 
                                        acc_tr_c=acc_tr_c, 
                                        Dt=Dt)
        T_wc, vel_tr_c = r[0], r[3]
# egomotion_int_IMUdata_fwd_Euler(...)

