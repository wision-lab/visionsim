# IMU.py

import numpy as np
import numpy.typing as npt
from typing import Any
from collections.abc import Iterable, Iterator

# Finite-difference formulas
# https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf
def numdiff_forward(
        f0: npt.NDArray[float],
        f1: npt.NDArray[float],
        f2: npt.NDArray[float],
        f3: npt.NDArray[float],
        Dx: float = 1)  -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    df = (-3*f0 + 4*f1 - f2) * (0.5/Dx)
    d2f = (2*f0 - 5*f1 + 4*f2 - f3) * (1.0/(Dx**2))
    return (df, d2f)
# numdiff_forward(...)

def numdiff_centered(
        fm2:    npt.NDArray[float],
        fm1:    npt.NDArray[float],
        f0:     npt.NDArray[float],
        fp1:    npt.NDArray[float],
        fp2:    npt.NDArray[float],
        Dx:     float = 1) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    df = (-fp2 + 8*fp1 - 8*fm1 + fm2) * (1.0/(12*Dx))
    d2f = (-fp2 + 16*fp1 - 30*f0 + 16*fm1 - fm2) * (1.0/(12*(Dx**2)))
    return (df, d2f)
# numdiff_centered(...)

def numdiff_backward(
        f0:     npt.NDArray[float],
        fm1:    npt.NDArray[float],     # this is f(x - Dx)
        fm2:    npt.NDArray[float],     #       f(x - 2*Dx)
        fm3:    npt.NDArray[float],
        Dx: float = 1)  -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    df = (3*f0 - 4*fm1 + fm2) * (0.5/Dx)
    d2f = (2*f0 - 5*fm1 + 4*fm2 - fm3) * (1.0/(Dx**2))
    return (df, d2f)
# numdiff_backward(...)


# Angular velocity vector to/from differential rotation matrix
def _xMat(v: npt.NDArray[float]) -> npt.NDArray[float]:
    return np.array([   [0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])
# _xMat(...)

def _xVec(M: npt.NDArray[float]) -> npt.NDArray[float]:
    return np.array([M[2,1], M[0,2], M[1,0]])
# _xVec(...)

# Pose derivatives 
# (translational and angular velocity, and translational acceleration)
def egomotion_numdiff(
        T:  list[npt.NDArray[float]],   # TODO: work with generator form
        Dt: float = 1)  -> Iterator[tuple[npt.NDArray[float],
                                        npt.NDArray[float],
                                        npt.NDArray[float]]]:
    assert(len(T) >= 5)
    # Uses fourth-order formula for translational velocity and acceleration
    # for the middle elements, otherwise second-order forward and backward
    # differences at the end-points.
    #
    # Angular velocity is always either a simple forward-/backward-difference
    # equivalent or centered-difference equivalent, and we make the assumption
    # that the rotation is small, for which we can use the approximation
    #   R(\omega dt) \approxeq (eye(3) + _xMat(\omega) * dt)
    # Angular acceleration is not computed at all.
    for n in range(len(T)):
        if n < 2:
            vel_tr_w, acc_tr_w = numdiff_forward(T[n][:3,3],
                                                    T[n+1][:3,3],
                                                    T[n+2][:3,3],
                                                    T[n+3][:3,3],
                                                Dx=Dt)
            rot_c = T[n][:3,:3].transpose() @ T[n+1][:3,:3]
        elif n >= len(T)-2:
            vel_tr_w, acc_tr_w = numdiff_backward(T[n][:3,3],
                                                    T[n-1][:3,3],
                                                    T[n-2][:3,3],
                                                    T[n-3][:3,3],
                                                Dx=Dt)
            rot_c = T[n-1][:3,:3].transpose() @ T[n][:3,:3]
        else:
            vel_tr_w, acc_tr_w = numdiff_centered(T[n-2][:3,3], T[n-1][:3,3],
                                                    T[n][:3,3],
                                                    T[n+1][:3,3], T[n+2][:3,3],
                                                Dx=Dt)
            rot_c_x2 = T[n-1][:3,:3].transpose() @ T[n+1][:3,:3]
            rot_c = np.eye(3) + 0.5*(rot_c_x2 - np.eye(3))

        vel_tr_c = T[n][:3,:3].transpose() @ vel_tr_w
        acc_tr_c = T[n][:3,:3].transpose() @ acc_tr_w

        vel_ang_c_xMat = (rot_c - np.eye(3)) * (1/Dt)
        vel_ang_c = _xVec(vel_ang_c_xMat)
        # TODO: angular acceleration

        yield (vel_tr_c, acc_tr_c, vel_ang_c)
# egomotion_numdiff(...)

""" Forster et al., "IMU Preintegration on Manifold for Efficient Visual-Inertial
                    Maximum-a-Posteriori Estimation", 2015.

    _w: in world coordinate frame
    _c: in camera-centered coordinate frame
"""
def egomotion_int_step_fwd_Euler(
        T_wc:       npt.NDArray[float] = np.eye(4),
        vel_tr_c:   npt.NDArray[float] = np.zeros((3,)),
        vel_ang_c:  npt.NDArray[float] = np.zeros((3,)),
        acc_tr_c:   npt.NDArray[float] = np.zeros((3,)),
        acc_ang_c:  npt.NDArray[float] = np.zeros((3,)),
        Dt:         float = 1) -> tuple[npt.NDArray[float],
                                        npt.NDArray[float],
                                        npt.NDArray[float],
                                        npt.NDArray[float],
                                        npt.NDArray[float]]:
    R_wc = T_wc[:3,:3]
    vel_tr_w = R_wc @ vel_tr_c
    acc_tr_w = R_wc @ acc_tr_c
    vel_tr_w_next = vel_tr_w + acc_tr_w*Dt
    p_w = T_wc[:3,3]
    p_w_next = p_w + (vel_tr_w*Dt) + 0.5*acc_tr_w*(Dt**2)
    p_w_next = p_w_next.reshape((3,1))

    assert(np.all(acc_ang_c == 0), "Not implemented!")
    vel_ang_w = R_wc @ vel_ang_c
    vel_ang_w_next = vel_ang_w
    # post-multiply rotates in camera coordinates
    # We make the small-angle approximation for rotation matrix again
    dR_c = np.eye(3) + _xMat(vel_ang_c*Dt)
    R_wc_next = R_wc @ dR_c

    T_wc_next = np.vstack((np.hstack((R_wc_next, p_w_next)),
                            np.array([0,0,0,1])))

    vel_tr_c_next = R_wc_next.transpose() @ vel_tr_w_next
    vel_ang_c_next = R_wc_next.transpose() @ vel_ang_w_next

    return (T_wc_next,
            vel_tr_w_next, vel_ang_w_next,
            vel_tr_c_next, vel_ang_c_next)
# egomotion_int_step_fwd_Euler(...)

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
            rng:            Any = np.random.default_rng()) \
                    -> Iterator[dict[str, npt.ArrayLike]]:
    # discrete-time noise from continuous-time process parameters (Crassidis)
    std_acc_discrete = std_acc / (Dt**0.5)
    std_gyro_discrete = std_gyro / (Dt**0.5)
    std_bias_acc_discrete = std_bias_acc * (Dt**0.5)
    std_bias_gyro_discrete = std_bias_gyro * (Dt**0.5)
    bias_acc = init_bias_acc
    bias_gyro = init_bias_gyro

    t = 0
    for T_wc, (v_tr_c, a_tr_c, v_ang_c) in zip(T_wc_seq,
                                            egomotion_numdiff(T_wc_seq, Dt)):
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

        data = {"acc_reading": sim_a_tr, "gyro_reading": sim_v_ang,
                "acc_bias": bias_acc, "gyro_bias": bias_gyro,
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

