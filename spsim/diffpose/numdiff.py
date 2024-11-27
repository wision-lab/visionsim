# numdiff.py    Numerical differentiation formulas 
#               for camera velocity and acceleration

import numpy as np
import numpy.typing as npt

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

