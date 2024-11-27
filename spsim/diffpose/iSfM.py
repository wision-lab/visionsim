# iSfM.py       instantaneous structure-from-motion formulation

import numpy as np
import numpy.typing as npt

""" References:
    H. C. Longuet-Higgins and K. Prazdny, “The interpretation of a moving 
        retinal image,” Proceedings of the Royal Society of London. 
        Series B., vol. 208, pp. 385--397, 1980.

    A. R. Bruss and B. K. P. Horn, “Passive Navigation,” Computer Vision, 
        Graphics, and Image Processing, vol. 21, pp. 3--20, 1983.

    K. Pauwels and M. M. Van Hulle, “Optimal instantaneous rigid motion 
        estimation insensitive to local minima,” Computer Vision and Image 
        Understanding, vol. 104, no. 1, pp. 77–86, Oct. 2006, 
        doi: 10.1016/j.cviu.2006.07.001.
"""
def iSfM_flow_tr_noZ(
        V:  npt.NDArray[float],     # translational velocity
        x:  npt.NDArray[float],     # pixel coordinates, _centered_
        y:  npt.NDArray[float],     #   (subtract principal point beforehand)
        f:  float = 1) -> tuple[npt.NDArray[float],     # f: focal length in px.
                                npt.NDArray[float]]:
    assert(V.size == 3)
    return (V[2]*x - f*V[0],
            V[2]*y - f*V[1])
# iSfM_flow_tr_noZ(...)

def iSfM_flow_rot(
        W:  npt.NDArray[float],     # angular velocity (rad/time unit)
        x:  npt.NDArray[float],     # pixel coordinates (centered)
        y:  npt.NDArray[float],
        f:  float = 1) -> tuple[npt.NDArray[float],     # f: focal length
                                npt.NDArray[float]]:
    assert(W.size == 3)
    xy = x * y
    x2 = x * x
    y2 = y * y
    if f != 1:
        xy = (1.0/f)*xy             # TODO: pre-compute and pass in?
        x2 = (1.0/f)*x2             # (stays constant throughout sequence)
        y2 = (1.0/f)*y2
    return ((W[0]*xy) - (W[1]*(f + x2)) + (W[2]*y),
            (W[0]*(f + y2)) - (W[1]*xy) - (W[2]*x))
# iSfM_flow_rot(...)

def iSfM_flow(
        Z:  npt.NDArray[float],     # depth (consistent units with V, W)
        V:  npt.NDArray[float],     # trans. vel.
        W:  npt.NDArray[float],     # ang. vel.
        x:  npt.NDArray[float],     # px. coords.
        y:  npt.NDArray[float],
        f:  float = 1) -> tuple[npt.NDArray[float],     # f: focal len.
                                npt.NDArray[float]]:
    u1_tr = iSfM_flow_tr_noZ(V, x, y, f=f)
    u_rot = iSfM_flow_rot(W, x, y, f=f)
    invZ = (1.0/Z)
    return (invZ*u1_tr[0] + u_rot[0],
            invZ*u1_tr[1] + u_rot[1])
# iSfM_flow(...)

def iSfM_invZ_2D(                   # returns (1.0/Z)
        V:  npt.NDArray[float],
        W:  npt.NDArray[float],
        flow:   tuple[npt.NDArray[float], npt.NDArray[float]],
        x:  npt.NDArray[float],
        y:  npt.NDArray[float],
        f:  float = 1) -> npt.NDArray[float]:
    u1_tr = iSfM_flow_tr_noZ(V, x, y, f=f)
    u1_tr_mag2 = (u1_tr[0]*u1_tr[0]) + (u1_tr[1]*u1_tr[1])
    u_rot = iSfM_flow_rot(W, x, y, f=f)

    N = (((flow[0] - u_rot[0])*u1_tr[0])
            + ((flow[1] - u_rot[1])*u1_tr[1]))
    return N / u1_tr_mag2
# iSfM_invZ_2D(...)

def iSfM_invZ_normal(
        V:  npt.NDArray[float],     # trans. vel.
        W:  npt.NDArray[float],     # ang. vel.
        flow_dir:   tuple[npt.NDArray[float], npt.NDArray[float]], # (Nx, Ny)
        flow_amt:   npt.NDArray[float],     # ||u_normal||
        x:  npt.NDArray[float],             # centered pixel coords.
        y:  npt.NDArray[float],
        f:  float = 1) -> npt.NDArray[float]:   # focal len.
    u1_tr = iSfM_flow_tr_noZ(V, x, y, f=f)
    u_rot = iSfM_flow_rot(W, x, y, f=f)

    un1_tr = flow_dir[0]*u1_tr[0] + flow_dir[1]*u1_tr[1]
    un_rot = flow_dir[0]*u_rot[0] + flow_dir[1]*u_rot[1]

    return (flow_amt - un_rot) / un1_tr
# iSfM_invZ_normal(...)

