from __future__ import annotations

import numpy as np
import numpy.typing as npt


def T_wc_camframe_gl2world(T_wc_w_gl: npt.NDArray) -> npt.NDArray:
    """Fix the coordinate convention in the camera pose matrix in transforms.json.

    The coordinate convention for the camera view is OpenGL:
        +x = right, 
        +y = up, 
        +z = out from the scene, 
    while the coordinate convention for the world frame is:
        +x = right, 
        +y = into the scene/viewing direction, 
        +z = up,
    and the matrix in transforms.json directly maps from the former to the
    latter, so we can not treat it as an [R | t] form.
    In turn, we also need to be careful about interpreting the pose
    "derivatives" we get from directly using that matrix, such as when
    simulating IMU data.
    To remove this confusion, here we convert the matrix to use the world
    coordinate convention for the camera too.

    Args:
        T_wc_w_gl (np.NDArray): 4 x 4 matrix representing camera pose, but also mapping directly
                                from OpenGL coordinate system to world

    Returns:
        T_wc_w_w (np.NDArray): also 4 x 4, but uses world frame on both sides
    """
    M_gl_w = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    return T_wc_w_gl @ M_gl_w
