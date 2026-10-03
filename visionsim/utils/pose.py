from __future__ import annotations

import numpy as np
import numpy.typing as npt


def tform_camcoord_gl2bl(T_bl_gl: npt.NDArray) -> npt.NDArray:
    """Fix the coordinate convention in the camera pose matrix in transforms.json.

    The coordinate convention for the camera view seems to be the OpenGL one:
        +x  = right
        +y  = up
        +z  = out from the scene,
    while the coordinate convention in Blender seems to be:
        +x  = right
        +y  = into the scene/viewing direction
        +z  = up,
    and the matrix in transforms.json appears to directly map from OpenGL
    coordinate system to Blender's, so we can not treat it as an [R | t] form.

    In turn, we also need to be careful about interpreting the pose
    "derivatives" we get from directly using that matrix, such as when
    simulating IMU data.

    To remove this confusion, here we convert the matrix to use Blender's 3D
    coordinate convention for the camera too.

    Note:
        For more info, see this `PR <https://github.com/wision-lab/visionsim/pull/24>`_ and also
        `this one <https://github.com/wision-lab/visionsim/pull/21>`_.

    Args:
        T_bl_gl (np.NDArray): 4 x 4 matrix representing camera pose, but also mapping directly
                            from OpenGL coordinate system to Blender

    Returns:
        T_bl_bl (np.NDArray): also 4 x 4, but represents only pose in Blender's convention
    """
    M_gl_bl = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    return T_bl_gl @ M_gl_bl
