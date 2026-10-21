from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Maps camera coordinates in the OpenGL convention (+x right, +y up, +z out of
# the scene) to Blender's convention (+x right, +y into the scene, +z up). The
# 3x3 block is an orthogonal rotation, so its transpose is also its inverse.
_M_GL_BL = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


def camcoord_opengl_to_blender(T_bl_gl: npt.NDArray) -> npt.NDArray:
    """Convert the camera coordinate convention of a pose matrix from OpenGL to Blender.

    The camera view convention used in ``transforms.json`` (and ``transforms.db``) is
    OpenGL's:

        +x  = right
        +y  = up
        +z  = out from the scene,

    while the convention used inside Blender is:

        +x  = right
        +y  = into the scene/viewing direction
        +z  = up.

    The matrix in ``transforms.json`` appears to directly map from the OpenGL camera
    frame into Blender's world frame, so it mixes a camera convention into a world
    matrix and can not be treated as a plain ``[R | t]`` pose. Interpreting its
    rotation as a camera-frame quantity, as when differentiating the pose to read an
    angular velocity while simulating IMU data, is then off. Rewriting the matrix so
    that both its world and its camera frame use Blender's convention removes that
    confusion.

    Note:
        For more info, see this `PR <https://github.com/wision-lab/visionsim/pull/24>`_ and also
        `this one <https://github.com/wision-lab/visionsim/pull/21>`_.

    Args:
        T_bl_gl (np.NDArray): 4 x 4 pose matrix from ``transforms.json``, mapping
                            directly from the OpenGL camera frame to Blender's world frame.

    Returns:
        T_bl_bl (np.NDArray): also 4 x 4, but expressing the same pose with the camera
                            frame converted to Blender's convention, so that it is a
                            plain ``[R | t]``.
    """
    return T_bl_gl @ _M_GL_BL


def camcoord_blender_to_opengl(T_bl_bl: npt.NDArray) -> npt.NDArray:
    """Convert the camera coordinate convention of a pose matrix from Blender back to OpenGL.

    This undoes :func:`camcoord_opengl_to_blender`: it takes a pose matrix whose camera
    frame is already in Blender's convention (+x right, +y into the scene, +z up) and
    rewrites it so the camera frame uses OpenGL's (+x right, +y up, +z out from the
    scene), which is the convention of the matrices as they appear in ``transforms.json``.

    The conversion matrix used by :func:`camcoord_opengl_to_blender` is an orthogonal
    rotation, so its inverse is simply its transpose. This function therefore post-multiplies
    by the transpose of that matrix.

    Note:
        For more info, see this `PR <https://github.com/wision-lab/visionsim/pull/24>`_ and also
        `this one <https://github.com/wision-lab/visionsim/pull/21>`_.

    Args:
        T_bl_bl (np.NDArray): 4 x 4 pose matrix whose camera frame follows Blender's
                            convention, as returned by :func:`camcoord_opengl_to_blender`.

    Returns:
        T_bl_gl (np.NDArray): also 4 x 4, but expressing the same pose with the camera
                            frame back in OpenGL's convention.
    """
    return T_bl_bl @ _M_GL_BL.T
