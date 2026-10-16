import numpy as np

from visionsim.utils.pose import camcoord_blender_to_opengl, camcoord_opengl_to_blender

# A pose matrix as it appears in transforms.json: a rotation plus a translation.
T_BL_GL = np.array(
    [
        [0.0, -1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def test_camcoord_opengl_to_blender_known_pose():
    # The conversion permutes the camera axes (OpenGL's +y ends up on Blender's +z,
    # and OpenGL's +z on Blender's -y), leaving the translation untouched.
    expected = np.array(
        [
            [0.0, 0.0, -1.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, -1.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    assert np.allclose(camcoord_opengl_to_blender(T_BL_GL), expected)


def test_camcoord_roundtrip_is_identity():
    converted = camcoord_opengl_to_blender(T_BL_GL)

    assert np.allclose(camcoord_blender_to_opengl(converted), T_BL_GL)


def test_camcoord_inverse_is_transpose():
    # The conversion matrices are rotations with no translation, so the inverse of
    # one is the transpose of the other.
    forward = camcoord_opengl_to_blender(np.eye(4))
    inverse = camcoord_blender_to_opengl(np.eye(4))

    assert np.allclose(inverse, forward.T)
    assert np.allclose(forward[:3, :3] @ forward[:3, :3].T, np.eye(3))
