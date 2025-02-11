import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from spsim.utils.utils import to_numpy


class pose_interp:
    """Linearly interpolate between 4x4 transformation matrices by interpolating it's components:
    Slerp the rotations and lerp the translations. This cannot do extrapolation (yet)."""

    def __init__(self, transforms, ts=None, kind="linear", normalize=False):
        ts = np.linspace(0, 1, len(transforms)) if ts is None else ts
        self.ts, self.transforms = to_numpy(ts, transforms)
        self.determinants = np.linalg.det(self.transforms[:, :3, :3])

        if normalize:
            self.transforms[:, :3, :3] /= self.determinants[:, None, None]
        elif not np.allclose(self.determinants, 1.0):
            raise RuntimeError(
                "Rotation matrices in poses must have determinant of 1. You may also try setting normalize to True."
            )

        self._rotation_interp = Slerp(ts, Rotation.from_matrix(self.transforms[:, :3, :3]))
        self._translation_interp = interp1d(ts, self.transforms[:, :3, -1], kind=kind, axis=0)

    def __call__(self, t):
        t_shape = np.shape(t)
        t = np.atleast_1d(t)

        bottom = np.array([0.0, 0.0, 0.0, 1.0])
        bottom = np.tile(bottom, (*t.shape, 1, 1))

        R = self._rotation_interp(t).as_matrix()
        t = self._translation_interp(t)[..., None]
        Rt = np.concatenate([R, t], axis=2)
        transforms = np.concatenate([Rt, bottom], axis=1)
        return transforms.reshape(*t_shape, 4, 4)
