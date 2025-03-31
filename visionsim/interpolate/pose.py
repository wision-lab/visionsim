from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation, RotationSpline
from typing_extensions import Literal, cast


class pose_interp:
    """Linearly interpolate between 4x4 (or 3x4) transformation matrices by interpolating it's components"""

    def __init__(self, transforms: npt.ArrayLike, ts: npt.ArrayLike | None = None, k=3, normalize=False) -> None:
        """Create a spline from the rotational and translational components of each pose.
        Specifically, we use scipy's RotationSpline and BSpline respectively.

        Args:
            transforms (npt.ArrayLike): Poses to interpolate as matrices.
            ts (npt.ArrayLike | None, optional): Time of each pose. Defaults to None, meaning linspace between [0, 1].
            k (int, optional): B-spline degree for translation. Default is cubic (i.e: 3).
            normalize (bool, optional): If true, normalize rotations by their determinants. Defaults to False.

        Raises:
            RuntimeError: raised when rotations are not normalized.
        """
        self.transforms = np.array(transforms)
        self.ts = np.linspace(0, 1, len(self.transforms)) if ts is None else np.array(ts)
        self.determinants = np.linalg.det(self.transforms[:, :3, :3])
        self.k = k

        if normalize:
            self.transforms[:, :3, :3] /= self.determinants[:, None, None]
        elif not np.allclose(self.determinants, 1.0):
            raise RuntimeError(
                "Rotation matrices in poses must have determinant of 1. You may also try setting normalize to True."
            )

        self._rotation_interp = RotationSpline(self.ts, Rotation.from_matrix(self.transforms[:, :3, :3]))
        self._translation_interp = make_interp_spline(self.ts, self.transforms[:, :3, -1], k=k, axis=0)

    def __call__(self, ts: npt.ArrayLike, order: Literal[0, 1, 2] = 0) -> npt.NDArray:
        """Compute interpolated poses, or their derivatives.

        Args:
            ts (npt.ArrayLike): Times of interest.
            order (Literal[0, 1, 2], optional): Order of differentiation:

                * 0 (default): return pose as 4x4 matrices
                * 1: return velocities, where the first row is the angular rates
                    in rad/sec and second row are positional velocities.
                * 2: return the accelerations, packaged as angular acceleration
                    (in rad/sec/sec) then positional.

                Defaults to 0.

        Returns:
            npt.NDArray: Interpolated poses (Tx4x4) or their derivatives (2xTx3)
        """
        if order not in [0, 1, 2]:
            raise ValueError(f"Order of derivative (order = {order}) must be 0, 1 or 2.")
        if order > self.k:
            raise ValueError(f"Order of derivative (order = {order}) must be <= order of spline (k = {self.k}).")

        t_shape = np.shape(ts)
        ts = np.atleast_1d(ts)

        R = self._rotation_interp(ts, order)
        t = self._translation_interp(ts, order)

        if order == 0:
            bottom = np.array([0.0, 0.0, 0.0, 1.0])
            bottom = np.tile(bottom, (*ts.shape, 1, 1))

            # Cast to make mypy happy
            Rt = np.concatenate([cast(Rotation, R).as_matrix(), t[..., None]], axis=2)
            transforms = np.concatenate([Rt, bottom], axis=1)
            return transforms.reshape(*t_shape, 4, 4)
        return np.stack([cast(np.ndarray, R), t], axis=0)
