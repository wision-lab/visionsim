from __future__ import annotations

import copy
import functools
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, ClassVar, Iterator, Literal

import numpy as np
from pydantic import AfterValidator, BaseModel, ConfigDict, model_validator
from typing_extensions import Self

from visionsim.simulate import schema
from visionsim.types import Matrix4x4, _Matrix4x4


def _validate_transform_matrix(matrix: _Matrix4x4) -> _Matrix4x4:
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("transform_matrix must be a 4x4 matrix")
    return matrix


class Camera(BaseModel):
    """Camera Intrinsics"""

    model_config = ConfigDict(extra="allow", frozen=True)

    camera_model: Literal["OPENCV", "OPENCV_FISHEYE"] | None = None
    """camera model type"""
    fl_x: float | None = None
    """focal length x"""
    fl_y: float | None = None
    """focal length y"""
    cx: float | None = None
    """principal point x"""
    cy: float | None = None
    """principal point y"""
    h: int | None = None
    """image height"""
    w: int | None = None
    """image width"""
    c: int | None = None
    """image channels"""
    k1: float | None = None
    """first radial distortion parameter, used by [OPENCV, OPENCV_FISHEYE]"""
    k2: float | None = None
    """second radial distortion parameter, used by [OPENCV, OPENCV_FISHEYE]"""
    k3: float | None = None
    """third radial distortion parameter, used by [OPENCV_FISHEYE]"""
    k4: float | None = None
    """fourth radial distortion parameter, used by [OPENCV_FISHEYE]"""
    p1: float | None = None
    """first tangential distortion parameter, used by [OPENCV]"""
    p2: float | None = None
    """second tangential distortion parameter, used by [OPENCV]"""
    fps: float | None = None
    """framerate of camera"""


class Data(BaseModel):
    """Frame data"""

    model_config = ConfigDict(extra="allow", frozen=True)

    file_path: Path | None = None
    """path to data, usually an image or ndarray file"""
    bitpack_dim: int | None = None
    """dimension that has been bitpacked"""


class Frame(Camera, Data):
    """Frame information"""

    model_config = ConfigDict(frozen=True)

    transform_matrix: Annotated[_Matrix4x4, AfterValidator(_validate_transform_matrix)]
    """camera pose (orientation and position) as a 4x4 matrix"""
    offset: int | None = None
    """index of frame, used when ``file_path`` is an ``.npy`` file"""


class Metadata(Camera):
    """A superset of the `Nerfstudio <https://docs.nerf.studio/quickstart/data_conventions.html#dataset-format>`_
    ``transforms.json`` format with a few additional fields such as additional data paths (eg: flow/segmentation)
    and a channels dimension."""

    _REQUIRED_FIELDS: ClassVar[tuple[str, ...]] = ("fl_x", "fl_y", "cx", "cy", "h", "w")
    _data_types: set[str]
    _cameras: set[Camera]
    _path: Path | None

    model_config = ConfigDict(extra="allow", frozen=True)

    frames: list[Frame]
    """per-frame data, intrinsics and extrinsics parameters"""

    @model_validator(mode="after")
    def _validate_data_paths(self) -> Self:
        per_frame_paths = set(
            tuple(field for field in Data.model_fields.keys() if getattr(frame, field)) for frame in self.frames
        )
        if len(per_frame_paths) != 1:
            raise ValueError("Some data paths are defined per-frame for some frames but not all.")

        self._data_types = set(per_frame_paths.pop())
        return self

    @model_validator(mode="after")
    def _validate_intrinsics_usage(self) -> Self:
        # Check camera intrinsics are either per-frame or global, allow mixed usage such as global focal-length and per-frame distortion.
        per_frame_intrinsics = set(
            tuple(field for field in Camera.model_fields.keys() if getattr(frame, field)) for frame in self.frames
        )
        if len(per_frame_intrinsics) != 1:
            raise ValueError("Some intrinsic fields are defined per-frame for some frames but not all.")

        per_frame_intrinsic_fields = set(per_frame_intrinsics.pop())
        redefined_intrinsics = [field for field in per_frame_intrinsic_fields if getattr(self, field)]

        if "camera_model" in per_frame_intrinsic_fields:
            raise ValueError("Per-frame `camera_model` is not supported.")
        if redefined_intrinsics:
            raise ValueError(f"Intrinsic '{', '.join(redefined_intrinsics)}' are defined both per-frame and globally.")

        missing_intrinsics = [
            field
            for field in self._REQUIRED_FIELDS
            if field not in per_frame_intrinsic_fields and getattr(self, field) is None
        ]
        if missing_intrinsics:
            raise ValueError(
                f"Intrinsics '{', '.join(missing_intrinsics)}' must be defined either globally or for all frames."
            )

        self._cameras = set(
            Camera.model_validate(
                self.model_dump(exclude="frames", exclude_unset=True)
                | f.model_dump(include=set(Camera.model_fields.keys()), exclude_unset=True)
            )
            for f in self.frames
        )
        return self

    @classmethod
    def load(cls, path: str | os.PathLike, rename_to: str = "file_path") -> Self:
        """Load metadata from a ``.json`` or ``.db`` transforms file.

        Args:
            path (str | os.PathLike): Path to load metadata from.
            rename_to (str, optional): Load data paths from a ``.db`` file as a different key.
                Defaults to "file_path".

        Raises:
            RuntimeError: raised if loading camera configurations fail.
            ValueError: raised if file format is not understood.

        Returns:
            Self: instantiated Metadata object
        """
        if Path(path).suffix.lower() == ".json":
            with open(path, "r") as f:
                data = json.load(f)
                instance = cls.model_validate(data)
                instance._path = Path(path).resolve()
                return instance
        elif Path(path).suffix.lower() == ".db":
            ds = schema.Metadata(path)
            dense_transforms = ds.to_dense_transforms(rename_to=rename_to)
            instance = cls.from_dense_transforms(dense_transforms)
            if len(instance.cameras) != len(ds.cameras):
                # Note: This really shouldn't occur, but better catch it early if it does!
                raise RuntimeError(
                    f"Unable to load metadata from {path}, original dataset has {len(ds.cameras)} "
                    f"unique cameras but only {len(instance.cameras)} where retained when loading."
                )
            instance._path = Path(path).resolve()
            return instance
        raise ValueError(
            f"Can only load metadata from `.json` or `.db`, tried to load a `{Path(path).suffix}` file (from {path})."
        )

    @classmethod
    def from_path(cls, path: str | os.PathLike, rename_to: str = "file_path") -> Self:
        """Same as :meth:`load` with the added bonus of path disambiguation,
        where ``path`` can also be the directory containing the metadata file."""

        try:
            instance = cls.load(path=path, rename_to=rename_to)
        except ValueError:
            candidates = list(Path(path).glob("*.db")) + list(Path(path).glob("*.json"))

            if not candidates:
                raise RuntimeError(f"No dataset found at '{path}'.")
            if len(candidates) != 1:
                raise RuntimeError(
                    f"Ambiguous dataset root. Found multiple metadata sources in {path} {tuple([str(c.relative_to(path)) for c in candidates])}."
                )
            instance = cls.load(path=candidates.pop(), rename_to=rename_to)
        return instance

    def save(self, path: str | os.PathLike, *, indent: int = 2) -> None:
        """Save metadata to a ``.json`` or ``.db`` transforms file.

        Args:
            path (str | os.PathLike): Path to save metadata to.
            indent (int, optional): Indent amount to use when saving JSON file. Defaults to 2.
        """
        if Path(path).suffix.lower() == ".json":
            with open(path, "w") as f:
                f.write(self.model_dump_json(exclude_unset=True, indent=indent))
        elif Path(path).suffix.lower() == ".db":
            data_type = next(iter(self.data_types))
            schema.Metadata.from_dense_transforms(path=path, transforms=self.iter_dense_transforms(data_type=data_type))
        else:
            raise ValueError(f"Can only save metadata as `.json` or `.db`, tried to save as `{Path(path).suffix}`.")

    @classmethod
    def from_dense_transforms(cls, transforms: Sequence[dict[str, Any]]) -> Self:
        """Load metadata from a sequence of dictionary which contain all frame and camera information.

        Args:
            transforms (Sequence[dict[str, Any]]): Dictionaries containing frame information such as
                "file_path", "transform_matrix" and camera parameters.

        Returns:
            Self: instantiated Metadata object
        """

        def is_equal(a, b):
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                return np.allclose(a, b)
            return a == b

        global_fields = functools.reduce(
            lambda a, b: {k: a[k] for k in set(a.keys()) & set(b.keys()) if is_equal(a[k], b[k])}, transforms
        )
        global_fields = copy.copy(global_fields)

        # Prevent non-camera fields from being global
        for field in set(Frame.model_fields.keys()) - set(Camera.model_fields.keys()):
            global_fields.pop(field, None)

        frames = [Frame.model_validate({k: v for k, v in t.items() if k not in global_fields}) for t in transforms]
        return cls(frames=frames, **global_fields)

    @classmethod
    def from_frames(
        cls, frames: Sequence[Frame] | Sequence[dict[str, Any]], camera: Camera | dict[str, Any] | None = None
    ) -> Self:
        """Load metadata from Frame objects (or their model dicts) and a single Camera object (or model dict).

        Args:
            frames (Sequence[Frame] | Sequence[dict[str, Any]]): Frame instances to load from.
            camera (Camera | dict[str, Any] | None, optional): Global camera to use, if multiple cameras are needed,
                pass them as parts of the frames. Defaults to None (use frame cameras).

        Returns:
            Self: instantiated Metadata object
        """
        return cls(
            frames=[Frame.model_validate(f) for f in frames],
            **Camera.model_validate(camera).model_dump(exclude_unset=True),
        )

    def iter_dense_transforms(
        self, data_type: str | None = None, rename_to: str = "path", relative_to: Path | None = None
    ) -> Iterator[dict[str, Any]]:
        """Yield dictionaries containing all frame and camera information, one per frame.

        Args:
            data_type (str | None, optional): Select which data type to iterate over, since there
                might be multiple ("file_path", "mask_path", etc). Defaults to None (all available).
            rename_to (str, optional): Rename key of iterated data, for instance from "file_path" to "path".
                Only used if ``data_type`` is set. Defaults to "path".
            relative_to (Path | None, optional): Make data paths relative to provided path.
                Defaults to not modifying paths (None).

        Yields:
            Iterator[dict[str, Any]]: Dictionaries containing all relevant frame data
        """
        if data_type:
            if data_type not in self.data_types:
                raise ValueError(f"Data type {data_type} is not defined for every frame, or at all.")
            exclude = set(Data.model_fields.keys())
            exclude.remove(data_type)
        else:
            exclude = set()

        for frame in self.frames:
            transform = self.model_dump(exclude_unset=True, exclude=exclude | {"frames"}) | frame.model_dump(
                exclude_unset=True, exclude=exclude
            )
            if relative_to:
                for dt in self.data_types:
                    # Note: Path(".").parent == Path('.')
                    transform[dt] = ((self._path or Path(".")).parent / transform[dt]).relative_to(relative_to)
            if data_type:
                transform[rename_to] = transform.pop(data_type)
            yield transform

    def to_dense_transforms(self, *args, **kwargs) -> list[dict[str, Any]]:
        """Same as :meth:`iter_dense_transforms` but returns a list instead of a generator."""
        return list(self.iter_dense_transforms(*args, **kwargs))

    def __len__(self) -> int:
        """Return the number of frames in dataset"""
        return len(self.frames)

    @property
    def data_types(self) -> set[str]:
        """Data types that are defined for each frame, such as ``file_path`` of ``depth_file_path``."""
        return self._data_types

    @property
    def cameras(self) -> set[Camera]:
        """Set of defined cameras."""
        return self._cameras

    @property
    def poses(self) -> list[Matrix4x4]:
        """Pose matrices of all frames."""
        return [np.array(f.transform_matrix) for f in self.frames]

    @property
    def path(self) -> Path | None:
        """Path to loaded metadata file, may be undefined."""
        return self._path

    @functools.cached_property
    def arclength(self) -> float:
        """Calculate the length of the trajectory"""
        points = np.array(self.poses)[:, :3, -1]
        dp = np.diff(points, axis=0)
        return np.sqrt((dp**2).sum(axis=1)).sum()
