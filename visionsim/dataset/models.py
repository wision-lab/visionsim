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
    model_config = ConfigDict(extra="allow", frozen=True)

    camera_model: Literal["OPENCV", "OPENCV_FISHEYE"] | None = None
    fl_x: float | None = None
    fl_y: float | None = None
    cx: float | None = None
    cy: float | None = None
    h: int | None = None
    w: int | None = None
    c: int | None = None
    k1: float | None = None
    k2: float | None = None
    k3: float | None = None
    k4: float | None = None
    p1: float | None = None
    p2: float | None = None


class Data(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    file_path: Path | None = None
    bitpack_dim: int | None = None


class Frame(Camera, Data):
    model_config = ConfigDict(frozen=True)

    transform_matrix: Annotated[_Matrix4x4, AfterValidator(_validate_transform_matrix)]
    offset: int | None = None


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

    @model_validator(mode="after")
    def validate_data_paths(self) -> Self:
        per_frame_paths = set(
            tuple(field for field in Data.model_fields.keys() if getattr(frame, field)) for frame in self.frames
        )
        if len(per_frame_paths) != 1:
            raise ValueError("Some data paths are defined per-frame for some frames but not all.")

        self._data_types = set(per_frame_paths.pop())
        return self

    @model_validator(mode="after")
    def validate_intrinsics_usage(self) -> Self:
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
    def load(cls, path: str | os.PathLike, as_data_type: str = "file_path") -> Self:
        if Path(path).suffix.lower() == ".json":
            with open(path, "r") as f:
                data = json.load(f)
                instance = cls.model_validate(data)
                instance._path = Path(path).resolve()
                return instance
        elif Path(path).suffix.lower() == ".db":
            ds = schema.Metadata(path)
            dense_transforms = list(ds.iter_dense_transforms(as_data_type=as_data_type))
            instance = cls.from_dense_transforms(dense_transforms)
            if len(instance.cameras) != len(ds.cameras):
                # Note: This really shouldn't occur, but better catch it early if it does!
                raise RuntimeError(
                    f"Unable to load metadata from {path}, original dataset has {len(ds.cameras)} "
                    f"unique cameras but only {len(instance.cameras)} where retained when loading."
                )
            instance._path = Path(path).resolve()
            return instance
        raise ValueError(f"Can only load metadata from `.json` or `.db`, tried to load a `{Path(path).suffix}` file.")

    @classmethod
    def from_path(cls, path: str | os.PathLike, as_data_type: str = "file_path") -> Self:
        """Same as :meth:`load` with the added bonus of path disambiguation,
        where `path` can also be the directory containing the metadata file."""

        try:
            instance = cls.load(path=path, as_data_type=as_data_type)
        except ValueError:
            candidates = list(Path(path).glob("*.db")) + list(Path(path).glob("*.json"))

            if len(candidates) != 1:
                raise RuntimeError(
                    f"Ambiguous dataset root. Found multiple metadata sources ({[c.relative_to(path) for c in candidates]})."
                )
            instance = cls.load(path=candidates.pop(), as_data_type=as_data_type)
        return instance

    def save(self, path: str | os.PathLike, *, indent: int = 2, data_type: str | None = None) -> None:
        if Path(path).suffix.lower() == ".json":
            with open(path, "w") as f:
                f.write(self.model_dump_json(exclude_unset=True, indent=indent))
        elif Path(path).suffix.lower() == ".db":
            if len(self.data_types) != 1:
                raise ValueError(
                    f"Can only save as a database when there is a single data type, got {self.data_types} instead."
                )

            data_type = next(iter(self.data_types))
            schema.Metadata.from_dense_transforms(path=path, transforms=self.iter_dense_transforms(data_type=data_type))
        else:
            raise ValueError(f"Can only save metadata as `.json` or `.db`, tried to save as `{Path(path).suffix}`.")

    @classmethod
    def from_dense_transforms(cls, transforms: Sequence[dict[str, Any]]) -> Self:
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
    def from_frames_and_camera(
        cls, camera: Camera | dict[str, Any], frames: Sequence[Frame] | Sequence[dict[str, Any]]
    ) -> Self:
        return cls(
            frames=[Frame.model_validate(f) for f in frames],
            **Camera.model_validate(camera).model_dump(exclude_unset=True),
        )

    def iter_dense_transforms(self, data_type: str | None = None, rename_to: str = "path") -> Iterator[dict[str, Any]]:
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

            if data_type:
                transform[rename_to] = transform.pop(data_type)
            yield transform

    def to_dense_transforms(self, *args, **kwargs) -> list[dict[str, Any]]:
        return list(self.iter_dense_transforms(*args, **kwargs))

    @property
    def data_types(self) -> set[str]:
        return self._data_types

    @property
    def cameras(self) -> set[Camera]:
        return self._cameras

    @property
    def poses(self) -> list[Matrix4x4]:
        return [np.array(f.transform_matrix) for f in self.frames]

    @property
    def path(self) -> Path | None:
        return self._path
