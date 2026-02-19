from __future__ import annotations

import os
from collections.abc import Iterator
from functools import cached_property
from pathlib import Path
from typing import Any

from peewee import (
    FloatField,
    ForeignKeyField,
    IntegerField,
    Model,
    TextField,
)
from playhouse.shortcuts import ThreadSafeDatabaseMetadata
from playhouse.sqlite_ext import JSONField, SqliteExtDatabase
from typing_extensions import Self

# https://docs.peewee-orm.com/en/latest/peewee/database.html#recommended-settings
DEFAULT_PRAGMAS = {
    "journal_mode": "wal",
    "cache_size": -1 * 64000,
    "foreign_keys": 1,
    "ignore_check_constraints": 0,
    "synchronous": 0,
}


class _BaseModel(Model):
    class Meta:
        model_metadata_class = ThreadSafeDatabaseMetadata


class Camera(_BaseModel):
    """A database model that mirrors :class:`models.Camera <visionsim.dataset.models.Camera>` with added blender-specific fields."""

    angle = FloatField(null=True)
    angle_x = FloatField(null=True)
    angle_y = FloatField(null=True)

    clip_start = FloatField(null=True)
    clip_end = FloatField(null=True)

    lens = FloatField(null=True)
    lens_unit = TextField(null=True)

    sensor_height = FloatField(null=True)
    sensor_width = FloatField(null=True)
    sensor_fit = TextField(null=True)

    shift_x = FloatField(null=True)
    shift_y = FloatField(null=True)

    type = TextField(null=True)

    h = IntegerField()
    w = IntegerField()
    c = IntegerField(null=True)

    fl_x = FloatField()
    fl_y = FloatField()

    cx = FloatField()
    cy = FloatField()
    fps = FloatField(null=True)
    keyframe_scale = FloatField(null=True)


class Data(_BaseModel):
    """A database model that mirrors :class:`models.Data <visionsim.dataset.models.Data>`."""

    path = TextField()
    bitpack_dim = IntegerField(null=True)


class Frame(_BaseModel):
    """A database model that mirrors :class:`models.Frame <visionsim.dataset.models.Frame>`."""

    data = ForeignKeyField(Data, backref="frames", on_delete="CASCADE", index=True)
    camera = ForeignKeyField(Camera, backref="frames", on_delete="CASCADE", index=True)
    transform_matrix = JSONField()
    offset = IntegerField(null=True)


MODELS: tuple[type[_BaseModel], ...] = (Camera, Data, Frame)


class Metadata:
    """The ``.db`` equivalent of :class:`models.Metadata <visionsim.dataset.models.Metadata>`"""

    def __init__(self, path: str | os.PathLike) -> None:
        """Initialize a metadata instance.

        Args:
            path (str | os.PathLike): Metadata dataset path
        """
        self.path = Path(path).resolve()

    @classmethod
    def load(cls, path: str | os.PathLike) -> Self:
        """Same as :meth:`__init__`, added to better mirror :class:`models.Metadata <visionsim.dataset.models.Metadata>`"""
        return cls(path)

    @classmethod
    def from_dense_transforms(cls, path: str | os.PathLike, transforms: Iterator[dict[str, Any]]) -> Self:
        """Instantiate a new dataset from the provided dense transforms, persisting all data to disk.

        Warning:
            This will overwrite any pre-existing dataset found at ``path``.

        Warning:
            This method can be slow, as it inserts rows into the database on at a time which is suboptimal.
            However, it ensures correctness as camera and data rows are checked for uniqueness.

        Args:
            path (str | os.PathLike): Metadata dataset path
            transforms (Iterator[dict[str, Any]]): Dictionary containing frame and camera data. Keys are filtered
                before inserting rows, so only keys that correspond to valid columns will be saved.

        Returns:
            Self: new metadata database object, with all data persisting on disk.
        """
        if Path(path).exists():
            Path(path).unlink()

        db = SqliteExtDatabase(path, pragmas=DEFAULT_PRAGMAS)

        with db.connection_context():
            with db.atomic():
                with db.bind_ctx(MODELS):
                    db.create_tables(MODELS, safe=True)

                    for index, transform in enumerate(transforms):
                        camera, _ = Camera.get_or_create(
                            **{k: v for k, v in transform.items() if k in Camera._meta.fields}  # type: ignore
                        )
                        data, _ = Data.get_or_create(**{k: v for k, v in transform.items() if k in Data._meta.fields})  # type: ignore
                        Frame.create(
                            id=index,
                            camera=camera,
                            data=data,
                            **{k: v for k, v in transform.items() if k in Frame._meta.fields},  # type: ignore
                        )
        return cls(path)

    def iter_dense_transforms(self, rename_to: str = "file_path", exclude_none: bool = True) -> Iterator[dict[str, Any]]:
        """Yield dictionaries containing all frame and camera information, one per frame.

        Args:
            rename_to (str, optional): What to rename "path" key to. Defaults to "file_path".
            exclude_none (bool, optional): If true, exclude columns that have a null value. Defaults to True.

        Yields:
            Iterator[dict[str, Any]]: Dictionaries containing all relevant frame data
        """
        db = SqliteExtDatabase(self.path, pragmas=DEFAULT_PRAGMAS)
        with db.connection_context():
            with db.bind_ctx(MODELS):
                for transform in Frame.select(*MODELS).join(Camera).switch(Frame).join(Data).dicts():
                    # Remove database-specific IDs and Foreign Keys
                    transform.pop("id")
                    transform.pop("camera")
                    transform.pop("data")

                    if exclude_none:
                        none_keys = [k for k, v in transform.items() if v is None]
                        for k in none_keys:
                            transform.pop(k)

                    transform[rename_to] = transform.pop("path")
                    yield transform

    def to_dense_transforms(self, *args, **kwargs) -> list[dict[str, Any]]:
        """Same as :meth:`iter_dense_transforms` but returns a list instead of a generator."""
        return list(self.iter_dense_transforms(*args, **kwargs))

    @cached_property
    def cameras(self) -> list[dict[str, Any]]:
        """List of defined cameras."""
        db = SqliteExtDatabase(self.path, pragmas=DEFAULT_PRAGMAS)
        with db.connection_context():
            with db.bind_ctx(MODELS):
                return list(Camera.select().dicts())
