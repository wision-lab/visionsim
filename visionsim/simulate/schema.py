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
    SqliteDatabase,
    TextField,
)
from playhouse.shortcuts import ThreadSafeDatabaseMetadata
from playhouse.sqlite_ext import JSONField
from typing_extensions import Self

# https://docs.peewee-orm.com/en/latest/peewee/database.html#recommended-settings
_DEFAULT_PRAGMAS = {
    "journal_mode": "wal",
    "cache_size": -1 * 64000,
    "foreign_keys": 1,
    "ignore_check_constraints": 0,
    "synchronous": 0,
}


class _BaseModel(Model):
    class Meta:
        model_metadata_class = ThreadSafeDatabaseMetadata


class _Camera(_BaseModel):
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


class _Data(_BaseModel):
    """A database model that mirrors :class:`models.Data <visionsim.dataset.models.Data>`."""

    path = TextField()
    bitpack_dim = IntegerField(null=True)


class _Frame(_BaseModel):
    """A database model that mirrors :class:`models.Frame <visionsim.dataset.models.Frame>`."""

    data = ForeignKeyField(_Data, backref="frames", on_delete="CASCADE", index=True)
    camera = ForeignKeyField(_Camera, backref="frames", on_delete="CASCADE", index=True)
    transform_matrix = JSONField()
    offset = IntegerField(null=True)


_MODELS: tuple[type[_BaseModel], ...] = (_Camera, _Data, _Frame)


class _Metadata:
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
        instance = cls(path)
        instance._migrate()
        return instance

    def _migrate(self) -> None:
        """Apply any necessary database migrations."""
        # TODO: Implement database migrations if needed. Below is an example of how to do it for a new "date" field in _Camera model.

        # from playhouse.migrate import SqliteMigrator, migrate

        # db = SqliteDatabase(self.path, pragmas=_DEFAULT_PRAGMAS)
        # with db.connection_context():
        #     columns = db.get_columns(_Camera._meta.table_name)
        #     column_names = [c.name for c in columns]

        #     if "date" not in column_names:
        #         migrator = SqliteMigrator(db)
        #         with db.atomic():
        #             migrate(migrator.add_column(_Camera._meta.table_name, "date", _Camera.date))

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

        db = SqliteDatabase(path, pragmas=_DEFAULT_PRAGMAS)

        with db.connection_context():
            with db.atomic():
                with db.bind_ctx(_MODELS):
                    db.create_tables(_MODELS, safe=True)

                    for index, transform in enumerate(transforms):
                        camera, _ = _Camera.get_or_create(
                            **{k: v for k, v in transform.items() if k in _Camera._meta.fields}  # type: ignore
                        )
                        data, _ = _Data.get_or_create(**{k: v for k, v in transform.items() if k in _Data._meta.fields})  # type: ignore
                        _Frame.create(
                            id=index,
                            camera=camera,
                            data=data,
                            **{k: v for k, v in transform.items() if k in _Frame._meta.fields},  # type: ignore
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
        db = SqliteDatabase(self.path, pragmas=_DEFAULT_PRAGMAS)
        with db.connection_context():
            with db.bind_ctx(_MODELS):
                for transform in _Frame.select(*_MODELS).join(_Camera).switch(_Frame).join(_Data).dicts():
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
        db = SqliteDatabase(self.path, pragmas=_DEFAULT_PRAGMAS)
        with db.connection_context():
            with db.bind_ctx(_MODELS):
                return list(_Camera.select().dicts())
