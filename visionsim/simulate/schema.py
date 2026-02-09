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


class BaseModel(Model):
    class Meta:
        model_metadata_class = ThreadSafeDatabaseMetadata


class Camera(BaseModel):
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
    c = IntegerField()

    fl_x = FloatField()
    fl_y = FloatField()

    cx = FloatField()
    cy = FloatField()


class Data(BaseModel):
    path = TextField()
    bitpack_dim = IntegerField(null=True)


class Frame(BaseModel):
    data = ForeignKeyField(Data, backref="frames", on_delete="CASCADE", index=True)
    camera = ForeignKeyField(Camera, backref="frames", on_delete="CASCADE", index=True)
    transform_matrix = JSONField()
    offset = IntegerField(null=True)


MODELS: tuple[type[BaseModel], ...] = (Camera, Data, Frame)


class Metadata:
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path).resolve()

    @classmethod
    def load(cls, path: str | os.PathLike) -> Self:
        return cls(path)

    @classmethod
    def from_dense_transforms(cls, path: str | os.PathLike, transforms: Iterator[dict[str, Any]]) -> Self:
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

    def iter_dense_transforms(
        self, as_data_type: str = "file_path", exclude_none: bool = True
    ) -> Iterator[dict[str, Any]]:
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

                    transform[as_data_type] = transform.pop("path")
                    yield transform

    def to_dense_transforms(self, *args, **kwargs) -> list[dict[str, Any]]:
        return list(self.iter_dense_transforms(*args, **kwargs))

    @cached_property
    def cameras(self) -> list[dict[str, Any]]:
        db = SqliteExtDatabase(self.path, pragmas=DEFAULT_PRAGMAS)
        with db.connection_context():
            with db.bind_ctx(MODELS):
                return list(Camera.select().dicts())
