from peewee import (
    FloatField,
    ForeignKeyField,
    IntegerField,
    Model,
    TextField,
)
from playhouse.shortcuts import ThreadSafeDatabaseMetadata
from playhouse.sqlite_ext import JSONField


class BaseModel(Model):
    class Meta:
        model_metadata_class = ThreadSafeDatabaseMetadata


class Camera(BaseModel):
    angle = FloatField()
    angle_x = FloatField()
    angle_y = FloatField()

    clip_start = FloatField()
    clip_end = FloatField()

    lens = FloatField()
    lens_unit = TextField()

    sensor_height = FloatField()
    sensor_width = FloatField()
    sensor_fit = TextField()

    shift_x = FloatField()
    shift_y = FloatField()

    type = TextField()

    h = IntegerField()
    w = IntegerField()
    c = IntegerField()

    fl_x = FloatField()
    fl_y = FloatField()

    cx = FloatField()
    cy = FloatField()


class Frame(BaseModel):
    idx = IntegerField(primary_key=True)
    path = TextField()
    transform_matrix = JSONField()

    camera = ForeignKeyField(
        Camera,
        backref="frames",
        on_delete="CASCADE",
    )
