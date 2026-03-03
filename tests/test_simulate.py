import itertools
import os
from pathlib import Path

import numpy as np
import OpenEXR
import pytest
from peewee import SqliteDatabase

from visionsim.dataset import Dataset, Metadata
from visionsim.simulate.blender import INDEX_PADDING, ITEMS_PER_SUBFOLDER, BlenderClients
from visionsim.simulate.schema import _MODELS, _Data


@pytest.mark.parametrize(
    "gt_type",
    [
        "composites",
        "frames",
        "depths",
        "normals",
        "flows",
        "segmentations",
        "previews/depths",
        "previews/normals",
        "previews/flows/forward",
        "previews/segmentations",
        "previews/materials",
        "materials",
        "diffuse/color",
        "diffuse/direct",
        "diffuse/indirect",
        # "diffuse/light",
        "specular/color",
        "specular/direct",
        "specular/indirect",
        # "specular/light",
        "points",
        "previews/points",
    ],
)
def test_render_layout(cube_dataset, gt_type):
    assert not (cube_dataset / "transforms.json").exists()
    assert not (cube_dataset / "transforms.db").exists()

    subdir = cube_dataset / gt_type
    assert subdir.exists()
    assert not (subdir / "transforms.json").exists()
    assert (subdir / "transforms.db").exists()

    if gt_type in ("frames", "composites") or "previews" in gt_type:
        assert len(list(subdir.glob("**/*.png"))) == 5
    else:
        assert len(list(subdir.glob("**/*.exr"))) == 5


@pytest.mark.parametrize(
    "subdir, channels",
    [
        ("depths", ["V"]),
        ("normals", ["RGB"]),
        ("flows", ["RGBA"]),
        ("segmentations", ["V"]),
        ("materials", ["V"]),
        ("diffuse/color", ["RGB"]),
        ("diffuse/direct", ["RGB"]),
        ("diffuse/indirect", ["RGB"]),
        # ("diffuse/light", ["RGB"]),
        ("specular/color", ["RGB"]),
        ("specular/direct", ["RGB"]),
        ("specular/indirect", ["RGB"]),
        # ("specular/light", ["RGB"]),
        ("points", ["RGB"]),
    ],
)
def test_groundtruth_exrs(cube_dataset, subdir, channels):
    for file in cube_dataset.glob(f"{subdir}/**/*.exr"):
        with OpenEXR.File(str(file)) as f:
            # Before v4 exr's couldn't be single channel, they were saved as
            # RGB with duplicated channels.
            if channels == ["V"] and "V" not in f.channels():
                assert "RGB" in f.channels()
                data = f.channels()["RGB"].pixels.transpose(2, 0, 1)
                assert all(np.allclose(a, b) for a, b in itertools.pairwise(data))
                channels = ["RGB"]
            else:
                assert list(f.channels().keys()) == channels

            for channel in channels:
                assert np.issubdtype(f.channels()[channel].pixels.dtype, np.floating)


@pytest.mark.parametrize(
    "subdir, shape, auto_collapse",
    [
        ("depths", (50, 50, 1), True),
        ("normals", (50, 50, 3), False),
        ("flows", (50, 50, 4), False),
        ("segmentations", (50, 50, 1), True),
        ("materials", (50, 50, 1), True),
        ("diffuse/color", (50, 50, 3), False),
        ("diffuse/direct", (50, 50, 3), False),
        ("diffuse/indirect", (50, 50, 3), False),
        # ("diffuse/light", (50, 50, 3)),
        ("specular/color", (50, 50, 3), False),
        ("specular/direct", (50, 50, 3), False),
        ("specular/indirect", (50, 50, 3), False),
        # ("specular/light", (50, 50, 3)),
        ("points", (50, 50, 3), False),
    ],
)
def test_load_exrs(cube_dataset, subdir, shape, auto_collapse):
    for file in cube_dataset.glob(f"{subdir}/**/*.exr"):
        assert Dataset.load_data(file, auto_collapse=auto_collapse).shape == shape


def test_transforms_schema(cube_dataset):
    for path in cube_dataset.glob("**/*.db"):
        Metadata.load(path)


def test_data_paths_exist(cube_dataset):
    for db_path in cube_dataset.glob("**/*.db"):
        db = SqliteDatabase(db_path)
        with db.connection_context():
            with db.bind_ctx(_MODELS):
                for data in _Data.select():
                    assert (db_path.parent / data.path).exists()


def test_database_threading(tmp_path_factory, executable):
    tmpdir = tmp_path_factory.mktemp("renders")
    log_dir = tmp_path_factory.mktemp("logs")
    scene = Path(__file__).parent / "test_files" / "scenes" / "cube.blend"

    # Spoof frames to bypass render, only save metadata, from a bunch of blender instances.
    # This forces a lot of database writes, which helps test for any potential "Database is locked" errors.
    with BlenderClients.spawn(jobs=os.cpu_count() or 5, executable=executable, timeout=30, log=log_dir) as clients:
        clients.initialize(scene.resolve(), tmpdir.resolve())
        clients.include_frames()
        clients.move_keyframes(scale=5)

        for idx in clients.common_animation_range():
            folder_index = f"{idx // ITEMS_PER_SUBFOLDER:04}"
            frame_index = f"{idx % ITEMS_PER_SUBFOLDER:0{INDEX_PADDING}}"
            frame = tmpdir / "frames" / folder_index / f"{frame_index}.png"
            frame.parent.mkdir(exist_ok=True, parents=True)
            frame.touch()
        clients.render_animation()


def test_metadata_roundtrip_from_db(cube_dataset):
    for path in cube_dataset.glob("**/*.db"):
        meta = Metadata.load(path)
        meta.save(path.parent / "transforms.json")

        assert Metadata.load(path.parent / "transforms.json").model_dump() == meta.model_dump()
