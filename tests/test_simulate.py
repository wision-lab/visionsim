import inspect
import itertools
import os
from dataclasses import fields
from pathlib import Path

import numpy as np
import OpenEXR
import pytest
from playhouse.sqlite_ext import SqliteExtDatabase

from visionsim.dataset import Dataset, Metadata
from visionsim.simulate import blender, config
from visionsim.simulate.blender import INDEX_PADDING, ITEMS_PER_SUBFOLDER, BlenderClients
from visionsim.simulate.schema import _MODELS, _Data


def test_render_layout(cube_dataset):
    assert not (cube_dataset / "transforms.json").exists()
    assert not (cube_dataset / "transforms.db").exists()

    for gt_type in [
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
    ]:
        subdir = cube_dataset / gt_type
        assert subdir.exists()
        assert not (subdir / "transforms.json").exists()
        assert (subdir / "transforms.db").exists()

        if gt_type in ("frames", "composites") or "previews" in gt_type:
            assert len(list(subdir.glob("**/*.png"))) == 5
        else:
            assert len(list(subdir.glob("**/*.exr"))) == 5


@pytest.mark.parametrize(
    "subdir, channels", [("depths", ["V"]), ("normals", ["RGB"]), ("flows", ["RGBA"]), ("segmentations", ["V"])]
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
    "subdir, shape",
    [("depths", (50, 50, 1)), ("normals", (50, 50, 3)), ("flows", (50, 50, 4)), ("segmentations", (50, 50, 1))],
)
def test_load_exrs(cube_dataset, subdir, shape):
    for file in cube_dataset.glob(f"{subdir}/**/*.exr"):
        assert Dataset.load_data(file).shape == shape


def test_transforms_schema(cube_dataset):
    for path in cube_dataset.glob("**/*.db"):
        Metadata.load(path)


@pytest.mark.parametrize(
    "func, conf",
    [
        (blender.BlenderService.exposed_include_composites, config.CompositesConfig),
        (blender.BlenderService.exposed_include_frames, config.FramesConfig),
        (blender.BlenderService.exposed_include_depths, config.DepthsConfig),
        (blender.BlenderService.exposed_include_normals, config.NormalsConfig),
        (blender.BlenderService.exposed_include_flows, config.FlowsConfig),
        (blender.BlenderService.exposed_include_segmentations, config.SegmentationsConfig),
    ],
)
def test_output_configs(func, conf):
    conf_params = {f.name: f.default for f in fields(conf)}
    sig_params = {name: val.default for name, val in inspect.signature(func).parameters.items()}
    sig_params.pop("self")

    assert sig_params == conf_params


def test_data_paths_exist(cube_dataset):
    for db_path in cube_dataset.glob("**/*.db"):
        db = SqliteExtDatabase(db_path)
        with db.connection_context():
            with db.bind_ctx(_MODELS):
                for data in _Data.select():
                    assert (db_path.parent / data.path).exists()


def test_database_threading(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("renders")
    log_dir = tmp_path_factory.mktemp("logs")
    scene = Path(__file__).parent / "test_files" / "scenes" / "cube.blend"

    # Spoof frames to bypass render, only save metadata, from a bunch of blender instances.
    # This forces a lot of database writes, which helps test for any potential "Database is locked" errors.
    with BlenderClients.spawn(jobs=os.cpu_count() or 5, timeout=30, log=log_dir) as clients:
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
