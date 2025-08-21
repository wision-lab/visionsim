import itertools

import numpy as np
import OpenEXR
import pytest

from visionsim.dataset import IMG_SCHEMA, read_and_validate


def test_render_layout(cube_dataset):
    assert (cube_dataset / "transforms.json").exists()

    for subdir in ["frames", "depths", "normals", "flows", "segmentations"]:
        subdir = cube_dataset / subdir
        assert subdir.exists()
        assert len(list(subdir.glob("*.png"))) == 5
        assert len(list(subdir.glob("*.exr"))) in (0, 5)


@pytest.mark.parametrize(
    "subdir, channels", [("depths", ["V"]), ("normals", ["RGB"]), ("flows", ["RGBA"]), ("segmentations", ["V"])]
)
def test_groundtruth_exrs(cube_dataset, subdir, channels):
    for file in cube_dataset.glob(f"{subdir}/*.exr"):
        with OpenEXR.File(str(file)) as f:
            # Before v4.3.0 exr's couldn't be single channel, they were saved as
            # RGB with duplicated channels. Yet, in 4.2 and 4.1, they still get saved as "V"
            if channels == ["V"] and "V" not in f.channels():
                assert "RGB" in f.channels()
                data = f.channels()["RGB"].pixels.transpose(2, 0, 1)
                assert all(np.allclose(a, b) for a, b in itertools.pairwise(data))
                channels = ["RGB"]
            else:
                assert list(f.channels().keys()) == channels

            for channel in channels:
                assert np.issubdtype(f.channels()[channel].pixels.dtype, np.floating)


def test_transforms_schema(cube_dataset):
    read_and_validate(path=cube_dataset / "transforms.json", schema=IMG_SCHEMA)
