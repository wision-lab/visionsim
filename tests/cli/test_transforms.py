import imageio.v3 as iio
import numpy as np
import OpenEXR
import pytest

from visionsim.cli.transforms import colorize_depths

HEADER = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}


def _write_exr(path, channel, data):
    with OpenEXR.File(HEADER, {channel: OpenEXR.Channel(data)}) as f:
        f.write(str(path))


@pytest.mark.parametrize("explicit_range", [False, True])
def test_colorize_depths_rejects_multichannel(tmp_path, explicit_range):
    src = tmp_path / "in"
    src.mkdir()
    (src / "0000").mkdir()
    data = np.stack([np.full((4, 5), v, dtype=np.float32) for v in (0.1, 0.2, 0.3)], axis=-1)
    _write_exr(src / "0000" / "0000.exr", "RGB", data)

    kwargs = {"vmin": 0.0, "vmax": 1.0} if explicit_range else {}
    with pytest.raises(ValueError, match="single-channel"):
        colorize_depths(src, tmp_path / "out", **kwargs)

    assert not list((tmp_path / "out").glob("**/*.png"))


def test_colorize_depths_accepts_scalar(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    data = np.linspace(0, 1, 20, dtype=np.float32).reshape(4, 5)
    (src / "0000").mkdir()
    _write_exr(src / "0000" / "0000.exr", "V", data)

    colorize_depths(src, tmp_path / "out")

    assert iio.imread(tmp_path / "out" / "0000.png").shape == (4, 5, 4)
