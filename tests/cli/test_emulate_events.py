import imageio.v3 as iio
import numpy as np
import pytest

from visionsim.cli import emulate

FPS = 29.97


def test_events_keep_fractional_fps(tmp_path, write_dataset):
    root = write_dataset(tmp_path / "in", frames=2, size=8)
    # Brighten the second frame so the emulator emits events across that interval.
    iio.imwrite(root / "0001.png", np.full((8, 8, 3), 200, np.uint8))

    emulate.events(root, tmp_path / "out", fps=FPS, leak_rate_hz=0, shot_noise_rate_hz=0)

    events = np.loadtxt(tmp_path / "out" / "events.txt", delimiter=",", ndmin=2)
    assert events.size
    # v2e spreads events across the interval; the last one lands at exactly 1/fps.
    assert events[:, 0].max() == int(1e6 / FPS)


def test_events_reject_non_positive_fps(tmp_path, write_dataset):
    root = write_dataset(tmp_path / "in", frames=1, size=8)

    with pytest.raises(ValueError, match="positive"):
        emulate.events(root, tmp_path / "out", fps=0)


def test_events_accept_grayscale_frames(tmp_path, write_dataset):
    root = write_dataset(tmp_path / "in", frames=3, size=8)
    for i, value in enumerate((10, 200, 200)):
        iio.imwrite(root / f"{i:04}.png", np.full((8, 8), value, np.uint8))

    emulate.events(root, tmp_path / "out", fps=1000, leak_rate_hz=0, shot_noise_rate_hz=0)

    events = np.loadtxt(tmp_path / "out" / "events.txt", delimiter=",", ndmin=2)
    assert events.size
