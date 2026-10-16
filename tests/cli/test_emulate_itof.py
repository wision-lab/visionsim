import json
import logging
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
import scipy.constants
from scipy.interpolate import PchipInterpolator

from visionsim.cli import emulate
from visionsim.dataset.models import Metadata
from visionsim.emulate.itof.coding import make_coding_functions
from visionsim.emulate.itof.simulation import compute_correlation_function
from visionsim.utils.color import srgb_to_linearrgb

_FREQ = 120e6
_D_MAX = scipy.constants.c / (2 * _FREQ)  # unambiguous range of a single-period scheme, ~1.25 m


def test_itof_writes_measurements_and_provenance(tmp_path: Path, write_dataset):
    input_dir = write_dataset(tmp_path / "in", depths=[0.4, 0.9])
    output_dir = tmp_path / "out"
    emulate.itof(input_dir=input_dir, output_dir=output_dir, scheme="convSin", n_captures=4, num_bins=200)

    assert sorted(p.name for p in output_dir.glob("*.npy")) == ["0000.npy", "0001.npy"]
    measurements = np.load(output_dir / "0000.npy")
    assert measurements.shape == (4, 32, 32)
    assert measurements.dtype == np.float32
    assert np.all(np.isfinite(measurements))

    # Acquisition parameters belong to the capture, not to any camera pose,
    # so they live in params.json rather than in the transforms schema.
    params = json.loads((output_dir / "params.json").read_text())
    assert params["scheme"] == "convSin"
    assert params["n_captures"] == 4
    assert params["freq_hz"] == pytest.approx(_FREQ)
    assert params["num_bins"] == 200
    assert params["unambiguous_range_m"] == pytest.approx(_D_MAX)
    assert "effective_range_m" not in params

    metadata = Metadata.load(output_dir / "transforms.json")
    assert len(metadata.frames) == 2
    assert not any(key.startswith("itof_") for key in metadata.frames[0].model_dump())


def test_itof_preview_taps(tmp_path: Path, write_dataset):
    input_dir = write_dataset(tmp_path / "in", depths=[0.4])
    output_dir = tmp_path / "out"
    emulate.itof(input_dir=input_dir, output_dir=output_dir, scheme="convSin", n_captures=3, num_bins=200, preview=True)
    for tap in range(3):
        assert (output_dir / "preview" / f"tap_{tap}" / "0000.png").is_file()


def test_itof_force_flag(tmp_path: Path, write_dataset):
    input_dir = write_dataset(tmp_path / "in", depths=[0.4])
    output_dir = tmp_path / "out"
    emulate.itof(input_dir=input_dir, output_dir=output_dir, scheme="convSin", n_captures=3, num_bins=200)
    with pytest.raises(FileExistsError, match="already exists"):
        emulate.itof(input_dir=input_dir, output_dir=output_dir, scheme="convSin", n_captures=3, num_bins=200)
    emulate.itof(input_dir=input_dir, output_dir=output_dir, scheme="convSin", n_captures=3, num_bins=200, force=True)


def test_itof_warns_when_out_of_range(tmp_path: Path, caplog, write_dataset):
    """Ramp schemes fold beyond c/4f, so the CLI has to warn about it."""
    input_dir = write_dataset(tmp_path / "in", depths=[0.9])
    with caplog.at_level(logging.WARNING, logger="rich"):
        emulate.itof(input_dir=input_dir, output_dir=tmp_path / "out", scheme="singleRamp", num_bins=200)
    assert any("unambiguous range" in record.getMessage() for record in caplog.records)
    # The recorded range is the one the warning compares against, halved for ramps.
    params = json.loads((tmp_path / "out" / "params.json").read_text())
    assert params["unambiguous_range_m"] == pytest.approx(0.5 * _D_MAX)


def test_itof_hilbert_parameters_are_forwarded(tmp_path: Path, write_dataset):
    input_dir = write_dataset(tmp_path / "in", depths=[0.4])
    output_dir = tmp_path / "out"
    emulate.itof(
        input_dir=input_dir,
        output_dir=output_dir,
        scheme="deltaHilbertDimTwo",
        n_captures=4,
        num_bins=200,
        hilbert_order=2,
        hilbert_delta=0.1,
    )
    params = json.loads((output_dir / "params.json").read_text())
    assert params["hilbert_order"] == 2
    assert params["hilbert_delta"] == pytest.approx(0.1)


def test_itof_shape_mismatch(tmp_path: Path, write_dataset):
    input_dir = write_dataset(tmp_path / "in", depths=[0.4])
    np.save(input_dir / "depth_0000.npy", np.full((16, 16), 0.4, np.float32)[None])
    with pytest.raises(ValueError, match="Shape mismatch"):
        emulate.itof(input_dir=input_dir, output_dir=tmp_path / "out", num_bins=100)


def test_itof_input_output_must_differ(tmp_path: Path, write_dataset):
    input_dir = write_dataset(tmp_path / "in", depths=[0.4])
    with pytest.raises(RuntimeError, match="cannot be the same"):
        emulate.itof(input_dir=input_dir, output_dir=input_dir, num_bins=100, force=True)


def test_itof_failed_run_leaves_no_output(tmp_path: Path, write_dataset):
    """A run that fails during argument validation must not create a partial tree.

    Otherwise the retry hits ``FileExistsError`` and demands ``--force`` even
    though nothing usable was produced.
    """
    input_dir = write_dataset(tmp_path / "in", depths=[0.4])
    output_dir = tmp_path / "out"
    with pytest.raises(ValueError):
        emulate.itof(input_dir=input_dir, output_dir=output_dir, scheme="multFreqSin", num_bins=100)
    assert not output_dir.exists()

    emulate.itof(input_dir=input_dir, output_dir=output_dir, scheme="convSin", n_captures=3, num_bins=100)
    assert (output_dir / "params.json").is_file()


def test_itof_measurement_matches_analytic_model(tmp_path: Path, write_dataset):
    """The written taps must equal the analytic model of the known depth and albedo.

    Finiteness and shape checks alone would pass for an all-zero regression, so
    this pins the actual value against the depth and albedo the frame was
    generated from.
    """
    depth = 0.4
    num_bins = 200
    input_dir = write_dataset(tmp_path / "in", depths=[depth])
    output_dir = tmp_path / "out"
    emulate.itof(input_dir=input_dir, output_dir=output_dir, scheme="convSin", n_captures=3, num_bins=num_bins)

    measurements = np.load(output_dir / "0000.npy")
    assert np.any(measurements > 0)

    # Reconstruct the albedo exactly as the CLI derives it: sRGB PNG -> linear -> luma.
    image = iio.imread(input_dir / "0000.png").astype(float) / 255.0
    linear = srgb_to_linearrgb(image)
    r, g, b = linear[..., 0], linear[..., 1], linear[..., 2]
    albedo = 0.0722 * b + 0.7152 * g + 0.2126 * r

    n_bins = num_bins + 1
    period = 1.0 / _FREQ
    time_resolution = period / n_bins
    max_depth = scipy.constants.c * period / 2
    distances = np.linspace(0, max_depth, n_bins, endpoint=False)
    modulation_codes, reference_codes = make_coding_functions("convSin", 3, n_bins)
    for tap in range(3):
        correlation = compute_correlation_function(
            modulation_codes[tap], reference_codes[tap], time_resolution, n_depths=n_bins
        )
        expected = (1.0 / period) * (albedo / depth**2) * PchipInterpolator(distances, correlation)(depth % max_depth)
        assert measurements[tap] == pytest.approx(expected, rel=1e-5)
