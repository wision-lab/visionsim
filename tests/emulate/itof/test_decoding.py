"""Tests for iToF depth decoding."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from tests.emulate.itof import D_MAX, FREQ, PERIOD, codes, roundtrip, tap_vectors
from visionsim.emulate.itof import (
    CodingScheme,
    decode,
    make_coding_functions,
    simulate_measurements,
    unambiguous_range,
)
from visionsim.emulate.itof.coding import make_gray_codes_reduced, make_max_min_run_length_gray_codes
from visionsim.emulate.itof.decoding import (
    _compute_segment_distance,
    _validate_captures,
    decode_hilbert,
    decode_mult_freq_sinusoid,
)

# ─────────────────────── Segment distance (decode helper) ──────────────────


def _naive_segment_distance_sq(start: npt.NDArray, end: npt.NDArray, points: npt.NDArray) -> npt.NDArray:
    """Squared distance from each point to a segment, computed by projection."""
    seg_len_sq: npt.NDArray[np.floating] = np.sum((start - end) ** 2)
    if seg_len_sq == 0:
        return np.sum((points - start[:, None]) ** 2, axis=0)
    t = np.sum((points - start[:, None]) * (end - start)[:, None], axis=0) / seg_len_sq
    t = np.clip(t, 0, 1)
    closest = start[:, None] + t[None, :] * (end - start)[:, None]
    return np.sum((points - closest) ** 2, axis=0)


@pytest.mark.parametrize("dim", [2, 3])
def test_segment_distance_matches_naive(dim: int):
    rng = np.random.default_rng(0)
    start, end = rng.normal(size=dim), rng.normal(size=dim)
    points = rng.normal(size=(dim, 500))
    assert _compute_segment_distance(start, end, points) == pytest.approx(_naive_segment_distance_sq(start, end, points))


def test_segment_distance_zero_on_segment():
    start, end = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    points = np.array([[0.5, 0.25], [0.5, 0.25]])  # both points lie on the segment
    assert _compute_segment_distance(start, end, points).tolist() == [0.0, 0.0]


def test_segment_distance_degenerate_segment():
    start = np.array([1.0, 2.0])
    points = np.array([[1.0, 0.0], [2.0, 2.0]]).T
    assert _compute_segment_distance(start, start.copy(), points) == pytest.approx(
        np.sum((points - start[:, None]) ** 2, axis=0)
    )


def test_segment_distance_clamped_beyond_endpoints():
    start, end = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    points = np.array([[3.0], [0.0]])
    assert _compute_segment_distance(start, end, points) == pytest.approx([4.0])  # distance to end


def test_segment_distance_precomputed_norm_fast_path():
    rng = np.random.default_rng(1)
    start, end = rng.normal(size=3), rng.normal(size=3)
    points = rng.normal(size=(3, 100))
    default = _compute_segment_distance(start, end, points)
    precomputed = _compute_segment_distance(start, end, points, np.sum(points**2, axis=0))
    assert default == pytest.approx(precomputed)


# ──────────────────────── Depth decoding / round-trips ─────────────────────

# (scheme, n_captures, tolerance in metres): schemes whose decode is exact for
# every depth within their unambiguous range.
_ROUNDTRIP_SCHEMES = [
    ("convSin", 3, 1e-3),
    ("convSin", 4, 1e-3),
    ("convSin", 5, 1e-3),
    ("deltaSin", 4, 1e-3),
    ("convSquare", 3, 1e-3),
    ("convSquare", 4, 1e-3),
    ("deltaHilbertDimTwo", 4, 5e-3),
    ("multFreqSin", 4, 1e-3),
    ("multFreqSin", 5, 1e-3),
    ("multFreqSin", 7, 1e-3),
]


@pytest.mark.parametrize(("scheme", "n_captures", "atol"), _ROUNDTRIP_SCHEMES)
def test_roundtrip_over_full_range(scheme: CodingScheme, n_captures: int, atol: float):
    depths = np.linspace(0.02, D_MAX - 0.02, 12)
    decoded, true = roundtrip(scheme, depths, n_captures=n_captures)
    assert np.abs(decoded - true).max() < atol


@pytest.mark.parametrize("n_captures", [3, 4])
def test_hilbert_dim_one_roundtrip(n_captures: int):
    depths = np.linspace(0.02, D_MAX - 0.02, 40)
    decoded, true = roundtrip("deltaHilbertDimOne", depths, n_captures=n_captures)
    assert np.abs(decoded - true).max() < 1e-3


def test_hilbert_dim_one_five_captures_within_one_interval():
    """n_captures=5 leaves a few ambiguous pixels, but never off by more than one interval."""
    interval_width = D_MAX / len(make_max_min_run_length_gray_codes())
    depths = np.linspace(0.02, D_MAX - 0.02, 200)
    decoded, true = roundtrip("deltaHilbertDimOne", depths, n_captures=5)
    assert np.abs(decoded - true).max() < 1.05 * interval_width


@pytest.mark.parametrize(
    ("scheme", "n_captures", "atol"),
    [("deltaHilbertDimTwo", 4, 5e-3), ("deltaHilbertDimTwo", 5, 1e-3), ("deltaHilbertDimThree", 5, 1e-3)],
)
def test_hilbert_higher_dim_robust_accuracy(scheme: CodingScheme, n_captures: int, atol: float):
    """Higher-dimensional Hilbert schemes are accurate for the vast majority of depths.

    The segment classifier normalizes each channel by its own range while its
    endpoints sit on the coarse Hilbert grid, so pixels sitting exactly between
    two segments can be misassigned. This checks the error distribution instead
    of the worst case -- see :func:`test_hilbert_dim_two_outlier_rate`.
    """
    depths = np.linspace(0.02, D_MAX - 0.02, 128)
    decoded, true = roundtrip(scheme, depths, n_captures=n_captures)
    err = np.abs(decoded - true)
    assert np.median(err) < atol
    assert np.percentile(err, 90) < atol


@pytest.mark.parametrize(("n_captures", "max_outliers"), [(4, 0), (5, 5)])
def test_hilbert_dim_two_outlier_rate(n_captures: int, max_outliers: int):
    """Pins the known misclassification rate of the dim=2 segment classifier."""
    depths = np.linspace(0.02, D_MAX - 0.02, 128)
    decoded, true = roundtrip("deltaHilbertDimTwo", depths, n_captures=n_captures)
    outliers = np.abs(decoded - true) > 0.01
    assert outliers.sum() <= max_outliers


def test_hilbert_dim_one_interval_indices_are_assigned():
    """Regression: interval 0 used to be mistaken for an unassigned pixel."""
    n_captures = 3
    depths = np.linspace(0.02, D_MAX - 0.02, 200)
    mod, ref = codes("deltaHilbertDimOne", n_captures, 1001)
    measurements = simulate_measurements(depths, np.full(depths.shape, 0.8), mod, ref, PERIOD)
    interval_indices, decoded = decode_hilbert(measurements, FREQ, dim=1)
    n_intervals = len(make_gray_codes_reduced(n_captures))
    assert np.all(interval_indices >= 0)  # nothing left unassigned
    assert np.all(interval_indices < n_intervals)
    assert set(np.unique(interval_indices)) == set(range(n_intervals))  # interval 0 included
    # the interval index must be consistent with the returned depth
    assert np.all(np.abs(interval_indices - decoded / D_MAX * n_intervals) <= 1)
    assert np.abs(decoded - depths).max() < 1e-3


@pytest.mark.parametrize(
    ("scheme", "n_captures", "hilbert_order", "hilbert_delta"),
    [
        ("deltaHilbertDimTwo", 4, 1, 0.25),
        ("deltaHilbertDimTwo", 4, 2, 0.4),
        ("deltaHilbertDimTwo", 5, 1, 0.25),
        ("deltaHilbertDimThree", 5, 1, 0.1),
    ],
)
def test_hilbert_higher_dims_and_parameters(
    scheme: CodingScheme, n_captures: int, hilbert_order: int, hilbert_delta: float
):
    depths = np.linspace(0.05, D_MAX - 0.05, 10)
    decoded, true = roundtrip(
        scheme, depths, n_captures=n_captures, hilbert_order=hilbert_order, hilbert_delta=hilbert_delta
    )
    assert np.abs(decoded - true).max() < 5e-3


@pytest.mark.parametrize("scheme", ["singleRamp", "doubleRamp"])
def test_ramp_roundtrip_within_half_range(scheme: CodingScheme):
    limit = unambiguous_range(scheme, FREQ)
    depths = np.linspace(0.01, limit - 0.005, 10)
    decoded, true = roundtrip(scheme, depths, n_captures=3)
    assert np.abs(decoded - true).max() < 5e-3


@pytest.mark.parametrize("scheme", ["singleRamp", "doubleRamp"])
def test_ramp_folds_beyond_half_range(scheme: CodingScheme):
    """Deeper than c/4f, ramp schemes mirror depth into the reported range."""
    depths = np.array([0.7, 0.9, 1.2])
    assert np.all(depths > unambiguous_range(scheme, FREQ))
    decoded, true = roundtrip(scheme, depths, n_captures=3)
    assert decoded == pytest.approx(D_MAX - true, abs=5e-3)


@pytest.mark.parametrize(
    ("scheme", "n_captures"),
    [("convSin", 4), ("deltaSin", 4), ("convSquare", 4), ("deltaHilbertDimOne", 4), ("deltaHilbertDimTwo", 4)],
)
def test_intensity_scale_invariance(scheme: CodingScheme, n_captures: int):
    """Decoding only uses relative intensities, so scaling is a no-op."""
    mod, ref = codes(scheme, n_captures, 1001)
    depths = np.array([0.3, 0.8])
    measurements = simulate_measurements(depths, np.full(2, 0.7), mod, ref, PERIOD)
    assert decode(scheme, measurements * 1e3, FREQ) == pytest.approx(decode(scheme, measurements, FREQ))


def test_batch_matches_per_pixel():
    depths = np.array([0.2, 0.5, 0.9])
    batch, true = roundtrip("deltaHilbertDimTwo", depths, n_captures=4)
    assert np.abs(batch - true).max() < 5e-3
    for i, depth in enumerate(depths):
        single, _ = roundtrip("deltaHilbertDimTwo", depth, n_captures=4)
        assert batch[i] == pytest.approx(single[0], abs=1e-9)


def test_mult_freq_k4_wide_range_not_supported():
    """The single low-frequency tap cannot widen the range (see docstring)."""
    freq_vec = np.array([0.5, 2.0, 2.0, 2.0])
    shifts_vec = np.array([0.0, 0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    mod, ref = make_coding_functions("multFreqSin", 4, 1001, freq_vec=freq_vec, shifts_vec=shifts_vec)
    depths = np.array([0.3, 0.5])
    measurements = simulate_measurements(depths, np.full(2, 0.8), mod, ref, PERIOD)
    with pytest.raises(NotImplementedError, match="not supported"):
        decode("multFreqSin", measurements, FREQ, freq_vec=freq_vec, shifts_vec=shifts_vec)


def test_mult_freq_direct_matches_dispatch():
    n_captures = 5
    freq_vec, shifts_vec = tap_vectors("multFreqSin", n_captures)
    depths = np.array([0.2, 1.0])
    mod, ref = codes("multFreqSin", n_captures, 1001)
    measurements = simulate_measurements(depths, np.full(2, 0.8), mod, ref, PERIOD)
    direct = decode_mult_freq_sinusoid(measurements, FREQ, freq_vec, shifts_vec)
    dispatched = decode("multFreqSin", measurements, FREQ, freq_vec=freq_vec, shifts_vec=shifts_vec)
    assert direct == pytest.approx(dispatched)
    assert direct == pytest.approx(depths, abs=1e-3)


def test_mult_freq_requires_vectors_to_decode():
    with pytest.raises(ValueError, match="requires freq_vec and shifts_vec"):
        decode("multFreqSin", np.ones((4, 3)), FREQ)


def test_mult_freq_tap_vectors_must_match_data():
    measurements = np.ones((4, 3))
    with pytest.raises(ValueError, match="must have shape"):
        decode("multFreqSin", measurements, FREQ, freq_vec=np.ones(5), shifts_vec=np.ones(5))


def test_mult_freq_taps_must_share_frequency():
    """Taps of one group have to carry the same frequency multiplier."""
    measurements = np.ones((5, 3))
    freq_vec = np.array([1.0, 1.0, 2.0, 2.0, 2.0])  # first group spans two frequencies
    with pytest.raises(ValueError, match="assumed to share a single frequency"):
        decode("multFreqSin", measurements, FREQ, freq_vec=freq_vec, shifts_vec=np.zeros(5))


@pytest.mark.parametrize(
    ("scheme", "n_captures"),
    [("convSin", 2), ("singleRamp", 4), ("deltaHilbertDimThree", 4), ("multFreqSin", 3), ("multFreqSin", 6)],
)
def test_unsupported_capture_counts(scheme: CodingScheme, n_captures: int):
    with pytest.raises(ValueError):
        decode(scheme, np.ones((n_captures, 3)), FREQ)


def test_unknown_scheme():
    with pytest.raises(NotImplementedError, match="No decoder implemented"):
        decode("nope", np.ones((3, 2)), FREQ)  # type: ignore[arg-type]


@pytest.mark.parametrize("measurements", [np.ones(4), np.ones((3, 2, 2)), [[1.0, 2.0]]])
def test_invalid_data_shape(measurements):
    with pytest.raises(ValueError, match="Expected measurements of shape"):
        decode("convSin", measurements, FREQ)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("scheme", "n_captures"),
    [("convSin", 3), ("convSin", 5), ("singleRamp", 3), ("deltaHilbertDimOne", 4), ("multFreqSin", 5)],
)
def test_validate_captures_accepts_supported(scheme: CodingScheme, n_captures: int):
    _validate_captures(scheme, n_captures)
