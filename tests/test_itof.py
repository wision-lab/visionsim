"""Tests for the iToF (indirect time-of-flight) emulation module.

These tests exercise the tracked API only: coding schemes, decoding, and the
measurement simulation. The scene/camera/light model layer lives in an untracked
module and is not imported here.

Round-trip tests synthetically capture noiseless measurements with
:func:`simulate_measurements` and check that :func:`decode` recovers the depths
they were generated from, for every supported coding scheme. Since the
simulation is our ground truth, this validates the encoder/decoder pair
end-to-end rather than reconstructing expected values by hand.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import numpy.typing as npt
import pytest
import scipy.constants
from scipy.interpolate import PchipInterpolator

from visionsim.emulate.itof import (
    CodingScheme,
    compute_correlation_function,
    decode,
    make_coding_functions,
    simulate_measurements,
    unambiguous_range,
)
from visionsim.emulate.itof.coding import (
    _hamiltonian_order,
    _hilbert_2d,
    _hilbert_3d,
    _normalize_and_expand,
    _perm_matrix_to_codes,
    make_conv_sinusoidal_codes,
    make_conv_square_codes,
    make_delta_sinusoidal_codes,
    make_double_ramp_codes,
    make_gray_codes,
    make_gray_codes_reduced,
    make_max_min_run_length_gray_codes,
    make_multi_freq_sinusoidal_codes,
    make_single_ramp_codes,
    make_tof_gray_codes,
    make_tof_hilbert_codes,
)
from visionsim.emulate.itof.decoding import (
    _compute_segment_distance,
    _validate_captures,
    decode_hilbert,
    decode_mult_freq_sinusoid,
)

_C = scipy.constants.c
_FREQ = 120e6
_D_MAX = _C / (2 * _FREQ)  # unambiguous range of a single-period scheme, ~1.25 m

# Tap layouts of the multi-frequency scheme, as documented by the reference
# implementation (see ``decode_mult_freq_sinusoid``).
_MULT_FREQ_LAYOUTS: dict[int, tuple[list[float], list[float]]] = {
    4: ([0.5, 1.0, 1.0, 1.0], [0.0, 0.0, 2 * np.pi / 3, 4 * np.pi / 3]),
    5: ([1.0, 1.0, 1.0, 2.0, 2.0], [0.0, 2 * np.pi / 3, 4 * np.pi / 3, 0.0, np.pi / 2]),
    7: (
        [1.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0],
        [0.0, 2 * np.pi / 3, 4 * np.pi / 3, 0.0, np.pi / 2, 0.0, np.pi / 2],
    ),
}


def _tap_vectors(scheme: CodingScheme, n_captures: int) -> tuple[npt.NDArray, npt.NDArray]:
    """Return the per-tap frequency multipliers and shifts of a scheme."""
    if scheme == "multFreqSin":
        freq_vec, shifts_vec = _MULT_FREQ_LAYOUTS[n_captures]
        return np.asarray(freq_vec), np.asarray(shifts_vec)
    return np.array([1.0]), np.array([0.0])


def _codes(
    scheme: CodingScheme,
    n_captures: int,
    n_depths: int = 1001,
    *,
    hilbert_order: int = 1,
    hilbert_delta: float = 0.25,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Build codes for a scheme, filling in multFreqSin's required tap vectors."""
    if scheme == "multFreqSin":
        freq_vec, shifts_vec = _tap_vectors(scheme, n_captures)
        return make_coding_functions(scheme, n_captures, n_depths, freq_vec=freq_vec, shifts_vec=shifts_vec)
    return make_coding_functions(scheme, n_captures, n_depths, hilbert_order=hilbert_order, hilbert_delta=hilbert_delta)


def _roundtrip(
    scheme: CodingScheme,
    depths: npt.ArrayLike,
    *,
    n_captures: int,
    n_depths: int = 1001,
    freq: float = _FREQ,
    hilbert_order: int = 1,
    hilbert_delta: float = 0.25,
    albedo: float = 0.8,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Synthetically capture ``depths`` with ``name`` and decode them back.

    Returns:
        Tuple of ``(decoded_depths, true_depths)``.
    """
    freq_vec, shifts_vec = _tap_vectors(scheme, n_captures)
    mod, ref = _codes(scheme, n_captures, n_depths, hilbert_order=hilbert_order, hilbert_delta=hilbert_delta)
    true_depths = np.atleast_1d(np.asarray(depths, dtype=float))
    measurements = simulate_measurements(true_depths, np.full(true_depths.shape, albedo), mod, ref, _D_MAX)
    mult_freq_kwargs = {"freq_vec": freq_vec, "shifts_vec": shifts_vec} if scheme == "multFreqSin" else {}
    decoded = decode(
        scheme, measurements, freq, hilbert_order=hilbert_order, hilbert_delta=hilbert_delta, **mult_freq_kwargs
    )
    return decoded, true_depths


# ───────────────────────────── Gray code tables ────────────────────────────


class TestGrayCodeTables:
    def test_gray_codes_shape(self):
        assert make_gray_codes(3).shape == (8, 3)

    def test_gray_codes_binary(self):
        assert set(np.unique(make_gray_codes(4))) == {0.0, 1.0}

    def test_gray_codes_hamming(self):
        """Adjacent Gray codes differ by exactly one bit."""
        gray_codes = make_gray_codes(3)
        for i in range(len(gray_codes) - 1):
            assert np.sum(np.abs(gray_codes[i] - gray_codes[i + 1])) == 1

    def test_max_min_run_length_is_hamiltonian_cycle(self):
        gray_codes = make_max_min_run_length_gray_codes()
        assert gray_codes.shape == (32, 5)  # 5 captures, 2**5 code words
        assert set(np.unique(gray_codes)) == {0.0, 1.0}
        assert len(np.unique(gray_codes, axis=0)) == len(gray_codes)
        for i in range(len(gray_codes)):
            assert np.sum(np.abs(gray_codes[(i + 1) % len(gray_codes)] - gray_codes[i])) == 1

    @pytest.mark.parametrize(("n_bits", "n_rows"), [(3, 6), (4, 12), (5, 30), (6, 60)])
    def test_reduced_gray_codes(self, n_bits: int, n_rows: int):
        """Reduced tables must be all-0/all-1 free Hamiltonian cycles."""
        gray_codes = make_gray_codes_reduced(n_bits)
        assert gray_codes.shape == (n_rows, n_bits)
        assert len(np.unique(gray_codes, axis=0)) == n_rows
        assert 0 not in gray_codes.sum(axis=1)
        assert n_bits not in gray_codes.sum(axis=1)
        for i in range(len(gray_codes)):
            assert np.sum(np.abs(gray_codes[(i + 1) % len(gray_codes)] - gray_codes[i])) == 1

    @pytest.mark.parametrize("n_bits", [2, 7])
    def test_reduced_gray_codes_unsupported(self, n_bits: int):
        with pytest.raises(ValueError, match="n_bits"):
            make_gray_codes_reduced(n_bits)

    def test_hamiltonian_order_permutes_rows(self):
        gray_codes = make_gray_codes(3)
        ordered = _hamiltonian_order(gray_codes)
        assert sorted(map(tuple, ordered)) == sorted(map(tuple, gray_codes))
        for i in range(len(ordered) - 1):
            assert np.sum(np.abs(ordered[i + 1] - ordered[i])) == 1

    def test_hamiltonian_order_without_cycle(self):
        gray_codes = np.array([[0.0, 0.0], [1.0, 1.0]])  # no two rows are adjacent
        with pytest.raises(RuntimeError, match="Hamiltonian"):
            _hamiltonian_order(gray_codes)

    def test_tof_gray_codes_structure(self):
        for n_captures in (3, 4, 5):
            n_segments = 32 if n_captures == 5 else len(make_gray_codes_reduced(n_captures))
            codes = make_tof_gray_codes(n_captures, 1000)
            assert codes.shape[0] == n_captures
            assert codes.shape[1] % n_segments == 0
            assert codes.shape[1] >= 1000
            assert codes.min() >= 0.0 and codes.max() <= 1.0
            # every segment ramps linearly, i.e. its differences are constant
            points_per_segment = codes.shape[1] // n_segments
            for i in range(0, codes.shape[1], points_per_segment):
                diff = np.diff(codes[:, i : i + points_per_segment], axis=1)
                assert np.allclose(diff, diff[:, :1])


# ──────────────────────────── Hilbert curves ───────────────────────────────


class TestHilbertCurves:
    @pytest.mark.parametrize("order", [0, 1, 2, 3])
    def test_hilbert_2d_shape(self, order: int):
        assert _hilbert_2d(order).shape == (2, 4**order)

    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_hilbert_3d_shape(self, order: int):
        assert _hilbert_3d(order).shape == (3, 8**order)

    @pytest.mark.parametrize("hilbert", [_hilbert_2d, _hilbert_3d])
    def test_hilbert_curve_properties(self, hilbert):
        X = hilbert(2)
        assert X.min() >= -0.5 and X.max() <= 0.5
        assert len(set(map(tuple, X.T))) == X.shape[1]  # visits every point once
        steps = np.abs(np.diff(X, axis=1)).sum(axis=0)
        assert np.allclose(steps, steps[0], atol=1e-9)  # constant step length
        # a Hilbert curve moves along exactly one axis at a time
        assert np.all(np.sum(np.abs(np.diff(X, axis=1)) > 0, axis=0) == 1)

    @pytest.mark.parametrize("hilbert_delta", [0.0, 0.25, 0.4])
    def test_normalize_and_expand(self, hilbert_delta: float):
        X = _normalize_and_expand(_hilbert_2d(2), hilbert_delta, 200)
        assert X.shape[0] == 2
        for row in X:
            assert row.min() == pytest.approx(hilbert_delta, abs=1e-9)
            assert row.max() == pytest.approx(1 - hilbert_delta, abs=1e-9)
        assert 0.5 * 200 <= X.shape[1] <= 2 * 200
        assert _normalize_and_expand(_hilbert_2d(2), hilbert_delta, 400).shape[1] > X.shape[1]

    def test_perm_matrix_to_codes(self):
        curve_points = np.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]])
        permutation_matrix = np.array([[0, 1, 2, -2], [3, 0, 1, -3]])
        codes = _perm_matrix_to_codes(permutation_matrix, curve_points)
        assert codes.shape == (2, 4 * curve_points.shape[1])
        # 0/1 are constants, 2/3 select a curve coordinate, negatives reverse it
        assert np.allclose(codes[0, 0:3], 0.0)
        assert np.allclose(codes[0, 3:6], 1.0)
        assert np.allclose(codes[0, 6:9], curve_points[0])
        assert np.allclose(codes[0, 9:12], curve_points[0][::-1])
        assert np.allclose(codes[1, 0:3], curve_points[1])
        assert np.allclose(codes[1, 9:12], curve_points[1][::-1])

    @pytest.mark.parametrize(("n_captures", "dim"), [(3, 2), (4, 3), (6, 2)])
    def test_tof_hilbert_codes_unsupported(self, n_captures: int, dim: int):
        with pytest.raises(ValueError, match="Unsupported"):
            make_tof_hilbert_codes(n_captures, dim, 1, 0.25, 1001)

    @pytest.mark.parametrize(
        ("n_captures", "dim", "n_segments", "n_sub_segments"), [(4, 2, 12, 3), (5, 2, 60, 3), (5, 3, 20, 7)]
    )
    def test_tof_hilbert_codes_structure(self, n_captures: int, dim: int, n_segments: int, n_sub_segments: int):
        codes = make_tof_hilbert_codes(n_captures, dim, 1, 0.25, 1001)
        assert codes.shape[0] == n_captures
        assert codes.shape[1] % n_segments == 0
        assert codes.shape[1] >= 1001
        assert codes.min() >= 0.0 and codes.max() <= 1.0
        # piecewise linear: slopes change only at sub-segment junctions
        diff = np.diff(codes, axis=1)
        for row in diff:
            assert np.count_nonzero(np.abs(np.diff(row)) > 1e-9) <= n_segments * (n_sub_segments + 1)


# ───────────────────────────── Coding schemes ──────────────────────────────


class TestCodingSchemes:
    @pytest.mark.parametrize("n_captures", [3, 4, 5])
    def test_conv_sin_normalization(self, n_captures: int):
        mod, ref = make_conv_sinusoidal_codes(n_captures, 1000)
        for i in range(n_captures):
            assert mod[i].sum() == pytest.approx(1.0, abs=1e-12)
            assert mod[i].min() >= 0.0
            assert ref[i].min() >= 0.0
            assert ref[i].max() <= 1.0
        # the first tap samples the cosine exactly at its extremes
        assert ref[0].max() == pytest.approx(1.0, abs=1e-12)
        assert ref[0].min() == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("n_captures", [3, 4, 5])
    def test_delta_sin_is_delta(self, n_captures: int):
        mod, ref = make_delta_sinusoidal_codes(n_captures, 500)
        for i in range(n_captures):
            assert mod[i, 0] == pytest.approx(1.0, abs=1e-12)
            assert np.allclose(mod[i, 1:], 0.0)
            assert ref[i].max() <= 1.0 and ref[i].min() >= 0.0
        assert ref[0].max() == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("n_captures", [3, 4])
    def test_conv_square_binary_ref(self, n_captures: int):
        mod, ref = make_conv_square_codes(n_captures, 500)
        assert set(np.unique(ref)) == {0.0, 1.0}
        for i in range(n_captures):
            assert mod[i].sum() == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("factory", [make_single_ramp_codes, make_double_ramp_codes])
    def test_ramp_shapes_and_normalization(self, factory):
        mod, ref = factory(500)
        assert mod.shape == ref.shape == (3, 2 * 500 - 1)
        assert mod.min() >= 0.0
        assert np.allclose(ref[2], 1.0)
        assert np.allclose(mod[2], 0.0)
        for i in (0, 1):
            assert mod[i].sum() == pytest.approx(1.0, abs=1e-12)
            assert ref[i].max() <= 1.0

    @pytest.mark.parametrize("n_captures", [4, 5, 7])
    def test_multi_freq_sinusoidal_codes(self, n_captures: int):
        freq_vec, shifts_vec = (np.asarray(v) for v in _MULT_FREQ_LAYOUTS[n_captures])
        mod, ref = make_multi_freq_sinusoidal_codes(freq_vec, shifts_vec, 1000)
        assert mod.shape == ref.shape == (n_captures, 1000)
        for i in range(n_captures):
            assert mod[i].sum() == pytest.approx(1.0, abs=1e-12)
            assert mod[i].min() >= 0.0
            assert ref[i].max() <= 1.0
            assert ref[i].min() >= 0.0
        # the zero-shift tap reaches the full cosine range
        assert ref[0].max() == pytest.approx(1.0, abs=1e-12)
        # a tap with a higher frequency multiplier crosses its mean more often
        crossings = [np.count_nonzero(np.diff(np.sign(m - m.mean()))) for m in mod]
        assert crossings[3] > crossings[0]

    @pytest.mark.parametrize(
        ("scheme", "n_captures", "expected"),
        [
            ("convSin", 4, 1001),
            ("deltaSin", 3, 1001),
            ("convSquare", 4, 1001),
            ("singleRamp", 3, 2001),
            ("doubleRamp", 3, 2001),
            ("deltaHilbertDimOne", 3, 1002),
            ("deltaHilbertDimOne", 5, 1024),
            ("deltaHilbertDimTwo", 4, 1008),
            ("deltaHilbertDimThree", 5, 1120),
        ],
    )
    def test_code_lengths(self, scheme: CodingScheme, n_captures: int, expected: int):
        """Returned codes are whole segments long, so they may exceed n_depths."""
        mod, ref = _codes(scheme, n_captures, 1001)
        assert mod.shape == ref.shape == (n_captures, expected)
        assert mod.min() >= 0.0 and mod.max() <= 1.0

    @pytest.mark.parametrize("n_captures", [4, 5, 7])
    def test_mult_freq_code_length(self, n_captures: int):
        mod, ref = _codes("multFreqSin", n_captures, 1001)
        assert mod.shape == ref.shape == (n_captures, 1001)

    @pytest.mark.parametrize(
        ("scheme", "n_captures"), [("singleRamp", 4), ("singleRamp", 5), ("doubleRamp", 4), ("doubleRamp", 5)]
    )
    def test_ramp_requires_k3(self, scheme: CodingScheme, n_captures: int):
        with pytest.raises(ValueError, match="requires n_captures=3"):
            make_coding_functions(scheme, n_captures, 1001)

    def test_unknown_scheme(self):
        with pytest.raises(ValueError, match="Unknown coding scheme"):
            make_coding_functions("convSin_", 4, 1001)  # type: ignore[arg-type]

    def test_mult_freq_requires_vectors(self):
        with pytest.raises(ValueError, match="requires freq_vec and shifts_vec"):
            make_coding_functions("multFreqSin", 5, 1001)

    def test_mult_freq_vector_lengths_must_match(self):
        with pytest.raises(ValueError, match="equal length"):
            make_coding_functions(
                "multFreqSin", 2, 1001, freq_vec=np.array([1.0, 2.0]), shifts_vec=np.array([0.0, 1.0, 2.0])
            )

    def test_mult_freq_k_must_match_vectors(self):
        with pytest.raises(ValueError, match="derives n_captures from freq_vec"):
            make_coding_functions(
                "multFreqSin", 5, 1001, freq_vec=np.array([1.0, 1.0, 2.0, 2.0]), shifts_vec=np.zeros(4)
            )

    @pytest.mark.parametrize("n_captures", [3, 6, 8])
    def test_mult_freq_k_layout(self, n_captures: int):
        with pytest.raises(ValueError, match=r"expects n_captures=4.*odd n_captures>=5"):
            make_coding_functions(
                "multFreqSin", n_captures, 1001, freq_vec=np.ones(n_captures), shifts_vec=np.zeros(n_captures)
            )

    @pytest.mark.parametrize("n_captures", [2, 7])
    def test_hilbert_dim_one_requires_known_k(self, n_captures: int):
        with pytest.raises(ValueError, match="n_bits"):
            _codes("deltaHilbertDimOne", n_captures, 1001)

    @pytest.mark.parametrize(("scheme", "n_captures"), [("deltaHilbertDimTwo", 3), ("deltaHilbertDimThree", 4)])
    def test_hilbert_requires_known_k_dim(self, scheme: CodingScheme, n_captures: int):
        with pytest.raises(ValueError, match="Unsupported n_captures="):
            _codes(scheme, n_captures, 1001)

    def test_unambiguous_range(self):
        for scheme in ("convSin", "deltaSin", "convSquare", "deltaHilbertDimOne", "deltaHilbertDimThree"):
            assert unambiguous_range(scheme, _FREQ) == pytest.approx(_D_MAX)  # type: ignore[arg-type]

    def test_ramp_range_is_half(self):
        """Ramp codes span two periods, so their usable range is halved."""
        assert unambiguous_range("singleRamp", _FREQ) == pytest.approx(0.5 * _D_MAX)
        assert unambiguous_range("doubleRamp", _FREQ) == pytest.approx(0.5 * unambiguous_range("convSin", _FREQ))


# ─────────────────────── Segment distance (decode helper) ──────────────────


def _naive_segment_distance_sq(start: npt.NDArray, end: npt.NDArray, points: npt.NDArray) -> npt.NDArray:
    """Reference implementation of ComputeSegmentDistance.m, in squared units."""
    seg_len_sq = np.sum((start - end) ** 2)
    if seg_len_sq == 0:
        return np.sum((points - start[:, None]) ** 2, axis=0)
    t = np.sum((points - start[:, None]) * (end - start)[:, None], axis=0) / seg_len_sq
    t = np.clip(t, 0, 1)
    closest = start[:, None] + t[None, :] * (end - start)[:, None]
    return np.sum((points - closest) ** 2, axis=0)


class TestSegmentDistance:
    @pytest.mark.parametrize("dim", [2, 3])
    def test_matches_naive_reference(self, dim: int):
        rng = np.random.default_rng(0)
        start, end = rng.normal(size=dim), rng.normal(size=dim)
        points = rng.normal(size=(dim, 500))
        assert _compute_segment_distance(start, end, points) == pytest.approx(
            _naive_segment_distance_sq(start, end, points)
        )

    def test_zero_on_segment(self):
        start, end = np.array([0.0, 0.0]), np.array([1.0, 1.0])
        points = np.array([[0.5, 0.25], [0.5, 0.25]])  # both points lie on the segment
        assert _compute_segment_distance(start, end, points).tolist() == [0.0, 0.0]

    def test_degenerate_segment(self):
        start = np.array([1.0, 2.0])
        points = np.array([[1.0, 0.0], [2.0, 2.0]]).T
        assert _compute_segment_distance(start, start.copy(), points) == pytest.approx(
            np.sum((points - start[:, None]) ** 2, axis=0)
        )

    def test_clamped_beyond_endpoints(self):
        start, end = np.array([0.0, 0.0]), np.array([1.0, 0.0])
        points = np.array([[3.0], [0.0]])
        assert _compute_segment_distance(start, end, points) == pytest.approx([4.0])  # distance to end

    def test_precomputed_norm_fast_path(self):
        rng = np.random.default_rng(1)
        start, end = rng.normal(size=3), rng.normal(size=3)
        points = rng.normal(size=(3, 100))
        default = _compute_segment_distance(start, end, points)
        precomputed = _compute_segment_distance(start, end, points, np.sum(points**2, axis=0))
        assert default == pytest.approx(precomputed)


# ───────────────────────────– Correlation function ─────────────────────────


class TestCorrelation:
    def test_shape(self):
        mod, ref = make_conv_sinusoidal_codes(4, 500)
        assert compute_correlation_function(mod[0], ref[0], 1e-9).shape == (500,)
        assert compute_correlation_function(mod[0], ref[0], 1e-9, n_depths=100).shape == (100,)

    def test_delta_mod_reproduces_reference_code(self):
        """An impulse modulation correlates to the reference code itself."""
        _, ref = make_delta_sinusoidal_codes(3, 200)
        mod = np.zeros(200)
        mod[0] = 1.0
        time_resolution = 1e-9
        assert compute_correlation_function(mod, ref[0], time_resolution) == pytest.approx(
            200 * ref[0] * time_resolution
        )

    def test_periodic_continuation(self):
        """More bins than the period repeat the correlation function."""
        mod, ref = make_conv_sinusoidal_codes(4, 300)
        one_period = compute_correlation_function(mod[0], ref[0], 1e-9)
        two_periods = compute_correlation_function(mod[0], ref[0], 1e-9, n_depths=600)
        assert two_periods[:300] == pytest.approx(one_period)
        assert two_periods[300:] == pytest.approx(one_period)
        # the sample at max_depth equals the sample at depth 0
        with_wrap = compute_correlation_function(mod[0], ref[0], 1e-9, n_depths=301)
        assert with_wrap[-1] == pytest.approx(with_wrap[0])

    def test_non_negative_for_sinusoidal_codes(self):
        mod, ref = make_conv_sinusoidal_codes(4, 500)
        assert np.all(compute_correlation_function(mod[0], ref[0], 1e-9) >= 0)

    def test_mismatched_code_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            compute_correlation_function(np.ones(4), np.ones(5), 1e-9)

    @pytest.mark.parametrize("n_depths", [0, -5])
    def test_invalid_n_depths(self, n_depths: int):
        with pytest.raises(ValueError, match="positive number of depth bins"):
            compute_correlation_function(np.ones(4), np.ones(4), 1e-9, n_depths=n_depths)


# ───────────────────────– Measurement simulation ───────────────────────────


class TestSimulation:
    def test_output_shape_1d_and_2d(self):
        mod, ref = _codes("convSin", 4, 501)
        depths_1d = np.array([0.3, 0.7, 1.2])
        assert simulate_measurements(depths_1d, np.full(3, 0.5), mod, ref, _D_MAX).shape == (4, 3)
        depths_2d = np.full((8, 6), 0.5)
        assert simulate_measurements(depths_2d, np.full((8, 6), 0.5), mod, ref, _D_MAX).shape == (4, 8, 6)

    @pytest.mark.parametrize("kwargs", [{"exposure_time": 2.5}, {"light_power": 3.0}, {"albedo": None}])
    def test_linear_terms(self, kwargs: dict):
        mod, ref = _codes("convSin", 4, 501)
        depths = np.array([0.3, 0.7])
        albedo = np.array([0.4, 0.9])
        baseline = simulate_measurements(depths, albedo, mod, ref, _D_MAX)
        if "albedo" in kwargs:
            scaled = simulate_measurements(depths, albedo * 2, mod, ref, _D_MAX)
            assert scaled == pytest.approx(2 * baseline)
        else:
            scaled = simulate_measurements(depths, albedo, mod, ref, _D_MAX, **kwargs)
            assert scaled == pytest.approx(list(kwargs.values())[0] * baseline)

    def test_inverse_square_falloff_and_phase_wrap(self):
        """A depth one period deeper has the same phase but a 1/d**2 falloff."""
        mod, ref = _codes("convSin", 4, 1001)
        depths = np.array([0.25, 0.25 + _D_MAX, 0.5, 0.5 + _D_MAX])
        measurements = simulate_measurements(depths, np.full(4, 0.8), mod, ref, _D_MAX)
        for i in (0, 2):
            assert measurements[:, i + 1] / measurements[:, i] == pytest.approx(
                (depths[i] / depths[i + 1]) ** 2, rel=1e-6
            )

    def test_ambient_term_is_depth_independent(self):
        mod, ref = _codes("convSin", 4, 501)
        depths = np.array([0.3, 1.1])
        ambient_only = simulate_measurements(
            depths, np.full(2, 0.5), mod, ref, _D_MAX, ambient_power=2.0, light_power=0.0
        )
        assert ambient_only[:, 0] == pytest.approx(ambient_only[:, 1])
        assert ambient_only == pytest.approx(
            2 * simulate_measurements(depths, np.full(2, 0.5), mod, ref, _D_MAX, ambient_power=1.0, light_power=0.0)
        )  # noqa: E501

    def test_matches_analytic_model(self):
        mod, ref = _codes("deltaSin", 4, 501)
        depths = np.array([0.4, 1.0])
        albedos = np.array([0.7, 0.3])
        exposure_time, ambient_power, light_power = 0.5, 2.0, 3.0
        measurements = simulate_measurements(
            depths,
            albedos,
            mod,
            ref,
            _D_MAX,
            exposure_time=exposure_time,
            ambient_power=ambient_power,
            light_power=light_power,
        )
        n = mod.shape[1]
        time_resolution = 2 * _D_MAX / (n * _C)
        period = n * time_resolution
        distances = np.linspace(0, _D_MAX, n, endpoint=False)
        for i in range(mod.shape[0]):
            correlation = compute_correlation_function(mod[i], ref[i], time_resolution, n_depths=n)
            kappa = ref[i].sum() * time_resolution
            expected = (exposure_time / period) * (
                light_power * (albedos / depths**2) * PchipInterpolator(distances, correlation)(depths % _D_MAX)
                + ambient_power * kappa * albedos
            )
            assert measurements[i] == pytest.approx(expected)

    def test_non_negative_without_ambient(self):
        mod, ref = _codes("convSin", 4, 501)
        measurements = simulate_measurements(np.array([0.2, 0.9]), np.full(2, 0.5), mod, ref, _D_MAX)
        assert np.all(measurements >= 0)

    def test_zero_depth_is_finite(self):
        mod, ref = _codes("convSin", 4, 501)
        measurements = simulate_measurements(np.array([0.0, 1e-9]), np.full(2, 0.5), mod, ref, _D_MAX)
        assert np.all(np.isfinite(measurements))

    def test_mismatched_code_lengths(self):
        mod, _ = _codes("convSin", 4, 501)
        _, ref = _codes("convSin", 4, 502)
        with pytest.raises(ValueError, match="same length"):
            simulate_measurements(np.array([0.5]), np.array([0.5]), mod, ref, _D_MAX)


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


class TestDecoding:
    @pytest.mark.parametrize(("scheme", "n_captures", "atol"), _ROUNDTRIP_SCHEMES)
    def test_roundtrip_over_full_range(self, scheme: CodingScheme, n_captures: int, atol: float):
        depths = np.linspace(0.02, _D_MAX - 0.02, 12)
        decoded, true = _roundtrip(scheme, depths, n_captures=n_captures)
        assert np.abs(decoded - true).max() < atol

    @pytest.mark.parametrize("n_captures", [3, 4])
    def test_hilbert_dim_one_roundtrip(self, n_captures: int):
        depths = np.linspace(0.02, _D_MAX - 0.02, 40)
        decoded, true = _roundtrip("deltaHilbertDimOne", depths, n_captures=n_captures)
        assert np.abs(decoded - true).max() < 1e-3

    def test_hilbert_dim_one_five_captures_within_one_interval(self):
        """n_captures=5 leaves a few ambiguous pixels, but never off by more than one interval."""
        interval_width = _D_MAX / len(make_max_min_run_length_gray_codes())
        depths = np.linspace(0.02, _D_MAX - 0.02, 200)
        decoded, true = _roundtrip("deltaHilbertDimOne", depths, n_captures=5)
        assert np.abs(decoded - true).max() < 1.05 * interval_width

    @pytest.mark.parametrize(
        ("scheme", "n_captures", "atol"),
        [("deltaHilbertDimTwo", 4, 5e-3), ("deltaHilbertDimTwo", 5, 1e-3), ("deltaHilbertDimThree", 5, 1e-3)],
    )
    def test_hilbert_higher_dim_robust_accuracy(self, scheme: CodingScheme, n_captures: int, atol: float):
        """Higher-dimensional Hilbert schemes are accurate for the vast majority of depths.

        The segment classifier normalizes each channel by its own range while its
        endpoints sit on the coarse Hilbert grid, mirroring the reference
        implementation. Pixels sitting exactly between two segments can therefore
        be misassigned, which is why this checks the error distribution instead of
        the worst case -- see :func:`test_hilbert_dim_two_outlier_rate`.
        """
        depths = np.linspace(0.02, _D_MAX - 0.02, 128)
        decoded, true = _roundtrip(scheme, depths, n_captures=n_captures)
        err = np.abs(decoded - true)
        assert np.median(err) < atol
        assert np.percentile(err, 90) < atol

    @pytest.mark.parametrize(("n_captures", "max_outliers"), [(4, 0), (5, 5)])
    def test_hilbert_dim_two_outlier_rate(self, n_captures: int, max_outliers: int):
        """Pins the known misclassification rate of the dim=2 segment classifier."""
        depths = np.linspace(0.02, _D_MAX - 0.02, 128)
        decoded, true = _roundtrip("deltaHilbertDimTwo", depths, n_captures=n_captures)
        outliers = np.abs(decoded - true) > 0.01
        assert outliers.sum() <= max_outliers

    def test_hilbert_dim_one_interval_indices_are_assigned(self):
        """Regression: interval 0 used to be mistaken for an unassigned pixel."""
        n_captures = 3
        depths = np.linspace(0.02, _D_MAX - 0.02, 200)
        mod, ref = _codes("deltaHilbertDimOne", n_captures, 1001)
        measurements = simulate_measurements(depths, np.full(depths.shape, 0.8), mod, ref, _D_MAX)
        interval_indices, decoded = decode_hilbert(measurements, _FREQ, dim=1)
        n_intervals = len(make_gray_codes_reduced(n_captures))
        assert np.all(interval_indices >= 0)  # nothing left unassigned
        assert np.all(interval_indices < n_intervals)
        assert set(np.unique(interval_indices)) == set(range(n_intervals))  # interval 0 included
        # the interval index must be consistent with the returned depth
        assert np.all(np.abs(interval_indices - decoded / _D_MAX * n_intervals) <= 1)
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
        self, scheme: CodingScheme, n_captures: int, hilbert_order: int, hilbert_delta: float
    ):
        depths = np.linspace(0.05, _D_MAX - 0.05, 10)
        decoded, true = _roundtrip(
            scheme, depths, n_captures=n_captures, hilbert_order=hilbert_order, hilbert_delta=hilbert_delta
        )
        assert np.abs(decoded - true).max() < 5e-3

    @pytest.mark.parametrize("scheme", ["singleRamp", "doubleRamp"])
    def test_ramp_roundtrip_within_half_range(self, scheme: CodingScheme):
        limit = unambiguous_range(scheme, _FREQ)
        depths = np.linspace(0.01, limit - 0.005, 10)
        decoded, true = _roundtrip(scheme, depths, n_captures=3)
        assert np.abs(decoded - true).max() < 5e-3

    @pytest.mark.parametrize("scheme", ["singleRamp", "doubleRamp"])
    def test_ramp_folds_beyond_half_range(self, scheme: CodingScheme):
        """Deeper than c/4f, ramp schemes mirror depth into the reported range."""
        depths = np.array([0.7, 0.9, 1.2])
        assert np.all(depths > unambiguous_range(scheme, _FREQ))
        decoded, true = _roundtrip(scheme, depths, n_captures=3)
        assert decoded == pytest.approx(_D_MAX - true, abs=5e-3)

    @pytest.mark.parametrize(
        ("scheme", "n_captures"),
        [("convSin", 4), ("deltaSin", 4), ("convSquare", 4), ("deltaHilbertDimOne", 4), ("deltaHilbertDimTwo", 4)],
    )
    def test_intensity_scale_invariance(self, scheme: CodingScheme, n_captures: int):
        """Decoding only uses relative intensities, so scaling is a no-op."""
        mod, ref = _codes(scheme, n_captures, 1001)
        depths = np.array([0.3, 0.8])
        measurements = simulate_measurements(depths, np.full(2, 0.7), mod, ref, _D_MAX)
        assert decode(scheme, measurements * 1e3, _FREQ) == pytest.approx(decode(scheme, measurements, _FREQ))

    def test_batch_matches_per_pixel(self):
        depths = np.array([0.2, 0.5, 0.9])
        batch, true = _roundtrip("deltaHilbertDimTwo", depths, n_captures=4)
        assert np.abs(batch - true).max() < 5e-3
        for i, depth in enumerate(depths):
            single, _ = _roundtrip("deltaHilbertDimTwo", depth, n_captures=4)
            assert batch[i] == pytest.approx(single[0], abs=1e-9)

    def test_mult_freq_k4_wide_range_not_supported(self):
        """The single low-frequency tap cannot widen the range (see docstring)."""
        freq_vec = np.array([0.5, 2.0, 2.0, 2.0])
        shifts_vec = np.array([0.0, 0.0, 2 * np.pi / 3, 4 * np.pi / 3])
        mod, ref = make_coding_functions("multFreqSin", 4, 1001, freq_vec=freq_vec, shifts_vec=shifts_vec)
        depths = np.array([0.3, 0.5])
        measurements = simulate_measurements(depths, np.full(2, 0.8), mod, ref, _D_MAX)
        with pytest.raises(NotImplementedError, match="not supported"):
            decode("multFreqSin", measurements, _FREQ, freq_vec=freq_vec, shifts_vec=shifts_vec)

    def test_mult_freq_direct_matches_dispatch(self):
        n_captures = 5
        freq_vec, shifts_vec = _tap_vectors("multFreqSin", n_captures)
        depths = np.array([0.2, 1.0])
        mod, ref = _codes("multFreqSin", n_captures, 1001)
        measurements = simulate_measurements(depths, np.full(2, 0.8), mod, ref, _D_MAX)
        direct = decode_mult_freq_sinusoid(measurements, _FREQ, freq_vec, shifts_vec)
        dispatched = decode("multFreqSin", measurements, _FREQ, freq_vec=freq_vec, shifts_vec=shifts_vec)
        assert direct == pytest.approx(dispatched)
        assert direct == pytest.approx(depths, abs=1e-3)

    def test_mult_freq_requires_vectors_to_decode(self):
        with pytest.raises(ValueError, match="requires freq_vec and shifts_vec"):
            decode("multFreqSin", np.ones((4, 3)), _FREQ)

    def test_mult_freq_tap_vectors_must_match_data(self):
        measurements = np.ones((4, 3))
        with pytest.raises(ValueError, match="must have shape"):
            decode("multFreqSin", measurements, _FREQ, freq_vec=np.ones(5), shifts_vec=np.ones(5))

    def test_mult_freq_taps_must_share_frequency(self):
        """Taps of one group have to carry the same frequency multiplier."""
        measurements = np.ones((5, 3))
        freq_vec = np.array([1.0, 1.0, 2.0, 2.0, 2.0])  # first group spans two frequencies
        with pytest.raises(ValueError, match="assumed to share a single frequency"):
            decode("multFreqSin", measurements, _FREQ, freq_vec=freq_vec, shifts_vec=np.zeros(5))

    @pytest.mark.parametrize(
        ("scheme", "n_captures"),
        [("convSin", 2), ("singleRamp", 4), ("deltaHilbertDimThree", 4), ("multFreqSin", 3), ("multFreqSin", 6)],
    )
    def test_unsupported_capture_counts(self, scheme: CodingScheme, n_captures: int):
        with pytest.raises(ValueError):
            decode(scheme, np.ones((n_captures, 3)), _FREQ)

    def test_unknown_scheme(self):
        with pytest.raises(NotImplementedError, match="No decoder implemented"):
            decode("nope", np.ones((3, 2)), _FREQ)  # type: ignore[arg-type]

    @pytest.mark.parametrize("measurements", [np.ones(4), np.ones((3, 2, 2)), [[1.0, 2.0]]])
    def test_invalid_data_shape(self, measurements):
        with pytest.raises(ValueError, match="Expected measurements of shape"):
            decode("convSin", measurements, _FREQ)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("scheme", "n_captures"),
        [("convSin", 3), ("convSin", 5), ("singleRamp", 3), ("deltaHilbertDimOne", 4), ("multFreqSin", 5)],
    )
    def test_validate_captures_accepts_supported(self, scheme: CodingScheme, n_captures: int):
        _validate_captures(scheme, n_captures)


# ──────────────────────────────── CLI command ──────────────────────────────


def _write_dataset(root: Path, depths: list[float], *, size: int = 32) -> Path:
    """Write a minimal Blender-free dataset (images, depth and transforms.json)."""
    from PIL import Image

    root.mkdir(parents=True, exist_ok=True)
    source = next(iter(sorted((Path(__file__).parent / "test_files" / "lego-gt").glob("*.png"))))
    image = np.asarray(Image.open(source).convert("RGB").resize((size, size)))
    frames = []
    for i, depth in enumerate(depths):
        iio.imwrite(root / f"{i:04}.png", image)
        np.save(root / f"depth_{i:04}.npy", np.full((size, size), depth, np.float32)[None])
        frames.append(
            {
                "file_path": f"{i:04}.png",
                "transform_matrix": np.eye(4).tolist(),
                "depth_file_path": f"depth_{i:04}.npy",
            }
        )
    (root / "transforms.json").write_text(
        json.dumps(
            {
                "camera_model": "OPENCV",
                "fl_x": float(size),
                "fl_y": float(size),
                "cx": size / 2,
                "cy": size / 2,
                "h": size,
                "w": size,
                "frames": frames,
            }
        )
    )
    return root


class TestItfCli:
    def test_writes_measurements_and_provenance(self, tmp_path: Path):
        from visionsim.cli import emulate
        from visionsim.dataset.models import Metadata

        input_dir = _write_dataset(tmp_path / "in", [0.4, 0.9])
        output_dir = tmp_path / "out"
        emulate.itof(input_dir=input_dir, output_dir=output_dir, scheme="convSin", n_captures=4, num_bins=200)

        assert sorted(p.name for p in output_dir.glob("*.npy")) == ["0000.npy", "0001.npy"]
        measurements = np.load(output_dir / "0000.npy")
        assert measurements.shape == (4, 32, 32)
        assert measurements.dtype == np.float32
        assert np.all(np.isfinite(measurements))

        metadata = Metadata.load(output_dir / "transforms.json")
        assert len(metadata.frames) == 2
        assert metadata.itof_scheme == "convSin"
        assert metadata.itof_captures == 4
        assert metadata.itof_freq_hz == pytest.approx(_FREQ)
        assert metadata.itof_num_bins == 200
        assert metadata.itof_effective_range_m == pytest.approx(_D_MAX)
        assert metadata.itof_unambiguous_range_m == pytest.approx(_D_MAX)

    def test_preview_taps(self, tmp_path: Path):
        from visionsim.cli import emulate

        input_dir = _write_dataset(tmp_path / "in", [0.4])
        output_dir = tmp_path / "out"
        emulate.itof(
            input_dir=input_dir, output_dir=output_dir, scheme="convSin", n_captures=3, num_bins=200, preview=True
        )
        for tap in range(3):
            assert (output_dir / "preview" / f"tap_{tap}" / "0000.png").is_file()

    def test_force_flag(self, tmp_path: Path):
        from visionsim.cli import emulate

        input_dir = _write_dataset(tmp_path / "in", [0.4])
        output_dir = tmp_path / "out"
        emulate.itof(input_dir=input_dir, output_dir=output_dir, scheme="convSin", n_captures=3, num_bins=200)
        with pytest.raises(FileExistsError, match="already exists"):
            emulate.itof(input_dir=input_dir, output_dir=output_dir, scheme="convSin", n_captures=3, num_bins=200)
        emulate.itof(
            input_dir=input_dir, output_dir=output_dir, scheme="convSin", n_captures=3, num_bins=200, force=True
        )

    def test_warns_when_out_of_range(self, tmp_path: Path, caplog):
        """Ramp schemes fold beyond c/4f, so the CLI has to warn about it."""
        from visionsim.cli import emulate

        input_dir = _write_dataset(tmp_path / "in", [0.9])
        with caplog.at_level(logging.WARNING, logger="rich"):
            emulate.itof(input_dir=input_dir, output_dir=tmp_path / "out", scheme="singleRamp", num_bins=200)
        assert any("unambiguous range" in record.getMessage() for record in caplog.records)

    def test_hilbert_parameters_are_forwarded(self, tmp_path: Path):
        from visionsim.cli import emulate
        from visionsim.dataset.models import Metadata

        input_dir = _write_dataset(tmp_path / "in", [0.4])
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
        metadata = Metadata.load(output_dir / "transforms.json")
        assert metadata.itof_hilbert_order == 2
        assert metadata.itof_hilbert_delta == pytest.approx(0.1)

    def test_shape_mismatch(self, tmp_path: Path):
        from visionsim.cli import emulate

        input_dir = _write_dataset(tmp_path / "in", [0.4])
        np.save(input_dir / "depth_0000.npy", np.full((16, 16), 0.4, np.float32)[None])
        with pytest.raises(ValueError, match="Shape mismatch"):
            emulate.itof(input_dir=input_dir, output_dir=tmp_path / "out", num_bins=100)

    def test_input_output_must_differ(self, tmp_path: Path):
        from visionsim.cli import emulate

        input_dir = _write_dataset(tmp_path / "in", [0.4])
        with pytest.raises(RuntimeError, match="cannot be the same"):
            emulate.itof(input_dir=input_dir, output_dir=input_dir, num_bins=100, force=True)
