"""Tests for iToF waveform generation and coding schemes."""

from __future__ import annotations

import numpy as np
import pytest

from tests.emulate.itof import D_MAX, FREQ, MULT_FREQ_LAYOUTS, codes
from visionsim.emulate.itof import CodingScheme, make_coding_functions, unambiguous_range
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

# ───────────────────────────── Gray code tables ────────────────────────────


def test_gray_codes_shape():
    assert make_gray_codes(3).shape == (8, 3)


def test_gray_codes_binary():
    assert set(np.unique(make_gray_codes(4))) == {0.0, 1.0}


def test_gray_codes_hamming():
    """Adjacent Gray codes differ by exactly one bit."""
    gray_codes = make_gray_codes(3)
    for i in range(len(gray_codes) - 1):
        assert np.sum(np.abs(gray_codes[i] - gray_codes[i + 1])) == 1


def test_max_min_run_length_is_hamiltonian_cycle():
    gray_codes = make_max_min_run_length_gray_codes()
    assert gray_codes.shape == (32, 5)  # 5 captures, 2**5 code words
    assert set(np.unique(gray_codes)) == {0.0, 1.0}
    assert len(np.unique(gray_codes, axis=0)) == len(gray_codes)
    for i in range(len(gray_codes)):
        assert np.sum(np.abs(gray_codes[(i + 1) % len(gray_codes)] - gray_codes[i])) == 1


@pytest.mark.parametrize(("n_bits", "n_rows"), [(3, 6), (4, 12), (5, 30), (6, 60)])
def test_reduced_gray_codes(n_bits: int, n_rows: int):
    """Reduced tables must be all-0/all-1 free Hamiltonian cycles."""
    gray_codes = make_gray_codes_reduced(n_bits)
    assert gray_codes.shape == (n_rows, n_bits)
    assert len(np.unique(gray_codes, axis=0)) == n_rows
    assert 0 not in gray_codes.sum(axis=1)
    assert n_bits not in gray_codes.sum(axis=1)
    for i in range(len(gray_codes)):
        assert np.sum(np.abs(gray_codes[(i + 1) % len(gray_codes)] - gray_codes[i])) == 1


@pytest.mark.parametrize("n_bits", [2, 7])
def test_reduced_gray_codes_unsupported(n_bits: int):
    with pytest.raises(ValueError, match="n_bits"):
        make_gray_codes_reduced(n_bits)


def test_hamiltonian_order_permutes_rows():
    gray_codes = make_gray_codes(3)
    ordered = _hamiltonian_order(gray_codes)
    assert sorted(map(tuple, ordered)) == sorted(map(tuple, gray_codes))
    for i in range(len(ordered) - 1):
        assert np.sum(np.abs(ordered[i + 1] - ordered[i])) == 1


def test_hamiltonian_order_without_cycle():
    gray_codes = np.array([[0.0, 0.0], [1.0, 1.0]])  # no two rows are adjacent
    with pytest.raises(RuntimeError, match="Hamiltonian"):
        _hamiltonian_order(gray_codes)


def test_tof_gray_codes_structure():
    for n_captures in (3, 4, 5):
        n_segments = 32 if n_captures == 5 else len(make_gray_codes_reduced(n_captures))
        codes_ = make_tof_gray_codes(n_captures, 1000)
        assert codes_.shape[0] == n_captures
        assert codes_.shape[1] % n_segments == 0
        assert codes_.shape[1] >= 1000
        assert codes_.min() >= 0.0 and codes_.max() <= 1.0
        # every segment ramps linearly, i.e. its differences are constant
        points_per_segment = codes_.shape[1] // n_segments
        for i in range(0, codes_.shape[1], points_per_segment):
            diff = np.diff(codes_[:, i : i + points_per_segment], axis=1)
            assert np.allclose(diff, diff[:, :1])


# ──────────────────────────── Hilbert curves ───────────────────────────────


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_hilbert_2d_shape(order: int):
    assert _hilbert_2d(order).shape == (2, 4**order)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_hilbert_3d_shape(order: int):
    assert _hilbert_3d(order).shape == (3, 8**order)


@pytest.mark.parametrize("hilbert", [_hilbert_2d, _hilbert_3d])
def test_hilbert_curve_properties(hilbert):
    curve = hilbert(2)
    assert curve.min() >= -0.5 and curve.max() <= 0.5
    assert len(set(map(tuple, curve.T))) == curve.shape[1]  # visits every point once
    steps = np.abs(np.diff(curve, axis=1)).sum(axis=0)
    assert np.allclose(steps, steps[0], atol=1e-9)  # constant step length
    # a Hilbert curve moves along exactly one axis at a time
    assert np.all(np.sum(np.abs(np.diff(curve, axis=1)) > 0, axis=0) == 1)


@pytest.mark.parametrize("hilbert_delta", [0.0, 0.25, 0.4])
def test_normalize_and_expand(hilbert_delta: float):
    expanded = _normalize_and_expand(_hilbert_2d(2), hilbert_delta, 200)
    assert expanded.shape[0] == 2
    for row in expanded:
        assert row.min() == pytest.approx(hilbert_delta, abs=1e-9)
        assert row.max() == pytest.approx(1 - hilbert_delta, abs=1e-9)
    assert 0.5 * 200 <= expanded.shape[1] <= 2 * 200
    assert _normalize_and_expand(_hilbert_2d(2), hilbert_delta, 400).shape[1] > expanded.shape[1]


def test_perm_matrix_to_codes():
    curve_points = np.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]])
    permutation_matrix = np.array([[0, 1, 2, -2], [3, 0, 1, -3]])
    codes_ = _perm_matrix_to_codes(permutation_matrix, curve_points)
    assert codes_.shape == (2, 4 * curve_points.shape[1])
    # 0/1 are constants, 2/3 select a curve coordinate, negatives reverse it
    assert np.allclose(codes_[0, 0:3], 0.0)
    assert np.allclose(codes_[0, 3:6], 1.0)
    assert np.allclose(codes_[0, 6:9], curve_points[0])
    assert np.allclose(codes_[0, 9:12], curve_points[0][::-1])
    assert np.allclose(codes_[1, 0:3], curve_points[1])
    assert np.allclose(codes_[1, 9:12], curve_points[1][::-1])


@pytest.mark.parametrize(("n_captures", "dim"), [(3, 2), (4, 3), (6, 2)])
def test_tof_hilbert_codes_unsupported(n_captures: int, dim: int):
    with pytest.raises(ValueError, match="Unsupported"):
        make_tof_hilbert_codes(n_captures, dim, 1, 0.25, 1001)


@pytest.mark.parametrize(
    ("n_captures", "dim", "n_segments", "n_sub_segments"), [(4, 2, 12, 3), (5, 2, 60, 3), (5, 3, 20, 7)]
)
def test_tof_hilbert_codes_structure(n_captures: int, dim: int, n_segments: int, n_sub_segments: int):
    codes_ = make_tof_hilbert_codes(n_captures, dim, 1, 0.25, 1001)
    assert codes_.shape[0] == n_captures
    assert codes_.shape[1] % n_segments == 0
    assert codes_.shape[1] >= 1001
    assert codes_.min() >= 0.0 and codes_.max() <= 1.0
    # piecewise linear: slopes change only at sub-segment junctions
    diff = np.diff(codes_, axis=1)
    for row in diff:
        assert np.count_nonzero(np.abs(np.diff(row)) > 1e-9) <= n_segments * (n_sub_segments + 1)


# ───────────────────────────── Coding schemes ──────────────────────────────


@pytest.mark.parametrize("n_captures", [3, 4, 5])
def test_conv_sin_normalization(n_captures: int):
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
def test_delta_sin_is_delta(n_captures: int):
    mod, ref = make_delta_sinusoidal_codes(n_captures, 500)
    for i in range(n_captures):
        assert mod[i, 0] == pytest.approx(1.0, abs=1e-12)
        assert np.allclose(mod[i, 1:], 0.0)
        assert ref[i].max() <= 1.0 and ref[i].min() >= 0.0
    assert ref[0].max() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("n_captures", [3, 4])
def test_conv_square_binary_ref(n_captures: int):
    mod, ref = make_conv_square_codes(n_captures, 500)
    assert set(np.unique(ref)) == {0.0, 1.0}
    for i in range(n_captures):
        assert mod[i].sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("factory", [make_single_ramp_codes, make_double_ramp_codes])
def test_ramp_shapes_and_normalization(factory):
    mod, ref = factory(500)
    assert mod.shape == ref.shape == (3, 2 * 500 - 1)
    assert mod.min() >= 0.0
    assert np.allclose(ref[2], 1.0)
    assert np.allclose(mod[2], 0.0)
    for i in (0, 1):
        assert mod[i].sum() == pytest.approx(1.0, abs=1e-12)
        assert ref[i].max() <= 1.0


@pytest.mark.parametrize("n_captures", [4, 5, 7])
def test_multi_freq_sinusoidal_codes(n_captures: int):
    freq_vec, shifts_vec = (np.asarray(v) for v in MULT_FREQ_LAYOUTS[n_captures])
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
def test_code_lengths(scheme: CodingScheme, n_captures: int, expected: int):
    """Returned codes are whole segments long, so they may exceed n_depths."""
    mod, ref = codes(scheme, n_captures, 1001)
    assert mod.shape == ref.shape == (n_captures, expected)
    assert mod.min() >= 0.0 and mod.max() <= 1.0


@pytest.mark.parametrize("n_captures", [4, 5, 7])
def test_mult_freq_code_length(n_captures: int):
    mod, ref = codes("multFreqSin", n_captures, 1001)
    assert mod.shape == ref.shape == (n_captures, 1001)


@pytest.mark.parametrize(
    ("scheme", "n_captures"), [("singleRamp", 4), ("singleRamp", 5), ("doubleRamp", 4), ("doubleRamp", 5)]
)
def test_ramp_requires_k3(scheme: CodingScheme, n_captures: int):
    with pytest.raises(ValueError, match="requires n_captures=3"):
        make_coding_functions(scheme, n_captures, 1001)


def test_unknown_scheme():
    with pytest.raises(ValueError, match="Unknown coding scheme"):
        make_coding_functions("convSin_", 4, 1001)  # type: ignore[arg-type]


def test_mult_freq_requires_vectors():
    with pytest.raises(ValueError, match="requires freq_vec and shifts_vec"):
        make_coding_functions("multFreqSin", 5, 1001)


def test_mult_freq_vector_lengths_must_match():
    with pytest.raises(ValueError, match="equal length"):
        make_coding_functions(
            "multFreqSin", 2, 1001, freq_vec=np.array([1.0, 2.0]), shifts_vec=np.array([0.0, 1.0, 2.0])
        )


def test_mult_freq_k_must_match_vectors():
    with pytest.raises(ValueError, match="derives n_captures from freq_vec"):
        make_coding_functions("multFreqSin", 5, 1001, freq_vec=np.array([1.0, 1.0, 2.0, 2.0]), shifts_vec=np.zeros(4))


@pytest.mark.parametrize("n_captures", [3, 6, 8])
def test_mult_freq_k_layout(n_captures: int):
    with pytest.raises(ValueError, match=r"expects n_captures=4.*odd n_captures>=5"):
        make_coding_functions(
            "multFreqSin", n_captures, 1001, freq_vec=np.ones(n_captures), shifts_vec=np.zeros(n_captures)
        )


@pytest.mark.parametrize("n_captures", [2, 7])
def test_hilbert_dim_one_requires_known_k(n_captures: int):
    with pytest.raises(ValueError, match="n_bits"):
        codes("deltaHilbertDimOne", n_captures, 1001)


@pytest.mark.parametrize(("scheme", "n_captures"), [("deltaHilbertDimTwo", 3), ("deltaHilbertDimThree", 4)])
def test_hilbert_requires_known_k_dim(scheme: CodingScheme, n_captures: int):
    with pytest.raises(ValueError, match="Unsupported n_captures="):
        codes(scheme, n_captures, 1001)


def test_unambiguous_range():
    for scheme in ("convSin", "deltaSin", "convSquare", "deltaHilbertDimOne", "deltaHilbertDimThree"):
        assert unambiguous_range(scheme, FREQ) == pytest.approx(D_MAX)  # type: ignore[arg-type]


def test_ramp_range_is_half():
    """Ramp codes span two periods, so their usable range is halved."""
    assert unambiguous_range("singleRamp", FREQ) == pytest.approx(0.5 * D_MAX)
    assert unambiguous_range("doubleRamp", FREQ) == pytest.approx(0.5 * unambiguous_range("convSin", FREQ))
