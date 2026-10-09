"""Waveform generation and coding schemes for iToF."""

from __future__ import annotations

import math
from typing import Callable, Literal

import numpy as np
import numpy.typing as npt
import scipy.constants


def unambiguous_range(scheme: CodingScheme, freq: float) -> float:
    """Return the unambiguous depth range of a coding scheme, in metres.

    Most schemes recover depth over a single code period, i.e. ``c / (2 f)``.
    The ramp codes (``singleRamp``, ``doubleRamp``) put two periods in one code
    array, so their correlation function is only monotonic over ``c / (4 f)``;
    depths beyond that fold back into the reported range (see
    :func:`visionsim.emulate.itof.decoding.decode_single_ramp`).

    Args:
        scheme: Coding scheme identifier.
        freq: Modulation frequency in Hz.

    Returns:
        Unambiguous depth range in metres.
    """
    if scheme in ("singleRamp", "doubleRamp"):
        return scipy.constants.c / (4 * freq)
    return scipy.constants.c / (2 * freq)


def make_conv_sinusoidal_codes(n_captures: int, n_depths: int) -> tuple[npt.NDArray, npt.NDArray]:
    """Build conventional sinusoidal modulation and reference codes.

    The emitter is modulated with ``0.5 * (1 + cos(2*pi*t/T))`` and every capture
    mixes the returning light with the same waveform shifted by ``2*pi*k/n_captures``,
    which is the classic continuous-wave iToF waveform pair. The modulation code
    is normalized so that its sum over one period is one (unit emitted energy),
    while the reference code spans ``[0, 1]``.

    Args:
        n_captures: Number of captures (phase shifts).
        n_depths: Number of time bins per period.

    Returns:
        Tuple of ``(modulation_codes, reference_codes)``, each of shape ``(n_captures, n_depths)``.

    See Also:
        :func:`visionsim.emulate.itof.decoding.decode` with ``scheme="convSin"``.
    """
    sample_indices: npt.NDArray[np.floating] = np.arange(n_depths, dtype=float)  # (n_depths,)
    phases = 2 * np.pi * np.arange(n_captures)[:, None] / n_captures  # (n_captures, 1)
    modulation = 0.5 + 0.5 * np.cos(2 * np.pi * sample_indices / n_depths)  # (n_depths,) – same for every capture
    modulation_codes = np.broadcast_to(modulation / modulation.sum(), (n_captures, n_depths)).copy()
    reference_codes = 0.5 + 0.5 * np.cos(2 * np.pi * sample_indices / n_depths - phases)  # (n_captures, n_depths)
    return modulation_codes, reference_codes


def make_delta_sinusoidal_codes(n_captures: int, n_depths: int) -> tuple[npt.NDArray, npt.NDArray]:
    """Build delta (impulse) sinusoidal modulation and reference codes.

    Instead of emitting a sinusoid, each capture emits a single short pulse and
    keeps the sinusoidal reference waveform of :func:`make_conv_sinusoidal_codes`.
    Concentrating the emitted energy in time maximizes the signal that reaches
    the sensor, at the cost of a much weaker suppression of ambient light.

    Args:
        n_captures: Number of captures (phase shifts).
        n_depths: Number of time bins per period.

    Returns:
        Tuple of ``(modulation_codes, reference_codes)``, each of shape ``(n_captures, n_depths)``.

    See Also:
        :func:`visionsim.emulate.itof.decoding.decode` with ``scheme="deltaSin"``.
    """
    sample_indices: npt.NDArray[np.floating] = np.arange(n_depths, dtype=float)
    phases = 2 * np.pi * np.arange(n_captures)[:, None] / n_captures  # (n_captures, 1)
    modulation_codes = np.zeros((n_captures, n_depths))
    modulation_codes[:, 0] = 1.0  # impulse at t=0 for every capture
    reference_codes = 0.5 + 0.5 * np.cos(2 * np.pi * sample_indices / n_depths - phases)  # (n_captures, n_depths)
    return modulation_codes, reference_codes


def make_conv_square_codes(n_captures: int, n_depths: int) -> tuple[npt.NDArray, npt.NDArray]:
    """Build conventional square-wave modulation and reference codes.

    The sinusoidal waveforms of :func:`make_conv_sinusoidal_codes` thresholded at
    their mid-level: square waves are cheap to generate and to demodulate in
    hardware, at the cost of harmonics that make the correlation piecewise
    linear rather than sinusoidal.

    Args:
        n_captures: Number of captures (phase shifts).
        n_depths: Number of time bins per period.

    Returns:
        Tuple of ``(modulation_codes, reference_codes)``, each of shape ``(n_captures, n_depths)``.

    See Also:
        :func:`visionsim.emulate.itof.decoding.decode` with ``scheme="convSquare"``.
    """
    sample_indices: npt.NDArray[np.floating] = np.arange(n_depths, dtype=float)
    phases = 2 * np.pi * np.arange(n_captures)[:, None] / n_captures  # (n_captures, 1)
    modulation = (0.5 + 0.5 * np.cos(2 * np.pi * sample_indices / n_depths) >= 0.5).astype(float)  # (n_depths,)
    modulation_codes = np.broadcast_to(modulation / modulation.sum(), (n_captures, n_depths)).copy()
    reference_codes = (0.5 + 0.5 * np.cos(2 * np.pi * sample_indices / n_depths - phases) >= 0.5).astype(
        float
    )  # (n_captures, n_depths)
    return modulation_codes, reference_codes


def make_single_ramp_codes(n_depths: int) -> tuple[npt.NDArray, npt.NDArray]:
    """Build single-ramp modulation and reference codes (``n_captures=3``).

    Three captures are combined: a square-wave illumination with a matching
    reference, a capture with uniform illumination and an all-ones reference
    (the ambient/offset reference), and a dark capture that measures ambient
    light only. The recovered depth is only unambiguous over ``c / (4 * freq)``,
    half of what the sinusoidal schemes cover at the same frequency, because the
    ramp spans two correlation periods.

    Args:
        n_depths: Base number of depth bins; the actual code length is ``2*n_depths - 1``.

    Returns:
        Tuple of ``(modulation_codes, reference_codes)``, each of shape ``(3, 2*n_depths - 1)``.

    See Also:
        :func:`unambiguous_range` and :func:`visionsim.emulate.itof.decoding.decode`
        with ``scheme="singleRamp"``.
    """
    n_bins = 2 * n_depths - 1
    sample_indices: npt.NDArray[np.floating] = np.arange(n_bins, dtype=float)
    modulation_codes = np.zeros((3, n_bins))
    reference_codes = np.zeros((3, n_bins))
    modulation = 0.5 + 0.5 * np.cos(2 * np.pi * sample_indices / n_bins - np.pi / 2)
    modulation = np.where(modulation >= 0.5, 1.0, 0.0)
    modulation_codes[0] = modulation / modulation.sum()
    reference = 0.5 + 0.5 * np.cos(2 * np.pi * sample_indices / n_bins - np.pi / 2)
    reference_codes[0] = np.where(reference >= 0.5, 1.0, 0.0)
    uniform_modulation = np.ones(n_bins)
    modulation_codes[1] = uniform_modulation / uniform_modulation.sum()
    reference_codes[1] = np.ones(n_bins)
    modulation_codes[2] = np.zeros(n_bins)
    reference_codes[2] = np.ones(n_bins)
    return modulation_codes, reference_codes


def make_double_ramp_codes(n_depths: int) -> tuple[npt.NDArray, npt.NDArray]:
    """Build double-ramp modulation and reference codes (``n_captures=3``).

    The double-ramp variant of :func:`make_single_ramp_codes`: the second capture
    uses a reference that is shifted by half a period, so the two ramps together
    cover the depth range in both directions. Its usable range is likewise
    ``c / (4 * freq)``.

    Args:
        n_depths: Base number of depth bins; the actual code length is ``2*n_depths - 1``.

    Returns:
        Tuple of ``(modulation_codes, reference_codes)``, each of shape ``(3, 2*n_depths - 1)``.

    See Also:
        :func:`unambiguous_range` and :func:`visionsim.emulate.itof.decoding.decode`
        with ``scheme="doubleRamp"``.
    """
    n_bins = 2 * n_depths - 1
    sample_indices: npt.NDArray[np.floating] = np.arange(n_bins, dtype=float)
    modulation_codes = np.zeros((3, n_bins))
    reference_codes = np.zeros((3, n_bins))
    modulation = 0.5 + 0.5 * np.cos(2 * np.pi * sample_indices / n_bins - np.pi / 2)
    modulation = np.where(modulation >= 0.5, 1.0, 0.0)
    modulation_codes[0] = modulation / modulation.sum()
    reference = 0.5 + 0.5 * np.cos(2 * np.pi * sample_indices / n_bins - np.pi / 2)
    reference_codes[0] = np.where(reference >= 0.5, 1.0, 0.0)
    modulation_codes[1] = modulation / modulation.sum()
    reference_codes[1] = np.roll(reference_codes[0], round(reference_codes.shape[1] / 2) - 1)
    modulation_codes[2] = np.zeros(n_bins)
    reference_codes[2] = np.ones(n_bins)
    return modulation_codes, reference_codes


def make_multi_freq_sinusoidal_codes(
    freq_vec: npt.NDArray,
    shifts_vec: npt.NDArray,
    n_depths: int,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Build multi-frequency sinusoidal modulation and reference codes.

    Every capture uses its own frequency multiplier and phase shift, so a single
    capture set probes the scene at several modulation frequencies. Multi-
    frequency coding is what extends the unambiguous range beyond ``c / (2*f)``
    of a single frequency: each frequency yields a wrapped phase and unwrapping
    them coarse-to-fine recovers the depth over the largest of their periods.

    The tap layout expected by the decoder is ``n_captures == 4`` (one
    low-frequency tap plus three uniformly shifted taps of one frequency) or
    ``n_captures >= 5`` (three shifts on the first frequency and two per further
    frequency).

    Args:
        freq_vec: Frequency multipliers for each capture, shape ``(n_captures,)``.
        shifts_vec: Phase shifts (radians) for each capture, shape ``(n_captures,)``.
        n_depths: Number of time bins per period.

    Returns:
        Tuple of ``(modulation_codes, reference_codes)``, each of shape ``(n_captures, n_depths)``.

    See Also:
        :func:`visionsim.emulate.itof.decoding.decode` with ``scheme="multFreqSin"``.
    """
    sample_indices: npt.NDArray[np.floating] = np.arange(n_depths, dtype=float)  # (n_depths,)
    waveform_phases = 2 * np.pi * sample_indices * freq_vec[:, None] / n_depths  # (n_captures, n_depths)
    modulation = 0.5 + 0.5 * np.cos(waveform_phases)  # (n_captures, n_depths)
    modulation_codes = modulation / modulation.sum(axis=1, keepdims=True)
    reference_codes = 0.5 + 0.5 * np.cos(waveform_phases - shifts_vec[:, None])
    return modulation_codes, reference_codes


def make_gray_codes(n_bits: int) -> npt.NDArray:
    """Generate the standard *n_bits*-bit reflected Gray code table.

    Consecutive rows differ in exactly one bit, which is what makes a
    Gray-coded capture set usable as a continuous-time code: while the scene
    moves between two code words, only one capture channel changes at a time.

    Args:
        n_bits: Number of bits; the returned table has ``2**n_bits`` rows.

    Returns:
        Boolean integer array of shape ``(2**n_bits, n_bits)``.
    """
    gray_codes = np.array([[0], [1]])
    for _ in range(1, n_bits):
        gray_codes = np.vstack(
            [
                np.hstack([np.zeros((len(gray_codes), 1)), gray_codes]),
                np.hstack([np.ones((len(gray_codes), 1)), gray_codes[::-1]]),
            ]
        )
    return gray_codes


def make_max_min_run_length_gray_codes() -> npt.NDArray:
    """Build the 5-capture max-min run-length Gray code table.

    A Hamiltonian cycle over the 32 code words of five captures that keeps the
    runs of constant illumination as short as possible, which spreads the
    emitted energy evenly over the depth range. It is the table used by
    :func:`make_tof_gray_codes` for ``n_captures == 5``.

    Returns:
        Float array of shape ``(32, 5)`` containing the code words.
    """
    base_pairs = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    toggle_sequence = [1, 3, 2, 3, 1, 2, 3, 2, 1, 3, 2, 3, 1, 2, 3, 2]
    states: npt.NDArray[np.integer] = np.zeros((len(toggle_sequence) + 1, 3), dtype=int)
    for i, t in enumerate(toggle_sequence):
        states[i + 1] = states[i].copy()
        states[i + 1, t - 1] = 1 - states[i + 1, t - 1]
    rows = []
    for i in range(16):
        ib = i % 4
        ib1 = (i + 1) % 4
        rows.append(np.concatenate([states[i], base_pairs[ib]]))
        rows.append(np.concatenate([states[i], base_pairs[ib1]]))
    return np.array(rows, dtype=float)


def make_gray_codes_reduced(n_bits: int) -> npt.NDArray:
    """Build a reduced Gray code table ordered as a Hamiltonian cycle.

    The reflected Gray table with the all-zero and all-one code words removed:
    those two words are the only ones that would keep every capture channel
    constant, so dropping them leaves a table in which every code word has at
    least one channel transitioning. For 4 and 6 captures no Hamiltonian cycle
    exists over the reduced table, so the cycle of the next smaller dimension is
    duplicated with a constant leading bit instead.

    Args:
        n_bits: Number of sensor captures; must be in ``{3, 4, 5, 6}``.

    Returns:
        Float array of shape ``(n_intervals, n_bits)`` containing the reduced,
        Hamiltonian-ordered Gray codes.

    Raises:
        ValueError: If *n_bits* is not in the supported set.
    """
    if n_bits not in {3, 4, 5, 6}:
        raise ValueError(f"Reduced Gray codes are only defined for n_bits in (3, 4, 5, 6), got n_bits={n_bits}")
    doubled = n_bits in {4, 6}
    dim = n_bits - int(doubled)
    gray_codes = make_gray_codes(dim)
    row_sums = gray_codes.sum(axis=1)
    gray_codes = _hamiltonian_order(gray_codes[(row_sums != 0) & (row_sums != dim)])
    if doubled:
        gray_codes = np.vstack(
            [
                np.hstack([np.zeros((len(gray_codes), 1)), gray_codes]),
                np.hstack([np.ones((len(gray_codes), 1)), gray_codes[::-1]]),
            ]
        )
    return gray_codes


def _hamiltonian_order(gray_codes: npt.NDArray) -> npt.NDArray:
    """Re-order the rows of *gray_codes* to follow a Hamiltonian cycle on its graph.

    Two rows are adjacent if they differ in exactly one position (Gray-code
    adjacency).

    Args:
        gray_codes: Code word table, shape ``(n, d)``.

    Returns:
        Re-ordered copy of *gray_codes* following a Hamiltonian cycle, same shape.

    Raises:
        RuntimeError: If no Hamiltonian cycle exists in the adjacency graph.
    """
    n = len(gray_codes)
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if np.sum(np.abs(gray_codes[i] - gray_codes[j])) == 1:
                adj[i][j] = adj[j][i] = True
    path = [0]
    visited = {0}

    def _dfs() -> bool:
        if len(path) == n:
            return adj[path[-1]][path[0]]
        for nxt in range(n):
            if nxt not in visited and adj[path[-1]][nxt]:
                path.append(nxt)
                visited.add(nxt)
                if _dfs():
                    return True
                path.pop()
                visited.discard(nxt)
        return False

    if not _dfs():
        raise RuntimeError("No Hamiltonian cycle found")
    return gray_codes[path]


def _hilbert_2d(order: int) -> npt.NDArray:
    """Recursively generate 2-D Hilbert curve coordinates.

    Args:
        n: Recursion order (curve resolution).

    Returns:
        Array of shape ``(2, 4**n)`` with x/y coordinates in ``[-0.5, 0.5]``.
    """
    if order <= 0:
        return np.zeros((2, 1))
    sub_curve = _hilbert_2d(order - 1)
    x = 0.5 * np.concatenate([-0.5 + sub_curve[1], -0.5 + sub_curve[0], 0.5 + sub_curve[0], 0.5 - sub_curve[1]])
    y = 0.5 * np.concatenate([-0.5 + sub_curve[0], 0.5 + sub_curve[1], 0.5 + sub_curve[1], -0.5 - sub_curve[0]])
    return np.vstack([x, y])


def _hilbert_3d(order: int) -> npt.NDArray:
    """Recursively generate 3-D Hilbert curve coordinates.

    Args:
        n: Recursion order (curve resolution).

    Returns:
        Array of shape ``(3, 8**n)`` with x/y/z coordinates in ``[-0.5, 0.5]``.
    """
    if order <= 0:
        return np.zeros((3, 1))
    sub_curve = _hilbert_3d(order - 1)
    x = 0.5 * np.concatenate(
        [
            0.5 + sub_curve[2],
            0.5 + sub_curve[1],
            -0.5 + sub_curve[1],
            -0.5 - sub_curve[0],
            -0.5 - sub_curve[0],
            -0.5 - sub_curve[1],
            0.5 - sub_curve[1],
            0.5 + sub_curve[2],
        ]
    )
    y = 0.5 * np.concatenate(
        [
            0.5 + sub_curve[0],
            0.5 + sub_curve[2],
            0.5 + sub_curve[2],
            0.5 + sub_curve[1],
            -0.5 + sub_curve[1],
            -0.5 - sub_curve[2],
            -0.5 - sub_curve[2],
            -0.5 - sub_curve[0],
        ]
    )
    z = 0.5 * np.concatenate(
        [
            0.5 + sub_curve[1],
            -0.5 + sub_curve[0],
            -0.5 + sub_curve[0],
            0.5 - sub_curve[2],
            0.5 - sub_curve[2],
            -0.5 + sub_curve[0],
            -0.5 + sub_curve[0],
            0.5 - sub_curve[1],
        ]
    )
    return np.vstack([x, y, z])


def _normalize_and_expand(curve: npt.NDArray, hilbert_delta: float, n_points: int) -> npt.NDArray:
    """Normalize Hilbert curve coordinates and interpolate to a target density.

    Every coordinate is linearly normalized to ``[hilbert_delta, 1 - hilbert_delta]``
    so that the curve stays away from the edges of the code range, and its
    segments are then sub-sampled proportionally to their arc length.

    Args:
        curve: Curve coordinates of shape ``(dim, n_curve_points)``.
        hilbert_delta: Margin applied to each side of the normalized range.
        n_points: Total number of output sample points.

    Returns:
        Expanded coordinate array of shape ``(dim, n_points)`` (approximately).
    """
    curve = curve.copy().astype(float)
    for i in range(curve.shape[0]):
        coordinate = curve[i]
        coordinate = (coordinate - coordinate.min()) / (coordinate.max() - coordinate.min() + 1e-30)
        curve[i] = coordinate * (1 - 2 * hilbert_delta) + hilbert_delta
    n_sub_segments = curve.shape[1] - 1
    segment_lengths = np.array([np.linalg.norm(curve[:, i] - curve[:, i + 1]) for i in range(n_sub_segments)])
    total_length = segment_lengths.sum()
    segment_points: list[npt.NDArray] = []
    for i in range(n_sub_segments):
        n_segment_points = max(1, math.ceil(segment_lengths[i] / total_length * n_points))
        segment = np.zeros((curve.shape[0], n_segment_points + 1))
        for j in range(curve.shape[0]):
            if curve[j, i] == curve[j, i + 1]:
                segment[j] = curve[j, i]
            else:
                segment[j] = np.linspace(curve[j, i], curve[j, i + 1], n_segment_points + 1)
        segment_points.append(segment[:, :-1])
    return np.concatenate(segment_points, axis=1)


def _perm_matrix_to_codes(permutation_matrix: npt.NDArray, curve_points: npt.NDArray) -> npt.NDArray:
    """Convert a permutation matrix to a concatenated code array.

    Args:
        perm: Permutation index matrix of shape ``(n_captures, n_seg)`` where each
            entry encodes which Hilbert coordinate (or constant) to use.
        xpts: Hilbert curve sample points of shape ``(D, points_per_segment)``.

    Returns:
        Concatenated code array of shape ``(n_captures, n_segments * points_per_segment)``.
    """
    n_captures, n_segments = permutation_matrix.shape
    points_per_segment = curve_points.shape[1]
    code_blocks: list[npt.NDArray] = []
    for i in range(n_segments):
        c = np.zeros((n_captures, points_per_segment))
        for j in range(n_captures):
            v = int(permutation_matrix[j, i])
            if v in (0, 1):
                c[j] = float(v)
            else:
                row = curve_points[abs(v) - 2]
                c[j] = row[::-1] if v < 0 else row
        code_blocks.append(c)
    return np.concatenate(code_blocks, axis=1)


def make_tof_gray_codes(n_captures: int, n_depths: int) -> npt.NDArray:
    """Build ToF reference codes based on the max-min run-length Gray code.

    Each code word is emitted over a run of consecutive time bins, and the runs
    are concatenated so that the reference code interpolates linearly between
    adjacent code words: a depth falling inside a run maps to a point on the line
    connecting its two words, which is what the interval-wise decoder inverts.

    Args:
        n_captures: Number of captures.
        n_depths: Requested number of time bins; the returned length is
            ``n_segments * ceil(n_depths / n_segments)``, which can exceed *n_depths*.

    Returns:
        Reference code array of shape ``(n_captures, n_bins)``.

    Note:
        The max-min run-length table is only defined for five captures, so it is
        used for ``n_captures == 5`` and the reduced Gray table otherwise.
        :func:`visionsim.emulate.itof.decoding.decode_hilbert` applies the same
        rule, which keeps the encoder and decoder consistent for every capture
        count.

    See Also:
        :func:`visionsim.emulate.itof.decoding.decode` with
        ``scheme="deltaHilbertDimOne"``.
    """
    gray_codes = make_max_min_run_length_gray_codes() if n_captures == 5 else make_gray_codes_reduced(n_captures)
    n_segments = gray_codes.shape[0]
    points_per_segment = math.ceil(n_depths / n_segments)
    code_blocks: list[npt.NDArray] = []
    for i in range(n_segments):
        next_index = (i + 1) % n_segments
        code_block = np.zeros((n_captures, points_per_segment))
        for j in range(n_captures):
            if gray_codes[i, j] == 0 and gray_codes[next_index, j] == 0:
                code_block[j] = 0.0
            elif gray_codes[i, j] == 1 and gray_codes[next_index, j] == 1:
                code_block[j] = 1.0
            elif gray_codes[i, j] == 0 and gray_codes[next_index, j] == 1:
                code_block[j] = np.linspace(0, 1, points_per_segment + 1)[:-1]
            elif gray_codes[i, j] == 1 and gray_codes[next_index, j] == 0:
                code_block[j] = np.linspace(1, 0, points_per_segment + 1)[:-1]
        code_blocks.append(code_block)
    return np.concatenate(code_blocks, axis=1)


# fmt: off
_PERM_K4_DIM2 = np.array(
    [
        [ 0,  0,  2, 1,  1, -2,  1, -2, 3, 0,  2, -2],
        [ 2,  1,  3, -3, 0, -3, -3, 1,  1, -2, 0,  0],
        [ 3, -3,  1, -2, 2,  0,  0, 0,  2, 1,  1, -3],
        [ 1, -2,  0,  0, 3,  1, -2, -3, 0, -3, 3,  1],
    ]
)
_PERM_K5_DIM2 = np.array(
    [
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 3, 3, 3, 3, 3, 3],
        [1, 1, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 3, 3, 3, 3, 3, 3, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0, 1, 1, 1],
        [2, 2, 2, 2, 2, 2, 1, 1, 0, 0, 0, 1, 3, 3, 3, 3, 3, 3, 1, 1, 0, 0, 0, 1, 3, 3, 3, 3, 3, 3, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 3, 3, 3, 3, 3, 3, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 0, 0, 0, 1, 3, 3, 3, 3, 3, 3, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 3, 3, 3, 3, 3, 3, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
    ]
)
_PERM_K5_DIM3 = np.array(
    [
        [0, 0, 0, 0, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2],
        [1, 2, 2, 2, 0, 0, 0, 0, 2, 1, 3, 3, 2, 1, 3, 3, 2, 1, 3, 3],
        [2, 1, 3, 3, 2, 1, 3, 3, 0, 0, 0, 0, 3, 3, 1, 4, 3, 3, 1, 4],
        [3, 3, 1, 4, 3, 3, 1, 4, 3, 3, 1, 4, 0, 0, 0, 0, 4, 4, 4, 1],
        [4, 4, 4, 1, 4, 4, 4, 1, 4, 4, 4, 1, 4, 4, 4, 1, 0, 0, 0, 0],
    ]
)
# fmt: on


def make_tof_hilbert_codes(
    n_captures: int, dim: int, hilbert_order: int, hilbert_delta: float, n_depths: int
) -> npt.NDArray:
    """Build ToF reference codes by sampling along a Hilbert curve.

    A Hilbert curve fills the ``dim``-dimensional space of code words, so
    walking along it visits neighbouring code words and therefore changes one
    capture channel at a time. Each code word is emitted over a run of time bins
    whose length follows the arc length of the corresponding curve segment, and
    the result is normalized to ``[hilbert_delta, 1 - hilbert_delta]`` so that
    no channel saturates. ``dim=1`` degenerates to the Gray-coded codes of
    :func:`make_tof_gray_codes`.

    Args:
        n_captures: Number of captures; ``(4, 2)``, ``(5, 2)`` and ``(5, 3)``
            have code tables, where the pair is ``(n_captures, dim)``.
        dim: Hilbert curve dimensionality (1, 2, or 3).
        hilbert_order: Hilbert curve recursion order; higher orders resolve the
            code words more finely.
        hilbert_delta: Hilbert curve normalization margin in ``[0, 0.5)``.
        n_depths: Requested number of time bins; the returned length is
            ``n_segments * ceil(n_depths / n_segments)``.

    Returns:
        Reference code array of shape ``(n_captures, n_bins)``.

    Raises:
        ValueError: If no code table exists for the requested ``(n_captures, dim)``.

    See Also:
        :func:`visionsim.emulate.itof.decoding.decode` with
        ``scheme="deltaHilbertDimTwo"`` or ``scheme="deltaHilbertDimThree"``.
    """
    if dim == 1:
        return make_tof_gray_codes(n_captures, n_depths)
    if n_captures == 4 and dim == 2:
        permutation_matrix = _PERM_K4_DIM2
    elif n_captures == 5 and dim == 2:
        permutation_matrix = _PERM_K5_DIM2
    elif n_captures == 5 and dim == 3:
        permutation_matrix = _PERM_K5_DIM3
    else:
        raise ValueError(f"Unsupported n_captures={n_captures}, dim={dim}")
    n_segments = permutation_matrix.shape[1]
    points_per_segment = math.ceil(n_depths / n_segments)
    hilbert_curve = _hilbert_3d if dim == 3 else _hilbert_2d
    curve = hilbert_curve(hilbert_order)
    curve_points = _normalize_and_expand(curve, hilbert_delta, points_per_segment)
    return _perm_matrix_to_codes(permutation_matrix, curve_points)


def make_delta_hilbert_codes(
    n_captures: int, dim: int, hilbert_order: int, hilbert_delta: float, n_depths: int
) -> tuple[npt.NDArray, npt.NDArray]:
    """Build delta modulation and Hilbert-based reference codes.

    Combines the impulse (delta) illumination of
    :func:`make_delta_sinusoidal_codes` with the Hilbert curve reference codes of
    :func:`make_tof_hilbert_codes`: the emitted energy is concentrated in a
    single bin while the reference code walks along the curve, which gives
    continuous-time codes with a large light budget. The reference codes are
    normalized so that every channel reaches one.

    Args:
        n_captures: Number of captures.
        dim: Hilbert curve dimensionality (1, 2, or 3).
        hilbert_order: Hilbert curve recursion order.
        hilbert_delta: Hilbert curve normalization margin in ``[0, 0.5)``.
        n_depths: Requested number of time bins.

    Returns:
        Tuple of ``(modulation_codes, reference_codes)``, each of shape ``(n_captures, n_bins)``.

    See Also:
        :func:`visionsim.emulate.itof.decoding.decode` with
        ``scheme="deltaHilbertDimOne"``, ``scheme="deltaHilbertDimTwo"`` or
        ``scheme="deltaHilbertDimThree"``.
    """
    ref = make_tof_hilbert_codes(n_captures, dim, hilbert_order, hilbert_delta, n_depths)
    mod = np.zeros_like(ref)
    mod[:, 0] = 1.0
    for i in range(n_captures):
        mod[i] /= mod[i].sum()
        ref[i] /= ref[i].max()
    return mod, ref


CodingScheme = Literal[
    "convSin",
    "deltaSin",
    "convSquare",
    "singleRamp",
    "doubleRamp",
    "deltaHilbertDimOne",
    "deltaHilbertDimTwo",
    "deltaHilbertDimThree",
    "multFreqSin",
]
"""Identifier of a supported coding scheme, used as the ``scheme`` argument of :func:`make_coding_functions`."""


def make_coding_functions(
    scheme: CodingScheme,
    n_captures: int,
    n_depths: int,
    *,
    hilbert_order: int = 1,
    hilbert_delta: float = 0.25,
    freq_vec: npt.NDArray | None = None,
    shifts_vec: npt.NDArray | None = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Return modulation and demodulation codes for the named coding scheme.

    Args:
        scheme: Coding scheme identifier; must be one of the ``CodingScheme``
            literals.
        n_captures: Number of captures (phase shifts / code words).
        n_depths: Requested number of depth (time) bins per code period. The
            returned length is the smallest multiple of the scheme's segment
            count that is at least *n_depths*, so it can exceed it: ``2*n_depths
            - 1`` for the ramp schemes and ``n_segments * ceil(n_depths /
            n_segments)`` for the Hilbert schemes; every other scheme returns
            exactly *n_depths*.
        hilbert_order: Hilbert curve recursion order (for deltaHilbertDim codes).
        hilbert_delta: Hilbert curve normalization margin in ``[0, 0.5)`` (for deltaHilbertDim codes).
        freq_vec: Frequency multipliers required by ``"multFreqSin"``; its
            length determines ``n_captures``.
        shifts_vec: Phase shifts (radians) required by ``"multFreqSin"``.

    Returns:
        Tuple of ``(modulation_codes, reference_codes)``, each of shape ``(n_captures, n_bins)``,
        where ``n_bins`` is as described for *n_depths* above.

    Raises:
        ValueError: If *scheme* is not a recognized coding scheme, if a
            scheme-specific constraint is violated (``n_captures != 3`` for the ramp
            schemes, missing or mismatched ``freq_vec``/``shifts_vec`` for
            ``"multFreqSin"``), or if no code table exists for the requested
            ``n_captures`` (or ``(n_captures, dim)``) combination.

    See Also:
        :func:`unambiguous_range` for the depth range each scheme can cover, and
        the per-scheme builders such as :func:`make_conv_sinusoidal_codes`.
    """
    if scheme in ("singleRamp", "doubleRamp") and n_captures != 3:
        raise ValueError(f"Scheme '{scheme}' requires n_captures=3, got n_captures={n_captures}")
    if scheme == "multFreqSin":
        if freq_vec is None or shifts_vec is None:
            raise ValueError("'multFreqSin' requires freq_vec and shifts_vec")
        if len(freq_vec) != len(shifts_vec):
            raise ValueError(
                f"'multFreqSin' expects freq_vec and shifts_vec of equal length, "
                f"got {len(freq_vec)} and {len(shifts_vec)}"
            )
        if len(freq_vec) != n_captures:
            raise ValueError(
                f"'multFreqSin' derives n_captures from freq_vec/shifts_vec, got n_captures={n_captures} but {len(freq_vec)} frequencies"
            )
        if n_captures != 4 and (n_captures < 5 or (n_captures - 3) % 2 != 0):
            raise ValueError(
                "'multFreqSin' expects n_captures=4 (two frequencies) or an odd n_captures>=5 "
                f"(three shifts on the first frequency, two per further frequency), got n_captures={n_captures}"
            )

    _dispatch: dict[str, Callable[[], tuple[npt.NDArray, npt.NDArray]]] = {
        "convSin": lambda: make_conv_sinusoidal_codes(n_captures, n_depths),
        "deltaSin": lambda: make_delta_sinusoidal_codes(n_captures, n_depths),
        "convSquare": lambda: make_conv_square_codes(n_captures, n_depths),
        "singleRamp": lambda: make_single_ramp_codes(n_depths),
        "doubleRamp": lambda: make_double_ramp_codes(n_depths),
        "deltaHilbertDimOne": lambda: make_delta_hilbert_codes(n_captures, 1, hilbert_order, hilbert_delta, n_depths),
        "deltaHilbertDimTwo": lambda: make_delta_hilbert_codes(n_captures, 2, hilbert_order, hilbert_delta, n_depths),
        "deltaHilbertDimThree": lambda: make_delta_hilbert_codes(n_captures, 3, hilbert_order, hilbert_delta, n_depths),
        "multFreqSin": lambda: make_multi_freq_sinusoidal_codes(freq_vec, shifts_vec, n_depths),
    }
    try:
        return _dispatch[scheme]()
    except KeyError:
        raise ValueError(f"Unknown coding scheme: {scheme!r}") from None
