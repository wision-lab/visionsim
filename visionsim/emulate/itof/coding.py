"""Waveform generation and coding schemes for iToF."""

from __future__ import annotations

import math
from typing import Callable, Literal

import numpy as np
import numpy.typing as npt


def _make_conv_sinusoidal_codes(K: int, N: int) -> tuple[npt.NDArray, npt.NDArray]:
    """Build conventional sinusoidal modulation and reference codes.

    Args:
        K: Number of captures (phase shifts).
        N: Number of time bins per period.

    Returns:
        Tuple of ``(mod_codes, ref_codes)``, each of shape ``(K, N)``.
    """
    X = np.arange(N, dtype=float)  # (N,)
    phases = 2 * np.pi * np.arange(K)[:, None] / K  # (K, 1)
    mc = 0.5 + 0.5 * np.cos(2 * np.pi * X / N)  # (N,) – same for every capture
    mod_codes = np.broadcast_to(mc / mc.sum(), (K, N)).copy()
    ref_codes = 0.5 + 0.5 * np.cos(2 * np.pi * X / N - phases)  # (K, N)
    return mod_codes, ref_codes


def _make_delta_sinusoidal_codes(K: int, N: int) -> tuple[npt.NDArray, npt.NDArray]:
    """Build delta (impulse) sinusoidal modulation and reference codes.

    Args:
        K: Number of captures (phase shifts).
        N: Number of time bins per period.

    Returns:
        Tuple of ``(mod_codes, ref_codes)``, each of shape ``(K, N)``.
    """
    X = np.arange(N, dtype=float)
    phases = 2 * np.pi * np.arange(K)[:, None] / K  # (K, 1)
    mod_codes = np.zeros((K, N))
    mod_codes[:, 0] = 1.0  # impulse at t=0 for every capture
    ref_codes = 0.5 + 0.5 * np.cos(2 * np.pi * X / N - phases)  # (K, N)
    return mod_codes, ref_codes


def _make_conv_square_codes(K: int, N: int) -> tuple[npt.NDArray, npt.NDArray]:
    """Build conventional square-wave modulation and reference codes.

    Args:
        K: Number of captures (phase shifts).
        N: Number of time bins per period.

    Returns:
        Tuple of ``(mod_codes, ref_codes)``, each of shape ``(K, N)``.
    """
    X = np.arange(N, dtype=float)
    phases = 2 * np.pi * np.arange(K)[:, None] / K  # (K, 1)
    mc = (0.5 + 0.5 * np.cos(2 * np.pi * X / N) >= 0.5).astype(float)  # (N,)
    mod_codes = np.broadcast_to(mc / mc.sum(), (K, N)).copy()
    ref_codes = (0.5 + 0.5 * np.cos(2 * np.pi * X / N - phases) >= 0.5).astype(float)  # (K, N)
    return mod_codes, ref_codes


def _make_single_ramp_codes(N: int) -> tuple[npt.NDArray, npt.NDArray]:
    """Build single-ramp modulation and reference codes (K=3).

    Args:
        N: Base number of depth bins; actual code length is ``2*N - 1``.

    Returns:
        Tuple of ``(mod_codes, ref_codes)``, each of shape ``(3, 2*N-1)``.
    """
    N2 = 2 * N - 1
    X = np.arange(N2, dtype=float)
    mod_codes = np.zeros((3, N2))
    ref_codes = np.zeros((3, N2))
    mc = 0.5 + 0.5 * np.cos(2 * np.pi * X / N2 - np.pi / 2)
    mc = np.where(mc >= 0.5, 1.0, 0.0)
    mod_codes[0] = mc / mc.sum()
    rc = 0.5 + 0.5 * np.cos(2 * np.pi * X / N2 - np.pi / 2)
    ref_codes[0] = np.where(rc >= 0.5, 1.0, 0.0)
    mc2 = np.ones(N2)
    mod_codes[1] = mc2 / mc2.sum()
    ref_codes[1] = np.ones(N2)
    mod_codes[2] = np.zeros(N2)
    ref_codes[2] = np.ones(N2)
    return mod_codes, ref_codes


def _make_double_ramp_codes(N: int) -> tuple[npt.NDArray, npt.NDArray]:
    """Build double-ramp modulation and reference codes (K=3).

    Args:
        N: Base number of depth bins; actual code length is ``2*N - 1``.

    Returns:
        Tuple of ``(mod_codes, ref_codes)``, each of shape ``(3, 2*N-1)``.
    """
    N2 = 2 * N - 1
    X = np.arange(N2, dtype=float)
    mod_codes = np.zeros((3, N2))
    ref_codes = np.zeros((3, N2))
    mc = 0.5 + 0.5 * np.cos(2 * np.pi * X / N2 - np.pi / 2)
    mc = np.where(mc >= 0.5, 1.0, 0.0)
    mod_codes[0] = mc / mc.sum()
    rc = 0.5 + 0.5 * np.cos(2 * np.pi * X / N2 - np.pi / 2)
    ref_codes[0] = np.where(rc >= 0.5, 1.0, 0.0)
    mod_codes[1] = mc / mc.sum()
    ref_codes[1] = np.roll(ref_codes[0], round(ref_codes.shape[1] / 2) - 1)
    mod_codes[2] = np.zeros(N2)
    ref_codes[2] = np.ones(N2)
    return mod_codes, ref_codes


def _make_multi_freq_sinusoidal_codes(
    freq_vec: npt.NDArray,
    shifts_vec: npt.NDArray,
    N: int,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Build multi-frequency sinusoidal modulation and reference codes.

    Args:
        freq_vec: Frequency multipliers for each capture, shape ``(K,)``.
        shifts_vec: Phase shifts (radians) for each capture, shape ``(K,)``.
        N: Number of time bins per period.

    Returns:
        Tuple of ``(mod_codes, ref_codes)``, each of shape ``(K, N)``.
    """
    X = np.arange(N, dtype=float)  # (N,)
    args = 2 * np.pi * X * freq_vec[:, None] / N  # (K, N)
    mc = 0.5 + 0.5 * np.cos(args)  # (K, N)
    mod_codes = mc / mc.sum(axis=1, keepdims=True)
    ref_codes = 0.5 + 0.5 * np.cos(args - shifts_vec[:, None])
    return mod_codes, ref_codes


def _make_gray_codes(n_bits: int) -> npt.NDArray:
    """Generate the standard *n_bits*-bit reflected Gray code table.

    Args:
        n_bits: Number of bits; the returned table has ``2**n_bits`` rows.

    Returns:
        Boolean integer array of shape ``(2**n_bits, n_bits)``.
    """
    G = np.array([[0], [1]])
    for _ in range(1, n_bits):
        G = np.vstack([np.hstack([np.zeros((len(G), 1)), G]), np.hstack([np.ones((len(G), 1)), G[::-1]])])
    return G


def _make_max_min_run_length_gray_codes() -> npt.NDArray:
    """Build the 5-capture max-min run-length Gray code table.

    Returns:
        Integer array of shape ``(32, 5)`` containing the code words.
    """
    B = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    T = [1, 3, 2, 3, 1, 2, 3, 2, 1, 3, 2, 3, 1, 2, 3, 2]
    W = np.zeros((len(T) + 1, 3), dtype=int)
    for i, t in enumerate(T):
        W[i + 1] = W[i].copy()
        W[i + 1, t - 1] = 1 - W[i + 1, t - 1]
    rows = []
    for i in range(16):
        ib = i % 4
        ib1 = (i + 1) % 4
        rows.append(np.concatenate([W[i], B[ib]]))
        rows.append(np.concatenate([W[i], B[ib1]]))
    return np.array(rows, dtype=float)


def _make_gray_codes_reduced(n_bits: int) -> npt.NDArray:
    """Build a reduced Gray code table ordered as a Hamiltonian cycle.

    Supports ``n_bits`` in ``{3, 4, 5, 6}``.

    Args:
        n_bits: Number of sensor captures.

    Returns:
        Float array of shape ``(n_intervals, n_bits)`` containing the
        reduced, Hamiltonian-ordered Gray codes.

    Raises:
        ValueError: If *n_bits* is not in the supported set.
    """
    if n_bits not in {3, 4, 5, 6}:
        raise ValueError(f"Unsupported n_bits={n_bits}")
    double = n_bits in {4, 6}
    dim = n_bits - int(double)
    G = _make_gray_codes(dim)
    row_sums = G.sum(axis=1)
    G = _hamiltonian_order(G[(row_sums != 0) & (row_sums != dim)])
    if double:
        G = np.vstack(
            [
                np.hstack([np.zeros((len(G), 1)), G]),
                np.hstack([np.ones((len(G), 1)), G[::-1]]),
            ]
        )
    return G


def _hamiltonian_order(G: npt.NDArray) -> npt.NDArray:
    """Re-order the rows of *G* to follow a Hamiltonian cycle on its graph.

    Two rows are adjacent if they differ in exactly one position (Gray-code
    adjacency).

    Args:
        G: Code word table, shape ``(n, d)``.

    Returns:
        Re-ordered copy of *G* following a Hamiltonian cycle, same shape.

    Raises:
        RuntimeError: If no Hamiltonian cycle exists in the adjacency graph.
    """
    n = len(G)
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if np.sum(np.abs(G[i] - G[j])) == 1:
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
    return G[path]


def _hilbert_2d(n: int) -> npt.NDArray:
    """Recursively generate 2-D Hilbert curve coordinates.

    Args:
        n: Recursion order (curve resolution).

    Returns:
        Array of shape ``(2, 4**n)`` with x/y coordinates in ``[-0.5, 0.5]``.
    """
    if n <= 0:
        return np.zeros((2, 1))
    X0 = _hilbert_2d(n - 1)
    x = 0.5 * np.concatenate([-0.5 + X0[1], -0.5 + X0[0], 0.5 + X0[0], 0.5 - X0[1]])
    y = 0.5 * np.concatenate([-0.5 + X0[0], 0.5 + X0[1], 0.5 + X0[1], -0.5 - X0[0]])
    return np.vstack([x, y])


def _hilbert_3d(n: int) -> npt.NDArray:
    """Recursively generate 3-D Hilbert curve coordinates.

    Args:
        n: Recursion order (curve resolution).

    Returns:
        Array of shape ``(3, 8**n)`` with x/y/z coordinates in ``[-0.5, 0.5]``.
    """
    if n <= 0:
        return np.zeros((3, 1))
    X0 = _hilbert_3d(n - 1)
    x = 0.5 * np.concatenate(
        [
            0.5 + X0[2],
            0.5 + X0[1],
            -0.5 + X0[1],
            -0.5 - X0[0],
            -0.5 - X0[0],
            -0.5 - X0[1],
            0.5 - X0[1],
            0.5 + X0[2],
        ]
    )
    y = 0.5 * np.concatenate(
        [
            0.5 + X0[0],
            0.5 + X0[2],
            0.5 + X0[2],
            0.5 + X0[1],
            -0.5 + X0[1],
            -0.5 - X0[2],
            -0.5 - X0[2],
            -0.5 - X0[0],
        ]
    )
    z = 0.5 * np.concatenate(
        [
            0.5 + X0[1],
            -0.5 + X0[0],
            -0.5 + X0[0],
            0.5 - X0[2],
            0.5 - X0[2],
            -0.5 + X0[0],
            -0.5 + X0[0],
            0.5 - X0[1],
        ]
    )
    return np.vstack([x, y, z])


def _normalize_and_expand(X: npt.NDArray, delta: float, pts_per_seg: int) -> npt.NDArray:
    """Normalise Hilbert curve coordinates and interpolate to a target density.

    Each row of *X* is linearly normalised to ``[delta, 1-delta]``.  Segments
    are then sub-sampled proportionally to their arc length.

    Args:
        X: Curve coordinates of shape ``(D, n_points)``.
        delta: Margin applied to each side of the normalised range.
        pts_per_seg: Total number of output sample points.

    Returns:
        Expanded coordinate array of shape ``(D, pts_per_seg)`` (approximately).
    """
    X = X.copy().astype(float)
    for i in range(X.shape[0]):
        v = X[i]
        v = (v - v.min()) / (v.max() - v.min() + 1e-30)
        X[i] = v * (1 - 2 * delta) + delta
    n_sub = X.shape[1] - 1
    lengths = np.array([np.linalg.norm(X[:, i] - X[:, i + 1]) for i in range(n_sub)])
    total = lengths.sum()
    all_pts: list[npt.NDArray] = []
    for i in range(n_sub):
        n_pts = max(1, int(math.ceil(lengths[i] / total * pts_per_seg)))
        seg = np.zeros((X.shape[0], n_pts + 1))
        for j in range(X.shape[0]):
            if X[j, i] == X[j, i + 1]:
                seg[j] = X[j, i]
            else:
                seg[j] = np.linspace(X[j, i], X[j, i + 1], n_pts + 1)
        all_pts.append(seg[:, :-1])
    return np.concatenate(all_pts, axis=1)


def _perm_matrix_to_codes(perm: npt.NDArray, xpts: npt.NDArray) -> npt.NDArray:
    """Convert a permutation matrix to a concatenated code array.

    Args:
        perm: Permutation index matrix of shape ``(K, n_seg)`` where each
            entry encodes which Hilbert coordinate (or constant) to use.
        xpts: Hilbert curve sample points of shape ``(D, pps)``.

    Returns:
        Concatenated code array of shape ``(K, n_seg * pps)``.
    """
    K, n_seg = perm.shape
    pps = xpts.shape[1]
    codes_list: list[npt.NDArray] = []
    for i in range(n_seg):
        c = np.zeros((K, pps))
        for j in range(K):
            v = int(perm[j, i])
            if v in (0, 1):
                c[j] = float(v)
            else:
                row = xpts[abs(v) - 2]
                c[j] = row[::-1] if v < 0 else row
        codes_list.append(c)
    return np.concatenate(codes_list, axis=1)


def _make_tof_gray_codes(K: int, N: int) -> npt.NDArray:
    """Build ToF reference codes based on the max-min run-length Gray code.

    Args:
        K: Number of captures.
        N: Total number of time bins.

    Returns:
        Reference code array of shape ``(K, N)``.
    """
    G = _make_max_min_run_length_gray_codes()
    n_seg = G.shape[0]
    pps = int(math.ceil(N / n_seg))
    codes_list: list[npt.NDArray] = []
    for i in range(n_seg):
        i_next = (i + 1) % n_seg
        ct = np.zeros((K, pps))
        for j in range(K):
            if G[i, j] == 0 and G[i_next, j] == 0:
                ct[j] = 0.0
            elif G[i, j] == 1 and G[i_next, j] == 1:
                ct[j] = 1.0
            elif G[i, j] == 0 and G[i_next, j] == 1:
                ct[j] = np.linspace(0, 1, pps + 1)[:-1]
            elif G[i, j] == 1 and G[i_next, j] == 0:
                ct[j] = np.linspace(1, 0, pps + 1)[:-1]
        codes_list.append(ct)
    return np.concatenate(codes_list, axis=1)


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


def _make_tof_hilbert_codes(K: int, dim: int, ord_: int, delta: float, N: int) -> npt.NDArray:
    """Build ToF reference codes by sampling along a Hilbert curve.

    Args:
        K: Number of captures.
        dim: Hilbert curve dimensionality (1, 2, or 3).
        ord_: Hilbert curve recursion order.
        delta: Normalisation margin in ``[0, 0.5)``.
        N: Total number of time bins.

    Returns:
        Reference code array of shape ``(K, N)``.
    """
    if dim == 1:
        return _make_tof_gray_codes(K, N)
    if K == 4 and dim == 2:
        perm = _PERM_K4_DIM2
    elif K == 5 and dim == 2:
        perm = _PERM_K5_DIM2
    elif K == 5 and dim == 3:
        perm = _PERM_K5_DIM3
    else:
        raise ValueError(f"Unsupported K={K}, dim={dim}")
    n_seg = perm.shape[1]
    pps = int(math.ceil(N / n_seg))
    hfn = _hilbert_3d if dim == 3 else _hilbert_2d
    X = hfn(ord_)
    xpts = _normalize_and_expand(X, delta, pps)
    return _perm_matrix_to_codes(perm, xpts)


def _make_delta_hilbert_codes(K: int, dim: int, ord_: int, delta: float, N: int) -> tuple[npt.NDArray, npt.NDArray]:
    """Build delta modulation and Hilbert-based reference codes.

    Args:
        K: Number of captures.
        dim: Hilbert curve dimensionality (1, 2, or 3).
        ord_: Hilbert curve recursion order.
        delta: Normalisation margin in ``[0, 0.5)``.
        N: Total number of time bins.

    Returns:
        Tuple of ``(mod_codes, ref_codes)``, each of shape ``(K, N)``.
    """
    ref = _make_tof_hilbert_codes(K, dim, ord_, delta, N)
    mod = np.zeros_like(ref)
    mod[:, 0] = 1.0
    for i in range(K):
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


def make_coding_functions(
    name: CodingScheme,
    K: int,
    n_depths: int,
    *,
    freq_vec: npt.NDArray | None = None,
    shifts_vec: npt.NDArray | None = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Return modulation and demodulation codes for the named coding scheme.

    Args:
        name: Coding scheme identifier; must be one of the ``CodingScheme``
            literals.
        K: Number of captures (phase shifts / code words).
        n_depths: Number of depth (time) bins per code period.
        freq_vec: Frequency multipliers required by ``"multFreqSin"``.
        shifts_vec: Phase shifts (radians) required by ``"multFreqSin"``.

    Returns:
        Tuple of ``(mod_codes, ref_codes)``, each of shape ``(K, n_depths)``.

    Raises:
        ValueError: If *name* is not a recognised coding scheme.
        AssertionError: If scheme-specific constraints (e.g. ``K == 3``) are
            violated.
    """
    if name in ("singleRamp", "doubleRamp") and K != 3:
        raise ValueError(f"Scheme '{name}' requires K=3, got K={K}")
    if name == "multFreqSin" and (freq_vec is None or shifts_vec is None):
        raise ValueError("'multFreqSin' requires freq_vec and shifts_vec")

    _dispatch: dict[str, Callable[[], tuple[npt.NDArray, npt.NDArray]]] = {
        "convSin": lambda: _make_conv_sinusoidal_codes(K, n_depths),
        "deltaSin": lambda: _make_delta_sinusoidal_codes(K, n_depths),
        "convSquare": lambda: _make_conv_square_codes(K, n_depths),
        "singleRamp": lambda: _make_single_ramp_codes(n_depths),
        "doubleRamp": lambda: _make_double_ramp_codes(n_depths),
        "deltaHilbertDimOne": lambda: _make_delta_hilbert_codes(K, 1, 1, 0.25, n_depths),
        "deltaHilbertDimTwo": lambda: _make_delta_hilbert_codes(K, 2, 1, 0.25, n_depths),
        "deltaHilbertDimThree": lambda: _make_delta_hilbert_codes(K, 3, 1, 0.25, n_depths),
        "multFreqSin": lambda: _make_multi_freq_sinusoidal_codes(freq_vec, shifts_vec, n_depths),
    }
    try:
        return _dispatch[name]()
    except KeyError:
        raise ValueError(f"Unknown coding scheme: {name!r}") from None
