"""Decoding functions for iToF depth recovery."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.constants

from .coding import (
    _PERM_K4_DIM2,
    _PERM_K5_DIM2,
    _PERM_K5_DIM3,
    CodingScheme,
    _hilbert_2d,
    _hilbert_3d,
    _make_gray_codes_reduced,
    _make_max_min_run_length_gray_codes,
    _perm_matrix_to_codes,
)


def _compute_segment_distance(
    p1: npt.NDArray,
    p2: npt.NDArray,
    pts: npt.NDArray,
) -> npt.NDArray:
    """Compute the squared distance from each point to a line segment.

    Args:
        p1: Start point of the segment, shape ``(D,)``.
        p2: End point of the segment, shape ``(D,)``.
        pts: Query points stored as column vectors, shape ``(D, N)``.

    Returns:
        Squared distances from each column of *pts* to the segment
        ``[p1, p2]``, shape ``(N,)``.
    """
    seg_len_sq = np.sum((p1 - p2) ** 2)
    if seg_len_sq == 0:
        return np.sum((pts - p1[:, None]) ** 2, axis=0)
    t = np.sum((pts - p1[:, None]) * (p2 - p1)[:, None], axis=0) / seg_len_sq
    t = np.clip(t, 0, 1)
    closest = p1[:, None] + t[None, :] * (p2 - p1)[:, None]
    return np.sum((pts - closest) ** 2, axis=0)


def _decode_sinusoid(data: npt.NDArray, freq: float) -> npt.NDArray:
    """Decode sinusoidal (conventional or delta) iToF measurements.

    Args:
        data: Raw measurement array of shape ``(K, N_pixels)``.
        freq: Modulation frequency in Hz.

    Returns:
        Recovered depth map of shape ``(N_pixels,)`` in metres.
    """
    K = data.shape[0]
    M = np.column_stack([np.ones(K), np.cos(2 * np.pi / K * np.arange(K)), np.sin(2 * np.pi / K * np.arange(K))])
    X = np.linalg.lstsq(M, data, rcond=None)[0]
    amps = np.sqrt(X[1] ** 2 + X[2] ** 2)
    phi = np.arccos(np.clip(X[1] / (amps + 1e-30), -1, 1))
    phi[X[2] < 0] = 2 * np.pi - phi[X[2] < 0]
    return (scipy.constants.c * phi) / (4 * np.pi * freq)


def _decode_square(data: npt.NDArray, freq: float) -> npt.NDArray:
    """Decode square-wave iToF measurements.

    Args:
        data: Raw measurement array of shape ``(K, N_pixels)``.
        freq: Modulation frequency in Hz.

    Returns:
        Recovered depth map of shape ``(N_pixels,)`` in metres.
    """
    K = data.shape[0]
    depth_range = scipy.constants.c / (2 * freq)
    n_int = 2 * K
    # Build piecewise-linear start/end/slope for each interval
    start = np.zeros((K, n_int))
    start[0] = np.abs(1 - np.arange(2 * K) / K)
    for i in range(1, K):
        start[i] = np.roll(start[i - 1], 2)
    end = np.roll(start, -1, axis=1)
    slopes = end - start
    mid = (start + end) / 2
    # Pairwise relative ordering of captures — vectorised (K*(K-1), n_int)
    j_idx, k_idx = np.array([(j, k) for j in range(K) for k in range(K) if j != k]).T
    pair_rel = (mid[j_idx, :] >= mid[k_idx, :]).astype(float)  # (K*(K-1), n_int)
    data_rel = (data[j_idx, :] >= data[k_idx, :]).astype(float)  # (K*(K-1), n_pixels)
    # Squared Hamming distance for every interval × pixel: (n_int, n_pixels)
    dist_mat = np.array([((data_rel - pair_rel[:, i : i + 1]) ** 2).sum(0) for i in range(n_int)])
    interval_idx = np.argmin(dist_mat, axis=0)
    depths = np.zeros(data.shape[1])
    for i in range(n_int):
        mask = interval_idx == i
        if not mask.any():
            continue
        M = np.column_stack([np.ones(K), start[:, i], slopes[:, i]])
        X = np.linalg.lstsq(M, data[:, mask], rcond=None)[0]
        t = np.clip(X[2] / (X[1] + 1e-30), 0, 1)
        depths[mask] = (i + t) / n_int * depth_range
    return depths


def _decode_single_ramp(data: npt.NDArray, freq: float) -> npt.NDArray:
    """Decode single-ramp iToF measurements (K=3).

    Args:
        data: Raw measurement array of shape ``(3, N_pixels)``.
        freq: Modulation frequency in Hz.

    Returns:
        Recovered depth map of shape ``(N_pixels,)`` in metres.
    """
    M = np.array([[-1, 1, 0.5], [0, 1, 1], [0, 0, 1]], dtype=float)
    X = np.linalg.lstsq(M, data, rcond=None)[0]
    nd = np.clip(X[0] / (X[1] + 1e-30), 0, 1)
    return nd * scipy.constants.c / (4 * freq)


def _decode_double_ramp(data: npt.NDArray, freq: float) -> npt.NDArray:
    """Decode double-ramp iToF measurements (K=3).

    Args:
        data: Raw measurement array of shape ``(3, N_pixels)``.
        freq: Modulation frequency in Hz.

    Returns:
        Recovered depth map of shape ``(N_pixels,)`` in metres.
    """
    M = np.array([[-1, 1, 0.5], [1, 0, 0.5], [0, 0, 1]], dtype=float)
    X = np.linalg.lstsq(M, data, rcond=None)[0]
    nd = np.clip(X[0] / (X[1] + 1e-30), 0, 1)
    return nd * scipy.constants.c / (4 * freq)


def _make_hilbert_code_endpoints(K: int, dim: int, ord_: int, delta: float) -> npt.NDArray:
    """Build Hilbert curve code endpoint array for the decode lookup (dim >= 2).

    Args:
        K: Number of captures.
        dim: Hilbert curve dimensionality (2 or 3).
        ord_: Hilbert curve recursion order.
        delta: Normalisation margin in ``[0, 0.5)``.

    Returns:
        Code endpoint array of shape ``(K, n_endpoints)``.

    Raises:
        ValueError: If the ``(K, dim)`` combination is not supported.
    """
    _PERM = {(4, 2): _PERM_K4_DIM2, (5, 2): _PERM_K5_DIM2, (5, 3): _PERM_K5_DIM3}
    if (K, dim) not in _PERM:
        raise ValueError(f"Unsupported K={K}, dim={dim}")
    hfn = _hilbert_3d if dim == 3 else _hilbert_2d
    X = hfn(ord_).astype(float)
    X = (X - X.min(axis=1, keepdims=True)) / (X.max(axis=1, keepdims=True) - X.min(axis=1, keepdims=True) + 1e-30)
    X = X * (1 - 2 * delta) + delta
    return _perm_matrix_to_codes(_PERM[(K, dim)], X)


def _decode_hilbert(
    data: npt.NDArray,
    freq: float,
    dim: int = 1,
    ord_: int = 1,
    delta: float = 0.25,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Decode Hilbert-coded iToF measurements for any supported dimensionality.

    Dispatches to a Gray-code interval classifier for ``dim=1`` and to a
    Hilbert-curve segment-distance classifier for ``dim=2`` or ``dim=3``.
    The final least-squares depth-recovery step is shared across all paths.

    Args:
        data: Raw measurement array of shape ``(K, N_pixels)``.
        freq: Modulation frequency in Hz.
        dim: Hilbert curve dimensionality.  ``1`` uses the Gray-code path;
            ``2`` or ``3`` use the Hilbert-curve path.  Defaults to ``1``.
        ord_: Hilbert curve recursion order (ignored for ``dim=1``).
            Defaults to ``1``.
        delta: Normalisation margin in ``[0, 0.5)`` (ignored for ``dim=1``).
            Defaults to ``0.25``.

    Returns:
        Tuple of ``(interval_idx, depths)`` where *interval_idx* has shape
        ``(N_pixels,)`` (integer bin index) and *depths* has shape
        ``(N_pixels,)`` in metres.

    Raises:
        ValueError: If the ``(K, dim)`` combination is not supported.
    """
    K, n_pixels = data.shape[0], data.shape[1]
    depth_range = scipy.constants.c / (2 * freq)

    if dim == 1:
        # Gray-code path
        G = _make_max_min_run_length_gray_codes() if K == 5 else _make_gray_codes_reduced(K)
        n_int = G.shape[0]
        start = G.T.astype(float)
        end = np.roll(start, -1, axis=1)
        slopes = end - start

        # Threshold each channel: 1.0 above mid-range, 0.0 below
        # -1.0 for the most-transitioning channel ("don't-care" marker)
        val_max, val_min = data.max(axis=0), data.min(axis=0)
        non_const_idx = np.argmax(np.minimum(np.abs(val_max - data), np.abs(val_min - data)), axis=0)
        thr = (data > (val_max + val_min) / 2).astype(float)
        thr[non_const_idx, np.arange(n_pixels)] = -1.0

        # Assign each pixel to the interval whose constant channels match.
        interval_idx = np.zeros(n_pixels, dtype=int)
        for i in range(n_int):
            const_rows = np.where(slopes[:, i] == 0)[0]
            mask = np.ones(n_pixels, dtype=bool)
            for j in const_rows:
                mask &= thr[j] == start[j, i]
            interval_idx[mask] = i

        # Fallback: nearest interval by squared threshold difference.
        unassigned = interval_idx == 0
        if unassigned.any():
            d_ua = data[:, unassigned]
            thr_ua = (d_ua > (d_ua.max(0) + d_ua.min(0)) / 2).astype(float)
            diffs = ((thr_ua[:, :, None] - start[:, None, :]) ** 2).sum(axis=0)  # (n_ua, n_int)
            interval_idx[unassigned] = np.argmin(diffs, axis=1)

    else:
        # Hilbert-curve path (dim = 2 or 3)
        _CFG: dict[tuple[int, int], tuple[int, set[int]]] = {
            (4, 2): (36, set(range(3, 48, 4))),
            (5, 2): (180, set(range(3, 240, 4))),
            (5, 3): (140, set(range(7, 160, 8))),
        }
        if (K, dim) not in _CFG:
            raise ValueError(f"Unsupported K={K}, dim={dim}")
        n_int, skip = _CFG[(K, dim)]
        ep = _make_hilbert_code_endpoints(K, dim, ord_, delta)
        ep_idx = [i for i in range(ep.shape[1]) if i not in skip]
        start = ep[:, ep_idx[:n_int]]
        end = ep[:, [i + 1 for i in ep_idx[:n_int]]]
        slopes = end - start

        val_max, val_min = data.max(axis=0), data.min(axis=0)
        norm_data = (data - val_min) / (val_max - val_min + 1e-30)
        interval_idx = np.zeros(n_pixels, dtype=int)
        min_dist = np.full(n_pixels, np.inf)
        for i in range(n_int):
            d = _compute_segment_distance(start[:, i], end[:, i], norm_data)
            better = d < min_dist
            interval_idx[better], min_dist[better] = i, d[better]

    # Shared depth-recovery: least-squares within each interval
    depths = np.zeros(n_pixels)
    for i in range(n_int):
        mask = interval_idx == i
        if not mask.any():
            continue
        M = np.column_stack([np.ones(K), start[:, i], slopes[:, i]])
        X = np.linalg.lstsq(M, data[:, mask], rcond=None)[0]
        t = np.clip(X[2] / (X[1] + 1e-30), 0, 1)
        depths[mask] = (i + t) / n_int * depth_range
    return interval_idx, depths


def decode(
    name: CodingScheme,
    data: npt.NDArray,
    freq: float,
    *,
    ord_: int = 1,
    delta: float = 0.25,
) -> npt.NDArray:
    """Decode iToF measurements using the specified coding scheme.

    This is the primary entry point for depth recovery. It dispatches the
    raw measurement data to the appropriate decoding algorithm based on the
    *name* of the coding scheme.

    Args:
        name: Identifier of the coding scheme used for acquisition.
        data: Raw measurement array of shape ``(K, N_pixels)``.
        freq: Modulation frequency in Hz.
        ord_: Hilbert curve recursion order (only for Hilbert schemes).
            Defaults to ``1``.
        delta: Normalisation margin in ``[0, 0.5)`` (only for Hilbert schemes).
            Defaults to ``0.25``.

    Returns:
        Recovered depth map of shape ``(N_pixels,)`` in metres.

    Raises:
        NotImplementedError: If the coding scheme does not have a decoder.
    """
    if name in ("convSin", "deltaSin"):
        return _decode_sinusoid(data, freq)
    if name == "convSquare":
        return _decode_square(data, freq)
    if name == "singleRamp":
        return _decode_single_ramp(data, freq)
    if name == "doubleRamp":
        return _decode_double_ramp(data, freq)
    if name == "deltaHilbertDimOne":
        return _decode_hilbert(data, freq, dim=1, ord_=ord_, delta=delta)[1]
    if name == "deltaHilbertDimTwo":
        return _decode_hilbert(data, freq, dim=2, ord_=ord_, delta=delta)[1]
    if name == "deltaHilbertDimThree":
        return _decode_hilbert(data, freq, dim=3, ord_=ord_, delta=delta)[1]

    raise NotImplementedError(f"No decoder implemented for scheme: {name}")
