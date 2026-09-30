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
    _perm_matrix_to_codes,
    make_gray_codes_reduced,
    make_max_min_run_length_gray_codes,
)


def _compute_segment_distance(
    start: npt.NDArray,
    end: npt.NDArray,
    points: npt.NDArray,
    points_sq_sum: npt.NDArray | None = None,
) -> npt.NDArray:
    """Compute the squared distance from each point to a line segment.

    Args:
        start: Start point of the segment, shape ``(dim,)``.
        end: End point of the segment, shape ``(dim,)``.
        points: Query points stored as column vectors, shape ``(dim, n_points)``.
        points_sq_sum: Optional precomputed ``sum(points**2, axis=0)``.

    Returns:
        Squared distances from each column of *points* to the segment
        ``[start, end]``, shape ``(n_points,)``.
    """
    v = end - start
    seg_len_sq = np.sum(v**2)
    if points_sq_sum is None:
        points_sq_sum = np.sum(points**2, axis=0)

    y_sq_sum = points_sq_sum + np.sum(start**2) - 2 * (start @ points)
    if seg_len_sq == 0:
        return y_sq_sum

    y_dot_v = v @ points - np.dot(v, start)
    t = np.clip(y_dot_v / seg_len_sq, 0, 1)
    return y_sq_sum + t * (t * seg_len_sq - 2 * y_dot_v)


def decode_sinusoid(measurements: npt.NDArray, freq: float) -> npt.NDArray:
    """Decode sinusoidal (conventional or delta) iToF measurements.

    The offset and the two quadrature components are recovered with a linear
    least-squares fit of the taps onto the basis ``[1, cos(phi_k), sin(phi_k)]``.
    The phase is the angle of the two quadrature coefficients, and it is turned
    into depth with ``d = c * phi / (4 * pi * freq)``.

    Args:
        measurements: Raw measurement array of shape ``(n_captures, n_pixels)``.
        freq: Modulation frequency in Hz.

    Returns:
        Recovered depth map of shape ``(n_pixels,)`` in metres.

    Note:
        The phase is taken with ``atan2``, so pixels with no modulation signal
        (both quadrature coefficients zero) decode to exactly ``0`` m rather
        than being flagged. Callers that need to detect signal-free pixels
        should check the recovered amplitude themselves.

    See Also:
        :func:`decode` for the scheme-dispatching entry point, and
        :func:`visionsim.emulate.itof.coding.make_conv_sinusoidal_codes` or
        :func:`visionsim.emulate.itof.coding.make_delta_sinusoidal_codes` for the
        matching codes.
    """
    n_captures = measurements.shape[0]
    design_matrix = np.column_stack(
        [
            np.ones(n_captures),
            np.cos(2 * np.pi / n_captures * np.arange(n_captures)),
            np.sin(2 * np.pi / n_captures * np.arange(n_captures)),
        ]
    )
    coefficients = np.linalg.lstsq(design_matrix, measurements, rcond=None)[0]
    phase = np.arctan2(coefficients[2], coefficients[1]) % (2 * np.pi)
    return (scipy.constants.c * phase) / (4 * np.pi * freq)


def decode_square(measurements: npt.NDArray, freq: float) -> npt.NDArray:
    """Decode square-wave iToF measurements.

    Square waves correlate into a piecewise-linear response, so the depth range
    is split into intervals whose ordering is fixed by the capture count; the
    interval is selected from the relative ordering of the taps and the depth is
    then interpolated inside it.

    Args:
        measurements: Raw measurement array of shape ``(n_captures, n_pixels)``.
        freq: Modulation frequency in Hz.

    Returns:
        Recovered depth map of shape ``(n_pixels,)`` in metres.

    See Also:
        :func:`decode` for the scheme-dispatching entry point and
        :func:`visionsim.emulate.itof.coding.make_conv_square_codes` for the
        matching codes.
    """
    n_captures = measurements.shape[0]
    depth_range = scipy.constants.c / (2 * freq)
    n_intervals = 2 * n_captures
    # Build piecewise-linear start/end/slope for each interval
    start = np.zeros((n_captures, n_intervals))
    start[0] = np.abs(1 - np.arange(2 * n_captures) / n_captures)
    for i in range(1, n_captures):
        start[i] = np.roll(start[i - 1], 2)
    end = np.roll(start, -1, axis=1)
    slopes = end - start
    interval_midpoints = (start + end) / 2
    # Pairwise relative ordering of captures — vectorised (n_captures*(n_captures-1), n_intervals)
    row_indices, column_indices = np.array(
        [(row, column) for row in range(n_captures) for column in range(n_captures) if row != column]
    ).T
    pair_relations = (interval_midpoints[row_indices, :] >= interval_midpoints[column_indices, :]).astype(
        float
    )  # (n_captures*(n_captures-1), n_intervals)
    measurement_relations = (measurements[row_indices, :] >= measurements[column_indices, :]).astype(
        float
    )  # (n_captures*(n_captures-1), n_pixels)
    # Squared Hamming distance for every interval × pixel: (n_intervals, n_pixels)
    distance_matrix = np.array(
        [((measurement_relations - pair_relations[:, i : i + 1]) ** 2).sum(0) for i in range(n_intervals)]
    )
    interval_indices = np.argmin(distance_matrix, axis=0)
    depths = np.zeros(measurements.shape[1])
    for i in range(n_intervals):
        mask = interval_indices == i
        if not mask.any():
            continue
        design_matrix = np.column_stack([np.ones(n_captures), start[:, i], slopes[:, i]])
        coefficients = np.linalg.lstsq(design_matrix, measurements[:, mask], rcond=None)[0]
        t = np.clip(coefficients[2] / (coefficients[1] + 1e-30), 0, 1)
        depths[mask] = (i + t) / n_intervals * depth_range
    return depths


def decode_single_ramp(measurements: npt.NDArray, freq: float) -> npt.NDArray:
    """Decode single-ramp iToF measurements (n_captures=3).

    Args:
        measurements: Raw measurement array of shape ``(3, n_pixels)``.
        freq: Modulation frequency in Hz.

    Returns:
        Recovered depth map of shape ``(n_pixels,)`` in metres.

    Note:
        The single-ramp code spans two correlation periods, so the recovered
        depth is only unambiguous over ``c / (4 * freq)`` -- half of the
        ``c / (2 * freq)`` covered by the other schemes (see
        :func:`visionsim.emulate.itof.coding.unambiguous_range`). Scene points
        deeper than that fold back into the reported range.

    See Also:
        :func:`decode` for the scheme-dispatching entry point and
        :func:`visionsim.emulate.itof.coding.make_single_ramp_codes` for the
        matching codes.
    """
    design_matrix = np.array([[-1, 1, 0.5], [0, 1, 1], [0, 0, 1]], dtype=float)
    coefficients = np.linalg.lstsq(design_matrix, measurements, rcond=None)[0]
    normalized_depth = np.clip(coefficients[0] / (coefficients[1] + 1e-30), 0, 1)
    return normalized_depth * scipy.constants.c / (4 * freq)


def decode_double_ramp(measurements: npt.NDArray, freq: float) -> npt.NDArray:
    """Decode double-ramp iToF measurements (n_captures=3).

    Args:
        measurements: Raw measurement array of shape ``(3, n_pixels)``.
        freq: Modulation frequency in Hz.

    Returns:
        Recovered depth map of shape ``(n_pixels,)`` in metres.

    Note:
        Like :func:`decode_single_ramp`, the recovered depth is only
        unambiguous over ``c / (4 * freq)``; deeper scene points fold back into
        that range.

    See Also:
        :func:`decode` for the scheme-dispatching entry point and
        :func:`visionsim.emulate.itof.coding.make_double_ramp_codes` for the
        matching codes.
    """
    design_matrix = np.array([[-1, 1, 0.5], [1, 0, 0.5], [0, 0, 1]], dtype=float)
    coefficients = np.linalg.lstsq(design_matrix, measurements, rcond=None)[0]
    normalized_depth = np.clip(coefficients[0] / (coefficients[1] + 1e-30), 0, 1)
    return normalized_depth * scipy.constants.c / (4 * freq)


def _make_hilbert_code_endpoints(n_captures: int, dim: int, hilbert_order: int, hilbert_delta: float) -> npt.NDArray:
    """Build Hilbert curve code endpoint array for the decode lookup (dim >= 2).

    Note:
        The endpoints are the raw ``4**hilbert_order``/``8**hilbert_order``
        grid points of the Hilbert curve, *not* the arc-length expansion that
        :func:`visionsim.emulate.itof.coding.make_tof_hilbert_codes` uses for
        encoding. Segment directions are unaffected, but the two grids differ
        near junctions, which is the source of the misclassification described in
        :func:`visionsim.emulate.itof.decoding.decode_hilbert`.

    Args:
        n_captures: Number of captures.
        dim: Hilbert curve dimensionality (2 or 3).
        hilbert_order: Hilbert curve recursion order.
        hilbert_delta: Hilbert curve normalization margin in ``[0, 0.5)``.

    Returns:
        Code endpoint array of shape ``(n_captures, n_endpoints)``.

    Raises:
        ValueError: If the ``(n_captures, dim)`` combination is not supported.
    """
    _PERM = {(4, 2): _PERM_K4_DIM2, (5, 2): _PERM_K5_DIM2, (5, 3): _PERM_K5_DIM3}
    if (n_captures, dim) not in _PERM:
        raise ValueError(f"Unsupported n_captures={n_captures}, dim={dim}")
    hilbert_curve = _hilbert_3d if dim == 3 else _hilbert_2d
    coefficients = hilbert_curve(hilbert_order).astype(float)
    coefficients = (coefficients - coefficients.min(axis=1, keepdims=True)) / (
        coefficients.max(axis=1, keepdims=True) - coefficients.min(axis=1, keepdims=True) + 1e-30
    )
    coefficients = coefficients * (1 - 2 * hilbert_delta) + hilbert_delta
    return _perm_matrix_to_codes(_PERM[(n_captures, dim)], coefficients)


def decode_hilbert(
    measurements: npt.NDArray,
    freq: float,
    dim: int = 1,
    hilbert_order: int = 1,
    hilbert_delta: float = 0.25,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Decode Hilbert-coded iToF measurements for any supported dimensionality.

    Dispatches to a Gray-code interval classifier for ``dim=1`` and to a
    Hilbert-curve segment-distance classifier for ``dim=2`` or ``dim=3``.
    The final least-squares depth-recovery step is shared across all paths.

    Args:
        measurements: Raw measurement array of shape ``(n_captures, n_pixels)``.
        freq: Modulation frequency in Hz.
        dim: Hilbert curve dimensionality.  ``1`` uses the Gray-code path;
            ``2`` or ``3`` use the Hilbert-curve path.  Defaults to ``1``.
        hilbert_order: Hilbert curve recursion order (ignored for ``dim=1``).
            Defaults to ``1``.
        hilbert_delta: Hilbert curve normalization margin in ``[0, 0.5)`` (ignored for ``dim=1``).
            Defaults to ``0.25``.

    Returns:
        Tuple of ``(interval_indices, depths)`` where *interval_indices* has shape
        ``(n_pixels,)`` (integer bin index) and *depths* has shape
        ``(n_pixels,)`` in metres.

    Note:
        The ``dim >= 2`` classifier compares each pixel's per-channel
        normalized measurements against segment endpoints that live on the
        coarse Hilbert grid, while the
        encoder expands the curve along its arc length. Pixels whose channels
        nearly vanish (they sit close to a segment junction) can therefore be
        misassigned, which shows up as isolated large depth errors for the
        ``n_captures=5`` codes. The ``dim == 1`` and ``n_captures=4`` paths are unaffected.

    Raises:
        ValueError: If the ``(n_captures, dim)`` combination is not supported.

    See Also:
        :func:`decode` for the scheme-dispatching entry point that returns only the
        depths, and :func:`visionsim.emulate.itof.coding.make_tof_hilbert_codes` for
        the matching reference codes.
    """
    n_captures, n_pixels = measurements.shape[0], measurements.shape[1]
    depth_range = scipy.constants.c / (2 * freq)

    if dim == 1:
        # Gray-code path
        gray_codes = make_max_min_run_length_gray_codes() if n_captures == 5 else make_gray_codes_reduced(n_captures)
        n_intervals = gray_codes.shape[0]
        start = gray_codes.T.astype(float)
        end = np.roll(start, -1, axis=1)
        slopes = end - start

        # Threshold each channel: 1.0 above the mid-range value, 0.0 below, and
        # -1.0 for the most-transitioning channel (a "don't-care" marker).
        value_max, value_min = measurements.max(axis=0), measurements.min(axis=0)
        transitioning_index = np.argmax(
            np.minimum(np.abs(value_max - measurements), np.abs(value_min - measurements)), axis=0
        )
        thresholded = (measurements > (value_max + value_min) / 2).astype(float)
        thresholded[transitioning_index, np.arange(n_pixels)] = -1.0

        # Assign each pixel to the first interval whose constant channels match.
        # Interval 0 is a real interval, so unassigned pixels are tracked with a
        # -1 sentinel rather than by testing against 0.
        interval_indices = np.full(n_pixels, -1, dtype=int)
        for i in range(n_intervals):
            todo = interval_indices < 0
            if not todo.any():
                break
            const_rows = np.where(slopes[:, i] == 0)[0]
            mask = todo.copy()
            for j in const_rows:
                mask &= thresholded[j] == start[j, i]
            interval_indices[mask] = i

        # Fallback for the few pixels that no interval claimed: nearest interval
        # by squared difference between their thresholded channels and the
        # interval's constant channels. Only truly unassigned pixels are passed
        # through this path.
        unassigned = interval_indices < 0
        if unassigned.any():
            unassigned_measurements = measurements[:, unassigned]
            unassigned_thresholded = (
                unassigned_measurements > (unassigned_measurements.max(0) + unassigned_measurements.min(0)) / 2
            ).astype(float)
            squared_differences = ((unassigned_thresholded[:, :, None] - start[:, None, :]) ** 2).sum(
                axis=0
            )  # (n_ua, n_intervals)
            interval_indices[unassigned] = np.argmin(squared_differences, axis=1)

    else:
        # Hilbert-curve path (dim = 2 or 3)
        _PERM = {(4, 2): _PERM_K4_DIM2, (5, 2): _PERM_K5_DIM2, (5, 3): _PERM_K5_DIM3}
        if (n_captures, dim) not in _PERM:
            raise ValueError(f"Unsupported n_captures={n_captures}, dim={dim}")
        permutation_matrix = _PERM[(n_captures, dim)]
        num_segments = permutation_matrix.shape[1]
        points_per_segment = (8**hilbert_order) if dim == 3 else (4**hilbert_order)
        total_points = num_segments * points_per_segment

        n_intervals = num_segments * (points_per_segment - 1)
        skip = set(range(points_per_segment - 1, total_points, points_per_segment))

        endpoints = _make_hilbert_code_endpoints(n_captures, dim, hilbert_order, hilbert_delta)
        endpoint_indices = [i for i in range(endpoints.shape[1]) if i not in skip]
        start = endpoints[:, endpoint_indices[:n_intervals]]
        end = endpoints[:, [i + 1 for i in endpoint_indices[:n_intervals]]]
        slopes = end - start

        # Normalized endpoints specifically for robust segment classification
        endpoint_max, endpoint_min = endpoints.max(axis=0), endpoints.min(axis=0)
        normalized_endpoints = (endpoints - endpoint_min) / (endpoint_max - endpoint_min + 1e-30)
        normalized_start = normalized_endpoints[:, endpoint_indices[:n_intervals]]
        normalized_end = normalized_endpoints[:, [i + 1 for i in endpoint_indices[:n_intervals]]]

        value_max, value_min = measurements.max(axis=0), measurements.min(axis=0)
        normalized_measurements = (measurements - value_min) / (value_max - value_min + 1e-30)

        points_sq_sum = np.sum(normalized_measurements**2, axis=0)
        interval_indices = np.zeros(n_pixels, dtype=int)
        min_dist = np.full(n_pixels, np.inf)

        for i in range(n_intervals):
            d = _compute_segment_distance(
                normalized_start[:, i], normalized_end[:, i], normalized_measurements, points_sq_sum
            )
            better = d < min_dist
            interval_indices[better] = i
            min_dist[better] = d[better]

    # Shared depth-recovery: least-squares within each interval
    depths = np.zeros(n_pixels)
    for i in range(n_intervals):
        mask = interval_indices == i
        if not mask.any():
            continue
        design_matrix = np.column_stack([np.ones(n_captures), start[:, i], slopes[:, i]])
        coefficients = np.linalg.lstsq(design_matrix, measurements[:, mask], rcond=None)[0]
        t = np.clip(coefficients[2] / (coefficients[1] + 1e-30), 0, 1)
        depths[mask] = (i + t) / n_intervals * depth_range
    return interval_indices, depths


#: Capture counts supported by each decoder. ``multFreqSin`` is validated
#: separately (``n_captures == 4``, or an odd ``n_captures >= 5``) since it accepts a family of
#: values derived from ``freq_vec``.
_SUPPORTED_K: dict[str, tuple[int, ...]] = {
    "convSin": (3, 4, 5),
    "deltaSin": (3, 4, 5),
    "convSquare": (3, 4, 5),
    "singleRamp": (3,),
    "doubleRamp": (3,),
    "deltaHilbertDimOne": (3, 4, 5, 6),
    "deltaHilbertDimTwo": (4, 5),
    "deltaHilbertDimThree": (5,),
}


def _validate_captures(scheme: CodingScheme, n_captures: int) -> None:
    """Validate the capture count of a measurement array against a scheme.

    Args:
        scheme: Coding scheme identifier.
        n_captures: Number of rows in the measurement array.

    Raises:
        NotImplementedError: If *scheme* has no decoder.
        ValueError: If *n_captures* is not supported by the scheme's decoder.
    """
    if scheme == "multFreqSin":
        if n_captures != 4 and (n_captures < 5 or (n_captures - 3) % 2 != 0):
            raise ValueError(
                f"Scheme 'multFreqSin' expects n_captures=4 or an odd n_captures>=5, got n_captures={n_captures}"
            )
        return
    if scheme not in _SUPPORTED_K:
        raise NotImplementedError(f"No decoder implemented for scheme: {scheme}")
    supported = _SUPPORTED_K[scheme]
    if n_captures not in supported:
        raise ValueError(f"Scheme '{scheme}' expects n_captures in {supported}, got n_captures={n_captures}")


def decode_mult_freq_sinusoid(
    measurements: npt.NDArray,
    freq: float,
    freq_vec: npt.NDArray,
    shifts_vec: npt.NDArray,
) -> npt.NDArray:
    """Decode multi-frequency sinusoidal iToF measurements.

    Frequencies are unwrapped coarse-to-fine: the phase of a group of uniformly
    shifted taps recovers a wrapped depth, and every finer (higher) frequency
    then selects the branch of its shorter period closest to that estimate. The
    result is unambiguous over the largest effective range of any group,
    ``c / (2 * freq * min(freq_vec))``.

    Note:
        The tap layout is ``n_captures == 4`` with three uniformly shifted taps
        on one frequency plus a single low-frequency tap, or ``n_captures >= 5``
        with three shifts on the first frequency and two per further frequency.
        For ``n_captures == 4`` the three-shift frequency must cover the full
        range (``freq_vec[1] == 1`` unit): if the single low-frequency tap would
        be needed to widen it, ``NotImplementedError`` is raised rather than
        returning aliased depths.

    Args:
        measurements: Raw measurement array of shape ``(n_captures, n_pixels)``.
        freq: Base modulation frequency in Hz. ``freq_vec`` are multipliers of
            it, so a tap with multiplier ``f`` has period ``c / (2 * freq * f)``.
        freq_vec: Frequency multiplier per tap, shape ``(n_captures,)``.
        shifts_vec: Phase shift (radians) per tap, shape ``(n_captures,)``.

    Returns:
        Recovered depth map of shape ``(n_pixels,)`` in metres.

    Raises:
        ValueError: If the tap vectors do not match the measurements, or if the tap
            layout cannot be decoded.

    See Also:
        :func:`decode` for the scheme-dispatching entry point and
        :func:`visionsim.emulate.itof.coding.make_multi_freq_sinusoidal_codes` for the
        matching codes.
    """
    freq_vec, shifts_vec = np.asarray(freq_vec, dtype=float), np.asarray(shifts_vec, dtype=float)
    n_captures = measurements.shape[0]
    if freq_vec.shape != (n_captures,) or shifts_vec.shape != (n_captures,):
        raise ValueError(
            f"freq_vec and shifts_vec must have shape ({n_captures},), got {freq_vec.shape} and {shifts_vec.shape}"
        )

    # Tap layout: n_captures == 4 is a single
    # low-frequency tap plus three uniformly shifted taps, n_captures >= 5 is three
    # shifts on the first frequency and two per further frequency.
    if n_captures == 4:
        group_taps = [[0], [1, 2, 3]]
    else:
        group_taps = [[0, 1, 2]] + [[2 * i + 1, 2 * i + 2] for i in range(1, (n_captures - 3) // 2 + 1)]

    multi_taps = [taps for taps in group_taps if len(taps) > 1]
    single_taps = [taps[0] for taps in group_taps if len(taps) == 1]
    if not multi_taps:
        raise ValueError("'multFreqSin' needs at least one frequency captured with multiple phase shifts")

    for taps in group_taps:
        multipliers = freq_vec[taps]
        if not np.allclose(multipliers, multipliers[0]):
            raise ValueError(f"Taps {taps} are assumed to share a single frequency, got {multipliers}")

    # Joint solve for one shared offset and a cos/sin pair per multi-shift
    # frequency: solving each group separately would be underdetermined for the
    # two-shift groups, since offset and amplitude cannot be separated from a
    # single sample, so the offset and the cos/sin coefficients are fitted jointly.
    rows = np.concatenate(multi_taps)
    positions = {tap: index for index, tap in enumerate(rows)}
    design = np.zeros((rows.size, 1 + 2 * len(multi_taps)))
    design[:, 0] = 1.0
    for group_index, taps in enumerate(multi_taps):
        index = [positions[tap] for tap in taps]
        design[index, 1 + 2 * group_index] = np.cos(shifts_vec[taps])
        design[index, 2 + 2 * group_index] = np.sin(shifts_vec[taps])
    coefficients = np.linalg.lstsq(design, measurements[rows], rcond=None)[0]

    # (effective range, phase, period, is_multi_shift) of every group.
    groups: list[tuple[float, npt.NDArray, float, bool]] = []
    offset, reference_amplitude = coefficients[0], None
    for group_index, taps in enumerate(multi_taps):
        period = scipy.constants.c / (2 * freq * float(freq_vec[taps[0]]))
        phase = np.arctan2(coefficients[2 + 2 * group_index], coefficients[1 + 2 * group_index]) % (2 * np.pi)
        groups.append((period, phase, period, True))
        if reference_amplitude is None:
            reference_amplitude = np.hypot(coefficients[1 + 2 * group_index], coefficients[2 + 2 * group_index])

    for tap in single_taps:
        if reference_amplitude is None:
            raise ValueError("A single-tap frequency can only be decoded alongside a multi-shift frequency")
        period = scipy.constants.c / (2 * freq * float(freq_vec[tap]))
        # Only |phase| is recoverable from one sample, which halves the range; the
        # amplitude is taken from the reconstructed multi-shift frequency.
        ratio = np.clip((measurements[tap] - offset) / (reference_amplitude + 1e-30), -1, 1)
        groups.append((period / 2, np.arccos(ratio), period, False))

    # Start from the widest effective range (multi-shift groups win ties, their
    # phase being sign-resolved), then let every finer multi-shift frequency
    # select the branch of its shorter period closest to the current estimate.
    groups.sort(key=lambda group: (group[0], group[3]), reverse=True)
    _, base_phase, base_period, base_is_multi_shift = groups[0]

    if not base_is_multi_shift:
        # The half-frequency tap of the n_captures == 4 layout is only redundant
        # in the freq_vec = [0.5, f, f, f] configuration, where a multi-shift
        # group already covers the full range. Whenever it would be needed to
        # widen the range it cannot be inverted, because half a period does not
        # yield a plain sinusoid. Refuse instead of silently aliasing.
        raise NotImplementedError(
            "'multFreqSin' with n_captures=4 requires the three-shift frequency to cover the full "
            "range (freq_vec[1] == 1 unit); widening it with the single low-frequency tap is not "
            "supported. Use n_captures>=5 or a higher freq_vec[1] instead."
        )

    depths = base_period * base_phase / (2 * np.pi)
    uncertainty = groups[0][0]
    for _, phase, period, is_multi_shift in groups[1:]:
        if not is_multi_shift or period >= uncertainty:
            continue
        branch = np.round(depths / period - phase / (2 * np.pi))
        depths = period * (phase / (2 * np.pi) + branch)
        uncertainty = period
    return depths


def decode(
    scheme: CodingScheme,
    measurements: npt.NDArray,
    freq: float,
    *,
    hilbert_order: int = 1,
    hilbert_delta: float = 0.25,
    freq_vec: npt.NDArray | None = None,
    shifts_vec: npt.NDArray | None = None,
) -> npt.NDArray:
    """Decode iToF measurements using the specified coding scheme.

    This is the primary entry point for depth recovery. It dispatches the
    raw measurement data to the appropriate decoding algorithm based on the
    *scheme* of the coding scheme.

    Args:
        scheme: Identifier of the coding scheme used for acquisition.
        measurements: Raw measurement array of shape ``(n_captures, n_pixels)``.
        freq: Modulation frequency in Hz. Note that the ramp schemes recover
            depth over ``c / (4 * freq)`` rather than ``c / (2 * freq)`` (see
            :func:`visionsim.emulate.itof.coding.unambiguous_range`).
        hilbert_order: Hilbert curve recursion order (only for Hilbert schemes).
            Defaults to ``1``.
        hilbert_delta: Hilbert curve normalization margin in ``[0, 0.5)`` (only for Hilbert schemes).
            Defaults to ``0.25``.
        freq_vec: Frequency multipliers per tap, required by ``"multFreqSin"``.
        shifts_vec: Phase shifts (radians) per tap, required by
            ``"multFreqSin"``.

    Returns:
        Recovered depth map of shape ``(n_pixels,)`` in metres.

    Raises:
        NotImplementedError: If the coding scheme does not have a decoder.
        ValueError: If *measurements* is not a 2-D array, if its capture count is not
            supported by the scheme (e.g. ``n_captures != 3`` for the ramp schemes), or
            if a required argument such as ``freq_vec`` is missing.

    See Also:
        The per-scheme decoders this function dispatches to: :func:`decode_sinusoid`
        (``convSin``, ``deltaSin``), :func:`decode_square` (``convSquare``),
        :func:`decode_single_ramp` (``singleRamp``), :func:`decode_double_ramp`
        (``doubleRamp``), :func:`decode_hilbert` (``deltaHilbertDimOne``,
        ``deltaHilbertDimTwo``, ``deltaHilbertDimThree``) and
        :func:`decode_mult_freq_sinusoid` (``multFreqSin``).
    """
    if not isinstance(measurements, np.ndarray) or measurements.ndim != 2:
        raise ValueError(
            f"Expected measurements of shape (n_captures, n_pixels), got {getattr(measurements, 'shape', type(measurements))}"
        )

    _validate_captures(scheme, measurements.shape[0])

    if scheme in ("convSin", "deltaSin"):
        return decode_sinusoid(measurements, freq)
    if scheme == "convSquare":
        return decode_square(measurements, freq)
    if scheme == "singleRamp":
        return decode_single_ramp(measurements, freq)
    if scheme == "doubleRamp":
        return decode_double_ramp(measurements, freq)
    if scheme == "deltaHilbertDimOne":
        return decode_hilbert(measurements, freq, dim=1, hilbert_order=hilbert_order, hilbert_delta=hilbert_delta)[1]
    if scheme == "deltaHilbertDimTwo":
        return decode_hilbert(measurements, freq, dim=2, hilbert_order=hilbert_order, hilbert_delta=hilbert_delta)[1]
    if scheme == "deltaHilbertDimThree":
        return decode_hilbert(measurements, freq, dim=3, hilbert_order=hilbert_order, hilbert_delta=hilbert_delta)[1]
    if scheme == "multFreqSin":
        if freq_vec is None or shifts_vec is None:
            raise ValueError("'multFreqSin' requires freq_vec and shifts_vec")
        return decode_mult_freq_sinusoid(measurements, freq, freq_vec, shifts_vec)

    raise NotImplementedError(f"No decoder implemented for scheme: {scheme}")
