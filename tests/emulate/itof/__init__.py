"""Shared helpers for the iToF emulation tests.

Round-trip tests synthetically capture noiseless measurements with
:func:`simulate_measurements` and check that :func:`decode` recovers the depths
they were generated from, for every supported coding scheme. Since the
simulation is our ground truth, this validates the encoder/decoder pair
end-to-end rather than reconstructing expected values by hand.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.constants

from visionsim.emulate.itof import CodingScheme, decode, make_coding_functions, simulate_measurements

C = scipy.constants.c
FREQ = 120e6
PERIOD = 1 / FREQ  # one code period, 1 / f
D_MAX = C / (2 * FREQ)  # unambiguous range of a single-period scheme, ~1.25 m

# Tap layouts of the multi-frequency scheme, as documented by
# ``decode_mult_freq_sinusoid``.
MULT_FREQ_LAYOUTS: dict[int, tuple[list[float], list[float]]] = {
    4: ([0.5, 1.0, 1.0, 1.0], [0.0, 0.0, 2 * np.pi / 3, 4 * np.pi / 3]),
    5: ([1.0, 1.0, 1.0, 2.0, 2.0], [0.0, 2 * np.pi / 3, 4 * np.pi / 3, 0.0, np.pi / 2]),
    7: (
        [1.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0],
        [0.0, 2 * np.pi / 3, 4 * np.pi / 3, 0.0, np.pi / 2, 0.0, np.pi / 2],
    ),
}


def tap_vectors(scheme: CodingScheme, n_captures: int) -> tuple[npt.NDArray, npt.NDArray]:
    """Return the per-tap frequency multipliers and shifts of a scheme."""
    if scheme == "multFreqSin":
        freq_vec, shifts_vec = MULT_FREQ_LAYOUTS[n_captures]
        return np.asarray(freq_vec), np.asarray(shifts_vec)
    return np.array([1.0]), np.array([0.0])


def codes(
    scheme: CodingScheme,
    n_captures: int,
    n_depths: int = 1001,
    *,
    hilbert_order: int = 1,
    hilbert_delta: float = 0.25,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Build codes for a scheme, filling in multFreqSin's required tap vectors."""
    if scheme == "multFreqSin":
        freq_vec, shifts_vec = tap_vectors(scheme, n_captures)
        return make_coding_functions(scheme, n_captures, n_depths, freq_vec=freq_vec, shifts_vec=shifts_vec)
    return make_coding_functions(scheme, n_captures, n_depths, hilbert_order=hilbert_order, hilbert_delta=hilbert_delta)


def roundtrip(
    scheme: CodingScheme,
    depths: npt.ArrayLike,
    *,
    n_captures: int,
    n_depths: int = 1001,
    freq: float = FREQ,
    hilbert_order: int = 1,
    hilbert_delta: float = 0.25,
    albedo: float = 0.8,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Synthetically capture ``depths`` with ``scheme`` and decode them back.

    Returns:
        Tuple of ``(decoded_depths, true_depths)``.
    """
    freq_vec, shifts_vec = tap_vectors(scheme, n_captures)
    mod, ref = codes(scheme, n_captures, n_depths, hilbert_order=hilbert_order, hilbert_delta=hilbert_delta)
    true_depths = np.atleast_1d(np.asarray(depths, dtype=float))
    measurements = simulate_measurements(true_depths, np.full(true_depths.shape, albedo), mod, ref, PERIOD)
    mult_freq_kwargs = {"freq_vec": freq_vec, "shifts_vec": shifts_vec} if scheme == "multFreqSin" else {}
    decoded = decode(
        scheme, measurements, freq, hilbert_order=hilbert_order, hilbert_delta=hilbert_delta, **mult_freq_kwargs
    )
    return decoded, true_depths
