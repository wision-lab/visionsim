"""Image simulation and projection pipeline for iToF."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.constants
from scipy.interpolate import PchipInterpolator
from scipy.linalg import circulant


def compute_correlation_function(
    modulation_code: npt.NDArray,
    reference_code: npt.NDArray,
    time_resolution: float,
    *,
    n_depths: int | None = None,
) -> npt.NDArray:
    """Compute the correlation function between one modulation/reference pair.

    Args:
        modulation_code: 1-D modulation code of length ``n_bins``.
        reference_code: 1-D reference code of length ``n_bins``.
        time_resolution: Time resolution (seconds per bin).
        n_depths: Number of depth bins to return. Defaults to ``len(modulation_code)``.
            The correlation function is periodic with the code period, so
            requesting more bins than that repeats it: ``len(modulation_code) + 1``
            bins give the value at exactly ``max_depth``, which equals the value
            at depth ``0``.

    Returns:
        Correlation values sampled at ``n_depths`` evenly spaced depth bins,
        shape ``(n_depths,)``.

    Raises:
        ValueError: If *modulation_code* and *reference_code* differ in length, or if
            *n_depths* is not positive.
    """
    if len(modulation_code) != len(reference_code):
        raise ValueError(
            f"modulation_code and reference_code must have the same length, got {len(modulation_code)} and {len(reference_code)}"
        )
    if n_depths is not None and n_depths < 1:
        raise ValueError(f"n_depths must be a positive number of depth bins, got {n_depths}")

    n_bins = len(modulation_code)
    period = n_bins * time_resolution
    # scaled_modulation represents the code scaled to period/time_resolution (index.e. n_bins bins)
    scaled_modulation = modulation_code * (period / time_resolution)
    circulant_matrix = circulant(scaled_modulation).T
    correlation_full = (circulant_matrix @ reference_code * time_resolution).ravel()
    if n_depths is None or n_depths == n_bins:
        return correlation_full
    return np.resize(correlation_full, n_depths)


def simulate_measurements(
    depths: npt.NDArray,
    albedos: npt.NDArray,
    modulation_codes: npt.NDArray,
    reference_codes: npt.NDArray,
    period: float,
    *,
    exposure_time: float = 1.0,
    ambient_power: float = 0.0,
    light_power: float = 1.0,
) -> npt.NDArray:
    """Directly simulate iToF measurements from depths and albedos.

    This implements a physically accurate direct model including $1/d^2$
    radiometric falloff and albedo scaling [1]_.

    Args:
        depths: Depth map in metres, arbitrary shape ``(*spatial,)``.
        albedos: Reflectance values (0-1), same shape as *depths*.
        modulation_codes: Modulation codes, shape ``(n_captures, n_bins)``.
        reference_codes: Reference (demodulation) codes, shape ``(n_captures, n_bins)``.
        period: Duration of one code period in seconds, i.e. ``1 / f``. The
            correlation function is periodic over it, so ``n_bins`` bins span the
            round-trip distance ``c * period / 2`` and depths wrap back over it.
        exposure_time: Camera exposure time in seconds. Defaults to 1.0.
        ambient_power: Average ambient irradiance. Defaults to 0.0.
        light_power: Peak active light intensity. Defaults to 1.0.

    Returns:
        Simulated measurement array of shape ``(n_captures, *spatial)``.

    References:
        .. [1] `Gupta et al. (2018), "What Are Optimal Coding Functions for Time-of-Flight Imaging?"
           <https://wisionlab.com/wp-content/uploads/2018/07/Gupta_ToG18_ToFOptimalCodingFunctions.pdf>`_
    """
    n_captures, n_bins = modulation_codes.shape
    # The codes span one period, so the correlation wraps over the distance
    # light travels in that time, and back out.
    max_depth = scipy.constants.c * period / 2
    time_resolution = period / n_bins

    # kappa is the integral of the reference functions in a period
    kappa = reference_codes.sum(axis=1) * time_resolution  # (n_captures,)
    beta = albedos / (depths**2 + 1e-30)

    correlation_distances = np.linspace(0, max_depth, n_bins, endpoint=False)
    measurements = np.zeros((n_captures, *depths.shape))

    for index in range(n_captures):
        # Correlation function scaled by light power
        correlation = light_power * compute_correlation_function(
            modulation_codes[index], reference_codes[index], time_resolution, n_depths=n_bins
        )
        interpolator = PchipInterpolator(correlation_distances, correlation)

        # The phase of the correlation function wraps around the code period,
        # while the 1/d**2 falloff below follows the true distance.
        correlation_samples = interpolator(depths % max_depth)

        # Reference formula: (T_exp / T_period) * (beta * correlation + ambient * kappa * albedo)
        measurements[index] = (exposure_time / period) * (
            beta * correlation_samples + ambient_power * kappa[index] * albedos
        )

    return measurements
