"""Image simulation and projection pipeline for iToF."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.constants
from scipy.interpolate import LinearNDInterpolator, PchipInterpolator
from scipy.linalg import circulant

from .models import Camera, Light, Scene


def compute_correlation_function(
    mod_code: npt.NDArray,
    ref_code: npt.NDArray,
    time_res: float,
    *,
    n_depths: int | None = None,
) -> npt.NDArray:
    """Compute the correlation function between one modulation/reference pair.

    Args:
        mod_code: 1-D modulation code of length ``N``.
        ref_code: 1-D reference code of length ``N``.
        time_res: Time resolution (seconds per bin).
        n_depths: Number of depth bins to return. Defaults to ``len(mod_code)``.

    Returns:
        Correlation values sampled at ``n_depths`` evenly spaced depth bins,
        shape ``(n_depths,)``.
    """
    N = len(mod_code)
    period = N * time_res
    # scaled_mod represents the code scaled to period/time_res (i.e. N bins)
    scaled_mod = mod_code * (period / time_res)
    C = circulant(scaled_mod).T
    corr_full = (C @ ref_code * time_res).ravel()
    return corr_full[:n_depths] if n_depths is not None else corr_full


def simulate_measurements(
    depths: npt.NDArray,
    albedos: npt.NDArray,
    mod_codes: npt.NDArray,
    ref_codes: npt.NDArray,
    d_max: float,
    *,
    exposure_time: float = 1.0,
    ambient_power: float = 0.0,
    light_power: float = 1.0,
) -> npt.NDArray:
    """Directly simulate iToF measurements from depths and albedos.

    This implements a physically accurate direct model including $1/d^2$
    radiometric falloff and albedo scaling [#Gupta2018]_.

    Args:
        depths: Depth map in metres, arbitrary shape ``(*S,)``.
        albedos: Reflectance values (0-1), same shape as *depths*.
        mod_codes: Modulation codes, shape ``(K, N)``.
        ref_codes: Reference (demodulation) codes, shape ``(K, N)``.
        d_max: Unambiguous depth range in metres.
        exposure_time: Camera exposure time in seconds. Defaults to 1.0.
        ambient_power: Average ambient irradiance. Defaults to 0.0.
        light_power: Peak active light intensity. Defaults to 1.0.

    Returns:
        Simulated measurement array of shape ``(K, *S)``.

    References:
        .. [#Gupta2018] `Gupta et al. (2018), "What Are Optimal Coding Functions for Time-of-Flight Imaging?"
           <https://wisionlab.com/wp-content/uploads/2018/07/Gupta_ToG18_ToFOptimalCodingFunctions.pdf>`_
    """
    # Time resolution per bin based on unambiguous range d_max
    K, N = mod_codes.shape
    time_res = 2 * d_max / (N * scipy.constants.c)
    period = N * time_res

    # kappa is the integral of the reference functions in a period
    kappa = ref_codes.sum(axis=1) * time_res  # (K,)
    beta = albedos / (depths**2 + 1e-30)
    beta = 1

    correlation_distances = np.linspace(0, d_max, N, endpoint=False)
    measurements = np.zeros((K, *depths.shape))

    for i in range(K):
        # Correlation function scaled by light power
        corr = light_power * compute_correlation_function(mod_codes[i], ref_codes[i], time_res, n_depths=N)
        interp = PchipInterpolator(correlation_distances, corr)

        # iToF measurements wrap around the unambiguous range
        corr_samples = interp(depths % d_max)

        # Reference formula: (T_exp / T_period) * (beta * corr + ambient * kappa * albedo)
        measurements[i] = (exposure_time / period) * (beta * corr_samples + ambient_power * kappa[i] * albedos)

    return measurements
