from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter
from typing_extensions import Literal

from visionsim.utils.color import raw_bayer_to_rgb, rgb_to_raw_bayer
from visionsim.utils.imgproc import unsharp_mask


def emulate_rgb_from_sequence(
    sequence: npt.ArrayLike,
    shutter_frac: float = 1.0,
    readout_std: float = 0.0,
    fwc: float = 10000.0,
    adc_bitdepth: int = 12,
    flux_gain: float = 2**12,
    sensor_gain: float = 1.0,
    mosaic: bool = False,
    demosaic: Literal["off", "bilinear", "MHC04"] = "off",
    denoise_sigma: float = 0.0,
    sharpen_weight: float = 0.0,
    rng: np.random.Generator | None = None,
) -> npt.NDArray:
    """Emulates a conventional RGB camera [#Hasinoff2010]_ from a sequence of intensity frames.

    For demosaicing details see :func:`raw_bayer_to_rgb <visionsim.utils.color.raw_bayer_to_rgb>`.

    Note:
        Motion-blur is approximated by averaging consecutive ground truth frames,
        this can be done more efficiently if optical flow is available.
        See :func:`emulate_rgb_from_flow <visionsim.emulate.rgb.emulate_rgb_from_flow>` for more.

    Args:
        sequence (npt.ArrayLike): Input sequence of linear-intensity frames, can be a collection of frames,
            or np/torch array with time as the first dimension.
        shutter_frac (float, optional): fraction of inter-frame duration the shutter is active. Range [0, 1]
        readout_std (float, optional): Standard deviation of zero mean Gaussian read noise. Defaults to 0.0.
        fwc (float, optional): Full well capacity, used for normalization. Defaults to 10000.0.
        adc_bitdepth (int, optional): Resolution of ADC in bits. Defaults to 12.
        flux_gain (float, optional): factor to scale the input [0, 1] image _before_ Poisson sampling
        sensor_gain (float, optional): factor to scale the photo-electron reading _after_ Poisson sampling.
            This setting is inversely proportional to ISO gain, with a camera-dependent proportionality constant.
            Specifically, ISO = U / sensor_gain where U is a camera-dependent constant [#Hasinoff2010]_.
        mosaic (bool, optional): implement one array with mosaiced R-/G-/B-sensitive pixels or an innately 3-channel sensor
        demosaic (string, optional): demosaicing method to use if "mosaic" is set (default "off")
        denoise_sigma (float, optional): Gaussian blur kernel sigma (disabled if 0.0)
        sharpen_weight (float, optional): sharpening weight (disabled if 0.0)
        rng (np.random.Generator, optional): Optional random number generator. Defaults to none.

    Returns:
        npt.NDArray: Quantized linear-intensity RGB patch as floating point array (range [0, 1])

    References:
        ..  [#Hasinoff2010] S. W. Hasinoff, F. Durand, and W. T. Freeman,
            “Noise-optimal capture for high dynamic range photography,”
            CVPR 2010.
    """
    # Get mean of linear-intensity frames.
    burst_size = int(max(1, np.ceil(len(sequence) * shutter_frac)))
    sequence = np.array(sequence[:burst_size])
    patch = np.mean(sequence, axis=0)

    # Convert to raw bayer if mosaic, this just samples the color channels
    if mosaic:
        patch = rgb_to_raw_bayer(patch)

    # So far patch is GT motion blurred frame, without any noise or gain applied
    # we need to scale it by flux gain and shutter fraction to get the correct amount of light
    patch *= flux_gain * shutter_frac

    # Roughly translating the model in Eqs. (1,2) and Fig. 1 of Hasinoff et al.
    # Perform poisson sampling
    rng = np.random.default_rng() if rng is None else rng
    patch = cast(np.ndarray, rng.poisson(patch)).astype(float)

    # Clip to full-well capacity, add readout noise, apply ISO gain
    patch = np.clip(patch, 0, fwc)
    patch += rng.normal(0, readout_std, size=patch.shape)
    patch *= sensor_gain

    # Assume perfect quantization in ADC
    patch = np.round(np.clip(patch, 0, (2**adc_bitdepth - 1)))
    patch = patch / (2**adc_bitdepth - 1)

    # De-mosaicing: necessary if data is mosaiced, so can't be `None`.
    # ("off" is not a no-op, it still creates a full 3-channel image from 1,
    # albeit a bad one)
    if mosaic:
        patch = raw_bayer_to_rgb(patch, method=demosaic)

    # De-noising and sharpening
    if denoise_sigma != 0.0:
        patch = gaussian_filter(patch, denoise_sigma)
    if sharpen_weight != 0.0:
        patch = unsharp_mask(patch, sigma=max(1, denoise_sigma), amount=sharpen_weight)
    return patch


def emulate_rgb_from_flow():
    """Not (Yet) Implemented"""
    raise NotImplementedError
