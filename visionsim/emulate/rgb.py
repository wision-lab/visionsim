from __future__ import annotations

import numpy as np
import numpy.typing as npt
from skimage.filters import gaussian, unsharp_mask
from typing_extensions import Literal

from visionsim.utils.color import linearrgb_to_srgb, raw2rgb_bayer, rgb2raw_bayer


def emulate_rgb_from_sequence(
    sequence: npt.ArrayLike,
    shutter_angle_degrees: float = 360.0,
    readout_std: float = 0.3,
    fwc: float = 10000.0,
    bitdepth: int = 12,
    scale_flux: float = 1.0,
    gain_ISO: float = 1.0,
    demosaic_method: Literal["off", "bilinear", "MHC04"] = "bilinear",
    denoise_sigma: float = 0.0,
    sharpen_weight: float = 0.0,
    rng: np.random.Generator | None = None,
) -> npt.NDArray:
    """Emulates a conventional RGB camera from a sequence of intensity frames.

    Note:
        Motion-blur is approximated by averaging consecutive ground truth frames,
        this can be done more efficiently if optical flow is available.
        See `emulate_rgb_from_flow` for more.

    Args:
        sequence (npt.ArrayLike): Input sequence of linear-intensity frames, can be a collection of frames,
            or np/torch array with time as the first dimension.
        shutter_angle_degrees (float, optional): fraction of inter-frame duration the shutter is active.
        readout_std (float, optional): Standard deviation of zero mean Gaussian read noise. Defaults to 20.0.
        fwc (float, optional): Full well capacity, used for normalization. Defaults to 500.0.
        bitdepth (int, optional): Resolution of ADC in bits. Defaults to 12.
        scale_flux (float, optional): factor to scale the input [0, 1] image _before_ Poisson rng
        gain_ISO (float, optional): factor to scale the photo-electron reading _after_ Poisson rng
        demosaic_method (string, optional): demosaicing method to use
        denoise_sigma (float, optional): Gaussian blur kernel sigma (disabled if 0.0)
        sharpen_weight (float, optional): sharpening weight (disabled if 0.0)
        rng (np.random.Generator, optional): Optional random number generator. Defaults to none.

    Returns:
        Quantized sRGB patch is returned
    """
    # Get sum of linear-intensity frames.
    burst_size = int(max(1, np.ceil(len(sequence) * (shutter_angle_degrees / 360.0))))
    sequence = np.array(sequence[:burst_size])
    patch = np.sum(sequence, axis=0) * scale_flux

    color = (len(patch.shape) > 2) and (patch.shape[2] > 1)
    if color:
        patch = rgb2raw_bayer(patch)

    # Roughly translating the model in Eqs. (1,2) and Fig. 1 of Hasinoff et al.:
    # S. W. Hasinoff, F. Durand, and W. T. Freeman,
    # “Noise-optimal capture for high dynamic range photography,” CVPR 2010.

    # Perform poisson sampling
    rng = np.random.default_rng() if rng is None else rng
    patch = rng.poisson(patch).astype(float)
    # full-well capacity
    patch = np.clip(patch, 0, fwc)
    # readout noise
    patch += rng.normal(0, readout_std, size=patch.shape)
    # apply ISO gain
    patch *= gain_ISO
    # assume perfect quantization in ADC
    patch = np.round(np.clip(patch, 0, (2**bitdepth - 1)))
    patch = patch * (1.0 / (2**bitdepth - 1))

    # de-mosaicing
    if color:
        patch = raw2rgb_bayer(patch, method=demosaic_method)

    # de-noising and sharpening
    if denoise_sigma != 0.0:
        patch = gaussian(patch, denoise_sigma)
    if sharpen_weight != 0.0:
        patch = unsharp_mask(patch, amount=sharpen_weight, channel_axis=2 if color else None)

    # Convert to sRGB color space for viewing and quantize to 8-bits
    patch = linearrgb_to_srgb(patch.astype(np.double))
    patch = np.round(patch * 255).astype(np.uint8)
    if not color:  # fake it anyway
        patch = np.repeat(patch, 3, axis=-1)
    return patch


def emulate_rgb_from_flow():
    """Not (Yet) Implemented"""
    raise NotImplementedError
