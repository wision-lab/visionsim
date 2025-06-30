from __future__ import annotations

import numpy as np
import numpy.typing as npt

from visionsim.utils.color import linearrgb_to_srgb


def emulate_rgb_from_sequence(
    sequence: npt.ArrayLike,
    frac_shutter_angle: float = 1.0,
    readout_std: float = 20.0,
    fwc: float = 10000.0,
    bitdepth: int = 12,
    scale_flux: float = 1.0,
    gain_ISO: float = 1.0,
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
        frac_shutter_angle (float, optional): fraction of inter-frame duration the shutter is active.
        readout_std (float, optional): Standard deviation of zero mean Gaussian read noise. Defaults to 20.0.
        fwc (float, optional): Full well capacity, used for normalization. Defaults to 500.0.
        bitdepth (int, optional): Resolution of ADC in bits. Defaults to 12.
        scale_flux (float, optional): factor to scale the input [0, 1] image _before_ Poisson rng
        gain_ISO (float, optional): factor to scale the photo-electron reading _after_ Poisson rng
        rng (np.random.Generator, optional): Optional random number generator. Defaults to none.

    Returns:
        Quantized sRGB patch is returned
    """
    # Get sum of linear-intensity frames.
    burst_size = int(max(1, np.ceil(len(sequence) * frac_shutter_angle)))
    sequence = np.array(sequence[:burst_size])

    patch = np.sum(sequence, axis=0) * scale_flux

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
    # assume a "perfect" ADC with no additional noise except quantization
    patch = np.round(np.clip(patch, 0, (2**bitdepth-1))) / (2**bitdepth - 1)

    # do not attempt to keep the same levels as the input

    # Convert to sRGB color space for viewing and quantize to 8-bits
    patch = linearrgb_to_srgb(patch)
    patch = np.round(patch * 2**8) / 2**8
    return patch


def emulate_rgb_from_flow():
    """Not (Yet) Implemented"""
    raise NotImplementedError
