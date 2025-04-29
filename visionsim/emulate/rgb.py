from __future__ import annotations

import numpy as np
import numpy.typing as npt

from visionsim.utils.color import linearrgb_to_srgb


def emulate_rgb_from_sequence(
    sequence: npt.ArrayLike,
    readout_std: float = 20.0,
    fwc: float = 500.0,
    bitdepth: int = 12,
    factor: float = 1.0,
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
        readout_std (float, optional): Standard deviation of zero mean Gaussian read noise. Defaults to 20.0.
        fwc (float, optional): Full well capacity, used for normalization. Defaults to 500.0.
        bitdepth (int, optional): Resolution of ADC in bits. Defaults to 12.
        factor (float, optional): Scaling factor to control intensity of output RGB image. Defaults to 1.0.
        rng (np.random.Generator, optional): Optional random number generator. Defaults to none.

    Returns:
        Quantized sRGB patch is returned
    """
    # Get sum of linear-intensity frames.
    sequence = np.array(sequence)
    burst_size = len(sequence)
    patch = np.sum(sequence, axis=0) * factor

    # Perform poisson sampling and add zero-mean gaussian read noise
    rng = np.random.default_rng() if rng is None else rng
    patch = rng.poisson(patch).astype(float)
    patch += rng.normal(0, readout_std * burst_size / 255.0, size=patch.shape)

    # Normalize by full well capacity, clip highlights, and quantize to 12-bits
    patch = np.clip(patch / fwc, 0, 1.0)
    patch = np.round(patch * (2**bitdepth - 1)) / (2**bitdepth - 1)

    # Multiply by gain to keep constant(-ish) brightness
    patch *= fwc / (burst_size * factor)

    # Convert to sRGB color space for viewing and quantize to 8-bits
    patch = linearrgb_to_srgb(patch)
    patch = np.round(patch * 2**8) / 2**8
    return patch


def emulate_rgb_from_flow():
    """Not (Yet) Implemented"""
    raise NotImplementedError
