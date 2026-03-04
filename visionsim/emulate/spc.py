from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
import torch


def emulate_spc(
    img: npt.NDArray[np.floating],
    flux_gain: float = 1.0,
    bitplanes: int = 1,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.integer]:
    """Perform bernoulli sampling on linearized RGB frames to yield binary frames.

    Args:
        img (npt.NDArray[np.floating]): Linear intensity image to sample binary frame from.
        flux_gain(float, optional): scale factor to convert img in [0, 1] range to other flux levels. Defaults to 1.0.
        bitplanes (int, optional): when bitplanes > 1, this represents a binomial SPAD sensor that sums binary samples internally
            for each measurement, operating at framerate ``1 / bitplanes`` of a binary SPAD sensor. Defaults to 1.
        rng (np.random.Generator, optional): Optional random number generator. Defaults to none.

    Returns:
        npt.NDArray[np.integer]: binomial-distributed single-photon frame
    """
    rng = np.random.default_rng() if rng is None else rng
    return rng.binomial(cast(npt.NDArray[np.integer], bitplanes), 1.0 - np.exp(-img * flux_gain))


def spc_avg_to_rgb(
    mean_binary_patch: torch.Tensor | npt.NDArray,
    factor: float = 1.0,
    epsilon: float = 1e-6,
    quantile: float | None = None,
) -> torch.Tensor | npt.NDArray:
    """Convert average of many single photon binary frames to conventional RGB image.

    Invert the process by which binary frames are emulated. The result can be either
    linear RGB values or sRGB values depending on how the binary frames were constructed.

    Assuming each binary patch was created by a Bernoulli process with p=1-exp(-factor*rgb),
    then the average of binary frames tends to p. We can therefore recover the original rgb
    values as -log(1-bin)/factor where bin is the average of many binary frames.

    Args:
        mean_binary_patch (torch.Tensor | npt.NDArray): Binary avg to convert to rgb, expects a np.ndarray or torch tensor.
        factor (float, optional): Arbitrary corrective brightness factor. Defaults to 1.0.
        epsilon (float, optional): Smallest allowed value before taking logarithm, needed for stability. Defaults to 1e-6.
        quantile (float, optional): If supplied, normalize by this quantile. For instance, if set to `0.9` then the
            ninety-percent brightest value will be used as the new maximum, with brighter values being clipped.
            Defaults to None (no normalization/clipping).

    Returns:
        torch.Tensor | npt.NDArray: Conventional intensity image
    """
    module = torch if torch.is_tensor(mean_binary_patch) else np
    intensity = -module.log(module.clip(1 - mean_binary_patch, epsilon, 1)) / factor

    if quantile is not None:
        intensity = intensity / module.quantile(intensity, quantile)
        intensity = module.clip(intensity, 0, 1)

    return intensity
