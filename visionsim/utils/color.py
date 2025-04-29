from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from typing_extensions import overload


@overload
def srgb_to_linearrgb(img: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]: ...


@overload
def srgb_to_linearrgb(img: torch.Tensor) -> torch.Tensor: ...


def srgb_to_linearrgb(img: torch.Tensor | npt.NDArray[np.floating]) -> torch.Tensor | npt.NDArray[np.floating]:
    """Performs sRGB to linear RGB color space conversion by reversing gamma
    correction and obtaining values that represent the scene's intensities.

    Args:
        img (torch.Tensor | npt.NDArray): Image to un-tonemap.

    Returns:
        linear rgb image.
    """
    # https://github.com/blender/blender/blob/master/source/blender/blenlib/intern/math_color.c
    module, img = (torch, img.clone()) if torch.is_tensor(img) else (np, np.copy(img))
    mask = img < 0.04045
    img[mask] = module.clip(img[mask], 0.0, module.inf) / 12.92
    img[~mask] = ((img[~mask] + 0.055) / 1.055) ** 2.4  # type: ignore
    return img


@overload
def linearrgb_to_srgb(img: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]: ...


@overload
def linearrgb_to_srgb(img: torch.Tensor) -> torch.Tensor: ...


def linearrgb_to_srgb(img: torch.Tensor | npt.NDArray) -> torch.Tensor | npt.NDArray:
    """Performs linear RGB to sRGB color space conversion to apply gamma correction for display purposes.

    Args:
        img (torch.Tensor | npt.NDArray): Image to tonemap.

    Returns:
        tonemapped rgb image.
    """
    # https://github.com/blender/blender/blob/master/source/blender/blenlib/intern/math_color.c
    module, img = (torch, img.clone()) if torch.is_tensor(img) else (np, np.copy(img))
    mask = img < 0.0031308
    img[img < 0.0] = 0.0
    img[mask] = img[mask] * 12.92  # type: ignore
    img[~mask] = module.clip(1.055 * img[~mask] ** (1.0 / 2.4) - 0.055, 0.0, 1.0)
    return img
