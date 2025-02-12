from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


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
    img[mask] = img[mask] * 12.92 # type: ignore
    img[~mask] = module.clip(1.055 * img[~mask] ** (1.0 / 2.4) - 0.055, 0.0, 1.0) 
    return img


def apply_alpha(img, alpha_color=(1.0, 1.0, 1.0), ret_alpha=True):
    """Performs alpha blending between image and background color
    Blend an image with a background color using the image's alpha channel

    Args:
        img: Np array to perform blending.
        alpha_color: Background color to blend. Defaults to (1.0,1.0,1.0).
        ret_alpha: Flag to return alpha value. Defaults to true.

    :returns:
        Blended image and alpha value, or just image if ret_alpha false.
    """
    if not np.issubdtype(img.dtype, np.floating):
        raise RuntimeError("Expected image to be of dtype float.")

    # At least 3d with added axis appended to end
    original_shape = img.shape
    img = np.expand_dims(img, -1 if img.ndim == 2 else ())

    # Get image and alpha
    img, alpha = np.split(img, [-1], axis=-1) if img.shape[-1] == 4 else (img, 1.0)

    if isinstance(alpha, np.ndarray) and (alpha.max() > 1.0 or alpha.min() < 0.0):
        raise RuntimeError("Expected alpha channel to be normalized to the range [0, 1].")

    # If image does not have 4 channels, pass through
    if original_shape[-1] != 4 or alpha_color is None:
        return (img, alpha) if ret_alpha else img

    img = img * alpha + np.array(alpha_color) * (1 - alpha)

    return (img, alpha) if ret_alpha else img
