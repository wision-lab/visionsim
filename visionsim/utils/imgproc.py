from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter


def unsharp_mask(
    img: npt.ArrayLike,
    sigma: float = 1.0,
    amount: float = 1.0,
) -> npt.NDArray:
    """Unsharp-masking to sharpen an image [#UnsharpMasking]_.

    Borrows interface from `scikit-image's version <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.unsharp_mask>`_.

    Args:
        img (npt.ArrayLike, required): input image, can be 2-, 3-, or 4-channel
        sigma (float, default = 1.0): controls extent of blurring
        amount (float, default = 1.0): controls how much details are amplified by

    Returns:
        output (npt.NDArray):   img with unsharp mask applied,
                                same shape and dtype as img

    References:
        .. [#UnsharpMasking] `Wikipedia. Unsharp masking <https://en.wikipedia.org/wiki/Unsharp_masking>`_
    """

    def _unsharp_mask1(img):
        img_smooth = gaussian_filter(img, sigma)
        if np.issubdtype(img.dtype, np.floating):
            np.clip(img + (amount * (img - img_smooth)), 0, 1, out=img)
        else:
            # work with copy and convert back
            if not isinstance(img, np.uint8):
                raise NotImplementedError("_unsharp_mask1 expects (float | uint8)")
            img_float = (1.0 / 255) * img.astype(np.float)
            img_smooth_float = (1.0 / 255) * img_smooth.astype(np.float)
            np.clip(img_float + (amount * (img_float - img_smooth_float)), 0, 1, out=img_float)
            img[...] = (255 * img_float).astype(np.uint8)
        return

    has_alpha = (img.ndim == 3) and (img.shape[2] in [2, 4])
    if has_alpha:
        alpha_channel = img[:, :, -1:]
        img = img[:, :, :-1]
    has_color = (img.ndim == 3) and (img.shape[2] == 3)
    if has_color:
        # do each channel individually
        for c in range(3):
            _unsharp_mask1(img[:, :, c])
    else:
        _unsharp_mask1(img)
    if has_alpha:
        img = np.concatenate((img, alpha_channel), axis=-1)
    return img
