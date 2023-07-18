import numpy as np
import torch


def binary_avg_to_rgb(mean_binary_patch, factor=1.0):
    """Invert the process by which binary frames are simulated. The result can be either
    linear RGB values or sRGB values depending on how the binary frames were constructed.

    Assuming each binary patch was created by a Bernoulli process with p=1-exp(-factor*rgb),
    then the average of binary frames tends to p. We can therefore recover the original rgb
    values as -log(1-bin)/factor.
    """
    module = torch if torch.is_tensor(mean_binary_patch) else np
    return -module.log(1 - mean_binary_patch) / factor


def srgb_to_linearrgb(img):
    # https://github.com/blender/blender/blob/master/source/blender/blenlib/intern/math_color.c
    module = torch if torch.is_tensor(img) else np
    mask = img < 0.04045
    img[mask] = module.clip(img[mask], 0.0, module.inf) / 12.92
    return ((img + 0.055) / 1.055) ** 2.4


def linearrgb_to_srgb(img):
    # https://github.com/blender/blender/blob/master/source/blender/blenlib/intern/math_color.c
    module = torch if torch.is_tensor(img) else np
    mask = img < 0.0031308
    img[mask] = module.clip(img[mask], 0.0, module.inf) * 12.92
    return module.clip(1.055 * img ** (1.0 / 2.4) - 0.055, 0.0, 1.0)


def apply_alpha(img, alpha_color=(1.0, 1.0, 1.0), ret_alpha=True):
    """Blend an image with a background color using the image's alpha channel"""
    if not np.issubdtype(img.dtype, np.float) or img.max() > 1.0 or img.min() < 0.0:
        raise RuntimeError("Expected image to be of dtype float and normalized to the range [0, 1].")

    # At least 3d with added axis appended to end
    img = np.expand_dims(img, -1 if img.ndim == 2 else ())

    img, alpha = np.split(img, [-1], axis=-1) if img.shape[-1] == 4 else (img, 1.0)
    img = img * alpha + np.array(alpha_color) * (1 - alpha)

    return (img, alpha) if ret_alpha else img


def emulate_rgb_from_merged(patch, burst_size=200, readout_std=20, fwc=500, factor=1.0, generator=None):
    # Input patch is average of `burst_size` linear-intensity frames, get sum by multiplying.
    patch = patch * burst_size

    # Above sum is in range [0, burst_size*factor]
    # Perform poisson sampling and add zero-mean gaussian read noise
    patch = torch.poisson(patch, generator=generator)
    patch += torch.normal(torch.zeros_like(patch), readout_std, generator=generator)

    # Normalize by full well capacity, clip highlights, and quantize to 12-bits
    patch = torch.clip(patch / fwc, 0, 1.0)
    patch = torch.round(patch * 2**12) / 2**12

    # Multiply by gain to keep constant(-ish) brightness
    patch *= fwc / (burst_size * factor)

    # Convert to sRGB color space for viewing and quantize to 8-bits
    patch = linearrgb_to_srgb(patch)
    patch = torch.round(patch * 2**8) / 2**8
    return patch
