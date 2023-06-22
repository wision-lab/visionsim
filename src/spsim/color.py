import torch


def binary2rgb(binary_patch, factor=1.0):
    """Invert the process by which binary frames are simulated. The result can be either
    linear RGB values or sRGB values depending on how the binary frames were constructed.

    Assuming each binary patch was created by a Bernoulli process with p=1-exp(-factor*rgb),
    then the average of binary frames tends to p. We can therefore recover the original rgb
    values as -log(1-bin)/factor.
    """
    return -torch.log(1 - binary_patch) / factor


def srgb_to_linearrgb(img):
    # https://github.com/blender/blender/blob/master/source/blender/blenlib/intern/math_color.c
    mask = img < 0.04045
    img[mask] = torch.clip(img[mask], 0.0, torch.inf) / 12.92
    return ((img + 0.055) / 1.055) ** 2.4


def linearrgb_to_srgb(img):
    # https://github.com/blender/blender/blob/master/source/blender/blenlib/intern/math_color.c
    mask = img < 0.0031308
    img[mask] = torch.clip(img[mask], 0.0, torch.inf) * 12.92
    return torch.clip(1.055 * img ** (1.0 / 2.4) - 0.055, 0.0, 1.0)


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
