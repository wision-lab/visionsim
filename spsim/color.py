import numpy as np
import torch


def binary_avg_to_rgb(mean_binary_patch, factor=1.0, epsilon=1e-6, quantile=None):
    """Convert average binary patches to RGB
    Invert the process by which binary frames are simulated. The result can be either
    linear RGB values or sRGB values depending on how the binary frames were constructed.

    Assuming each binary patch was created by a Bernoulli process with p=1-exp(-factor*rgb),
    then the average of binary frames tends to p. We can therefore recover the original rgb
    values as -log(1-bin)/factor.

    Args:
        mean_binary_patch: Binary avg to convert to rgb
        factor: Arbitrary Brightness factor. Defaults to 1.0.
    :return:
        RGB value corresponding to specificed factor.
    """
    module = torch if torch.is_tensor(mean_binary_patch) else np
    intensity = -module.log(module.clip(1 - mean_binary_patch, epsilon, 1)) / factor

    if quantile is not None:
        intensity = intensity / module.quantile(intensity, quantile)
        intensity = module.clip(intensity, 0, 1)

    return intensity


def srgb_to_linearrgb(img):
    """Performs sRGB to linear RGB color space conversion by reversing gamma
    correction and obtaining values that represent the scene's intensities.

    Args:
        img: Tensor or np array to perform conversion.
    :returns:
        linear rgb image tensor or np array.

    """
    # https://github.com/blender/blender/blob/master/source/blender/blenlib/intern/math_color.c
    module, img = (torch, img.clone()) if torch.is_tensor(img) else (np, np.copy(img))
    mask = img < 0.04045
    img[mask] = module.clip(img[mask], 0.0, module.inf) / 12.92
    img[~mask] = ((img[~mask] + 0.055) / 1.055) ** 2.4
    return img


def linearrgb_to_srgb(img):
    """Performs linear RGB to sRGB inverse color space conversion to apply gamma correction or display purposes

    Args:
        img: Tensor or np array to perform conversion.
    :returns:
        srgb image tensor or np array.
    """
    # https://github.com/blender/blender/blob/master/source/blender/blenlib/intern/math_color.c
    module, img = (torch, img.clone()) if torch.is_tensor(img) else (np, np.copy(img))
    mask = img < 0.0031308
    img[img < 0.0] = 0.0
    img[mask] = img[mask] * 12.92
    img[~mask] = module.clip(1.055 * img[~mask] ** (1.0 / 2.4) - 0.055, 0.0, 1.0)
    return img


def apply_alpha(img, alpha_color=(1.0, 1.0, 1.0), ret_alpha=True):
    """Performs alpha blending between image and background color
    Blend an image with a background color using the image's alpha channel

    Args:
        img: Np array to perform blending.
        alpha_color: Backgroud color to blend. Defaults to (1.0,1.0,1.0).
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

    if isinstance(alpha, np.ndarray) and alpha.max() > 1.0 or alpha.min() < 0.0:
        raise RuntimeError("Expected alpha channel to be normalized to the range [0, 1].")

    # If image does not have 4 channels, pass through
    if original_shape[-1] != 4 or alpha_color is None:
        return (img, alpha) if ret_alpha else img

    img = img * alpha + np.array(alpha_color) * (1 - alpha)

    return (img, alpha) if ret_alpha else img


def emulate_rgb_from_merged(patch, burst_size=200, readout_std=20, fwc=500, factor=1.0, generator=None):
    """Simulates process of creating RGB image from merged intensity frames.Quantized sRGB patch is returned

    Args:
        patch: Input patch is average of burst_size linear-intensity frames.
        burst_size: Number of frames used for averaging. Defaults to 200.
        readout_std: Standard deviation of zero mean Gaussian read noise. Defaults to 20.
        fwc: Full well capacity, used for normalization. Defaults to 500.
        factor: Scaling factor to control intesnity of output RGB image. Defaults to 1.0.
        generator: Optional random number generator. Defaults to none.

    :returns:
        Processed path is returned as an np array.

    """
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
