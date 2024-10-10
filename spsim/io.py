import os

import imageio.v3 as iio
import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def read_img(in_file, apply_alpha=True, grayscale=False, alpha_color=(1.0, 1.0, 1.0)):
    """Reads imagine file, converts from grayscale to 3D, apply alpha blend with specified background color
    Args:
        in_file: Path to image file.
        apply_alpha: Boolean flag to indicate whether bgcolor should be blended, or just rgb channels should be returned.
        grayscale: Boolean flag to manually grayscale as conversion to floating point pixel has already been done.
        bgcolor: Background color to blend in alpha. Defaults to (1.0,1.0,1.0).

    :returns:
        Processed image data and alpha channels data.
    """
    img = iio.imread(in_file)

    if img.ndim == 2:
        img = img[..., None]

    img = img / (1.0 if str(in_file).endswith(".exr") else 255.0)
    # checks if image has 4 channels(it has alpha).
    alpha = img[:, :, -1][..., None] if img.shape[2] == 4 else 1.0
    # img = img[:, :, :3] if not apply_alpha else img[:, :, :3] * alpha + np.array(bgcolor) * (1 - alpha)
    img = img[:, :, :3] if not apply_alpha else img[:, :, :3] * alpha + np.array(alpha_color) * (1 - alpha)

    if grayscale:
        # Manually grayscale as we've already converted to floating point pixel values
        # Values from http://en.wikipedia.org/wiki/Grayscale
        b, g, r = np.transpose(img, (2, 0, 1))
        img = 0.0722 * b + 0.7152 * g + 0.2126 * r
        img = img[..., None]

    return img, alpha


def write_img(out_file, img):
    """Writes imagines to a specific output file.

    Args:
        out_file: Path to desired output file.
        img: The image data to be written.
    """
    iio.imwrite(out_file, img)
