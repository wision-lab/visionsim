import imageio.v3 as iio
import numpy as np


def read_img(in_file, apply_alpha=True, grayscale=False):
    img = iio.imread(in_file)

    if img.ndim == 2:
        img = img[..., None]

    img = img / (1.0 if str(in_file).endswith(".exr") else 255.0)
    alpha = img[:, :, -1][..., None] if img.shape[2] == 4 else 1.0
    img = img if not apply_alpha else img * alpha

    if grayscale:
        # Manually grayscale as we've already converted to floating point pixel values
        # Values from http://en.wikipedia.org/wiki/Grayscale
        b, g, r, _ = np.transpose(img, (2, 0, 1))
        img = 0.0722 * b + 0.7152 * g + 0.2126 * r
        img = img[..., None]

    return img[:, :, :3], alpha


def write_img(out_file, img):
    iio.imwrite(out_file, img)
