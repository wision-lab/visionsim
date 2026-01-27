from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from scipy.ndimage import correlate
from typing_extensions import Literal


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
    img[mask] = img[mask] * 12.92  # type: ignore
    img[~mask] = module.clip(1.055 * img[~mask] ** (1.0 / 2.4) - 0.055, 0.0, 1.0)
    return img


def rgb_to_raw_bayer(
    rgb: torch.Tensor | npt.NDArray, cfa_pattern: Literal["rggb"] = "rggb"
) -> torch.Tensor | npt.NDArray:
    """Mosaicing

    Realizes a mosaiced CFA image as would be sampled by a real Bayer-patterned sensor

    Args:
        rgb: hypothetical true RGB signal
        cfa_pattern: Bayer pattern (only RGGB implemented for now)

    Returns:
        raw: CFA mosaiced data
    """
    if cfa_pattern == "rggb":
        raw = np.copy(rgb[:, :, 0])
        raw[::2, 1::2] = rgb[::2, 1::2, 1]
        raw[1::2, ::2] = rgb[1::2, ::2, 1]
        raw[1::2, 1::2] = rgb[1::2, 1::2, 2]
    else:
        raise NotImplementedError
    return raw


def raw_to_rgb_bayer(
    raw: torch.Tensor | npt.NDArray,
    cfa_pattern: Literal["rggb"] = "rggb",
    method: Literal["off", "bilinear", "MHC04"] = "bilinear",
) -> torch.Tensor | npt.NDArray:
    """Demosaicing

    Convolution-based implementation as suggested by Malvar et al. [1].

    Bilinear is simpler but visually not as nice as Malvar et al.'s method.

    Alternative implementations are also available from OpenCV:
        rgb = cv2.cvtColor(<uint16_array>, cv2.COLOR_BAYER_BG2BGR)[:,:,::-1],
    and the colour-demosaicing library (https://pypi.org/project/colour-demosaicing):
        rgb = demosaicing_CFA_Bayer_bilinear(raw, pattern="RGGB")
        rgb = demosaicing_CFA_Bayer_Malvar2004(raw, pattern="RGGB"),
    which appear to give similar results but could run faster (not benchmarked).

    Args:
        raw: input array (has to be exactly 2D)
        cfa_pattern: Bayer pattern shape (only RGGB implemented for now)

    Returns:
        rgb: demosaiced RGB image

    References:
        .. [1] Malvar et al. (2004), "High-quality linear interpolation for demosaicing of Bayer-patterned color images", ICASSP 2004. <https://home.cis.rit.edu/~cnspci/references/dip/demosaicking/malvar2004.pdf>
    """

    if cfa_pattern == "rggb":
        if any(((L % 2) != 0) for L in raw.shape):
            raise RuntimeError(f"Expected even-length raw array, got {(raw.shape)}")

        if torch.is_tensor(raw):
            raw = raw.numpy()

        if not np.issubdtype(raw.dtype, np.floating):
            raise RuntimeError(f"Expected floating-point array, got {raw.dtype}")

        R, G1, G2, B = raw[::2, ::2], raw[::2, 1::2], raw[1::2, ::2], raw[1::2, 1::2]

        rgb = np.zeros(raw.shape + (3,), dtype=raw.dtype)
        rgb[::2, ::2, 0] = R
        rgb[::2, 1::2, 1] = G1
        rgb[1::2, ::2, 1] = G2
        rgb[1::2, 1::2, 2] = B

        if method == "off":
            return rgb

        if method in ["bilinear", "MHC04"]:
            h_avg_fw_x = np.array([[0.0, 0.5, 0.5]])
            h_avg_bw_x = np.fliplr(h_avg_fw_x)

            rgb[::2, ::2, 1] = 0.5 * (
                correlate(G1, h_avg_bw_x, mode="nearest") + correlate(G2, h_avg_bw_x.T, mode="nearest")
            )
            rgb[::2, ::2, 2] = correlate(correlate(B, h_avg_bw_x, mode="nearest"), h_avg_bw_x.T, mode="nearest")

            rgb[::2, 1::2, 0] = correlate(R, h_avg_fw_x, mode="nearest")
            rgb[::2, 1::2, 2] = correlate(B, h_avg_bw_x.T, mode="nearest")
            rgb[1::2, ::2, 0] = correlate(R, h_avg_fw_x.T, mode="nearest")
            rgb[1::2, ::2, 2] = correlate(B, h_avg_bw_x, mode="nearest")

            rgb[1::2, 1::2, 0] = correlate(correlate(R, h_avg_fw_x, mode="nearest"), h_avg_fw_x.T, mode="nearest")
            rgb[1::2, 1::2, 1] = 0.5 * (
                correlate(G1, h_avg_fw_x.T, mode="nearest") + correlate(G2, h_avg_fw_x, mode="nearest")
            )

            if method == "bilinear":
                return rgb

            if method == "MHC04":
                # Kernels in Fig. 2 of Malvar et al., along with weight factors for combination.
                # This implementation is a little idiosyncratic because it realizes the 2D convolution
                # via a sequence of 1D convolutions, probably not worth it in hindsight...
                h_avg_nhood_x = np.array([[0.5, 0.0, 0.5]])

                av_R = correlate(correlate(R, h_avg_nhood_x, mode="nearest"), h_avg_nhood_x.T, mode="nearest")
                diff_R = R - av_R

                av_B = correlate(correlate(B, h_avg_nhood_x, mode="nearest"), h_avg_nhood_x.T, mode="nearest")
                diff_B = B - av_B

                rgb[::2, ::2, 1] += (1 / 2) * diff_R
                rgb[::2, ::2, 2] += (3 / 4) * diff_B
                rgb[1::2, 1::2, 1] += (1 / 2) * diff_B
                rgb[1::2, 1::2, 0] += (3 / 4) * diff_R

                av_G2_at_G1 = correlate(correlate(G2, h_avg_fw_x, mode="nearest"), h_avg_bw_x.T, mode="nearest")

                av_G1_at_G1_R = correlate(G1, h_avg_nhood_x, mode="nearest") - correlate(
                    G1, 0.5 * h_avg_nhood_x.T, mode="nearest"
                )
                av_G1_R = (1 / 5) * (2 * av_G1_at_G1_R + 4 * av_G2_at_G1)
                diff_G1_R = G1 - av_G1_R
                rgb[::2, 1::2, 0] += (5 / 8) * diff_G1_R

                av_G1_at_G1_B = correlate(G1, h_avg_nhood_x.T, mode="nearest") - correlate(
                    G1, 0.5 * h_avg_nhood_x, mode="nearest"
                )
                av_G1_B = (1 / 5) * (2 * av_G1_at_G1_B + 4 * av_G2_at_G1)
                diff_G1_B = G1 - av_G1_B
                rgb[::2, 1::2, 2] += (5 / 8) * diff_G1_B

                av_G1_at_G2 = correlate(correlate(G1, h_avg_bw_x, mode="nearest"), h_avg_fw_x.T, mode="nearest")

                av_G2_at_G2_R = correlate(G2, h_avg_nhood_x.T, mode="nearest") - correlate(
                    G2, 0.5 * h_avg_nhood_x, mode="nearest"
                )
                av_G2_R = (1 / 5) * (4 * av_G1_at_G2 + 2 * av_G2_at_G2_R)
                diff_G2_R = G2 - av_G2_R
                rgb[1::2, ::2, 0] += (5 / 8) * diff_G2_R

                av_G2_at_G2_B = correlate(G2, h_avg_nhood_x, mode="nearest") - correlate(
                    G2, 0.5 * h_avg_nhood_x.T, mode="nearest"
                )
                av_G2_B = (1 / 5) * (4 * av_G1_at_G2 + 2 * av_G2_at_G2_B)
                diff_G2_B = G2 - av_G2_B
                rgb[1::2, ::2, 2] += (5 / 8) * diff_G2_B

                return rgb
    else:
        raise NotImplementedError
    return rgb
