import warnings

import more_itertools as mitertools
import numpy as np
import torch
import torchgeometry as tgm
import torchvision
import torchvision.transforms.functional as TF
from scipy import linalg as SLA


def to_cpu(*tensors):
    if len(tensors) == 1:
        return tensors[0].detach().cpu()
    return [t.detach().cpu() for t in tensors]


def to_numpy(*tensors):
    if len(tensors) == 1:
        if torch.is_tensor(tensors[0]):
            return tensors[0].detach().cpu().numpy()
        else:
            return np.array(tensors[0])
    return [t.detach().cpu().numpy() if torch.is_tensor(t) else np.array(t) for t in tensors]


def to_list(*tensors):
    if len(tensors) == 1:
        return tensors[0].detach().cpu().tolist()
    return [t.detach().cpu().tolist() for t in tensors]


def to_tensor(*tensors, dtype=torch.float32, device=None):
    if len(tensors) == 1:
        if torch.is_tensor(tensors[0]):
            return tensors[0].to(dtype=dtype, device=device)
        return torch.tensor(tensors[0], dtype=dtype, device=device)
    return [
        t.to(dtype=dtype, device=device) if torch.is_tensor(t) else torch.tensor(t, dtype=dtype, device=device)
        for t in tensors
    ]


def img_to_tensor(*imgs, batched=True, dtype=torch.float32, device=None):
    tensors = [tgm.utils.image_to_tensor(np.array(img)) if not torch.is_tensor(img) else img for img in imgs]
    tensors = [t.squeeze().to(dtype=dtype, device=device) for t in tensors]
    tensors = [t[None, ...] if t.ndim < 3 else t for t in tensors]
    return torch.stack(tensors) if batched else tensors


def tensor_to_img(*tensors):
    imgs = [
        tgm.utils.tensor_to_image(i.detach().cpu().squeeze()) if torch.is_tensor(i) else i.squeeze() for i in tensors
    ]
    imgs = [np.clip(i, 0, 255).astype(np.uint8) if np.nanmax(i) > 1.0 else i.astype(float) for i in imgs]
    imgs = [np.nan_to_num(np.atleast_3d(i)) for i in imgs]
    return imgs if len(imgs) != 1 else imgs[0]


def torch_logm(A, safe=True):
    # Note: Technically unstable implementation, but should work fine for homographies.
    lam, V = torch.linalg.eig(A)
    V_inv = torch.inverse(V).to(torch.complex128)
    V = V.to(torch.complex128)

    if not torch.allclose(A.to(torch.complex128), V @ torch.diag(lam).to(torch.complex128) @ V_inv):
        if safe:
            warnings.warn("Matrix is not diagonalizable, falling back on scipy.")
            return to_tensor(SLA.logm(to_numpy(A)), device=A.device)
        raise ValueError("Matrix is not diagonalizable, cannot compute matrix logarithm!")

    log_A_prime = torch.diag(lam.log()).to(torch.complex128)
    return V @ log_A_prime @ V_inv


def translation_matrix(dx, dy=None, device=None):
    device = getattr(dx, "device", device)
    t = torch.eye(3, dtype=torch.float32, device=device)
    if dy is None:
        t[:2, -1] = to_tensor(dx)
    else:
        t[0, -1] = dx
        t[1, -1] = dy
    return t if torch.is_tensor(dx) else to_numpy(t)


def scale_matrix(sx, sy=None, device=None):
    if sy is None:
        sx, sy = sx if getattr(sx, "__len__", lambda: 0)() == 2 else (sx, sx)
        device = getattr(sx, "device", device)
    s = torch.eye(3, dtype=torch.float32, device=device)
    s[0, 0] = sx
    s[1, 1] = sy
    return s if torch.is_tensor(sx) else to_numpy(s)


def cyclic_pairwise(values):
    for src, dst in mitertools.zip_offset(values, values, offsets=(0, 1), fillvalue=values[0], longest=True):
        if src != dst:
            yield (src, dst)


def sobel_grad(img, sigma=0, kernel_size=11):
    """Compute gradient of image using sobel operator, optionally pre-smooth image
    NB: This actually does not work with multichannel images, and even with grayscale ones
        it returns a grad with shape (N, 2, H, W) instead of (N, 2, C, H, W).
        Use `torch_grad` where ever possible.
    """
    kernel = (
        1
        / 8
        * torch.tensor(
            [[[+1, 0, -1], [+2, 0, -2], [+1, 0, -1]], [[+1, +2, +1], [0, 0, 0], [-1, -2, -1]]],
            dtype=torch.float32,
            device=img.device,
        )
    )

    if sigma:
        img = torchvision.transforms.functional.gaussian_blur(img, kernel_size, sigma=sigma)

    # Note the minus sign below. Conv2d actually perform cross-correlation,
    # so it is needed in oder for the gradients to have correct sign.
    *_, c, h, w = img.shape
    kernel = torch.tile(kernel.unsqueeze(1), (1, c, 1, 1))
    return -torch.nn.functional.conv2d(img, kernel, padding=1)


def torch_grad(img, sigma=0, kernel_size=11):
    if sigma:
        img = torchvision.transforms.functional.gaussian_blur(img, kernel_size, sigma=sigma)

    *c, h, w = img.shape
    *dc, dy, dx = torch.gradient(img.squeeze())
    return torch.stack([dx, dy]).reshape(-1, 2, c.pop() if c else 1, h, w)


def downsample2x(img, minsize=20):
    *c, h, w = img.shape

    if max(h, w) < minsize:
        raise RuntimeError(
            f"Cannot downsample image with shape {img.shape} any more as it's side length is "
            f"smaller than `minsize`={minsize}."
        )
    return TF.resize(img, [h // 2, w // 2], antialias=True)
