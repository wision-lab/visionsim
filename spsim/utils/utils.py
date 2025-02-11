import numpy as np
import torch
import torchgeometry as tgm


def to_numpy(*tensors):
    if len(tensors) == 1:
        if torch.is_tensor(tensors[0]):
            return tensors[0].detach().cpu().numpy()
        else:
            return np.array(tensors[0])
    return [t.detach().cpu().numpy() if torch.is_tensor(t) else np.array(t) for t in tensors]


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
