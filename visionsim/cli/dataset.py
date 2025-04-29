from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def imgs_to_npy(
    input_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    bitpack: bool = False,
    bitpack_dim: int | None = None,
    batch_size: int = 4,
    alpha_color: str = "(255, 255, 255)",
    is_grayscale: bool = False,
    force: bool = False,
):
    """Convert an image folder based dataset to a NPY dataset

    Args:
        input_dir: directory in which to look for frames
        output_dir: directory in which to save npy file
        bitpack: if true, each chunk of 8 binary pixels will by packed into a single byte. Only enable if data is binary valued
        bitpack_dim: axis along which to pack bits (H=1, W=2)
        batch_size: number of frames to write at once
        alpha_color: if set, blend with this background color and do not store alpha channel
        is_grayscale: If set, assume images are grayscale and only save first channel
        force: if true, overwrite output file(s) if present
    """
    # TODO: Add data integrity check
    #   - If npy file is present, scan through it to check if any data is missing
    #   - If so, fill in those frames only and skip the rest (add allow-skips arg?)
    # TODO: Add check to stop the user from bitpacking non-binary data
    # TODO: Support more types of frames, depth, normals, etc...
    import ast
    import copy

    import numpy as np
    from rich.progress import Progress
    from torch.utils.data import DataLoader

    from visionsim.dataset import ImgDataset, NpyDatasetWriter, default_collate

    from . import _validate_directories

    input_path, output_path, *_ = _validate_directories(input_dir, output_dir)
    dataset = ImgDataset(input_path)
    transforms_new = copy.deepcopy(dataset.transforms or {})

    if ".exr" in set(Path(p).suffix for p in dataset.paths):
        # TODO: This is due to the alpha blending below, we need alpha in [0, 1] to blend.
        raise NotImplementedError("Task does not yet support EXRs")
    if dataset.transforms:
        if any(any(k != "file_path" and "path" in k for k in f.keys()) for f in dataset.transforms["frames"]):
            raise NotImplementedError(
                "Only color frames are supported for now. Keys such as 'depth_file_path' "
                "or 'mask_path' are not supported."
            )
    alpha_color = ast.literal_eval(alpha_color) if alpha_color else None
    shape = np.array(dataset.full_shape)
    shape[-1] = 1 if is_grayscale else (shape[-1] - int(alpha_color is not None))

    # Bitpack if either is set, defaults to bitpacking width dimension
    if bitpack or bitpack_dim is not None:
        bitpack_dim = bitpack_dim if bitpack_dim is not None else 2
        bitpack = True

        if bitpack_dim == 0 or bitpack_dim >= 3:
            raise NotImplementedError("Can only bitpack along H or W.")

        transforms_new["bitpack"] = bitpack
        transforms_new["bitpack_dim"] = bitpack_dim
        shape[bitpack_dim] /= 8

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=os.cpu_count() or 1, collate_fn=default_collate)

    with (
        NpyDatasetWriter(
            output_path, shape=np.ceil(shape).astype(int), transforms=transforms_new, force=force
        ) as writer,
        Progress() as progress,
    ):
        task1 = progress.add_task("Writing frames...", total=len(dataset))

        for i, (idxs, imgs, poses) in enumerate(loader):
            if alpha_color is not None and imgs.ndim == 4 and imgs.shape[-1] == 4:
                alpha = imgs[..., -1][..., None] / 255.0
                imgs = imgs[..., :-1] * alpha + alpha_color * (1 - alpha)
            if bitpack:
                imgs = imgs >= 128
                imgs = np.packbits(imgs, axis=bitpack_dim)
            if is_grayscale:
                imgs = imgs[..., :1]
            writer[idxs] = (imgs, poses)
            progress.update(task1, advance=len(idxs))


def npy_to_imgs(
    input_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    batch_size: int = 4,
    pattern: str = "frame_{:06}.png",
    step: int = 1,
    force: bool = False,
):
    """Convert an NPY based dataset to an image-folder dataset

    Args:
        input_dir: directory in which to look for frames
        output_dir: directory in which to save npy file
        batch_size: number of frames to write at once
        pattern: filenames of frames will match this
        step: skip some frames when converting between formats
        force: if true, overwrite output file(s) if present

    """
    import copy

    from rich.progress import Progress
    from torch.utils.data import DataLoader

    from visionsim.dataset import ImgDatasetWriter, NpyDataset, default_collate

    from . import _validate_directories

    input_path, output_path, *_ = _validate_directories(input_dir, output_dir)
    dataset = NpyDataset(input_path)

    transforms_new = copy.deepcopy(dataset.transforms or {})
    transforms_new.pop("file_path", None)
    transforms_new.pop("bitpack", None)
    transforms_new.pop("bitpack_dim", None)

    sampler = range(0, len(dataset) - 1, step)
    loader = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size, num_workers=os.cpu_count() or 1, collate_fn=default_collate
    )

    with (
        ImgDatasetWriter(output_path, transforms=transforms_new, force=force, pattern=pattern) as writer,
        Progress() as progress,
    ):
        task1 = progress.add_task("Writing frames...", total=len(sampler))

        for idxs, imgs, poses in loader:
            if np.issubdtype(imgs.dtype, np.uint8):
                writer[idxs] = (imgs, poses)
            else:
                writer[idxs] = (np.repeat((imgs * 255).astype(np.uint8), 3, -1), poses)
            progress.update(task1, advance=len(idxs))


def info(input_dir: str | os.PathLike, as_json: bool = False):
    """Print information about the dataset

    Args:
        input_dir: directory in which to look for dataset
        as_json: print the output in a json-formatted string
    """
    import json

    from visionsim.dataset import Dataset

    from . import _validate_directories

    input_path, *_ = _validate_directories(input_dir=input_dir)
    dataset = Dataset.from_path(input_path)

    if as_json:
        print(
            json.dumps(
                {
                    "arclength": dataset.arclength,
                    "full_shape": dataset.full_shape,
                    "paths": [str(p) for p in dataset.paths],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"arclength: {dataset.arclength}")
        print(f"full_shape: {dataset.full_shape}")
        print(f"paths: {[str(p) for p in dataset.paths]}")
