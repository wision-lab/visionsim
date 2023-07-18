import functools

import numpy as np
from invoke import task


def _spad_collate(batch, *, mode, rng, factor, alpha_color):
    """Use default collate function on batch and then simulate SPAD, enabling compute to be done in threads"""
    from spsim.color import apply_alpha, srgb_to_linearrgb
    from spsim.dataset import default_collate

    idxs, imgs, poses = default_collate(batch)

    # Assume image has been tonemapped and undo mapping
    imgs = srgb_to_linearrgb((imgs / 255.0).astype(float))
    imgs, alpha = apply_alpha(imgs, alpha_color=alpha_color, ret_alpha=True)

    # Perform bernoulli sampling (equivalent to binomial w/ n=1)
    binary_img = rng.binomial(1, 1.0 - np.exp(-imgs * factor)) * 255
    binary_img = binary_img.astype(np.uint8)

    if mode.lower() == "npy":
        binary_img = binary_img >= 128
        binary_img = np.packbits(binary_img, axis=2)
    return idxs, binary_img, poses


@task(
    help={
        "input_dir": "directory in which to look for frames",
        "output_dir": "directory in which to save binary frames",
        "alpha_color": "if set, blend with this background color and do not store "
        "alpha channel. default: '(1.0, 1.0, 1.0)'",
        "pattern": "filenames of frames should match this, default: 'frame_{:06}.png'",
        "factor": "multiplicative factor controlling dynamic range of output, default: 1.0",
        "seed": "random seed to use while sampling, ensures reproducibility. default: 2147483647",
        "mode": "how to save binary frames, either as 'img' or as 'npy', default: 'npy'",
        "batch_size": "number of frames to write at once, default: 4",
        "force": "if true, overwrite output file(s) if present, default: False",
    }
)
def spad(
    c,
    input_dir,
    output_dir,
    alpha_color="(1.0, 1.0, 1.0)",
    pattern="frame_{:06}.png",
    factor=1.0,
    seed=2147483647,
    mode="npy",
    batch_size=4,
    force=False,
):
    """Perform bernoulli sampling on linearized RGB frames to yield binary frames"""
    import ast
    import copy

    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    from spsim.dataset import ImgDatasetWriter, NpyDatasetWriter, dataset_dispatch

    from .common import _validate_directories

    input_dir, output_dir = _validate_directories(input_dir, output_dir)
    dataset = dataset_dispatch(input_dir)
    alpha_color = ast.literal_eval(alpha_color) if alpha_color else None
    transforms_new = copy.deepcopy(dataset.transforms or {})
    shape = np.array(dataset.full_shape)
    shape[-1] = transforms_new["c"] = transforms_new.pop("c", 4) - 1

    if mode.lower() == "img":
        ...
    elif mode.lower() == "npy":
        # Default to bitpacking width
        transforms_new["bitpack"] = True
        transforms_new["bitpack_dim"] = 2
        shape[2] /= 8
    else:
        raise ValueError(f"Mode should be one of 'img' or 'npy', got {mode}.")

    if any(str(p).endswith(".exr") for p in getattr(dataset, "paths", [])):
        # TODO: This is due to the alpha blending below, we need alpha in [0, 1] to blend.
        raise NotImplementedError("Task does not yet support EXRs")

    rng = np.random.default_rng(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=c.get("max_threads"),
        collate_fn=functools.partial(_spad_collate, mode=mode, rng=rng, factor=factor, alpha_color=alpha_color),
    )
    pbar = tqdm(total=len(dataset))

    with ImgDatasetWriter(
        output_dir, transforms=transforms_new, force=force, pattern=pattern
    ) if mode.lower() == "img" else NpyDatasetWriter(
        output_dir, np.ceil(shape).astype(int), transforms=transforms_new, force=force
    ) as writer:
        for i, (idxs, imgs, poses) in enumerate(loader):
            writer[idxs] = (imgs, poses)
            pbar.update(len(idxs))


@task(
    help={
        "input_dir": "directory in which to look for frames",
        "output_dir": "directory in which to save binary frames",
        "chunk_size": "number of consecutive frames to average together, default: 10",
        "factor": "multiply image's linear intensity by this weight, default: 1.0",
        "readout_std": "standard deviation of gaussian read noise, default: 20",
        "fwc": "full well capacity of sensor in arbitrary units (relative to factor & chunk_size), default: 500",
        "alpha_color": "if set, blend with this background color and do not store "
                       "alpha channel. default: '(1.0, 1.0, 1.0)'",
        "pattern": "filenames of frames should match this, default: 'frame_{:06}.png'",
        "mode": "how to save binary frames, either as 'img' or as 'npy', default: 'npy'",
        "force": "if true, overwrite output file(s) if present, default: False",
    }
)
def rgb(
    c,
    input_dir,
    output_dir,
    chunk_size=10,
    factor=1.0,
    readout_std=20,
    fwc=500,
    alpha_color="(1.0, 1.0, 1.0)",
    pattern="frame_{:06}.png",
    mode="npy",
    force=False,
):
    """Simulate real camera, adding read/poisson noise and tonemapping"""
    import ast
    import copy

    import more_itertools as mitertools
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    from spsim.color import apply_alpha, emulate_rgb_from_merged, srgb_to_linearrgb
    from spsim.dataset import ImgDatasetWriter, NpyDatasetWriter, dataset_dispatch, default_collate
    from spsim.interpolate import pose_interp
    from spsim.utils import img_to_tensor, tensor_to_img  # Lazy Load

    from .common import _validate_directories

    input_dir, output_dir = _validate_directories(input_dir, output_dir)
    dataset = dataset_dispatch(input_dir)
    alpha_color = ast.literal_eval(alpha_color) if alpha_color else None
    transforms_new = copy.deepcopy(dataset.transforms or {})
    shape = np.array(dataset.full_shape)
    shape[-1] = transforms_new["c"] = transforms_new.pop("c", 4) - int(alpha_color is not None)
    shape[0] = np.ceil(shape[0] / chunk_size).astype(int)

    if mode.lower() not in ("img", "npy"):
        raise ValueError(f"Mode should be one of 'img' or 'npy', got {mode}.")

    if any(str(p).endswith(".exr") for p in getattr(dataset, "paths", [])):
        # TODO: This is due to the alpha blending below, we need alpha in [0, 1] to blend.
        raise NotImplementedError("Task does not yet support EXRs")

    loader = DataLoader(dataset, batch_size=1, num_workers=c.get("max_threads"), collate_fn=default_collate)
    pbar = tqdm(total=len(dataset))

    with ImgDatasetWriter(
        output_dir, transforms=transforms_new, force=force, pattern=pattern
    ) if mode.lower() == "img" else NpyDatasetWriter(
        output_dir, np.ceil(shape).astype(int), transforms=transforms_new, force=force
    ) as writer:
        for i, batch in enumerate(mitertools.ichunked(loader, chunk_size)):
            # Batch is an iterable of (idx, img, pose) that we need to reduce
            idxs, imgs, poses = mitertools.unzip(batch)
            imgs = sum((i / 255.0).astype(float) for i in imgs)
            idxs, poses = np.concatenate(list(idxs)), np.concatenate(list(poses))
            imgs = imgs.squeeze() / len(idxs)

            # Assume image has been tonemapped and undo mapping
            imgs = srgb_to_linearrgb(imgs)
            imgs, alpha = apply_alpha(imgs, alpha_color=alpha_color, ret_alpha=True)

            rgb_img = emulate_rgb_from_merged(
                img_to_tensor(imgs * factor), burst_size=chunk_size, readout_std=readout_std, fwc=fwc, factor=factor
            )
            rgb_img = tensor_to_img(rgb_img * 255)
            pose = pose_interp(poses)(0.5)

            writer[i] = (rgb_img.astype(np.uint8), pose)
            pbar.update(len(idxs))
