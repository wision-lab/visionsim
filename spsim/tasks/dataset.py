from invoke import task


@task(
    help={
        "input_dir": "directory in which to look for frames",
        "output_dir": "directory in which to save npy file",
        "bitpack": "if true, each chunk of 8 binary pixels will by packed into a single byte. "
        "Only enable if data is binary valued. Default: False",
        "bitpack_dim": "axis along which to pack bits (H=1, W=2), default: 2",
        "batch_size": "number of frames to write at once, default: 4",
        "alpha_color": "if set, blend with this background color and do not store "
        "alpha channel. default: (255, 255, 255)",
        "force": "if true, overwrite output file(s) if present, default: False",
    }
)
def imgs_to_npy(
    c,
    input_dir,
    output_dir,
    bitpack=False,
    bitpack_dim=None,
    batch_size=4,
    alpha_color="(255, 255, 255)",
    force=False,
):
    """Convert an image folder based dataset to a NPY dataset"""
    # TODO: Add data integrity check
    #   - If npy file is present, scan through it to check if any data is missing
    #   - If so, fill in those frames only and skip the rest (add allow-skips arg?)
    # TODO: Add check to stop the user from bitpacking non-binary data
    # TODO: Support more types of frames, depth, normals, etc...
    import ast

    import numpy as np
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    from spsim.dataset import ImgDataset, NpyDatasetWriter, default_collate

    from .common import _validate_directories

    input_dir, output_dir = _validate_directories(input_dir, output_dir)
    dataset = ImgDataset(input_dir)
    transforms_new = dataset.transforms or {}

    if dataset.transforms:
        if any(any(k != "file_path" and "path" in k for k in f.keys()) for f in dataset.transforms["frames"]):
            raise NotImplementedError(
                "Only color frames are supported for now. Keys such as 'depth_file_path' "
                "or 'mask_path' are not supported."
            )
    alpha_color = ast.literal_eval(alpha_color) if alpha_color else None

    if dataset.transforms:
        h, w = dataset.transforms["h"], dataset.transforms["w"]
    else:
        # Peak into dataset to get H/W
        _, im, _ = dataset[0]
        h, w, _ = im.shape
    shape = np.array([len(dataset), int(h), int(w), 3])

    # Bitpack if either is set, defaults to bitpacking width dimension
    if bitpack or bitpack_dim is not None:
        bitpack_dim = bitpack_dim if bitpack_dim is not None else 2
        bitpack = True

        if bitpack_dim == 0 or bitpack_dim >= 3:
            raise NotImplementedError("Can only bitpack along H or W.")

        transforms_new["bitpack"] = bitpack
        transforms_new["bitpack_dim"] = bitpack_dim
        shape[bitpack_dim] /= 8

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=c.get("max_threads"), collate_fn=default_collate)
    pbar = tqdm(total=len(dataset))

    with NpyDatasetWriter(
        output_dir, shape=np.ceil(shape).astype(int), transforms=transforms_new, force=force
    ) as writer:
        for i, (idxs, imgs, poses) in enumerate(loader):
            if alpha_color is not None and imgs.ndim == 4 and imgs.shape[-1] in (2, 4):
                alpha = imgs[..., -1][..., None]
                imgs = imgs[..., :-1] * alpha + alpha_color * (1 - alpha)
            if bitpack:
                imgs = imgs >= 128
                imgs = np.packbits(imgs, axis=bitpack_dim)
            writer[idxs] = (imgs, poses)
            pbar.update(len(idxs))
