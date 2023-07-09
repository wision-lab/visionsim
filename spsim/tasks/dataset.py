from invoke import task


@task(
    help={
        "input_dir": "directory in which to look for frames",
        "output_dir": "directory in which to save npy file",
        "file_name": "name of file containing transforms, default: 'transforms.json'",
        "bitpack": "if true, each chunk of 8 binary pixels will by packed into a single byte. "
        "Only enable if data is binary valued. Default: False",
        "bitpack_dim": "Axis along which to pack bits (H=1, W=2), default: 2",
    }
)
def frames_to_npy(c, input_dir, output_dir, file_name="transforms.json", bitpack=False, bitpack_dim=None):
    """Convert an image folder based dataset to a NPY dataset"""
    # TODO: Add data integrity check
    #   - If npy file is present, scan through it to check if any data is missing
    #   - If so, fill in those frames only and skip the rest (add allow-skips arg?)
    # TODO: Add check to stop the user from bitpacking non-binary data
    # TODO: Support more types of frames, depth, normals, etc...
    from pathlib import Path

    import numpy as np
    from natsort import natsorted
    from numpy.lib.format import open_memmap
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    from spsim.dataset import ImgDataset
    from spsim.schema import NPY_SCHEMA, NS_SCHEMA, _read_and_validate, _validate_and_write
    from spsim.tasks.common import _validate_directories

    input_dir, output_dir = _validate_directories(input_dir, output_dir)
    transforms = _read_and_validate(path=input_dir / file_name, schema=NS_SCHEMA)

    # Extract paths and ensure they are lexicographically sorted
    frames = natsorted(transforms["frames"], key=lambda f: f["file_path"])
    img_paths = [str(input_dir / f["file_path"]) for f in frames]
    exts = set(Path(p).suffix for p in img_paths)

    if any(any(k != "file_path" and "path" in k for k in f.keys()) for f in frames):
        raise NotImplementedError(
            "Only color frames are supported for now. Keys such as 'depth_file_path' "
            "or 'mask_path' are not supported."
        )

    if len(exts) != 1:
        raise RuntimeError(f"All images must have same extension but found {exts}.")

    # Bitpack if either is set, defaults to bitpacking width dimension
    if bitpack or bitpack_dim is not None:
        bitpack = True
        bitpack_dim = bitpack_dim if bitpack_dim is not None else 2

        if bitpack_dim == 0 or bitpack_dim >= 3:
            raise NotImplementedError("Can only bitpack along H or W.")

        transforms["bitpack"] = bitpack
        transforms["bitpack_dim"] = bitpack_dim

    h, w = transforms["h"], transforms["w"]
    shape = np.array([len(img_paths), int(h), int(w), 3])

    if bitpack:
        shape[bitpack_dim] /= 8

    frames_array = open_memmap(
        output_dir / "frames.npy", mode="w+", dtype=np.uint8, shape=tuple(np.ceil(shape).astype(int))
    )

    dataset = ImgDataset(img_paths, apply_alpha=True, bitpack=bitpack, bitpack_dim=bitpack_dim)
    loader = DataLoader(
        dataset, batch_size=4, num_workers=c.get("max_threads"), prefetch_factor=1, collate_fn=lambda x: x
    )
    pbar = tqdm(total=len(dataset))

    for batch in loader:
        for i, im in batch:
            pbar.update(1)
            frames_array[i] = im

    transforms["file_path"] = "frames.npy"
    new_frames = [{k: v for k, v in f.items() if "file_path" not in k} for f in frames]
    transforms["frames"] = new_frames
    transforms["frames"] = transforms.pop("frames")  # place frames at bottom

    _validate_and_write(schema=NPY_SCHEMA, path=output_dir / file_name, transforms=transforms)
