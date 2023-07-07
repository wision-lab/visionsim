import json

import numpy as np
from invoke import task

from spsim.tasks.common import _validate_directories


@task(
    help={
        "input_dir": "directory in which to look for frames",
        "output_dir": "directory in which to save npy file",
        "file_name": "name of file containing transforms, default: 'transforms.json'",
        "bitpack": "if true, each chunk of 8 binary pixels will by packed into a single byte. "
                   "Only enable if data is binary valued. Default: False",
        "bitpack_dim": "Axis along which to pack bits (H=1, W=2), default: 2"
    }
)
def frames_to_npy(_, input_dir, output_dir, file_name="transforms.json", bitpack=False, bitpack_dim=None):
    """Convert an image folder based dataset to a NPY dataset (experimental)"""
    from natsort import natsorted
    from numpy.lib.format import open_memmap
    from tqdm.auto import tqdm

    from spsim.io import read_img

    input_dir, output_dir = _validate_directories(input_dir, output_dir)

    with (input_dir / file_name).open("r") as f:
        transforms = json.load(f)

    # Expect either blender or nerf style transforms, giving priority to blender-style.
    img_paths = [f["file_paths"][0] if "file_paths" in f else f["file_path"] for f in transforms["frames"]]
    frames = natsorted(transforms["frames"], key=lambda f: f["file_paths"][0] if "file_paths" in f else f["file_path"])
    img_paths = natsorted(str(input_dir / p) for p in img_paths)

    # Ensure w/h are set in transforms.json
    w = transforms["w"] if "w" in transforms else 2 * (transforms["camera"]["cx"] - transforms["camera"]["shift_x"])
    h = transforms["h"] if "h" in transforms else 2 * (transforms["camera"]["cy"] - transforms["camera"]["shift_y"])
    transforms["w"] = w
    transforms["h"] = h

    # Bitpack if either is set, defaults to bitpacking width dimension
    if bitpack or bitpack_dim is not None:
        bitpack = True
        bitpack_dim = bitpack_dim if bitpack_dim is not None else 2

        if bitpack_dim == 0 or bitpack_dim >= 3:
            raise NotImplementedError("Can only bitpack along H or W.")

        transforms["bitpack"] = bitpack
        transforms["bitpack_dim"] = bitpack_dim
        shape = np.array([len(img_paths), int(h), int(w), 3])
        shape[bitpack_dim] /= 8
    else:
        shape = np.array([len(img_paths), int(h), int(w), 3])

    frames_array = open_memmap(
        output_dir / "frames.npy", mode="w+", dtype=np.uint8, shape=tuple(np.ceil(shape).astype(int))
    )

    for i, path in enumerate(tqdm(img_paths)):
        im, _ = read_img(path, apply_alpha=True)

        if bitpack:
            im = im >= 0.5
            im = np.packbits(im, axis=bitpack_dim - 1)
        else:
            im = (im * 255).astype(np.uint8)
        frames_array[i] = im

    transforms["file_path"] = "frames.npy"
    new_frames = [{k: v for k, v in f.items() if k not in ("file_path", "file_paths")} for f in frames]
    transforms["frames"] = new_frames

    with (output_dir / file_name).open("w") as f:
        json.dump(transforms, f, indent=2)
