import copy
from pathlib import Path
from typing import Collection, List, Tuple, Union

import imageio
import imageio.v3 as iio
import numpy as np
from jsonschema.exceptions import ValidationError
from natsort import natsorted
from numpy.lib.format import open_memmap
from torch.utils.data import Dataset

from .schema import IMG_SCHEMA, NPY_SCHEMA, _read_and_validate, _validate_and_write


def packbits(image: np.ndarray, dim: int = 1) -> np.ndarray:
    """Pack a single binary frame of shape (H, W, C) into (ceil(H/8), W, C) or (H, ceil(W/8), C)"""
    if dim not in (0, 1):
        raise NotImplementedError(f"Bit packing can only be done along height (dim=0) or height (dim=1), got {dim}.")
    if image.dtype != np.uint8:
        raise TypeError(f"Can only bitpack uint8 types, got {image.dtype}.")
    return np.packbits(image, axis=dim)


def unpackbits(image: np.ndarray, dim: int = 1) -> np.ndarray:
    """Unpack a single binary frame of shape (H, W, C) into (H*8, W, C) or (H, W*8, C)
    Note: If the original frame's size long `dim` is not a multiple of 8, then unpack(pack)
        might lead to a different image shape, which has been rounded off to a multiple of 8.
    """
    if dim not in (0, 1):
        raise NotImplementedError(f"Bit unpacking can only be done along height (dim=0) or height (dim=1), got {dim}.")
    if image.dtype != np.uint8:
        raise TypeError(f"Can only unpack uint8 types, got {image.dtype}.")
    return np.unpackbits(image, axis=dim)


def _resolve_root(root: Union[str, Path], mode: str) -> Tuple[List, np.ndarray, Union[dict, None]]:
    """Given a dataset root path, resolve it into:
    - The path of an npy file, or list of img paths
    - A list of poses, or list[None]
    - The contents of the transforms file or None
    """
    root = Path(root)

    if root.is_dir():
        if (root / "transforms.json").is_file():
            transforms_path = root / "transforms.json"
            data_path = None
        elif mode.lower() == "img":
            img_exts = set(imageio.core.format.known_extensions.keys())
            found_exts = set(p.suffix for p in root.glob("*"))

            if found_exts & img_exts:
                transforms_path = None
                data_path = root
            else:
                raise FileNotFoundError(f"No image files found in {root}.")
        elif mode.lower() == "npy":
            if (root / "frames.npy").is_file():
                transforms_path = None
                data_path = root / "frames.npy"
            else:
                raise FileNotFoundError("Expected one of 'transforms.json' or 'frames.npy'. Please give full path.")
        else:
            raise ValueError(f"Mode should be one of 'img' or 'npy', got {mode}.")
    elif root.is_file():
        if root.suffix == ".json":
            transforms_path = root
            data_path = None
        elif root.suffix == ".npy" and mode.lower() == "npy":
            transforms_path = None
            data_path = root
        else:
            errmsg = "a folder" if mode.lower() == "img" else "the path of an npy file"
            raise ValueError(
                f"Dataset root ({root}) not understood. "
                f"It should be a json file, the parent directory of a 'transforms.json' file,"
                f"or {errmsg} containing a collection of images."
            )
    else:
        raise FileNotFoundError(f"Dataset root ({root}) not found!")

    if mode.lower() == "img":
        # Extract paths and ensure they are lexicographically sorted
        if transforms_path:
            transforms = _read_and_validate(path=transforms_path, schema=IMG_SCHEMA)
            frames = natsorted(transforms["frames"], key=lambda f: f["file_path"])
            data_paths = [transforms_path.parent / f["file_path"] for f in frames]
            poses = [f["transform_matrix"] for f in frames]
        else:
            data_paths = natsorted(data_path.glob("*"))
            poses = [None] * len(data_paths)
            transforms = None

        if len(exts := set(Path(p).suffix for p in data_paths)) != 1:
            raise RuntimeError(f"All images must have same extension but found {exts}.")
    else:
        if transforms_path:
            transforms = _read_and_validate(path=transforms_path, schema=NPY_SCHEMA)
            data_paths = [transforms_path.parent / transforms["file_path"]]
            poses = [f["transform_matrix"] for f in transforms["frames"]]
        else:
            data_paths = [data_path]
            poses = [None] * len(np.load(str(data_path), mmap_mode="r"))
            transforms = None

    return data_paths, np.array(poses), transforms


def default_collate(batch):
    """Collate function that takes in a batch of [(img_idx, img, pose), ...] and returns (img_idxs, imgs, poses)"""
    img_idxs, imgs, poses = zip(*batch)
    img_idxs = np.concatenate([np.atleast_1d(idx) for idx in img_idxs])
    imgs = np.stack([np.atleast_1d(img) for img in imgs])
    poses = np.stack([np.atleast_1d(pose) for pose in poses])
    return img_idxs, imgs, poses


def dataset_dispatch(root, *args, mode=None, **kwargs):
    """Given a dataset root, resolve it and instantiate the correct dataset type"""
    if mode is not None:
        if mode.lower() == "img":
            return ImgDataset(root, *args, **kwargs)
        elif mode.lower() == "npy":
            return NpyDataset(root, *args, **kwargs)
        else:
            raise ValueError(f"Mode should be one of 'img' or 'npy', got {mode}.")

    for klass in (NpyDataset, ImgDataset):
        try:
            return klass(root, *args, **kwargs)
        except (FileNotFoundError, ValueError, RuntimeError, ValidationError):
            pass
    raise RuntimeError(f"Could not determine type of dataset at {root}")


class ImgDataset(Dataset):
    """Dataset to iterate over frames (stored as image files) and optionally poses (as .json)

    Args:
        root: Path of transforms.json file or parent directory, or image folder (if no pose info)
    """

    def __init__(self, root, *args, **kwargs):
        self.paths, self.poses, self.transforms = _resolve_root(root, mode="img")
        self._idxs = np.arange(len(self))
        self._idxs.setflags(write=False)
        self._full_shape = None
        super().__init__()

    @property
    def full_shape(self):
        if self._full_shape is None:
            if self.transforms:
                h, w, c = self.transforms["h"], self.transforms["w"], self.transforms["c"]
            else:
                # Peak into dataset to get H/W
                _, im, _ = self[0]
                h, w, c = im.shape
            self._full_shape = (len(self), h, w, c)
        return self._full_shape

    def __len__(self):
        return len(self.paths)

    def __getitem__(
        self, idx: Union[int, Collection[Union[int, slice]]]
    ) -> Tuple[Union[int, List[int]], Union[np.ndarray, List[np.ndarray]], np.ndarray]:
        if any(i is ... or i is np.newaxis for i in np.atleast_1d(idx)):
            raise NotImplementedError("Only basic indexing is currently supported.")

        if any(isinstance(i, (list, np.ndarray, tuple)) for i in np.atleast_1d(idx)):
            raise NotImplementedError("Integer array indexing is not yet supported.")

        # Split index into the idx of the imag path and the img slice
        idx = np.atleast_1d(idx)
        img_idx, *img_slice = idx
        img_idx = self._idxs[img_idx]

        if isinstance(img_idx, (int, np.integer)):
            # We must read and decode the whole image even if we are only indexing a pixel...
            im = iio.imread(self.paths[img_idx])
            pose = self.poses[img_idx]
            im = im[tuple(img_slice)] if img_slice else im
            return img_idx, im, pose

        if img_idx.size:
            img_idxs, imgs, poses = zip(*(self[tuple(np.atleast_1d(i).tolist() + img_slice)] for i in img_idx))
            return img_idxs, imgs, np.array(poses)

        return [], [], np.array([]).reshape((0, 4, 4))


class ImgDatasetWriter:
    """ImgDataset writer implemented as a context manager

    Args:
        root: directory in which to save dataset (both frames/*.png and optionally .json)
        transforms: transforms of source dataset, "frames" are discarded and camera info is kept
        pattern: frame filename pattern, will be formatted with frame index
        force: if true, overwrite output file(s) if present
    """

    def __init__(self, root, transforms=None, pattern="frame_{:06}.png", force=False):
        if ((Path(root) / "transforms.json").is_file() or (Path(root) / "frames").is_dir()) and not force:
            raise FileExistsError(f"Either 'frames/' directory or 'transforms.json' file already exists in {root}")

        self.root = Path(root)
        Path(root / "frames").mkdir(exist_ok=True, parents=True)
        self.transforms = copy.deepcopy(transforms or {})
        self.transforms.pop("frames", None)
        self.pattern = pattern
        self.frames = {}

    def __enter__(self):
        return self

    def __setitem__(self, idx: Union[int, List[int]], value: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]):
        if isinstance(idx, (int, np.integer)):
            path = self.root / "frames" / self.pattern.format(idx)
            self.frames[idx] = {"file_path": str(path.relative_to(self.root))}

            if isinstance(value, tuple):
                data, pose = value
                iio.imwrite(path, data)
                self.frames[idx]["transform_matrix"] = np.array(pose).tolist()
            elif self.transforms:
                raise RuntimeError("Expected image and pose tuple.")
            else:
                iio.imwrite(path, value)
        else:
            if isinstance(value, tuple):
                for i, data, pose in zip(idx, *value):
                    self[i] = (data, pose)
            else:
                for i, data in zip(idx, value):
                    self[i] = data

    def __exit__(self, exc_type, exc_val, exc_tb):
        # TODO: Handle any errors...
        if self.transforms:
            self.transforms["frames"] = list(self.frames.values())
            _validate_and_write(schema=IMG_SCHEMA, path=self.root / "transforms.json", transforms=self.transforms)


class NpyDataset(Dataset):
    def __init__(self, root, *args, **kwargs):
        (self.path,), self.poses, self.transforms = _resolve_root(root, mode="npy")
        self.data = np.load(str(self.path), mmap_mode="r")

        if self.transforms and self.transforms.get("bitpack"):
            self.bitpack_dim = self.transforms.get("bitpack_dim")
            self.full_shape = (
                len(self.transforms["frames"]),
                self.transforms.get("h"),
                self.transforms.get("w"),
                self.transforms.get("c"),
            )
        else:
            self.bitpack_dim = None
            self.full_shape = self.data.shape

        self._idxs = [np.arange(size).astype(int) for size in self.full_shape]

        # Ensure these don't get changed!
        for array in self._idxs:
            array.setflags(write=False)
        super().__init__()

    def __len__(self):
        return self.full_shape[0]

    def __getitem__(
        self, idx: Union[int, Collection[Union[int, slice]]]
    ) -> Tuple[Union[int, List[int]], np.ndarray, np.ndarray]:
        if any(i is ... or i is np.newaxis for i in np.atleast_1d(idx)):
            raise NotImplementedError("Only basic indexing is currently supported.")

        if any(isinstance(i, (list, np.ndarray, tuple)) for i in np.atleast_1d(idx)):
            # Only one index can be a list of integers, otherwise, if we slice with multiple ones
            # numpy will understand it as individual point coordinates instead of coords
            # of chunks, i.e:
            #   np.arange(4).reshape(2, 2)[0:2, 0:2] == array([[0, 1], [2, 3]])
            #                         ---- VS. ----
            #   np.arange(4).reshape(2, 2)[[0, 1], [0, 1]] == array([0, 3])
            # To address this problem, we should probably convert every idx into a list of
            # indices and find the open mesh-grid of all points (i.e: with `np.ix_`), and only
            # do so when needed (for perf), but for now we raise.
            raise NotImplementedError("Integer array indexing is not yet supported.")

        if self.bitpack_dim is not None:
            # Expand index over all dimensions
            idx = np.atleast_1d(idx).tolist()
            idx += [slice(None)] * (self.data.ndim - len(idx))
            img_idx, *_ = idx

            # If the index of a given dimension is an integer, that dimension gets collapsed.
            # We keep track of which dims need to be squeezed and only squeeze them at the end.
            collapsed_dims = [isinstance(dim_idx, (int, np.integer)) for dim_idx in idx]

            # Replace all idxs that would collapse a dimension with a slice of size 1.
            # Note: The "i+1 or None" is important here, if i=-1 and we let the slice end be zero,
            #   then the resulting slice (-1:0) will always be empty!
            idx = [slice(i, i + 1 or None) if isinstance(i, (int, np.integer)) else i for i in idx]

            # The index along the packed dimension might be a slice so we get all indices the slice
            # would correspond to by using a proxy array.
            # Note: If we don't copy here, numpy might return a view which will be modified below
            #   leading to bizarre and inconsistent _silent_ errors.
            idx[self.bitpack_dim] = np.copy(self._idxs[self.bitpack_dim][idx[self.bitpack_dim]]).flatten()

            # Compute the packed index and a secondary bit idx.
            # If we were unpacking into a new dimension then the bit index would simply be %8 of
            # the packed index. However, numpy's unpack bits does not do this, instead it lengths
            # the axis along which we unpack by a factor of 8. We need to shift the bit indices
            # by this lengthening factor.
            # Ex:
            #   Real indices along packed axis:         [31, 8, 121]
            #   Packed indices:                         [ 3,  1, 15]
            #   Bit indices (no shift):                 [ 0,  0,  1]
            #   Bit indices (shifted correct amount):   [ 0,  8, 17]
            # Note: The `bit_idx` is made into at least a 1d idx as to preserve dimensionality,
            #   otherwise it would mess up the `collapsed_dims` above.
            bit_idx = idx[self.bitpack_dim] % 8
            idx[self.bitpack_dim] //= 8
            bit_idx += np.arange(bit_idx.size) * 8

            # Perform the indexing, unpacking, bit indexing and dimensionality reduction.
            data = np.unpackbits(self.data[tuple(idx)], axis=self.bitpack_dim)
            data = np.take(data, bit_idx, axis=self.bitpack_dim)
            squeeze_dims = tuple(
                i for i, (size, collapsed) in enumerate(zip(collapsed_dims, data.shape)) if collapsed and size == 1
            )
            data = data.squeeze(axis=squeeze_dims)
        else:
            # If not bitpacked, just index...
            data = self.data[idx]
            img_idx, *_ = np.atleast_1d(idx)

        # Idx of img might be a slice...
        img_idx = self._idxs[0][img_idx]
        return img_idx, data, self.poses[img_idx]


class NpyDatasetWriter:
    """NpyDataset writer implemented as a context manager

    Usage:
        src_dataset = ImgDataset(input_dir)
        loader = DataLoader(src_dataset, ...)

        with NpyDatasetWriter(root, shape, transforms=...) as writer:
            for idxs, data, poses in loader:
                # Apply any transforms here
                writer[idxs] = (data, poses)

    Args:
        root: directory in which to save dataset (both .npy and optionally .json)
        shape: shape of resulting array, must be known ahead of time for npy file creation
        transforms: transforms of source dataset, "frames" are discarded and camera info is kept
        force: if true, overwrite output file(s) if present
        strict: if true, throw error if the whole dataset has not been filled
    """

    def __init__(self, root, shape, transforms=None, force=False, strict=True):
        if any((Path(root) / name).is_file() for name in ("frames.npy", "transforms.json")) and not force:
            raise FileExistsError(f"Either 'frames.npy' or 'transforms.json' file already exists in {root}")

        self.root = Path(root)
        Path(root).mkdir(exist_ok=True)
        self.data = open_memmap(root / "frames.npy", mode="w+", dtype=np.uint8, shape=tuple(shape))
        self.poses = np.zeros((len(self.data), 4, 4))
        self.transforms = copy.deepcopy(transforms or {})
        self.transforms.pop("frames", None)
        self.strict = strict

        self._setidxs = np.zeros(len(self.data))

    def __enter__(self):
        return self

    def __setitem__(self, idx: Union[int, List[int]], value: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]):
        if isinstance(value, tuple):
            data, poses = value
            self.data[idx] = data
            self.poses[idx] = poses
        elif self.transforms:
            raise RuntimeError("Expected image and pose tuple.")
        else:
            self.data[idx] = value
        self._setidxs[idx] = 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        # TODO: Handle any errors...
        self.data.flush()

        if self.transforms:
            self.transforms["file_path"] = "frames.npy"
            self.transforms["frames"] = [{"transform_matrix": mat.tolist()} for mat in self.poses]

            _validate_and_write(schema=NPY_SCHEMA, path=self.root / "transforms.json", transforms=self.transforms)

        if self.strict and not np.all(self._setidxs):
            raise RuntimeError("Not all idxs were set, dataset is incomplete!")
