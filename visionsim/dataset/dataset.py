from __future__ import annotations

import copy
import os
from collections.abc import Iterable
from functools import cached_property
from pathlib import Path

import imageio
import imageio.v3 as iio
import numpy as np
import numpy.typing as npt
import torch.utils.data
from jsonschema.exceptions import ValidationError
from natsort import natsorted
from numpy.lib.format import open_memmap
from typing_extensions import Any, Literal, cast

from .schema import IMG_SCHEMA, NPY_SCHEMA, read_and_validate, validate_and_write


def _resolve_root(root: str | os.PathLike, mode: Literal["img", "npy"]) -> tuple[list, npt.NDArray, dict | None]:
    """Resolve a dataset root path into:
        - The path of an npy file, or list of img paths
        - A list of poses, or list[None]
        - The contents of the transforms file or None

    Args:
        root (str | os.PathLike): Root path of the dataset.
        mode (Literal['img', 'npy']): type of dataset to process. Can be either 'img' or 'npy'.

    Returns:
        data_paths: list of paths to data files in dataset.
        poses: array of pose information for each data file, if present, else empty array.
        transforms: dictionary of information from json file, if present, else None.
    """
    root = Path(root)

    if root.is_dir():
        if (root / "transforms.json").is_file():
            transforms_path = root / "transforms.json"
            data_path = None
        # if found, transforms path is none and data path to root.
        elif mode.lower() == "img":
            img_exts = set(imageio.config.extensions.known_extensions.keys())
            found_exts = set(p.suffix for p in root.glob("*"))

            if found_exts & img_exts:
                transforms_path = None
                data_path = root
            else:
                raise FileNotFoundError(f"No image files found in {root}.")
        # if found, transforms path is none, data path is frames.npy
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
            err_msg = "a folder" if mode.lower() == "img" else "the path of an npy file"
            raise ValueError(
                f"Dataset root ({root}) not understood. "
                f"It should be a json file, the parent directory of a 'transforms.json' file,"
                f"or {err_msg} containing a collection of images."
            )
    else:
        raise FileNotFoundError(f"Dataset root ({root}) not found!")

    if mode.lower() == "img":
        # Extract paths and ensure they are lexicographically sorted
        if transforms_path:
            transforms = read_and_validate(path=transforms_path, schema=IMG_SCHEMA)
            frames = natsorted(transforms["frames"], key=lambda f: f["file_path"])
            data_paths = [transforms_path.parent / f["file_path"] for f in frames]
            poses = [f["transform_matrix"] for f in frames]
        elif data_path:
            data_paths = natsorted(data_path.glob("*"))
            poses = [None] * len(data_paths)
            transforms = None
        else:
            raise RuntimeError("This should be unreachable!")

        if len(exts := set(Path(p).suffix for p in data_paths)) != 1:
            raise RuntimeError(f"All images must have same extension but found {exts}.")
    else:
        if transforms_path:
            transforms = read_and_validate(path=transforms_path, schema=NPY_SCHEMA)
            data_paths = [transforms_path.parent / transforms["file_path"]]
            poses = [f["transform_matrix"] for f in transforms["frames"]]
        elif data_path:
            data_paths = [data_path]
            poses = [None] * len(np.load(str(data_path), mmap_mode="r"))
            transforms = None
        else:
            raise RuntimeError("This should be unreachable!")

    return data_paths, np.array(poses), transforms


def default_collate(
    batch: Iterable[tuple[int, npt.NDArray, npt.NDArray]],
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Collate function that takes in a batch of [(img_idx, img, pose), ...] and returns (img_idxs, imgs, poses)

    Args:
        batch (Iterable[tuple[int, npt.NDArray, npt.NDArray]]): Iterable of tuples (img_idx, img, pose).

    Returns
        Collated numpy arrays of img_idxs, imgs, poses.
    """
    img_idxs, imgs, poses = zip(*batch)
    img_idxs = np.concatenate([np.atleast_1d(idx) for idx in img_idxs])
    imgs = np.stack([np.atleast_1d(img) for img in imgs])
    poses = np.stack([np.atleast_1d(pose) for pose in poses])
    return img_idxs, imgs, poses


class Dataset(torch.utils.data.Dataset):
    """Base dataset implementation"""

    def __init__(self) -> None:
        # Add type hints for variables that should be defined by subclasses
        self.poses: npt.NDArray
        self.transforms: dict | None

        raise TypeError(
            "Cannot instantiate Dataset directly, use Dataset.from_path() to instantiate a concrete dataset type instead."
        )

    @classmethod
    def from_path(cls, root: str | os.PathLike, mode: Literal["img", "npy"] | None = None) -> NpyDataset | ImgDataset:
        """Given a dataset root, resolve it and instantiate the correct dataset type.

        Args:
            root (str | os.PathLike): path to dataset, either containing folder or `transforms.json`.
            mode (Literal['img', 'npy'] | None, optional): if type of dataset is known, it can be provided, otherwise it
                will try to be inferred. Expects either 'img' or 'npy'. Defaults to None (infer).

        Raises:
            ValueError: raise if `mode` is not understood.
            RuntimeError: raised if dataset type can not be determined.

        Returns:
            dataset instance, either a `NpyDataset` or `ImgDataset`.
        """
        if mode is not None:
            if mode.lower() == "img":
                return ImgDataset(root)
            elif mode.lower() == "npy":
                return NpyDataset(root)
            else:
                raise ValueError(f"Mode should be one of 'img' or 'npy', got {mode}.")

        for klass in (NpyDataset, ImgDataset):
            try:
                return klass(root)
            except (FileNotFoundError, ValueError, RuntimeError, ValidationError):
                pass

        raise RuntimeError(
            f"Could not determine type of dataset at {root}. Try opening dataset with concrete type `NpyDataset` or `ImgDataset`."
        )

    @cached_property
    def arclength(self) -> float:
        """Calculate the length of the trajectory"""

        if not self.transforms:
            return np.nan

        points = self.poses[:, :3, -1]
        dp = np.gradient(points, axis=0)
        dist = np.sqrt((dp**2).sum(axis=1)).sum()

        return dist


class ImgDataset(Dataset):
    """Dataset to iterate over frames (stored as image files) and optionally poses (as .json)."""

    def __init__(self, root: str | os.PathLike) -> None:
        """Initialize an `ImgDataset`, same as `Dataset.from_path(root, mode="img")`.

        Args:
            root (str | os.PathLike): path to dataset, either containing folder or `transforms.json`.
        """
        self.paths, self.poses, self.transforms = _resolve_root(root, mode="img")
        self._idxs = np.arange(len(self))
        self._idxs.setflags(write=False)
        self._full_shape: tuple | None = None

    @cached_property
    def full_shape(self) -> tuple[int, int, int, int]:
        """Get shape of dataset as (N, H, W, C)."""
        if self._full_shape is None:
            if self.transforms:
                h, w, c = self.transforms["h"], self.transforms["w"], self.transforms["c"]
            else:
                # Peak into dataset to get H/W
                _, im, _ = self[0]
                h, w, c = np.array(im).shape
            self._full_shape = (len(self), h, w, c)
        return self._full_shape

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: npt.ArrayLike) -> tuple[int | list[int], npt.ArrayLike, npt.NDArray]:
        if any(i is ... or i is np.newaxis for i in np.atleast_1d(idx)):
            raise NotImplementedError("Only basic indexing is currently supported.")

        if any(isinstance(i, (list, np.ndarray, tuple)) for i in np.atleast_1d(idx)):
            raise NotImplementedError("Integer array indexing is not yet supported.")

        # Split index into the idx of the image path and the img slice
        img_idx, *img_slice = np.atleast_1d(idx)
        img_idx = self._idxs[img_idx]

        if isinstance(img_idx, (int, np.integer)):
            # We must read and decode the whole image even if we are only indexing a pixel...
            im = iio.imread(self.paths[img_idx])
            pose = self.poses[img_idx]
            pixels = im[tuple(img_slice)] if img_slice else im
            return cast(int, img_idx), pixels, pose

        if img_idx.size:
            img_idxs, imgs, poses = zip(*(self[tuple(np.atleast_1d(i).tolist() + img_slice)] for i in img_idx))
            return img_idxs, imgs, np.array(poses)

        return [], [], np.array([]).reshape((0, 4, 4))


class ImgDatasetWriter:
    """ImgDataset writer implemented as a context manager.

    Example:
        .. code-block:: python

            with NpyDatasetWriter(root, transforms=...) as writer:
                for idxs, data, poses in dataset:
                    # Apply any transforms here
                    writer[idxs] = (data, poses)
    """

    def __init__(
        self, root: str | os.PathLike, transforms: dict | None = None, pattern: str = "frame_{:06}.png", force=False
    ) -> None:
        """Initialize `ImgDatasetWriter`.

        Args:
            root (str | os.PathLike): directory in which to save dataset (both frames/*.png and optionally .json)
            transforms (dict | None, optional): transforms of source dataset, `frames` are discarded and camera info is kept. Defaults to None.
            pattern (str, optional): frame filename pattern, will be formatted with frame index. Defaults to "frame_{:06}.png".
            force (bool, optional): if true, overwrite output file(s) if present. Defaults to False.

        Raises:
            FileExistsError: raised if dataset exists at requested location and force is false.
        """
        if ((Path(root) / "transforms.json").is_file() or (Path(root) / "frames").is_dir()) and not force:
            raise FileExistsError(f"Either 'frames/' directory or 'transforms.json' file already exists in {root}")

        self.root = Path(root)
        Path(self.root / "frames").mkdir(exist_ok=True, parents=True)
        self.transforms = copy.deepcopy(transforms or {})
        self.transforms.pop("frames", None)
        self.frames: dict[int, dict[str, Any]] = {}
        self.pattern = pattern

    def __enter__(self):
        return self

    def __setitem__(self, idx: int | list[int], value: np.ndarray | tuple[npt.NDArray, npt.NDArray]):
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
            self.transforms["frames"] = [v for _, v in sorted(self.frames.items())]
            validate_and_write(schema=IMG_SCHEMA, path=self.root / "transforms.json", transforms=self.transforms)


class NpyDataset(Dataset):
    """Dataset to iterate over frames (stored as a possibly bitpacked npy file) and optionally poses (as .json)."""

    def __init__(self, root: str | os.PathLike) -> None:
        """Initialize an `NpyDataset`, same as `Dataset.from_path(root, mode="npy")`.

        Args:
            root (str | os.PathLike): path to dataset, either containing folder or `transforms.json`.
        """
        (self.paths,), self.poses, self.transforms = _resolve_root(root, mode="npy")
        self.data = np.load(str(self.paths), mmap_mode="r")

        if self.transforms and self.transforms.get("bitpack"):
            self.bitpack_dim = self.transforms.get("bitpack_dim")
            self.full_shape: tuple[int, int, int, int] = (
                len(self.transforms["frames"]),
                self.transforms["h"],
                self.transforms["w"],
                self.transforms["c"],
            )
        else:
            self.bitpack_dim = None
            self.full_shape = self.data.shape

        self._idxs = [np.arange(size).astype(int) for size in self.full_shape]

        # Ensure these don't get changed!
        for array in self._idxs:
            array.setflags(write=False)

    def __len__(self) -> int:
        return self.full_shape[0]

    def __getitem__(self, idx: npt.ArrayLike) -> tuple[int | list[int], npt.NDArray, npt.NDArray]:
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
            # Typing bug in numpy-2.2.4: https://github.com/numpy/numpy/issues/27944
            idx_list: Any = np.atleast_1d(idx).tolist()
            idx_list += [slice(None)] * (self.data.ndim - len(idx_list))
            img_idx, *_ = idx_list

            # If the index of a given dimension is an integer, that dimension gets collapsed.
            # We keep track of which dims need to be squeezed and only squeeze them at the end.
            collapsed_dims = [isinstance(dim_idx, (int, np.integer)) for dim_idx in idx_list]

            # Replace all idxs that would collapse a dimension with a slice of size 1.
            # Note: The "i+1 or None" is important here, if i=-1 and we let the slice end be zero,
            #   then the resulting slice (-1:0) will always be empty!
            idx_list = [slice(i, i + 1 or None) if isinstance(i, (int, np.integer)) else i for i in idx_list]

            # The index along the packed dimension might be a slice so we get all indices the slice
            # would correspond to by using a proxy array.
            # Note: If we don't copy here, numpy might return a view which will be modified below
            #   leading to bizarre and inconsistent _silent_ errors.
            idx_list[self.bitpack_dim] = np.copy(self._idxs[self.bitpack_dim][idx_list[self.bitpack_dim]]).flatten()

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
            bit_idx = idx_list[self.bitpack_dim] % 8
            idx_list[self.bitpack_dim] //= 8
            bit_idx += np.arange(bit_idx.size) * 8

            # Perform the indexing, unpacking, bit indexing and dimensionality reduction.
            data = np.unpackbits(self.data[tuple(idx_list)], axis=self.bitpack_dim)
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
    """NpyDataset writer implemented as a context manager.

    Example:
        .. code-block:: python

            src_dataset = ImgDataset(input_dir)
            loader = DataLoader(src_dataset, ...)

            with NpyDatasetWriter(root, shape, transforms=...) as writer:
                for idxs, data, poses in loader:
                    # Apply any transforms here
                    writer[idxs] = (data, poses)
    """

    def __init__(
        self, root: str | os.PathLike, shape: tuple[int, ...], transforms=None, force=False, strict=True
    ) -> None:
        """Initialize `NpyDatasetWriter`.

        Args:
            root (str | os.PathLike): directory in which to save dataset (both .npy and optionally .json)
            shape (tuple[int, ...]): shape of resulting array, must be known ahead of time for npy file creation
            transforms (dict | None, optional): transforms of source dataset, `frames` are discarded and camera info is kept. Defaults to None.
            force (bool, optional): if true, overwrite output file(s) if present. Defaults to False.
            strict (bool, optional): if true, throw error if the whole dataset has not been filled. Defaults to True.

        Raises:
            FileExistsError: raised if dataset exists at requested location and force is false.
        """
        if any((Path(root) / name).is_file() for name in ("frames.npy", "transforms.json")) and not force:
            raise FileExistsError(f"Either 'frames.npy' or 'transforms.json' file already exists in {root}")

        self.root = Path(root)
        Path(root).mkdir(exist_ok=True)
        # Shape needs to be cast to primitive types, see https://github.com/numpy/numpy/issues/28334
        self.data = open_memmap(
            self.root / "frames.npy", mode="w+", dtype=np.uint8, shape=tuple(np.atleast_1d(shape).tolist())
        )
        self.poses = np.zeros((len(self.data), 4, 4))
        self.transforms = copy.deepcopy(transforms or {})
        self.transforms.pop("frames", None)
        self.strict = strict

        self._setidxs = np.zeros(len(self.data))

    def __enter__(self):
        return self

    def __setitem__(self, idx: int | list[int], value: np.ndarray | tuple[npt.NDArray, npt.NDArray]):
        if isinstance(value, tuple):
            data, poses = value
            if self.transforms:
                self.poses[idx] = poses
            self.data[idx] = data
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

            validate_and_write(schema=NPY_SCHEMA, path=self.root / "transforms.json", transforms=self.transforms)

        if self.strict and not np.all(self._setidxs):
            raise RuntimeError("Not all idxs were set, dataset is incomplete!")
