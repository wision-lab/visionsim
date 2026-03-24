from __future__ import annotations

import copy
import os
from collections.abc import Callable, Sequence
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, cast

import imageio.v3 as iio
import more_itertools as mitertools
import natsort
import numpy as np
import numpy.typing as npt
import OpenEXR  # type: ignore
import torch.utils.data
from typing_extensions import Self

from visionsim.dataset.models import Camera, Metadata
from visionsim.types import Matrix4x4


class PathTransforms:
    """Given a sequence of paths to load from, yield the minimal transforms dictionary for each path.

    Specifically, for image paths we just yield ``{"file_path": paths[idx]}`` for every index, but if some of the paths
    are numpy arrays, and ``iter_npys`` is true, we unpack the array's first dimension, and return the corresponding
    ``offset`` as well. For instance, if ``paths`` points to a png, a npy of shape (4, H, W, C), and another png, an
    index of 3 will return the path to the numpy file and an offset of 3."""

    def __init__(self, paths: Sequence[Path], iter_npys: bool = True, **kwargs) -> None:
        """Initialize a sequence of "dummy" transform dictionaries from a set of paths.

        Args:
            paths (Sequence[Path]): Paths to yield from
            iter_npys (bool, optional): If true, yield from numpy arrays will before
                moving on to next path. Defaults to True.
            **kwargs (dict[str, Any]): Additional key/value pairs to include in each transform dict.
        """
        if iter_npys:
            lengths = [len(np.load(str(path), mmap_mode="r")) if path.suffix.lower() == ".npy" else 1 for path in paths]
        else:
            lengths = [len(paths)]

        self.iter_npys = iter_npys
        self.cumulative_lengths = np.cumsum(lengths)
        self.lengths = lengths
        self.paths = paths
        self.kwargs = kwargs

    def __len__(self) -> int:
        """Length of dataset"""
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx: int) -> dict[str, int | Path]:
        """Return dummy transform dict at provided index.

        Args:
            idx (int): Index of item to return

        Returns:
            dict[str, int | Path]: transforms dictionary containing "file_path" of data
                and "offset" amount if loading from a numpy array.
        """
        if self.iter_npys:
            path_idx = int(np.searchsorted(self.cumulative_lengths, idx, side="right"))
            transform = {"file_path": self.paths[path_idx]}

            if self.paths[path_idx].suffix.lower() == ".npy":
                transform["offset"] = (idx - self.cumulative_lengths[path_idx]) % self.lengths[path_idx]
        else:
            transform = {"file_path": self.paths[idx]}
        return transform | self.kwargs


class Dataset(torch.utils.data.Dataset):
    """Main dataset class for loading a ``.db``/``.json`` dataset or a set of image/exr/npy files."""

    def __init__(
        self,
        transforms: Sequence[dict[str, Any]],
        root: str | os.PathLike | None = None,
        cameras: set[Camera] | None = None,
    ) -> None:
        """Initialize a dataset object.

        Note:
            No data validation is performed here, you likely want to use one of the
            classmethods such as :meth:`from_path` or :meth:`from_pattern` instead.

        Args:
            transforms (Sequence[dict[str, Any]]): A sequence of transforms dicts, which at a
                minimum should have a ``file_path`` key defined.
            root (str | os.PathLike | None, optional): Dataset root directory, if supplied all ``file_path``\\s
                are assumed to be relative to it. Defaults to None.
            cameras (set[Camera] | None, optional): Set of camera objects. Defaults to None.
        """
        self.transforms = transforms
        self.root = Path(root).resolve() if root else None
        self.cameras = cameras

    @classmethod
    def from_path(cls, root: str | os.PathLike) -> Self:
        """Load a dataset from a path.

        Args:
            root (str | os.PathLike): Path to dataset file (either a ``.db`` or ``.json`` file)
                or a directory containing a valid dataset.

        Raises:
            RuntimeError: raised if a dataset is not found at the provided path, or if multiple datasets are found.

        Returns:
            Self: instantiated Dataset object
        """
        root = Path(root).resolve()

        if root.is_dir():
            candidates = list(root.glob("*.db")) + list(root.glob("*.json"))

            if len(candidates) > 1:
                raise RuntimeError(
                    f"Ambiguous dataset root. Found multiple metadata sources ({[c.relative_to(root) for c in candidates]})."
                )
            elif len(candidates) == 0:
                raise RuntimeError(f"No dataset found at {root}.")
            metadata_path, *_ = candidates
        else:
            metadata_path = root
            root = root.parent

        metadata = Metadata.load(metadata_path)
        transforms = metadata.to_dense_transforms()
        return cls(root=root, cameras=metadata.cameras, transforms=transforms)

    @classmethod
    def from_paths(
        cls,
        paths: Sequence[Path],
        iter_npys: bool = True,
        root: str | os.PathLike | None = None,
        cameras: set[Camera] | None = None,
        **kwargs,
    ) -> Self:
        """Create a dataset object from a collection of data files.

        Args:
            paths (Sequence[Path]): Paths to load data from.
            iter_npys (bool, optional): If true, step into the first dimension of any numpy files
                when iterating over data. Defaults to True.
            root (str | os.PathLike | None, optional): Dataset root directory, if supplied all ``file_path``\\s
                are assumed to be relative to it. Defaults to None.
            cameras (set[Camera] | None, optional): Set of camera objects. Defaults to None.
            **kwargs (dict[str, Any]): Optional keyword arguments passed to :class:`PathTransforms`

        Raises:
            ValueError: raised if provided paths do not exist.

        Returns:
            Self: instantiated Dataset object
        """
        if any(not (Path(root or "") / p).exists() for p in paths):
            raise ValueError("Some paths do not exist!")

        transforms = PathTransforms(paths=[Path(p) for p in paths], iter_npys=iter_npys, **kwargs)
        return cls(transforms=cast(Sequence, transforms), root=root, cameras=cameras)

    @classmethod
    def from_pattern(
        cls,
        root: str | os.PathLike,
        pattern: str,
        cameras: set[Camera] | None = None,
        iter_npys: bool = True,
        key: Callable[[Any], Any] = natsort.natsort_key,
        **kwargs,
    ) -> Self:
        """Same as :meth:`from_paths` but will search for all paths that match the provided pattern
        (as found by `pathlib's glob <https://docs.python.org/3/library/pathlib.html#pathlib.Path.glob>`_)"""
        paths = [p.relative_to(root) for p in sorted(Path(root).glob(pattern), key=key)]
        return cls.from_paths(paths=paths, iter_npys=iter_npys, root=root, cameras=cameras, **kwargs)

    @cached_property
    def paths(self) -> list[Path]:
        """List of all data file paths (normalized)"""
        return [(self.root or Path("")) / t["file_path"] for t in self.transforms]

    @cached_property
    def poses(self) -> list[Matrix4x4] | None:
        """List of all camera poses, if available"""
        poses = [np.array(t["transform_matrix"]) for t in self.transforms if "transform_matrix" in t]

        if len(self) == len(poses):
            return poses
        return None

    @staticmethod
    def _slice_bitpacked_array(
        data: npt.NDArray,
        idx: tuple[int | slice, ...] = tuple(),
        bitpack_dim: Literal[0, 1, 2] | None = None,
        unpacked_size: int | None = None,
    ) -> int | npt.NDArray:
        if any(i is ... or i is np.newaxis for i in idx):
            raise NotImplementedError("Only basic indexing is currently supported.")

        if any(isinstance(i, (list, np.ndarray, tuple)) for i in idx):
            raise NotImplementedError("Integer and boolean array indexing is not yet supported.")

        if bitpack_dim is not None:
            # Expand index over all dimensions
            # Typing bug in numpy-2.2.4: https://github.com/numpy/numpy/issues/27944
            idx_list: Any = np.atleast_1d(idx).tolist()
            idx_list += [slice(None)] * (data.ndim - len(idx_list))

            # If the index of a given dimension is an integer, that dimension gets collapsed.
            # We keep track of which dims need to be squeezed and only squeeze them at the end.
            collapsed_dims = [isinstance(dim_idx, (int, np.integer)) for dim_idx in idx_list]

            # Replace all idxs that would collapse a dimension with a slice of size 1.
            # Note: The "i+1 or None" is important here, if i=-1 and we let the slice end be zero,
            #   then the resulting slice (-1:0) will always be empty!
            idx_list = [slice(i, i + 1 or None) if isinstance(i, (int, np.integer)) else i for i in idx_list]

            # The index along the packed dimension might be a slice so we get all indices the slice
            # would correspond to by using a proxy array.
            idx_list[bitpack_dim] = (
                np.arange(unpacked_size or (data.shape[bitpack_dim] * 8)).astype(int)[idx_list[bitpack_dim]].flatten()
            )

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
            bit_idx = idx_list[bitpack_dim] % 8
            idx_list[bitpack_dim] //= 8
            bit_idx += np.arange(bit_idx.size) * 8

            # Perform the indexing, unpacking, bit indexing and dimensionality reduction.
            data = np.unpackbits(data[tuple(idx_list)], axis=bitpack_dim)
            data = np.take(data, bit_idx, axis=bitpack_dim)
            squeeze_dims = tuple(
                i for i, (size, collapsed) in enumerate(zip(collapsed_dims, data.shape)) if collapsed and size == 1
            )
            return data.squeeze(axis=squeeze_dims)
        return data[idx]

    @staticmethod
    def load_data(
        path: str | os.PathLike,
        idx: tuple[int | slice, ...] = tuple(),
        auto_collapse: bool = True,
        bitpack_dim: Literal[0, 1, 2] | None = None,
        unpacked_size: int | None = None,
    ) -> int | float | npt.NDArray:
        """Load data from provided path, optionally slicing it.

        Support various image formats, as provided by `imageio's imread <https://imageio.readthedocs.io/en/
        stable/_autosummary/imageio.v3.imread.html>`_, exrs files, and numpy arrays (optionally bitpacked).

        Note:
            This function uses `OpenEXR <https://openexr.com/en/latest/python.html#the-openexr-python-module>`_
            to read exr files as both imageio and opencv cannot read an exr file when the data is stored in any
            other channel than RGB(A). As of Blender v4 single-channel data, such as depth maps, are correctly
            saved as single channel exrs, in the V channel. Previously, Blender just saved these as RGB by
            duplicating the data channel-wise. This function (optionally) auto-detects this issue and returns
            only a single channel numpy array.

        Note:
            Numpy arrays are not loaded into memory, instead they are memory mapped, making this
            function safe to use with very large arrays.

        Args:
            path (str | os.PathLike): Path to the image file or numpy array.
            idx (tuple[int | slice], optional): If present, slice the data using this index.
                In most cases this is equivalent to slicing the data after loading it, but
                for bitpacked numpy arrays, the slice needs to be modified first.
                Defaults to empty tuple (no slicing).
            auto_collapse (bool, optional): If true, when loading an EXR file that has duplicated
                channels, collapse them down into a single channel. See note for more. Only used when
                loading an EXR file that is saved using the "RGB" channel. Defaults to True.
            bitpack_dim (Literal[0, 1, 2] | None, optional): Axis along which to bits have been packed.
                Only used when loading data from a numpy file. Defaults to None.
            unpacked_size (int | None, optional): Length of bitpacked axis once unpacked, if not specified
                data will be returned in a larger array that is a multiple of 8. Only used when loading
                from a numpy array that is bitpacked.

        Returns:
            int | float | npt.NDArray: Data loaded from path
        """
        if Path(path).suffix.lower() == ".npy":
            data = np.load(str(path), mmap_mode="r")
            return Dataset._slice_bitpacked_array(data, idx=idx, bitpack_dim=bitpack_dim, unpacked_size=unpacked_size)
        elif Path(path).suffix.lower() == ".exr":
            with OpenEXR.File(str(path)) as f:
                if len(f.channels()) == 1 and list(f.channels().keys())[0] in ("RGBA", "RGB", "V"):
                    data = list(f.channels().values())[0].pixels

                    if data.ndim == 2:
                        data = data[..., np.newaxis]
                    elif (
                        auto_collapse
                        and data.ndim == 3
                        and all(np.allclose(a, b) for a, b in mitertools.pairwise(data.transpose(2, 0, 1)))
                    ):
                        data = data[..., :1]
                else:
                    raise RuntimeError(f"Cannot read EXR with channels {list(f.channels().keys())}.")
        else:
            data = iio.imread(path)
        return data[tuple(idx)] if idx else data

    def __len__(self) -> int:
        """Length of dataset"""
        return len(self.paths)

    def __getitem__(
        self, idx: npt.ArrayLike
    ) -> tuple[
        int | float | npt.NDArray | tuple[int | float | npt.NDArray, ...], dict[str, Any] | tuple[dict[str, Any], ...]
    ]:
        """Fetch an item from the dataset and return it's data and associated metadata.

        Args:
            idx (npt.ArrayLike): Index of item, usually an integer, but more complex indices are supported.

        Raises:
            NotImplementedError: raised when trying to slice using Ellipses, np.NewAxis, or integer/boolean arrays.

        Returns:
            tuple[int | float | npt.NDArray | tuple[int | float | npt.NDArray, ...], dict[str, Any] | tuple[dict[str, Any], ...]]:
                returns a tuple containing the data (as an array or number) and a dictionary containing metadata such as
                the "file_path" data was loaded from, and camera info if applicable. If the requested index spans multiple
                files, a tuple of data and tuple of dicts will be returned.
        """
        # Split index into the idx of the frame and the frame sub-slice
        frame_idx, *sub_slice = idx = np.atleast_1d(idx)
        frame_idx = np.arange(len(self))[frame_idx]

        if any(i is ... or i is np.newaxis for i in idx):
            raise NotImplementedError("Only basic indexing is currently supported.")

        if any(isinstance(i, (list, np.ndarray, tuple)) for i in idx):
            raise NotImplementedError("Integer and boolean array indexing is not yet supported.")

        # Return data from single frame/path
        # We must read and decode the whole image even if we are only indexing a pixel...
        if isinstance(frame_idx, (int, np.integer)):
            transform = copy.copy(self.transforms[int(frame_idx)])
            transform["file_path"] = self.paths[frame_idx]

            if self.poses:
                transform["transform_matrix"] = self.poses[frame_idx]

            sub_slice = tuple([transform["offset"]] + sub_slice) if "offset" in transform else sub_slice

            bitpack_dim = transform.get("bitpack_dim")
            full_shape = (len(self), transform.get("h"), transform.get("w"), transform.get("c"))
            unpacked_size = full_shape[bitpack_dim] if bitpack_dim is not None else None

            data = self.load_data(
                transform["file_path"], sub_slice, bitpack_dim=bitpack_dim, unpacked_size=unpacked_size
            )
            return data, transform
        elif frame_idx.size:
            data, transform = zip(*(self[tuple(np.atleast_1d(i).tolist() + sub_slice)] for i in frame_idx))
            return data, transform
        return tuple(), tuple()
