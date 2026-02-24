from __future__ import annotations

import functools
import shutil
from pathlib import Path

import numpy as np


def convert(
    input_dir: Path,
    output_dir: Path | None = None,
    force: bool = False,
) -> None:
    """Convert a ``.db`` database to a ``.json`` or vice-versa.

    Args:
        input_dir: directory in which to look for dataset
        output_dir: directory in which to save new dataset.
            If not set, save new metadata file in same directory,
            otherwise copy over all data to a new directory.
        force: if true, overwrite output file(s) if present
    """
    from visionsim.dataset import Metadata

    if output_dir:
        if input_dir.resolve() == output_dir.resolve():
            raise RuntimeError("Input and output directory cannot be the same!")
        if output_dir.exists() and not force:
            raise FileExistsError("Output directory already exists.")
        else:
            shutil.rmtree(output_dir, ignore_errors=True)

    meta = Metadata.from_path(input_dir)
    assert meta.path is not None

    rel_path = meta.path.relative_to(input_dir.resolve())
    meta_path = rel_path.with_suffix(".db" if meta.path.suffix == ".json" else ".json")

    if output_dir:
        shutil.copytree(input_dir, output_dir)
        meta.save(output_dir / meta_path)
        (output_dir / rel_path).unlink(missing_ok=True)
    else:
        meta.save(input_dir / meta_path)


def merge(input_files: list[Path], names: list[str] | None = None, output_file: Path = Path("combined.json")) -> None:
    """Merge one or more dataset files.

    Typically there will be dataset file per data type (frames, depth, etc) but these can be combined if they
    are compatible (same number of frames, same camera, etc) to yield Nerfstudio-compatible "transform.json"
    files that might have "depth_file_path" or "mask_path" in addition to a "file_path". This does not touch the
    underlying data, only modifies the transforms files.

    This can be used to rename a data type for a single file, merge multiple metadata files that already have distinct
    data type names, or merge and rename many metadata files altogether.

    Args:
        input_files (list[Path]): List of datasets to merge, can either be the path of a metadata file or it's directory.
        names (list[str] | None, optional): What to rename each "path" argument to. Defaults to None.
        output_file (Path, optional): Where to save metadata file, should be a ``.json`` file. Defaults to "combined.json".
    """
    import more_itertools as mitertools

    from visionsim.dataset import Metadata

    metas = [Metadata.from_path(p) for p in input_files]
    data_types = [dt for m in metas for dt in m.data_types]

    if set(len(m.cameras or []) for m in metas) != {1}:
        raise ValueError("Cannot merge datasets that have multiple cameras.")

    if len(set(cam.model_copy(update=dict(c=None)) for m in metas for cam in (m.cameras or []))) != 1:
        # Allow merge if data types have different number of channels
        raise ValueError("Cannot merge datasets that have different cameras.")

    if len(set(len(m) for m in metas)) != 1:
        raise ValueError("Datasets cannot be merged as they are not the same size.")

    if names is None:
        if len(set(data_types)) != len(data_types):
            raise ValueError(f"Data types must be unique, got {data_types}.")

        # Disable renaming of data types since they are unique and there might be many per metadata file
        data_types = [None] * len(input_files)  # type: ignore
        names = [None] * len(input_files)  # type: ignore
    else:
        if len(names) != len(input_files):
            raise ValueError(
                f"Expected as many names as input files, got {len(names)} and {len(input_files)} respectively."
            )
        if len(input_files) != len(data_types):
            raise ValueError("Cannot rename data types when multiple types exist in a file.")

    def merge_transform(transforms):
        # We already checked that cameras are the same, datasets are the same length, and (renamed) data
        # types are unique, so we check for poses and per-frame args then just merge dicts
        if any(
            not np.allclose(a["transform_matrix"], b["transform_matrix"]) for a, b in mitertools.pairwise(transforms)
        ):
            raise ValueError("Frames in datasets do not share a common pose.")
        if any(t.get("offset") or t.get("bitpack_dim") for t in transforms):
            raise ValueError("Cannot merge datasets that use additional per-frame attributes.")
        return functools.reduce(lambda a, b: a | b, transforms)

    merged_transforms = [
        merge_transform(transforms)
        for transforms in zip(
            *[
                m.iter_dense_transforms(data_type=dt, rename_to=name, relative_to=output_file.resolve().parent)
                for m, dt, name in zip(metas, data_types, names)
            ]
        )
    ]

    Metadata.from_dense_transforms(merged_transforms).save(output_file)
