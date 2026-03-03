from __future__ import annotations

import contextlib
import functools
import shutil
from pathlib import Path
from typing import cast, Any

import numpy as np


@contextlib.contextmanager
def _ply_stream(path: Path, binary: bool = False):
    """Context manager for streaming PLY data to a file."""
    total_points = 0
    mode = "wb" if binary else "w"
    with open(path, mode) as f:
        if binary:
            f.write(b"ply\nformat binary_little_endian 1.0\n")
            count_pos = f.tell()
            f.write(f"element vertex {' ' * 15}\n".encode())
            f.write(b"property float x\nproperty float y\nproperty float z\n")
            f.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write(b"end_header\n")
        else:
            f.write("ply\nformat ascii 1.0\n")
            count_pos = f.tell()
            # element vertex {placeholder} - 15 spaces allows for up to 999 trillion points
            f.write(f"element vertex {' ' * 15}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")

        def write_points(points: np.ndarray, colors: np.ndarray):
            nonlocal total_points
            if binary:
                num = len(points)
                dtype = np.dtype([("pos", "f4", (3,)), ("color", "u1", (3,))])
                data: np.ndarray = np.empty(num, dtype=dtype)
                data["pos"] = points.astype(np.float32)
                data["color"] = colors[..., :3].astype(np.uint8)
                f.write(data.tobytes())
            else:
                for (px, py, pz), (r, g, b, *_) in zip(points, colors):
                    f.write(f"{px} {py} {pz} {int(r)} {int(g)} {int(b)}\n")
            total_points += len(points)

        yield write_points

        if total_points > 0:
            f.seek(count_pos)
            msg = f"element vertex {total_points:<15d}"
            if binary:
                f.write(msg.encode())
            else:
                f.write(msg)
        else:
            path.unlink(missing_ok=True)


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


def to_pointcloud(
    colors: Path,
    depths: Path | None = None,
    points: Path | None = None,
    output: Path = Path("pointcloud.ply"),
    p: float = 0.15,
    binary: bool = True,
    force: bool = False,
) -> None:
    """Generate a ``.ply`` point cloud from datasets.

    Args:
        colors: path to dataset to use for point colors, must contain RGB data that is assumed to be in uint8.
        depths: path to dataset to use for depth-based 3D points. A pinhole camera model is used to project
            depth values to 3D points.
        points: path to dataset to use for world-space 3D points. If set, this will be used
            instead of depth-based points.
        output: path to save PLY file to.
        p: probability of sampling a pixel.
        binary: If true, save as a binary PLY file (smaller and faster).
        force: If true, overwrite output file if present.
    """
    from rich.progress import track

    from visionsim.cli import _log
    from visionsim.dataset import Dataset

    if output.exists() and not force:
        raise FileExistsError("Output file already exists.")
    else:
        output.unlink(missing_ok=True)

    # Filter out large depths, this is a render bug in CYCLES
    # See: https://blender.stackexchange.com/questions/325007
    DEPTH_CUTOFF = 10000000000
    np.random.seed(76986489)

    ds_colors = Dataset.from_path(colors)
    num_frames = len(ds_colors)

    ds_points = Dataset.from_path(points) if points else None
    ds_depths = Dataset.from_path(depths) if depths else None

    if (ds_points and len(ds_points) != num_frames) or (ds_depths and len(ds_depths) != num_frames):
        raise ValueError("Colors and depths/points datasets must have the same number of frames.")

    canonical_cameras = [
        [cam.model_copy(update=dict(c=None)) for cam in (m.cameras or [])]
        for m in [ds_colors, ds_points, ds_depths]
        if m is not None
    ]
    if any(c1 != c2 for c1, c2 in zip(*canonical_cameras)):
        raise ValueError("Cannot generate point cloud from datasets with different cameras.")

    if output.suffix != ".ply":
        raise ValueError(f"Output file must have a .ply extension, instead got {output.suffix}.")

    _log.info(f"Generating point cloud from {num_frames} frames (p={p})...")

    with _ply_stream(output, binary=binary) as write_points:
        for i in track(range(num_frames), description="Processing frames..."):
            colors_data, _ = cast(tuple[np.ndarray, dict[str, Any]], ds_colors[i])

            if ds_points is not None:
                points_data, points_meta = cast(tuple[np.ndarray, dict[str, Any]], ds_points[i])
            elif ds_depths is not None:
                points_data, points_meta = cast(tuple[np.ndarray, dict[str, Any]], ds_depths[i])
            else:
                raise ValueError("Either `points` or `depths` must be provided.")

            if points_data.shape[:2] != colors_data.shape[:2]:
                _log.warning(
                    f"Points and colors must have same resolution for frame {i}, "
                    f"got {points_data.shape[:2]} and {colors_data.shape[:2]}."
                )
                continue

            h, w = points_data.shape[:2]
            mask = np.random.rand(h, w) < p
            points_sampled = points_data[mask]
            colors_sampled = colors_data[mask]

            if ds_depths and ds_points is None:
                # Project depth to world-space
                depth = points_sampled[..., 0]
                # Filter out zero depth
                valid_mask = (depth > 0) & (depth < DEPTH_CUTOFF)
                depth = depth[valid_mask]
                colors_sampled = colors_sampled[valid_mask]

                # Get coordinates for valid mask
                yy, xx = np.where(mask)
                yy = yy[valid_mask]
                xx = xx[valid_mask]

                # Intrinsics
                fl_x = points_meta["fl_x"]
                fl_y = points_meta["fl_y"]
                cx = points_meta["cx"]
                cy = points_meta["cy"]

                # Camera space points (Blender/OpenGL: +X right, +Y up, -Z forward)
                # So depth is -Z. Y is inverted between image and camera space (up is positive)
                z = -depth
                x = -(xx - cx) * z / fl_x
                y = (yy - cy) * z / fl_y

                # World space
                c2w = np.array(points_meta["transform_matrix"])
                points_cam = np.stack([x, y, z, np.ones_like(x)], axis=-1)
                points_world = (c2w @ points_cam.T).T
                points_world = points_world[:, :3] / points_world[:, -1:]
            else:
                # Assume points are already in world-space
                points_world = points_sampled

                # Filter out zero points (often background in position pass if not careful)
                points_x, points_y, points_z = points_world.T
                valid_mask = (points_x != 0) | (points_y != 0) | (points_z != 0)
                points_world = points_world[valid_mask]
                colors_sampled = colors_sampled[valid_mask]

            # Write points for this frame to file
            write_points(points_world, colors_sampled)
