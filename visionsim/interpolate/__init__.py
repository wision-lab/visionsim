from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
from natsort import natsorted
from rich.progress import Progress

from .pose import pose_interp  # noqa: F401
from .rife.inference_img import interpolate_img as rife  # noqa: F401


def interpolate_poses(transforms, normalize: bool = False, n: int = 2, k: int = 3):
    """Interpolate between pose matrices

    Args:
        transforms: List of pose matrices to interpolate between
        normalize (bool): Whether the interpolation should be normalized or not
        n (int): Number of poses to interpolate between existing poses
        k (int): Order of spline interpolation, see :class:`pose_interp <visionsim.interpolate.pose.pose_interp>`

    :return:
        List of interpolated poses
    """
    # Process from frames folder
    frames = natsorted(transforms["frames"], key=lambda f: f["file_path"])

    # Run the pose interpolation
    num_frames = len(transforms["frames"])
    indices = np.linspace(0, num_frames - 1, num_frames)
    interp_indices = np.linspace(0, num_frames - 1, n * num_frames - (n - 1))
    pose_spline = pose_interp([f["transform_matrix"] for f in frames], ts=indices, normalize=normalize, k=k)
    new_poses = pose_spline(interp_indices)

    return new_poses


def interpolate_frames(
    input_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    interpolation_method: Literal["rife"] = "rife",
    n: int = 2,
):
    """Interpolate between image frames

    Note: Currently only RIFE is supported for frame interpolation but we intend to add more

    Args:
        input_dir (pathlib.Path): Path to directory containing images to interpolate
        output_dir (pathlib.Path): Path to directory to output interpolated frames
        interpolation_method (str): The image interpolation method used
        n (int): Number of frames to interpolate between existing frames
    """

    if n < 2 or not n & (n - 1) == 0:
        raise ValueError(f"Can only interpolate by a power of 2, greater or equal to 2, not {n}.")

    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: We only process pngs right now
    img_paths = [str(p) for p in input_dir.glob("frames/*.png")]

    # Route request to proper image interpolation method
    with Progress() as progress:
        if interpolation_method.lower() == "rife":
            task = progress.add_task("Interpolating with rife...")
            rife(img_paths, output_dir / "frames", exp=np.log2(n).astype(int), update_fn=partial(progress.update, task))
        else:
            raise NotImplementedError("Requested interpolation method is not supported at this time.")


def poses_and_frames_to_json(transforms, new_poses, output_dir: str | os.PathLike, file_name: str = "transforms.json"):
    """Combines interpolated poses (matrices) and frames into a new transforms.json file

    Args:
        transforms: Original transforms JSON file
        new_poses: List of interpolated pose matrices
        output_dir (pathlib.Path): Path to directory containing interpolated frames
        file_name: Name of new transforms JSON file
    """

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: We only output png files in frames dir
    new_paths = natsorted(output_dir.glob("frames/*.png"))

    if len(new_paths) != len(new_poses):
        raise RuntimeError(f"Image and pose mismatch! Found {len(new_poses)} new poses and {len(new_paths)} new images.")

    new_frames = [
        {"file_path": str(path.relative_to(output_dir)), "transform_matrix": pose.tolist()}
        for path, pose in zip(new_paths, new_poses)
    ]
    transforms["frames"] = new_frames

    with (output_dir / file_name).open("w") as f:
        json.dump(transforms, f, indent=2)
