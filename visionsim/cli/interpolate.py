from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from typing_extensions import Literal

from visionsim.dataset import IMG_SCHEMA, read_and_validate
from visionsim.interpolate import interpolate_frames, interpolate_poses, poses_and_frames_to_json


def video(input_file: str | os.PathLike, output_file: str | os.PathLike, method: str = "rife", n: int = 2):
    """Interpolate video by extracting all frames, performing frame-wise interpolation and re-assembling video

    Args:
        input_file: path to video file from which to extract frames
        output_file: path in which to save interpolated video
        method: interpolation method to use, only RIFE (ECCV22) is supported for now, default: 'rife'
        n: interpolation factor, must be a multiple of 2, default: 2
    """
    import tempfile

    from natsort import natsorted

    from visionsim.cli import _log
    from visionsim.interpolate import rife

    from .ffmpeg import animate, count_frames, duration, extract

    if method.lower() not in ("rife",):
        raise NotImplementedError("Only rife is currently supported as an interpolation method.")
    if n < 2 or not n & (n - 1) == 0:
        raise ValueError(f"Can only interpolate by a power of 2, greater or equal to 2, not {n}.")

    avg_fps = count_frames(input_file) / duration(input_file)
    _log.info(f"Video has average frame rate of {avg_fps}")

    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as dst_dir:
        # Extract all frames
        extract(input_file, src_dir, pattern="frames_%06d.png")

        # Interpolate them
        img_paths = [str(p) for p in natsorted(Path(src_dir).glob("frames_*.png"))]
        rife(img_paths, dst_dir, exp=np.log2(n).astype(int))

        # Assemble final video at correct frame-rate
        animate(dst_dir, pattern="frames_*.png", outfile=output_file, fps=avg_fps)


def frames(
    input_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    method: Literal["rife"] = "rife",
    file_name: str = "transforms.json",
    n: int = 2,
):
    """Interpolate poses and frames separately, then combine into transforms.json file

    Args:
        input_dir: directory in which to look for frames,
        output_dir: directory in which to save interpolated frames,
        method: interpolation method to use, only RIFE (ECCV22) is supported for now, default: 'rife',
        file_name: name of file containing transforms, default: 'transforms.json',
        n: interpolation factor, must be a multiple of 2, default: 2,
    """
    from visionsim.cli import _log, _validate_directories

    # Extract transforms from transforms.json file
    input_path, output_path, *_ = _validate_directories(input_dir, output_dir)
    transforms = read_and_validate(path=input_path / file_name, schema=IMG_SCHEMA)

    _log.info("Interpolating poses...")
    interpolated_poses = interpolate_poses(transforms, n=n)

    _log.info("Interpolating frames...")
    interpolate_frames(input_path, output_path, method, n)

    _log.info(f"Generating {file_name}")
    poses_and_frames_to_json(transforms, interpolated_poses, output_path, file_name="transforms.json")
