from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np


def video(input_file: Path, output_file: Path, method: str = "rife", n: int = 2):
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
    from visionsim.utils.progress import ElapsedProgress

    from .ffmpeg import animate, count_frames, duration, extract

    if method.lower() not in ("rife",):
        raise NotImplementedError("Only rife is currently supported as an interpolation method.")
    if n < 2 or not n & (n - 1) == 0:
        raise ValueError(f"Can only interpolate by a power of 2, greater or equal to 2, not {n}.")

    avg_fps = count_frames(input_file) / duration(input_file)
    _log.info(f"Video has average frame rate of {avg_fps}")

    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as dst_dir:
        # Extract all frames
        extract(Path(input_file), Path(src_dir), pattern="frames_%06d.png")

        # Interpolate them
        with ElapsedProgress() as progress:
            task = progress.add_task("Interpolating with rife...")
            img_paths = [str(p) for p in natsorted(Path(src_dir).glob("frames_*.png"))]
            rife(img_paths, dst_dir, exp=np.log2(n).astype(int), update_fn=partial(progress.update, task))

        # Assemble final video at correct frame-rate
        animate(Path(dst_dir), pattern="frames_*.png", outfile=output_file, fps=int(avg_fps))


def dataset(
    input_dir: Path,
    output_dir: Path,
    pattern: str | None = None,
    method: Literal["rife"] = "rife",
    n: int = 2,
):
    """Interpolate between a series of frames or a dataset (both it's images and poses)

    Note:
        This only works if the dataset has a single camera, as interpolating camera settings
        or types is not possible. Further, the data needs to be saved as images.

    Args:
        input_dir: directory in which to look for frames
        output_dir: directory in which to save interpolated frames
        pattern: used to find source image files to interpolate from,
            not needed when `input_dir` points to a valid dataset.
        method: interpolation method to use, only RIFE (ECCV22) is supported for now, default: 'rife'
        n: interpolation factor, must be a multiple of 2, default: 2
    """
    from natsort import natsorted

    from visionsim.cli import _log
    from visionsim.dataset import Dataset, Metadata
    from visionsim.interpolate import rife
    from visionsim.interpolate.pose import interpolate_poses
    from visionsim.utils.progress import ElapsedProgress

    if pattern:
        dataset = Dataset.from_pattern(input_dir, pattern)
    else:
        dataset = Dataset.from_path(input_dir)

    with ElapsedProgress() as progress:
        task = progress.add_task(f"Interpolating with {method}...")

        if method.lower() == "rife":
            rife(
                input_dir,
                output_dir,
                input_files=dataset.paths,
                exp=np.log2(n).astype(int),
                update_fn=partial(progress.update, task),
            )
        else:
            raise NotImplementedError("Requested interpolation method is not supported at this time.")

    if dataset.cameras is None or len(dataset.cameras) != 1:
        _log.warning("Cannot emulate an RGB camera from multiple cameras, not saving transforms.")
    elif dataset.poses is not None:
        _log.info("Interpolating poses...")
        interp_poses = interpolate_poses(dataset.poses, n=n)
        interp_paths = natsorted(output_dir.glob("**/*.png"))
        Metadata.from_frames(
            frames=[
                dict(file_path=p.relative_to(output_dir), transform_matrix=m) for p, m in zip(interp_paths, interp_poses)
            ],
            camera=next(iter(dataset.cameras)),
        ).save(output_dir / "transforms.json")
