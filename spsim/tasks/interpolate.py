from pathlib import Path

import numpy as np
from invoke import task

from spsim.interpolate import interpolate_poses, interpolate_frames, poses_and_frames_to_json
from spsim.schema import _read_and_validate, IMG_SCHEMA
from spsim.tasks.common import _validate_directories


@task(
    help={
        "input_file": "path to video file from which to extract frames",
        "output_file": "path in which to save interpolated video",
        "method": "interpolation method to use, only RIFE (ECCV22) is supported for now, default: 'rife'",
        "n": "interpolation factor, must be a multiple of 2, default: 2",
    }
)
def video(c, input_file, output_file, method="rife", n=2):
    """Interpolate video by extracting all frames, performing frame-wise interpolation and re-assembling video"""
    import tempfile

    from natsort import natsorted

    from spsim.interpolate import rife

    from .ffmpeg import animate, count_frames, duration, extract

    if method.lower() not in ("rife",):
        raise NotImplementedError("Only rife is currently supported as an interpolation method.")
    if n < 2 or not n & (n - 1) == 0:
        raise ValueError(f"Can only interpolate by a power of 2, greater or equal to 2, not {n}.")

    avg_fps = count_frames(c, input_file) / duration(c, input_file)
    print(f"Video has average frame rate of {avg_fps}")

    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as dst_dir:
        # Extract all frames
        extract(c, input_file, src_dir, pattern="frames_%06d.png")

        # Interpolate them
        img_paths = [str(p) for p in natsorted(Path(src_dir).glob("frames_*.png"))]
        rife(img_paths, dst_dir, exp=np.log2(n).astype(int))

        # Assemble final video at correct frame-rate
        animate(c, dst_dir, pattern="frames_*.png", outfile=output_file, fps=avg_fps)

# TODO: reaname to interpolate_cli or better name
@task(
    help={
        "input_dir": "directory in which to look for frames",
        "output_dir": "directory in which to save interpolated frames",
        # added file type for frames normals depths
        "method": "interpolation method to use, only RIFE (ECCV22) is supported for now, default: 'rife'",
        "file_name": "name of file containing transforms, default: 'transforms.json'",
        "n": "interpolation factor, must be a multiple of 2, default: 2",
    }
)
def frames(_, input_dir, output_dir, method="rife", file_name="transforms.json", n=2):
    """Interpolate poses and frames seperately, then combine into transforms.json file
    """

    # Extract transforms from transforms.json file
    input_dir, output_dir = _validate_directories(input_dir, output_dir)
    transforms = _read_and_validate(path=input_dir / file_name, schema=IMG_SCHEMA)


    print("Interpolating poses")
    interpolated_poses = interpolate_poses(transforms, n)

    print("Interpolating frames")
    interpolate_frames(input_dir, output_dir, method, n)

    print(f"Generating {file_name}")
    poses_and_frames_to_json(transforms, interpolated_poses, output_dir, file_name="transforms.json")