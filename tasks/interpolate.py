import json
from pathlib import Path

import numpy as np
from invoke import task

from tasks.common import _validate_directories


@task(
    help={
        "input_dir": "directory in which to look for frames",
        "output_dir": "directory in which to save interpolated frames",
        "method": "interpolation method to use, only RIFE (ECCV22) is supported for now",
        "file_name": "name of file containing transforms, default: 'transforms.json'",
        "n": "interpolation factor, must be a multiple of 2, default: 2",
    }
)
def frames(_, input_dir, output_dir, method="rife", file_name="transforms.json", n=2):
    """Interpolate between frames and poses (up to 16x) using RIFE (ECCV22)"""
    # TODO: Enable interpolation of only transforms or only frames
    from natsort import natsorted

    from spsim.interpolate import pose_interp, rife

    if method.lower() not in ("rife",):
        raise NotImplementedError(f"Only rife is currently supported as an interpolation method.")
    if n < 2 or not n & (n - 1) == 0:
        raise ValueError(f"Can only interpolate by a power of 2, greater or equal to 2, not {n}.")
    input_dir, output_dir = _validate_directories(input_dir, output_dir)

    with (input_dir / file_name).open("r") as f:
        transforms = json.load(f)

    # Expect either blender or nerf style transforms, giving priority to blender-style.
    img_paths = [f["file_paths"][0] if "file_paths" in f else f["file_path"] for f in transforms["frames"]]
    frames = natsorted(transforms["frames"], key=lambda f: f["file_paths"][0] if "file_paths" in f else f["file_path"])
    img_paths = natsorted(str(input_dir / p) for p in img_paths)
    is_blender = all("file_paths" in f for f in frames)
    is_nerf = all("file_path" in f for f in frames)
    exts = set(Path(p).suffix for p in img_paths)

    if len(exts) != 1:
        raise RuntimeError(f"All images must have same extension but found {exts}.")

    if not is_blender and not is_nerf:
        raise ValueError(f"Format not understood.")

    # Perform pose interpolation
    #   Ex for 4 frames, and n=4:
    #       [0.0, 1.0, 2.0, 3.0, 4.0]
    #       [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]
    num_frames = len(transforms["frames"])
    indices = np.linspace(0, num_frames - 1, num_frames)
    interp_indices = np.linspace(0, num_frames - 1, n * num_frames - (n - 1))
    pose_spline = pose_interp([f["transform_matrix"] for f in frames], ts=indices)
    new_poses = pose_spline(interp_indices)
    keys = set().union(*[f.keys() for f in frames])

    if len(keys) != 2:
        raise RuntimeError(
            f"Expected only two keys per frame ('transform_matrix' and 'file_paths'/'file_path')" f"but got {keys}."
        )

    # Perform image interpolation
    rife(img_paths, output_dir / "frames", exp=np.log2(n).astype(int))

    # Assemble new transforms.json
    new_paths = natsorted(output_dir.glob(f"frames/*{exts.pop()}"))

    if len(new_paths) != len(new_poses):
        raise RuntimeError(
            f"Image and pose mismatch! Found {len(new_poses)} new poses " f"and {len(new_paths)} new images."
        )

    new_frames = [
        {"file_path": str(path), "transform_matrix": pose.tolist()}
        if is_nerf
        else {"file_paths": [str(path)], "transform_matrix": pose.tolist()}
        for path, pose in zip(new_paths, new_poses)
    ]
    transforms["frames"] = new_frames

    with (output_dir / file_name).open("w") as f:
        json.dump(transforms, f, indent=2)
