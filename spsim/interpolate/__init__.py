import json
from pathlib import Path

import numpy as np
from natsort import natsorted

from .pose import pose_interp  # noqa: F401
from .rife.inference_img import interpolate_img as rife  # noqa: F401


def interpolate_poses(transforms, n=2):
    """
    Interpolate between poses.
    Returns the interpolated poses in matrices.
    """
    # Process from frames folder
    frames = natsorted(transforms["frames"], key=lambda f: f["file_path"])

    # Run the pose interpolation
    num_frames = len(transforms["frames"])
    indices = np.linspace(0, num_frames - 1, num_frames)
    interp_indices = np.linspace(0, num_frames - 1, n * num_frames - (n - 1))
    pose_spline = pose_interp([f["transform_matrix"] for f in frames], ts=indices)
    new_poses = pose_spline(interp_indices)

    return new_poses


def interpolate_frames(input_dir, output_dir, interpolation_method="rife", n=2):
    """
    Interpolate between image frames.
    Returns the file extension of the images

    Currently only RIFE is supported for frame interpolation but we intend
    to add more. This method is future proofing the addition of more image interpolation
    methods
    """
    if n < 2 or not n & (n - 1) == 0:
        raise ValueError(f"Can only interpolate by a power of 2, greater or equal to 2, not {n}.")

    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: We only process pngs right now
    img_paths = [str(p) for p in input_dir.glob("frames/*.png")]

    # Route request to proper image interpolation method
    if interpolation_method.lower() == "rife":
        rife(img_paths, output_dir / "frames", exp=np.log2(n).astype(int))
    else:
        raise NotImplementedError("Requested interpolation method is not supported at this time.")


def poses_and_frames_to_json(transforms, new_poses, output_dir, file_name="transforms.json"):
    """
    Combine interpolated poses (matrices) and frames into a
    new transforms.json file

    Interpolated IMG frames are found in the output_dir and the interpolated poses
    are passed to this method
    """

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: We only output png files in frames dir
    new_paths = natsorted(output_dir.glob("frames/*.png"))

    if len(new_paths) != len(new_poses):
        raise RuntimeError(
            f"Image and pose mismatch! Found {len(new_poses)} new poses " f"and {len(new_paths)} new images."
        )

    new_frames = [
        {"file_path": str(path.relative_to(output_dir)), "transform_matrix": pose.tolist()}
        for path, pose in zip(new_paths, new_poses)
    ]
    transforms["frames"] = new_frames

    with (output_dir / file_name).open("w") as f:
        json.dump(transforms, f, indent=2)
