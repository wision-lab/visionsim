import json
import numpy as np

from natsort import natsorted
from pathlib import Path


from .schema import _read_and_validate, IMG_SCHEMA
from .interpolate import pose_interp, rife



def interpolate_poses(input_dir, file_type="frames", file_name="transforms.json", n = 2):
    """
    Interpolate between poses. 
    Returns the interpolated poses in matrices.
    """
    if n < 2 or not n & (n - 1) == 0:
        raise ValueError(f"Can only interpolate by a power of 2, greater or equal to 2, not {n}.")
    
    # dict converts file_type input key to transforms key
    types = {"frames": "file_path", "normals": "normals_file_path", "depths": "depth_file_path"}

    if file_type not in types:
        raise ValueError("Interpolation must be on rgb, normal, or depth files")
    # file_path holds transforms key
    file_path = types[file_type]

    input_dir = Path(input_dir).resolve()

    transforms = _read_and_validate(path=input_dir / file_name, schema=IMG_SCHEMA)
    frames = natsorted(transforms["frames"], key=lambda f: f[file_path])

    # Run the pose interpolation
    num_frames = len(transforms["frames"])
    indices = np.linspace(0, num_frames - 1, num_frames)
    interp_indices = np.linspace(0, num_frames - 1, n * num_frames - (n - 1))
    pose_spline = pose_interp([f["transform_matrix"] for f in frames], ts=indices)
    new_poses = pose_spline(interp_indices)

    return new_poses


def interpolate_frames(input_dir, output_dir, file_type="frames", file_name="transforms.json", interpolation_method = "rife", n = 2):
    """
    Interpolate between image frames.
    Returns the file extension of the images

    Currently only RIFE is supported for frame interpolation but we intend
    to add more. This method is future proofing the addition of more image interpolation
    methods
    """
    if n < 2 or not n & (n - 1) == 0:
        raise ValueError(f"Can only interpolate by a power of 2, greater or equal to 2, not {n}.")
    
    # dict converts file_type input key to transforms key
    types = {"frames": "file_path", "normals": "normals_file_path", "depths": "depth_file_path"}

    if file_type not in types:
        raise ValueError("Interpolation must be on rgb, normal, or depth files")
    # file_path holds transforms key
    file_path = types[file_type]

    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    transforms = _read_and_validate(path=input_dir / file_name, schema=IMG_SCHEMA)

    frames = natsorted(transforms["frames"], key=lambda f: f[file_path])
    img_paths = [str(input_dir / f[file_path]) for f in frames]
    exts = set(Path(p).suffix for p in img_paths)

    if len(exts) != 1:
        raise RuntimeError(f"All images must have same extension but found {exts}.")

    # Route request to proper image interpolation method
    if(interpolation_method.lower() == "rife"):
        rife(img_paths, output_dir / file_type, exp=np.log2(n).astype(int))
    else:
        raise NotImplementedError("Requested interpolation method is not supported at this time.")
    
    return exts



def poses_and_frames_to_json(input_dir, output_dir, new_poses, exts, file_type="frames", file_name="transforms.json"):
    """
    Combine interpolated poses (matrices) and frames into a
    new transforms.json file

    Interpolated IMG frames are found in the output_dir and the interpolated poses
    are passed to this method
    """
    # TODO: maybe don't make this reliant on input_dir. Why do we use transforms from input_dir?

    # dict converts file_type input key to transforms key
    types = {"frames": "file_path", "normals": "normals_file_path", "depths": "depth_file_path"}

    if file_type not in types:
        raise ValueError("Interpolation must be on rgb, normal, or depth files")
    # file_path holds transforms key
    file_path = types[file_type]

    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    transforms = _read_and_validate(path=input_dir / file_name, schema=IMG_SCHEMA)

    new_paths = natsorted(output_dir.glob(f"{file_type}/*{exts.pop()}"))

    if len(new_paths) != len(new_poses):
        raise RuntimeError(
            f"Image and pose mismatch! Found {len(new_poses)} new poses " f"and {len(new_paths)} new images."
        )

    new_frames = [
        # {"file_path": str(path.relative_to(output_dir)), "transform_matrix": pose.tolist()}
        {file_path: str(path.relative_to(output_dir)), "transform_matrix": pose.tolist()}
        # {"depth_file_path": str(path.relative_to(output_dir)), "transform_matrix": pose.tolist()}
        for path, pose in zip(new_paths, new_poses)
    ]
    transforms["frames"] = new_frames

    with (output_dir / file_name).open("w") as f:
        json.dump(transforms, f, indent=2)