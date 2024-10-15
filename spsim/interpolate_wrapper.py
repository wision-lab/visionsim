import json
import numpy as np

from natsort import natsorted
from pathlib import Path


from schema import _read_and_validate, IMG_SCHEMA
from interpolate import pose_interp, rife


def interpolate_poses(input_dir, output_dir, file_path, file_type, frames, img_paths, file_name = "transforms.json", n=2):
    """
    Interpolate between poses
    """
    if n < 2 or not n & (n - 1) == 0:
        raise ValueError(f"Can only interpolate by a power of 2, greater or equal to 2, not {n}.")
    
    transforms = _read_and_validate(path=input_dir / file_name, schema=IMG_SCHEMA)

    num_frames = len(transforms["frames"])
    indices = np.linspace(0, num_frames - 1, num_frames)
    interp_indices = np.linspace(0, num_frames - 1, n * num_frames - (n - 1))
    pose_spline = pose_interp([f["transform_matrix"] for f in frames], ts=indices)
    new_poses = pose_spline(interp_indices)

    exts = set(Path(p).suffix for p in img_paths)

    if len(exts) != 1:
        raise RuntimeError(f"All images must have same extension but found {exts}.")

    # Assemble new transforms.json
    new_paths = natsorted(output_dir.glob(f"{file_type}/*{exts.pop()}"))

    new_frames = [
        # {"file_path": str(path.relative_to(output_dir)), "transform_matrix": pose.tolist()}
        {file_path: str(path.relative_to(output_dir)), "transform_matrix": pose.tolist()}
        # {"depth_file_path": str(path.relative_to(output_dir)), "transform_matrix": pose.tolist()}
        for path, pose in zip(new_paths, new_poses)
    ]
    transforms["frames"] = new_frames

    with (output_dir / file_name).open("w") as f:
        json.dump(transforms, f, indent=2)

    

def interpolate_frames(output_dir, file_type, img_paths, n=2):
    """
    Interpolate between frames using RIFE
    """
    
    # Perform image interpolation
    rife(img_paths, output_dir / file_type, exp=np.log2(n).astype(int))

    


def interpolate_wrapper(interpolation_type, input_dir, output_dir, file_type, method, file_name, n):
    """
    Interpolate wrapper for both frames and poses
    
    interpolation_type specifies whether poses should be interpolated, frames should be interpolated, or both
    """
    # Authenticate interpolation method and number of frames to interpolate by
    if method.lower() not in ("rife",):
        raise NotImplementedError("Only rife is currently supported as an interpolation method.")
    if n < 2 or not n & (n - 1) == 0:
        raise ValueError(f"Can only interpolate by a power of 2, greater or equal to 2, not {n}.")

    # dict converts file_type input key to transforms key
    types = {"frames": "file_path", "normals": "normals_file_path", "depths": "depth_file_path"}

    if file_type not in types:
        raise ValueError("Interpolation must be on rgb, normal, or depth files")
    # file_path holds transforms key
    file_path = types[file_type]

    print(file_type)
    print(file_path)

    # input_dir, output_dir = _validate_directories(input_dir, output_dir)
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    transforms = _read_and_validate(path=input_dir / file_name, schema=IMG_SCHEMA)

    frames = natsorted(transforms["frames"], key=lambda f: f[file_path])
    img_paths = [str(input_dir / f[file_path]) for f in frames]

    # Validate user input interpolation type
    interpolation_types = ["frames", "poses", "both"]

    if interpolation_type.lower() not in interpolation_types:
        raise ValueError(f"Can only interpolate poses, frames, or both")
    
    # Route request to the correct interpolation type
    interpolation_type = interpolation_type.lower()
    if(interpolation_type == "poses"):
        # Just poses
        interpolate_poses(input_dir, output_dir, file_path, file_type, frames, img_paths, file_name, n)
    elif(interpolation_type == "frames"):
        # Just frames
        interpolate_frames(output_dir, file_type, img_paths, n)
    elif(interpolation_type == "both"):
        # Both poses and frames
        interpolate_poses(input_dir, output_dir, file_path, file_type, frames, img_paths, file_name, n)
        interpolate_frames(output_dir, file_type, img_paths, n)