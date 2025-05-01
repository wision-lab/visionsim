from visionsim.dataset import IMG_SCHEMA, read_and_validate
from visionsim.interpolate import interpolate_frames, interpolate_poses, poses_and_frames_to_json

if __name__ == "__main__":
    # Path to directory that has transforms.py file and img frames/ directory
    input_dir = "path/to/input_dir"
    # Path to the directory we want to output the interpolated transforms and image frames
    output_dir = "path/to/output_dir"
    # Name of transforms file
    transforms_file_name = "transforms.json"

    # Validate that the transforms.json file complies with our IMG_SCHEMA schema
    # ALso read transforms into dictionary
    transforms = read_and_validate(path=input_dir / transforms_file_name, schema=IMG_SCHEMA)

    # Interpolate poses using spsim API
    print("Interpolating poses")
    interpolated_poses = interpolate_poses(transforms, n=2)

    # Interpolate frames using spsim API (only supports RIFE interpolation at the moment)
    print("Interpolating frames")
    interpolate_frames(input_dir, output_dir, method="rife", n=2)

    # Optionally save the interpolated frames and transforms to a new directory
    print(f"Generating {transforms_file_name}")
    poses_and_frames_to_json(transforms, interpolated_poses, output_dir, file_name="transforms.json")