import os
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from spsim.interpolate import interpolate_frames, interpolate_poses
from spsim.schema import IMG_SCHEMA, _read_and_validate
from PIL import Image


def test_interpolate_poses():
    test_transforms_path = Path("test_files/interpolate_transforms/")

    # Load in initial transforms file
    transforms = _read_and_validate(path=test_transforms_path / "transforms.json", schema=IMG_SCHEMA)
    # Interpolate the poses
    new_poses = interpolate_poses(transforms)

    # Extract matrices from expected transform file
    with open(test_transforms_path / "expected_transforms.json", "r") as f:
        transforms = json.load(f)

    expected_poses = list()
    for frame in transforms["frames"]:
        expected_poses.append(frame["transform_matrix"])
    
    # Compare them
    for expected_pose, interpolated_pose in zip(expected_poses, new_poses):
        # Cast as numpy array for comparison
        expected_pose = np.array(expected_pose)
        interpolated_pose = np.array(interpolated_pose)

        assert np.allclose(expected_pose, interpolated_pose)


def test_interplate_rotation_matrix():
    # Pose at origin with no rotation
    start_pose = np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ])
    # Pose at (1,1,1) with 45 degree rotation
    end_pose = np.array([
        [45,0,0,1],
        [0,45,0,1],
        [0,0,45,1],
        [0,0,0,1]
    ])
    end_pose = end_pose/np.linalg.det(end_pose)
    # Put test matrices in transforms format
    transforms = {
        "frames": [
            {
                "file_path": "0",
                "transform_matrix": start_pose
            },
            {
                "file_path": "1",
                "transform_matrix": end_pose
            }
        ]
    }

    # Interpolate the poses
    new_poses = interpolate_poses(transforms)
    print(new_poses)



def test_interpolate_frames(tmp_path):
    # Generate white and black frames
    white_square = Image.new('RGB', (100, 100), 'white')
    black_square = Image.new('RGB', (100, 100), 'black')

    # Save the image as a PNG file
    os.mkdir(tmp_path / "frames")
    white_square.save(tmp_path / "frames/0.png")
    black_square.save(tmp_path / "frames/1.png")

    # Interpolate the test frames
    interpolate_frames(tmp_path, tmp_path, n=8)

    # Check to make sure each image is getting darker
    # In greyscale white=255, black=0
    prev_mean_color = 255 # Start at white
    img_paths = sorted(tmp_path.glob("frames/*.png"))
    for img_path in img_paths:
        mean_color = calculate_mean_color(img_path)

        # Make sure this image is darker (mean color moving towards 0 for black)
        assert mean_color <= prev_mean_color
        prev_mean_color = mean_color


def calculate_mean_color(img_path):
    # TODO: Maybe get mean pixel color

    # Get Greyscale of image
    img = Image.open(img_path).convert("L")

    # Convert to numpy array for easier processing
    img_array = np.array(img)
    
    # Calculate mean pixel color
    mean_color = img_array.mean()
    print(mean_color)

    return mean_color