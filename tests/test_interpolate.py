import os

import numpy as np
from numpy import float64
from PIL import Image
from scipy.spatial.transform import Rotation as R

from visionsim.interpolate import interpolate_frames, interpolate_poses


def test_interpolate_rotation_matrix():
    # Pose at origin with no rotation
    start_pose = np.eye(4, dtype=float64)
    # Pose at (1,1,1) with 45 degree rotation around x axis
    end_pose = np.eye(4, dtype=float64)
    end_pose[:3, -1] = [1, 1, 1]
    # Add rotation matrix
    end_pose[:3, :3] = R.from_euler("x", 45, degrees=True).as_matrix()
    # Put test matrices in transforms format
    transforms = {
        "frames": [{"file_path": "0", "transform_matrix": start_pose}, {"file_path": "1", "transform_matrix": end_pose}]
    }
    # Interpolate the poses
    new_poses = interpolate_poses(transforms, normalize=True, k=1)
    interpolated_start, interpolated_pose, interpolated_end = new_poses

    # Make sure first and last pose haven't changed
    assert np.allclose(start_pose, interpolated_start)
    assert np.allclose(end_pose, interpolated_end)

    # The interpolated position should be at (.5,.5,.5)
    test_position = np.array([0.5, 0.5, 0.5])
    assert np.allclose(interpolated_pose[:3, 3], test_position)

    # The interpolated rotation should be 22.5 degrees
    # Extract interpolated rotation matrix from interpolated poses
    interpolated_rotation = R.from_matrix(interpolated_pose[:3, :3])
    test_rotation = R.from_euler("x", 22.5, degrees=True)
    assert interpolated_rotation.approx_equal(test_rotation)


def test_interpolate_frames(tmp_path):
    # Generate white and black frames
    white_square = Image.new("RGB", (100, 100), "white")
    black_square = Image.new("RGB", (100, 100), "black")

    # Save the image as a PNG file
    os.mkdir(tmp_path / "frames")
    white_square.save(tmp_path / "frames/0.png")
    black_square.save(tmp_path / "frames/1.png")

    # Interpolate the test frames
    interpolate_frames(tmp_path, tmp_path, n=8)

    # Check to make sure each image is getting darker
    # In grayscale white=255, black=0
    prev_mean_color = 255  # Start at white
    img_paths = sorted(tmp_path.glob("frames/*.png"))
    for img_path in img_paths:
        mean_color = calculate_mean_color(img_path)

        # Make sure this image is darker (mean color moving towards 0 for black)
        assert mean_color <= prev_mean_color
        prev_mean_color = mean_color


def calculate_mean_color(img_path):
    # Get Grayscale of image
    img = Image.open(img_path).convert("L")
    # Convert to numpy array for easier processing
    img_array = np.array(img)
    # Calculate mean pixel color
    mean_color = img_array.mean()

    return mean_color
