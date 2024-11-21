import json
import numpy as np
from spsim.interpolate import interpolate_frames, interpolate_poses
from spsim.schema import IMG_SCHEMA, _read_and_validate
from pathlib import Path
from PIL import Image
import shutil


def test_interpolate_poses(test_transforms_path, tmp_path="tmp/"):
    # Load in initial transforms file
    transforms = _read_and_validate(path=test_transforms_path+"transforms.json", schema=IMG_SCHEMA)
    # Interpolate the poses
    new_poses = interpolate_poses(transforms)

    # Extract matrices from expected transform file
    with open(test_transforms_path + "expected_transforms.json", "r") as f:
        transforms = json.load(f)

    expected_poses = list()
    for frame in transforms["frames"]:
        expected_poses.append(frame["transform_matrix"])
    
    # Compare them
    for expected_pose, interpolated_pose in zip(expected_poses, new_poses):
        # Cast as numpy array for comparison
        expected_pose = np.array(expected_pose)
        interpolated_pose = np.array(interpolated_pose)

        assert np.array_equal(expected_pose, interpolated_pose)


def test_interpolate_frames(test_frames_path, tmp_path="tmp/"):
    # Interpolate the test frames
    interpolate_frames(test_frames_path, tmp_path, n=8)

    # Check to make sure each image is getting darker
    prev_dark_percentage = -1 # Make sure that first frame is "darker"
    img_paths = sorted(Path(tmp_path).glob("frames/*.png"))
    for img_path in img_paths:
        dark_percentage, _ = calculate_dark_and_light(img_path)

        # Make sure this image is darker
        assert dark_percentage >= prev_dark_percentage
        prev_dark_percentage = dark_percentage


    # Get rid of interpolated frames
    shutil.rmtree(tmp_path)


def calculate_dark_and_light(img_path, darkness_threshold=128):
    # Get Greyscale of image
    img = Image.open(img_path).convert("L")

    # Convert to numpy array for easier processing
    img_array = np.array(img)
    
    # Calculate percentages
    total_pixels = img_array.size
    dark_pixels = np.sum(img_array < darkness_threshold)
    light_pixels = np.sum(img_array >= darkness_threshold)
    
    dark_percentage = (dark_pixels / total_pixels) * 100
    light_percentage = (light_pixels / total_pixels) * 100

    return dark_percentage, light_percentage