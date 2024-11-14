import numpy as np
from spsim.interpolate import interpolate_frames, interpolate_poses
from pathlib import Path
from PIL import Image
import shutil


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