from spsim.interpolate_wrapper import *


def test_interpolate_poses_4():
    """
    Test to make sure interpolate_poses produces the correct transforms.json

    Interpolate 4 times
    """

    # Initialize test file paths
    input_dir = "test_files/lego100/"
    output_dir = "tmp/lego400/"

    print("Interpolating poses")
    interpolated_poses = interpolate_poses(input_dir, n=4)
    print("Interpolating frames")
    exts = interpolate_frames(input_dir, output_dir, n=4)

    poses_and_frames_to_json(input_dir, output_dir, interpolated_poses, exts)


test_interpolate_poses_4()