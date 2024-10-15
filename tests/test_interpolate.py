from spsim.interpolate_wrapper import interpolate_wrapper


def test_interpolate_poses_4():
    """
    Test to make sure interpolate_poses produces the correct transforms.json

    Interpolate 4 times
    """

    # TODO: create tmp folder for output files

    # Initialize test file paths
    baseline_transforms_json_dir = "test_files/lego100/"
    new_transforms_json_dir = "tmp/lego400/"


    interpolate_wrapper("poses", baseline_transforms_json_dir, new_transforms_json_dir, file_type="frames", method="rife", file_name="transforms.json", n=4)


    # TODO: Add check between output and input transforms.json

    # TODO: get rid of tmp file after test is run


test_interpolate_poses_4()