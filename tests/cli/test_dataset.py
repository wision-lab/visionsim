from visionsim.cli import dataset
from visionsim.dataset.models import Metadata


def test_dataset_merge(cube_dataset):
    # Rename single dataset
    dataset.merge(
        input_files=[cube_dataset / "frames"],
        names=["custom_file_path"],
        output_file=(frames_path := cube_dataset / "frames" / "transforms.json"),
    )
    Metadata.load(frames_path)

    # Merge with other dataset that is already renamed
    dataset.merge(
        input_files=[cube_dataset / "depths", frames_path],
        names=None,
        output_file=(ds_path := cube_dataset / "combined.json"),
    )
    Metadata.load(ds_path)
    frames_path.unlink()
    ds_path.unlink()

    # Merge with renames
    dataset.merge(
        input_files=[cube_dataset / "frames", cube_dataset / "depths"],
        names=["file_path", "depth_file_path"],
        output_file=(ds_path := cube_dataset / "combined.json"),
    )
    Metadata.load(ds_path)
    ds_path.unlink()
