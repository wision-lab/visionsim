import numpy as np

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


def _read_ply_points(path):
    """Read the float3 positions out of a ply written by :func:`dataset.to_pointcloud`."""
    header, _, body = path.read_bytes().partition(b"end_header\n")
    count = int(next(ln for ln in header.splitlines() if ln.startswith(b"element vertex")).split()[-1])
    dtype = np.dtype([("pos", "f4", (3,)), ("color", "u1", (3,))])
    return np.frombuffer(body, dtype=dtype, count=count)["pos"]


def test_dataset_up_axis(cube_dataset, tmp_path):
    # Blender renders a Z-up world; nerfstudio-family trainers such as Brush assume Y-up.
    z_to_y = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    original = Metadata.from_path(cube_dataset / "frames")

    for up_axis, expected_up in (
        ("x", (1.0, 0.0, 0.0)),
        ("-x", (-1.0, 0.0, 0.0)),
        ("y", (0.0, 1.0, 0.0)),
        ("-y", (0.0, -1.0, 0.0)),
        ("z", (0.0, 0.0, 1.0)),
        ("-z", (0.0, 0.0, -1.0)),
    ):
        dataset.convert(input_dir=cube_dataset / "frames", output_dir=tmp_path / up_axis, up_axis=up_axis)
        converted = Metadata.from_path(tmp_path / up_axis)

        # Recover what the command did to the renderer's world, then check where it put the up axis.
        rotate = np.array(converted.poses[0]) @ np.linalg.inv(np.array(original.poses[0]))
        assert np.allclose(rotate[:3, :3] @ [0.0, 0.0, 1.0], expected_up)
        assert np.isclose(np.linalg.det(rotate[:3, :3]), 1.0)  # a rotation, not a mirror
        # And that it did the same thing to every pose.
        assert np.allclose(np.array(converted.poses), np.array([rotate @ p for p in original.poses]))

    assert np.allclose(
        np.array(Metadata.from_path(tmp_path / "y").poses), np.array([z_to_y @ p for p in original.poses])
    )

    # With no up axis requested, the poses come through untouched.
    dataset.convert(input_dir=cube_dataset / "frames", output_dir=tmp_path / "as-is")
    assert np.allclose(np.array(Metadata.from_path(tmp_path / "as-is").poses), np.array(original.poses))

    # merge writes the same convention, and to-pointcloud rotates its points to match.
    # (Copy the datasets out of the shared fixture, since merge rewrites paths relative
    # to its output file.)
    for name in ("frames", "depths"):
        dataset.convert(input_dir=cube_dataset / name, output_dir=tmp_path / "ds" / name)
    dataset.merge(
        input_files=[tmp_path / "ds" / "frames", tmp_path / "ds" / "depths"],
        names=["file_path", "depth_file_path"],
        output_file=(merged := tmp_path / "ds" / "transforms.json"),
        up_axis="y",
    )
    assert np.allclose(np.array(Metadata.load(merged).poses), np.array([z_to_y @ p for p in original.poses]))

    dataset.to_pointcloud(
        colors=cube_dataset / "frames",
        depths=cube_dataset / "depths",
        output=(z_up := tmp_path / "z.ply"),
        p=0.5,
        force=True,
    )
    dataset.to_pointcloud(
        colors=cube_dataset / "frames",
        depths=cube_dataset / "depths",
        output=(y_up := tmp_path / "y.ply"),
        p=0.5,
        up_axis="y",
        force=True,
    )
    assert np.allclose(_read_ply_points(y_up), _read_ply_points(z_up) @ z_to_y[:3, :3].T)

    # Viewers such as Brush orient the scene from that header comment, and guess wrong for
    # a camera orbit that lies in a single plane, so it has to be recorded.
    assert b"comment Vertical axis: y" in y_up.read_bytes()
    assert b"Vertical axis" not in z_up.read_bytes()
