import json
import shutil

import imageio.v3 as iio
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import basic_indices, integer_array_indices

from visionsim.dataset import Dataset, ImgDataset, NpyDataset


def setup_dataset(tmp_path, mode="img", w=100, h=100, c=3, n=1, bitpack_dim=None):
    np.random.seed(123456789)
    transforms = dict(
        **{
            "fl_x": 0,
            "fl_y": 0,
            "cx": 0,
            "cy": 0,
            "h": h,
            "w": w,
            "c": c,
            "frames": [
                dict(
                    transform_matrix=np.random.rand(4, 4).tolist(),
                    **({"file_path": f"frames/frame_{i:04}.png"} if mode.lower() == "img" else {}),
                )
                for i in range(n)
            ],
        },
        **(
            {"file_path": "frames.npy", "bitpack_dim": bitpack_dim, "bitpack": bitpack_dim is not None}
            if mode.lower() == "npy"
            else {}
        ),
    )

    with open(tmp_path / "transforms.json", "w") as f:
        json.dump(transforms, f)

    data = np.random.randint(0, 255, size=(n, h, w, c), dtype=np.uint8)

    if bitpack_dim is not None:
        data = (data > 128).astype(np.uint8)

    if mode.lower() == "npy":
        if bitpack_dim is not None:
            packed_data = np.packbits(data.astype(bool), axis=bitpack_dim)
            np.save(str(tmp_path / "frames.npy"), packed_data)
        else:
            np.save(str(tmp_path / "frames.npy"), data)
    elif mode.lower() == "img":
        (tmp_path / "frames").mkdir(exist_ok=True)
        for i, frame in enumerate(transforms["frames"]):
            iio.imwrite(tmp_path / frame["file_path"], data[i])
    else:
        raise ValueError("Invalid mode!")

    return data, transforms


def test_imgds_valid_resolve_root(tmp_path):
    _, gt_transforms = setup_dataset(tmp_path, mode="img", w=100, h=100, n=1, bitpack_dim=None)
    gt_poses = [f["transform_matrix"] for f in gt_transforms["frames"]]

    # Test when given direct path to json file
    dataset = Dataset.from_path(tmp_path / "transforms.json", mode="img")
    img_paths, poses, transforms = dataset.paths, dataset.poses, dataset.transforms
    assert img_paths == [tmp_path / "frames/frame_0000.png"]
    assert np.allclose(poses, gt_poses)
    assert transforms == gt_transforms

    # Test when given parent directory of json
    dataset = Dataset.from_path(tmp_path, mode="img")
    img_paths, poses, transforms = dataset.paths, dataset.poses, dataset.transforms
    assert img_paths == [tmp_path / "frames/frame_0000.png"]
    assert np.allclose(poses, gt_poses)
    assert transforms == gt_transforms

    # Test when given image directory only
    dataset = Dataset.from_path(tmp_path / "frames", mode="img")
    img_paths, poses, transforms = dataset.paths, dataset.poses, dataset.transforms
    assert img_paths == [tmp_path / "frames/frame_0000.png"]
    assert poses == [None]
    assert transforms is None


def test_imgds_invalid_resolve_root(tmp_path):
    _, gt_transforms = setup_dataset(tmp_path, mode="img", w=100, h=100, n=1, bitpack_dim=None)

    # No file named `transforms.json`
    (tmp_path / "transforms.json").rename(tmp_path / "poses.yaml")
    with pytest.raises(ValueError, match="not understood"):
        Dataset.from_path(tmp_path / "poses.yaml", mode="img")

    # Images in folder do not all have the same extension
    with pytest.raises(RuntimeError, match="images must have same extension"):
        shutil.copy(tmp_path / "frames/frame_0000.png", tmp_path / "frames/frame_0000.jpg")
        Dataset.from_path(tmp_path / "frames", mode="img")

    # Folder provided is not filled with images or a transforms file
    with pytest.raises(FileNotFoundError, match="No image files found"):
        Dataset.from_path(tmp_path, mode="img")

    # No dataset exists
    with pytest.raises(FileNotFoundError, match="not found"):
        Dataset.from_path(tmp_path / "transforms.json", mode="img")

    with pytest.raises(FileNotFoundError, match="not found"):
        Dataset.from_path("some/path/that/does/not/exist", mode="img")


def test_npyds_valid_resolve_root(tmp_path):
    _, gt_transforms = setup_dataset(tmp_path, mode="npy", w=100, h=100, n=1, bitpack_dim=None)
    gt_poses = [f["transform_matrix"] for f in gt_transforms["frames"]]

    # Test when given direct path to json file
    dataset = Dataset.from_path(tmp_path / "transforms.json", mode="npy")
    npy_paths, poses, transforms = dataset.paths, dataset.poses, dataset.transforms
    assert npy_paths == tmp_path / "frames.npy"
    assert np.allclose(poses, gt_poses)
    assert transforms == gt_transforms

    # Test when given parent directory of json
    dataset = Dataset.from_path(tmp_path, mode="npy")
    npy_paths, poses, transforms = dataset.paths, dataset.poses, dataset.transforms
    assert npy_paths == tmp_path / "frames.npy"
    assert np.allclose(poses, gt_poses)
    assert transforms == gt_transforms

    # Test when given npy file directly
    dataset = Dataset.from_path(tmp_path / "frames.npy", mode="npy")
    npy_paths, poses, transforms = dataset.paths, dataset.poses, dataset.transforms
    assert npy_paths == tmp_path / "frames.npy"
    assert poses == [None]
    assert transforms is None

    # Test when given no transforms file
    (tmp_path / "transforms.json").rename(tmp_path / "poses.yaml")
    dataset = Dataset.from_path(tmp_path, mode="npy")
    npy_paths, poses, transforms = dataset.paths, dataset.poses, dataset.transforms
    assert npy_paths == tmp_path / "frames.npy"
    assert poses == [None]
    assert transforms is None


def test_npyds_invalid_resolve_root(tmp_path):
    _, gt_transforms = setup_dataset(tmp_path, mode="npy", w=100, h=100, n=1, bitpack_dim=None)

    # No file named `transforms.json`
    (tmp_path / "transforms.json").rename(tmp_path / "poses.yaml")
    with pytest.raises(ValueError, match="not understood"):
        Dataset.from_path(tmp_path / "poses.yaml", mode="npy")

    # Folder provided is not filled with npy or a transforms file
    (tmp_path / "frames.npy").rename(tmp_path / "images.npy")
    with pytest.raises(FileNotFoundError, match="one of 'transforms.json' or 'frames.npy'"):
        Dataset.from_path(tmp_path, mode="npy")

    # No dataset exists
    with pytest.raises(FileNotFoundError, match="not found"):
        Dataset.from_path(tmp_path / "transforms.json", mode="npy")

    with pytest.raises(FileNotFoundError, match="not found"):
        Dataset.from_path("some/path/that/does/not/exist", mode="npy")


@pytest.mark.parametrize(
    "mode, bitpack_dim",
    [
        ("img", None),
        ("npy", None),
        ("npy", 0),
        ("npy", 1),
        ("npy", 2),
        ("npy", 3),  # No point in bitpacking if channels < 8, but it works.
    ],
)
@given(idx=basic_indices((10, 50, 50, 3), allow_newaxis=False, allow_ellipsis=False))
def test_dataset_slicing(tmp_path_factory, mode, bitpack_dim, idx):
    tmp_path = tmp_path_factory.mktemp(f"{mode}-{bitpack_dim}")
    gt_data, gt_transforms = setup_dataset(tmp_path, mode=mode, w=50, h=50, n=10, bitpack_dim=bitpack_dim)
    frames = np.array([f["transform_matrix"] for f in gt_transforms["frames"]])
    ds = Dataset.from_path(tmp_path)

    gt = gt_data[idx]
    gt_idx, *_ = np.atleast_1d(idx)
    gt_poses = frames[gt_idx]
    gt_idx = np.arange(10)[gt_idx]
    im_idx, im, im_poses = ds[idx]

    # Since ImgDataset returns a list on ndarrays, and gt is just an ndarray,
    # we can have gt.shape == (0, x, x, x) and np.array(im).shape == (0,) which
    # do not broadcast together and cause the allclose below to fail.
    if isinstance(ds, ImgDataset) and gt.size == 0:
        im = np.array(im).reshape(gt.shape)

    assert np.allclose(gt_idx, im_idx)
    assert np.allclose(gt_poses, im_poses)
    assert np.allclose(gt, np.array(im))


@pytest.mark.parametrize(
    "ds_klass, mode, bitpack_dim",
    [
        (ImgDataset, "img", None),
        (NpyDataset, "npy", None),
        (NpyDataset, "npy", 0),
        (NpyDataset, "npy", 1),
        (NpyDataset, "npy", 2),
        (NpyDataset, "npy", 3),
    ],
)
@given(
    idx=st.one_of(
        basic_indices((1, 50, 50, 3), allow_newaxis=True, allow_ellipsis=True).filter(
            lambda shape: any(i in (np.newaxis, None, ...) for i in shape)
        ),
        integer_array_indices((1, 50, 50, 3)),
    )
)
def test_dataset_slicing_notimplemented(tmp_path_factory, ds_klass, mode, bitpack_dim, idx):
    tmp_path = tmp_path_factory.mktemp(f"{mode}-{bitpack_dim}")
    setup_dataset(tmp_path, mode=mode, w=50, h=50, n=1, bitpack_dim=bitpack_dim)
    ds = ds_klass(tmp_path)

    with pytest.raises(NotImplementedError):
        _, _, _ = ds[idx]


def test_dataset_direct_instantiation(tmp_path):
    with pytest.raises(TypeError):
        Dataset(tmp_path)
