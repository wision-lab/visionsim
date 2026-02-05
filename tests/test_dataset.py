import json

import imageio.v3 as iio
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import basic_indices, integer_array_indices

from visionsim.dataset import Dataset, Metadata, PathTransforms


def setup_dataset(tmp_path, mode="img", w=100, h=100, c=3, n=1, bitpack_dim=None):
    np.random.seed(123456789)
    transforms = dict(
        **{
            "fl_x": 123,
            "fl_y": 456,
            "cx": w / 2,
            "cy": h / 2,
            "h": h,
            "w": w,
            "c": c,
            "frames": [
                dict(
                    transform_matrix=np.random.rand(4, 4).tolist(),
                    **(
                        {"file_path": f"frames/frame_{i:04}.png"}
                        if mode.lower() == "img"
                        else {"file_path": "frames.npy", "bitpack_dim": bitpack_dim, "offset": i}
                    ),
                )
                for i in range(n)
            ],
        }
    )

    with open(tmp_path / "transforms.json", "w") as f:
        json.dump(transforms, f, indent=2, sort_keys=True)

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
    gt_poses = np.array([f["transform_matrix"] for f in gt_transforms["frames"]])
    ds = Dataset.from_path(tmp_path)
    frame_idx, *_ = np.atleast_1d(idx)
    im, transform = ds[idx]
    poses = (
        transform["transform_matrix"] if not isinstance(transform, tuple) else [t["transform_matrix"] for t in transform]
    )
    assert np.allclose(gt_poses[frame_idx], np.array(poses).reshape((-1, 4, 4)))

    # Since `Dataset` returns a tuple of ndarrays, and im is just an ndarray,
    # we can have im.shape == (0, x, x, x) and np.array(im).shape == (0,) which
    # do not broadcast together and cause the allclose below to fail.
    if gt_data[idx].size == 0:
        assert np.array(im).size == 0
    else:
        assert np.allclose(gt_data[idx], np.array(im))


@pytest.mark.parametrize(
    "mode, bitpack_dim",
    [
        ("img", None),
        ("npy", None),
        ("npy", 0),
        ("npy", 1),
        ("npy", 2),
        ("npy", 3),
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
def test_dataset_slicing_notimplemented(tmp_path_factory, mode, bitpack_dim, idx):
    tmp_path = tmp_path_factory.mktemp(f"{mode}-{bitpack_dim}")
    setup_dataset(tmp_path, mode=mode, w=50, h=50, n=1, bitpack_dim=bitpack_dim)
    ds = Dataset.from_path(tmp_path)

    with pytest.raises(NotImplementedError):
        ds[idx]


def test_metadata_roundtrip_from_json(tmp_path):
    setup_dataset(tmp_path)
    meta = Metadata.load(tmp_path / "transforms.json")
    meta.save(tmp_path / "transforms.db")

    assert Metadata.load(tmp_path / "transforms.db").model_dump() == meta.model_dump()


def test_path_transforms_simple(tmp_path):
    for i in range(10):
        (tmp_path / f"{i:04}.png").touch()

    paths = sorted(tmp_path.glob("*"))
    assert [t["file_path"] for t in PathTransforms(paths=paths)] == paths


def test_path_transforms_numpy(tmp_path):
    for i in range(10):
        np.save(tmp_path / f"{i:04}.npy", np.ones((i, 5, 5)) * i)

    paths = sorted(tmp_path.glob("*"))
    assert np.allclose(
        np.array([f[0, 0] for f, _ in Dataset.from_paths(paths=paths, iter_npys=True)]),
        np.concatenate([[i] * i for i in range(10)]),
    )
