import pytest

from visionsim.cli.interpolate import dataset


def test_dataset_rejects_non_power_of_two(tmp_path):
    with pytest.raises(ValueError, match="power of 2"):
        dataset(tmp_path / "in", tmp_path / "out", n=3)
