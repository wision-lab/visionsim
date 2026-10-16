from pathlib import Path

from visionsim.cli import emulate
from visionsim.dataset.models import Metadata


def test_rgb_trailing_single_frame_chunk(tmp_path: Path, write_dataset):
    """A sequence whose length is not a multiple of ``chunk_size`` ends on a chunk of
    one frame, which has no pair of poses to interpolate; the run must still finish."""
    input_dir = write_dataset(tmp_path / "in", 5)
    output_dir = tmp_path / "out"
    emulate.rgb(input_dir=input_dir, output_dir=output_dir, chunk_size=2, readout_std=0)

    assert len(sorted(output_dir.rglob("*.png"))) == 3
    metadata = Metadata.load(output_dir / "transforms.json")
    assert len(metadata.frames) == 3
