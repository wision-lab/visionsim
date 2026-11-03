from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

from visionsim.cli import blender as cli
from visionsim.dataset import Metadata
from visionsim.simulate.blender import INDEX_PADDING
from visionsim.simulate.config import RenderConfig

SCENE = Path(__file__).parent.parent / "test_files" / "scenes" / "cube.blend"
FRAME = 11


def test_render_frame(tmp_path, executable):
    """``blender.render-frame`` renders the requested frame and records its camera pose."""
    config = RenderConfig(include_frames=True, executable=executable, width=50, height=50)
    frames_dir = tmp_path / "frames"

    cli.render_frame(SCENE, tmp_path, config, frame=FRAME)

    frame = frames_dir / "0000" / f"{FRAME:0{INDEX_PADDING}}.png"
    assert frame.exists()
    assert iio.imread(frame).shape[:2] == (50, 50)
    assert sorted(frames_dir.glob("**/*.png")) == [frame]

    metadata = Metadata.load(frames_dir / "transforms.db")
    assert [f.file_path.name for f in metadata.frames] == [frame.name]
    assert np.array(metadata.frames[0].transform_matrix).shape == (4, 4)


def test_render_frame_matches_animation(tmp_path, executable):
    """A single-frame render matches what ``render-animation`` produces for the same index."""
    config = RenderConfig(include_frames=True, executable=executable, width=50, height=50)

    cli.render_frame(SCENE, tmp_path / "frame", config, frame=FRAME)
    cli.render_animation(SCENE, tmp_path / "anim", config, frame_start=FRAME, frame_end=FRAME)

    single = Metadata.load(tmp_path / "frame" / "frames" / "transforms.db")
    animation = Metadata.load(tmp_path / "anim" / "frames" / "transforms.db")

    assert [f.file_path for f in single.frames] == [f.file_path for f in animation.frames]
    assert single.frames[0].transform_matrix == animation.frames[0].transform_matrix

    # Rendered files differ in embedded text metadata, and Cycles is not bit-exact across
    # processes (thread/denoise rounding): the two renders can differ by a level or two.
    diff = np.abs(
        iio.imread(tmp_path / "frame" / "frames" / single.frames[0].file_path).astype(int)
        - iio.imread(tmp_path / "anim" / "frames" / animation.frames[0].file_path).astype(int)
    )
    assert diff.max() <= 2, f"renders differ by {diff.max()} levels"
    assert diff.mean() < 0.01, f"renders differ on average by {diff.mean():.3f}"


def test_render_frame_warns_on_multiple_jobs(tmp_path, executable, caplog):
    """``blender.render-frame`` warns that job fan-out is ignored, but still renders."""
    config = RenderConfig(include_frames=True, executable=executable, jobs=3, width=50, height=50)

    with caplog.at_level("WARNING"):
        cli.render_frame(SCENE, tmp_path, config, frame=FRAME)

    assert "always uses a single render job" in caplog.text
    assert (tmp_path / "frames" / "0000" / f"{FRAME:0{INDEX_PADDING}}.png").exists()


def test_autoscale_requires_max_job_vram():
    with pytest.raises(ValueError, match="max_job_vram"):
        cli._resolve_autoscale(RenderConfig(autoscale=True))


def test_autoscale_rejects_non_positive_max_job_vram():
    with pytest.raises(ValueError, match="positive"):
        cli._resolve_autoscale(RenderConfig(autoscale=True, max_job_vram=0))


def test_autoscale_without_gpu_uses_single_job(monkeypatch, caplog):
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)

    with caplog.at_level("WARNING"):
        config = cli._resolve_autoscale(RenderConfig(autoscale=True, max_job_vram=1024))

    assert config.jobs == 1
    assert not config.autoscale
    assert "No GPU" in caplog.text


@pytest.mark.skipif(not cli.torch.cuda.is_available(), reason="requires CUDA-enabled torch")
def test_autoscale_single_gpu_divides_free_vram(monkeypatch):
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cli.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(cli.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(cli.torch.cuda, "mem_get_info", lambda device: (10_000, 16_000))
    monkeypatch.setattr(cli.torch.cuda, "get_device_name", lambda device: "fake gpu")

    config = cli._resolve_autoscale(RenderConfig(autoscale=True, max_job_vram=2_500))

    assert config.jobs == 4
    assert config.autoscale
