import json
import subprocess

import imageio.v3 as iio
import numpy as np

from visionsim.dataset import Metadata
from visionsim.simulate.blender import INDEX_PADDING

FRAMES = [f"{f:0{INDEX_PADDING}}.png" for f in range(10, 15)]


def test_playblast_layout(playblast_dataset):
    playblast_dir = playblast_dataset / "playblast"
    assert playblast_dir.exists()
    assert not (playblast_dataset / "frames").exists()

    # Frame-sequence mode produces PNGs plus a metadata database
    assert sorted(p.name for p in playblast_dir.glob("*.png")) == FRAMES
    assert (playblast_dir / "transforms.db").exists()


def test_playblast_metadata(playblast_dataset):
    metadata = Metadata.load(playblast_dataset / "playblast" / "transforms.db")

    assert len(metadata.frames) == 5
    for frame in metadata.frames:
        transform = np.array(frame.transform_matrix)
        assert transform.shape == (4, 4)
        assert (playblast_dataset / "playblast" / frame.file_path).exists()


def test_playblast_images(playblast_dataset):
    images = sorted((playblast_dataset / "playblast").glob("*.png"))
    assert len(images) == 5

    image = iio.imread(images[0])
    assert image.dtype == np.uint8
    assert image.shape[:2] == (50, 50)
    assert image.shape[2] in (3, 4)


def test_playblast_video_layout(playblast_video_dataset):
    playblast_dir = playblast_video_dataset / "playblast"
    assert playblast_dir.exists()
    assert not (playblast_video_dataset / "frames").exists()

    # Video mode writes a single container, and no frame sequence or metadata database
    assert [p.name for p in playblast_dir.iterdir()] == ["playblast.mp4"]
    assert not list(playblast_dir.glob("*.png"))
    assert not (playblast_dir / "transforms.db").exists()


def test_playblast_video_contents(playblast_video_dataset):
    video = playblast_video_dataset / "playblast" / "playblast.mp4"
    assert video.exists()
    assert video.stat().st_size > 0

    # Decode through the ffmpeg CLI: no python video backend is a dependency of this project.
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,width,height",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert int(stream["nb_read_frames"]) == 5
    assert (stream["width"], stream["height"]) == (50, 50)

