from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image
from syrupy.extensions.image import PNGImageSnapshotExtension
from syrupy.extensions.json import JSONSnapshotExtension

from visionsim.emulate.dvs.v2e.emulator import EventEmulator


class ImageSnapshotExtension(PNGImageSnapshotExtension):
    def serialize(self, data, **kwargs):
        if isinstance(data, (str, Path)):
            with open(data, "rb") as f:
                data = f.read()
        return super().serialize(data, **kwargs)

    def matches(self, *, serialized_data, snapshot_data) -> bool:
        serialized_im = iio.imread(serialized_data)
        snapshot_im = iio.imread(snapshot_data)
        return np.allclose(serialized_im, snapshot_im)


def test_emulate_events(snapshot, tmp_path):
    emulator = EventEmulator(
        pos_thres=0.1,
        neg_thres=0.1,
        sigma_thres=0.0,
        cutoff_hz=0.0,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=0.0,
        seed=42087,
    )

    lego_gt_path = Path(__file__).parent / "test_files" / "lego-gt"
    frames = sorted(lego_gt_path.glob("*.png"))

    fps = 24.0
    delta_time = 1.0 / fps
    all_events = []
    summary_stats = []

    for i, frame_path in enumerate(frames):
        # Read image, convert to grayscale, and make it float32
        img = np.array(Image.open(frame_path).convert("L"), dtype=np.float32)
        events = emulator.generate_events(img, i * delta_time)

        if events is not None:
            all_events.append(events)
            summary_stats.append(
                {
                    "frame": i,
                    "total_events": len(events),
                    "pos_events": int(np.sum(events[:, -1] == 1)),
                    "neg_events": int(np.sum(events[:, -1] == -1)),
                }
            )

    assert len(all_events) > 0, "No events were generated from the frames."

    for i, events in enumerate(all_events):
        if i % 5 == 0:
            viz = np.ones((*img.shape, 3), dtype=np.uint8) * 255
            pos_mask = events[:, -1] == 1
            neg_mask = events[:, -1] == -1
            _, px, py, _ = events[pos_mask].T.astype(int)
            _, nx, ny, _ = events[neg_mask].T.astype(int)
            viz[ny, nx] = [255, 0, 0]
            viz[py, px] = [0, 0, 255]

            # Syrupy only compares byte strings, so we dump the image to disk and read it back
            # This helps debug too as we can visually inspect the generated preview
            temp_preview_path = tmp_path / f"events_preview_{i:04d}.png"
            iio.imwrite(temp_preview_path, viz)

    # Compare event statistics
    assert summary_stats == snapshot(extension_class=JSONSnapshotExtension)

    # Compare event preview
    for path in sorted(tmp_path.glob("events_preview_*.png")):
        assert path == snapshot(extension_class=ImageSnapshotExtension), (
            "Generated preview does not match the reference snapshot"
        )

    # Test reset functionality
    emulator.reset()
    assert emulator.base_log_frame is None
    assert emulator.frame_counter == 0
