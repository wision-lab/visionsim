from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image
from syrupy.extensions.image import PNGImageSnapshotExtension
from syrupy.extensions.json import JSONSnapshotExtension

from visionsim.emulate.dvs.v2e.emulator import EventEmulator


def test_emulate_events(snapshot, tmp_path):
    emulator = EventEmulator(
        pos_thres=0.1,
        neg_thres=0.1,
        sigma_thres=0.0,
        cutoff_hz=0.0,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=0.0,
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
    assert summary_stats == snapshot(extension_class=JSONSnapshotExtension)

    viz = np.ones((*img.shape, 3), dtype=np.uint8) * 255
    first_batch = all_events[0]
    pos_mask = first_batch[:, -1] == 1
    neg_mask = first_batch[:, -1] == -1
    _, px, py, _ = first_batch[pos_mask].T.astype(int)
    _, nx, ny, _ = first_batch[neg_mask].T.astype(int)
    viz[ny, nx] = [255, 0, 0]
    viz[py, px] = [0, 0, 255]

    # Syrupy only campares byte strings, so we dump the image to disk and read it back
    # This helps debug too as we can visually inspect the generated preview
    temp_preview_path = tmp_path / "events_preview.png"
    iio.imwrite(temp_preview_path, viz)

    with open(temp_preview_path, "rb") as f:
        viz_bytes = f.read()
        assert viz_bytes == snapshot(extension_class=PNGImageSnapshotExtension), (
            "Generated preview does not match the reference snapshot"
        )

    # Test reset functionality
    emulator.reset()
    assert emulator.base_log_frame is None
    assert emulator.frame_counter == 0
