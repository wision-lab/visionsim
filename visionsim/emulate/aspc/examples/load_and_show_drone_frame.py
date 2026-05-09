"""Load and display a specific RGB and depth frame from the drone dataset."""

from pathlib import Path

import OpenEXR
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted

from visionsim.dataset import Dataset

DATASET_ROOT = Path("examples/renders/drone")
FRAME_IDX = 0  # Change this to load a different frame


def main():
    rgb_dataset = Dataset.from_path(DATASET_ROOT / "frames" / "0000", mode="img")
    depth_paths = natsorted((DATASET_ROOT / "depths" / "0000").glob("*.exr"))

    print(f"Total RGB frames : {len(rgb_dataset)}")
    print(f"Total depth frames: {len(depth_paths)}")

    _, rgb_frame, _ = rgb_dataset[FRAME_IDX]
    rgb_frame = np.array(rgb_frame)

    with OpenEXR.File(str(depth_paths[FRAME_IDX])) as exr:
        depth_frame = exr.parts[0].channels["V"].pixels

    print(f"RGB frame shape  : {rgb_frame.shape}, dtype: {rgb_frame.dtype}")
    print(f"Depth frame shape: {depth_frame.shape}, dtype: {depth_frame.dtype}")
    print(f"Depth min={depth_frame.min():.4f}, max={depth_frame.max():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Drone dataset — frame {FRAME_IDX}")

    axes[0].imshow(rgb_frame)
    axes[0].set_title("RGB frame")
    axes[0].axis("off")

    depth_plot = axes[1].imshow(depth_frame, cmap="plasma")
    axes[1].set_title("Depth frame")
    axes[1].axis("off")
    fig.colorbar(depth_plot, ax=axes[1], label="Depth (m)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
