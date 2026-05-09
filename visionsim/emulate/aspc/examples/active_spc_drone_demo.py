"""Active SPC simulation demo using the drone dataset.

Loads RGB and depth frames from examples/renders/drone/, runs the Active SPAD
Camera pipeline, and saves a reconstruction plot.
"""

import os
from pathlib import Path

import OpenEXR
import cv2
import numpy as np
import torch
from natsort import natsorted

from visionsim.dataset import Dataset
from visionsim.emulate.aspc.camera import Camera
from visionsim.emulate.aspc.examples.ascp_plot_utils import plot_spad_sensor_grid
from visionsim.emulate.aspc.utils import ureg

DATASET_ROOT = Path("examples/renders/drone")
CONFIG_PATH = "visionsim/emulate/aspc/examples/active_spc_drone_demo.yaml"
FRAME_IDX = 89  # Change this to load a different frame


def load_drone_data(data_path: Path, config: dict, device, requires_grad: bool = False):
    """Load drone RGB and depth frames, resize, and convert to tensors.

    Mirrors preproc_albedo_intensity_depth_frames but handles the drone
    dataset layout (frames/0000/*.png, depths/0000/*.exr with a mixed-file dir).
    """
    Nr, Nc = config["sensor"]["size"]
    max_depth = config["histogrammer"]["max_depth"].to(ureg.meter).magnitude

    rgb_dataset = Dataset.from_path(data_path / "frames" / "0000", mode="img")
    depth_paths = natsorted((data_path / "depths" / "0000").glob("*.exr"))

    _, rgb_img, _ = rgb_dataset[FRAME_IDX]
    rgb_img = np.array(rgb_img)

    with OpenEXR.File(str(depth_paths[FRAME_IDX])) as exr:
        depth_img = exr.parts[0].channels["V"].pixels.copy()

    # Mask sky/background sentinel values (Blender sets missing geometry to 1e10)
    # then inpaint only those pixels, leaving valid depth values untouched.
    sky_mask = (depth_img > 1e9).astype(np.uint8)
    if sky_mask.any():
        depth_img = cv2.inpaint(depth_img.astype(np.float32), sky_mask, 5, cv2.INPAINT_TELEA)

    # Clip any remaining out-of-range values to max_depth
    depth_img = np.clip(depth_img, 0.0, max_depth)

    print(f"Depth min={depth_img.min():.3f} m, max={depth_img.max():.3f} m")

    rgb_img = cv2.resize(rgb_img, (Nc, Nr))
    depth_img = cv2.resize(depth_img, (Nc, Nr))

    rgb = torch.tensor(rgb_img.astype(float), device=device, requires_grad=requires_grad) / 255.0
    depth = torch.tensor(depth_img.astype(float), device=device, requires_grad=requires_grad).unsqueeze(0)

    # Red channel as albedo/intensity proxy (consistent with other demos)
    albedo = intensity = rgb[..., 0].unsqueeze(0)

    albedo_frames = albedo
    intensity_frames = intensity
    depth_frames = depth * ureg.meter

    return albedo_frames, intensity_frames, depth_frames


class DroneCamera(Camera):
    """Camera subclass that loads data from the drone dataset layout."""

    def _load_data(self, data_path, requires_grad=False):
        return load_drone_data(data_path, self.config, self.device, requires_grad)


def generate_sliding_window_fovs(image_shape, kernel_size, stride):
    H, W = image_shape
    kh, kw = kernel_size
    sh, sw = stride
    fov_list = []
    for y in range(0, H - kh + 1, sh):
        for x in range(0, W - kw + 1, sw):
            fov_list.append([y / H, (y + kh) / H, x / W, (x + kw) / W])
    print(f"Generated {len(fov_list)} FOV regions.")
    return fov_list


def run_simulation_scenario(camera, new_config, scenario_name, output_dir, spad_grid_shape):
    print(f"\n--- Running Scenario: {scenario_name} ---")
    camera.reconfigure(new_config)

    albedo_frames, depth_frames = camera.albedo_frames, camera.depth_frames
    transients, _ = camera.get_transients()
    arrival_rates = camera.get_arrival_rates()
    ewh_list = camera.get_ewh()

    output_path = os.path.join(output_dir, f"{scenario_name}_reconstruction.png")
    plot_spad_sensor_grid(
        camera.histogrammer,
        camera.get_fov_masks(),
        spad_grid_shape,
        albedo_frames[0],
        depth_frames[0].magnitude,
        transients,
        arrival_rates,
        ewh_list,
        save_path=output_path,
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    camera = DroneCamera(DATASET_ROOT, CONFIG_PATH, device)

    output_dir = "results_active_spc_drone_demo_fullres"
    os.makedirs(output_dir, exist_ok=True)

    sensor_h, sensor_w = camera.sensor.h, camera.sensor.w
    k_h, k_w = 60, 60
    s_h, s_w = 60, 60
    generated_fovs = generate_sliding_window_fovs((sensor_h, sensor_w), (k_h, k_w), (s_h, s_w))
    grid_rows = len(range(0, sensor_h - k_h + 1, s_h))
    grid_cols = len(range(0, sensor_w - k_w + 1, s_w))
    spad_grid_shape = (grid_rows, grid_cols)

    cfg = {
        "active_source": {"pulsed_laser": {"enabled": True, "avg_watts": 0.01 * ureg.watt}},
        "histogrammer": {
            "pixel_fov_list": generated_fovs,
            "n_pulses": 10000,
            "n_bins": 1500,
            "bin_width": 0.4 * ureg.meters,
            "max_depth": 600.0 * ureg.meters,
            "vignette": True,
            "dead_time_s": 1000 * ureg.nanoseconds,
            "free_running": False,
            "fast_sim": True,
        },
        "sensor": {
            "size": [1080, 1920],
            "pixel_pitch": 10.0 * ureg.micrometers,
            "f_number": 1.4,
            "fov": [90.5 * ureg.degree, 90.5 * ureg.degree],
        },
        "ambient_source": {"sun": {"enabled": True, "light_conditions": "BRIGHT_SUNLIGHT"}},
        # "ambient_source": {"sun": {"enabled": True, "light_conditions": "OVERCAST"}},
    }
    run_simulation_scenario(camera, cfg, "drone_high_fidelity", output_dir, spad_grid_shape)
