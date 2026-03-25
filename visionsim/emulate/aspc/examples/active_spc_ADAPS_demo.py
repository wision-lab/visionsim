import os
from pathlib import Path

import numpy as np
import torch

from visionsim.emulate.aspc.camera import Camera
from visionsim.emulate.aspc.examples.ascp_plot_utils import plot_spad_sensor_grid
from visionsim.emulate.aspc.utils import ureg


def generate_sliding_window_fovs(image_shape, kernel_size, stride):
    """
    Generates a list of FOV coordinates [y_min, y_max, x_min, x_max]
    for sliding windows over the image.

    Args:
        image_shape (tuple): (height, width) of the sensor in pixels (e.g., [100, 100])
        kernel_size (tuple): (height, width) of the window in pixels (e.g., [10, 10])
        stride (tuple): (vertical_step, horizontal_step) in pixels

    Returns:
        list: List of [y_min, y_max, x_min, x_max] normalized (0.0-1.0).
    """
    H, W = image_shape
    kh, kw = kernel_size
    sh, sw = stride

    fov_list = []

    # Iterate over rows
    for y in range(0, H - kh + 1, sh):
        # Iterate over columns
        for x in range(0, W - kw + 1, sw):
            # Calculate pixel bounds
            y_min_px, y_max_px = y, y + kh
            x_min_px, x_max_px = x, x + kw

            # Normalize to 0.0 - 1.0
            norm_box = [
                y_min_px / H,  # y_min
                y_max_px / H,  # y_max
                x_min_px / W,  # x_min
                x_max_px / W,  # x_max
            ]
            fov_list.append(norm_box)

    print(f"Generated {len(fov_list)} FOV regions.")
    return fov_list


def load_data(albedo_path, depth_path, device):
    # albedo_frames = torch.zeros((1, 10, 10), dtype=torch.float64, device=device)
    # depth_frames = torch.zeros((1, 10, 10), dtype=torch.float64, device=device)
    # mid_x, mid_y = 256 // 2, 192 // 2
    numpy_albedo = np.flipud(np.load(albedo_path)["arr"].sum(axis=-1))
    numpy_albedo = np.maximum((numpy_albedo / numpy_albedo.max() - 0.075), 0)
    numpy_depth = np.load(depth_path)["arr"]

    # Ensure both inputs are [frames, H, W].
    if numpy_albedo.ndim == 2:
        numpy_albedo = numpy_albedo[None, ...]
    if numpy_depth.ndim == 2:
        numpy_depth = numpy_depth[None, ...]

    albedo_frames = torch.from_numpy(numpy_albedo.copy()).to(device=device, dtype=torch.float64)
    depth_frames = torch.from_numpy(numpy_depth.copy()).to(device=device, dtype=torch.float64)
    # albedo_frames[0, ...] = torch.from_numpy(numpy_albedo.copy())[mid_y - 5 : mid_y + 5, mid_x - 5 : mid_x + 5]
    # depth_frames[0, ...] = torch.from_numpy(np.load(depth_path)["arr"])[mid_y - 5 : mid_y + 5, mid_x - 5 : mid_x + 5]
    depth_frames = ureg.Quantity(depth_frames, "meter")
    albedo_frames = ureg.Quantity(albedo_frames, "dimensionless")
    return albedo_frames, depth_frames


def run_simulation_scenario(camera, new_config, scenario_name, output_dir, spad_grid_shape):
    """Runs the simulation with the specific config object and saves results."""
    print(f"\n--- Running Scenario: {scenario_name} ---")

    # Apply scenario config to rebuild sources/sensor/histogrammer
    camera.reconfigure(new_config)

    # Get data
    albedo_frames, depth_frames = camera.albedo_frames, camera.depth_frames
    transients, _ = camera.get_transients()
    arrival_rates = camera.get_arrival_rates()
    print("Starting EWH simulation")
    ewh_list = camera.get_ewh()
    print("EWH simulation completed")
    # 5. Save Plot
    output_path = os.path.join(output_dir, f"{scenario_name}_ADAPS_normalized_sub.png")
    plot_spad_sensor_grid(
        camera.histogrammer,
        camera.get_fov_masks(),
        spad_grid_shape,
        albedo_frames[0],
        depth_frames[0].magnitude,
        transients,
        arrival_rates,
        ewh_list,
        save_path=output_path,  # Save instead of show
    )


if __name__ == "__main__":
    data_dir = Path("examples/renders/scene1/")
    config_path = "visionsim/emulate/aspc/examples/config_ADAPS_spc.yaml"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    camera = Camera(data_dir, config_path, device)

    output_dir = "results_comparison_ADAPS"
    os.makedirs(output_dir, exist_ok=True)

    # Load npz data
    albedo_path = Path("examples/raw_data/sum_raw_histogram.npz")
    depth_path = Path("examples/raw_data/depth_frame.npz")
    albedo_frames, depth_frames = load_data(albedo_path, depth_path, device)
    camera.albedo_frames, camera.depth_frames = albedo_frames, depth_frames

    # --- AUTOMATIC FOV GENERATION ---
    sensor_h, sensor_w = camera.sensor.h, camera.sensor.w
    k_h, k_w = 1, 1
    s_h, s_w = 1, 1
    generated_fovs = generate_sliding_window_fovs((sensor_h, sensor_w), (k_h, k_w), (s_h, s_w))
    grid_rows = len(range(0, sensor_h - k_h + 1, s_h))
    grid_cols = len(range(0, sensor_w - k_w + 1, s_w))
    spad_grid_shape = (grid_rows, grid_cols)
    # -------------------------------------------------

    # --- DEFINE SCENARIOS ---
    BASE_PULSED = {"enabled": True}
    BASE_SUN = {"enabled": True}
    BASE_LIGHT_CONDITIONS = {"light_conditions": "AVERAGE_SUNLIGHT"}
    BASE_HIST = {"pixel_fov_list": generated_fovs}
    cfg = {
        "active_source": {"pulsed_laser": {**BASE_PULSED}},
        "histogrammer": {**BASE_HIST},
        "ambient_source": {"sun": {**BASE_SUN, **BASE_LIGHT_CONDITIONS}},
    }
    run_simulation_scenario(camera, cfg, "ADAPS_demo", output_dir, spad_grid_shape)

    print("All scenarios completed. Check the 'results_comparison' folder.")
