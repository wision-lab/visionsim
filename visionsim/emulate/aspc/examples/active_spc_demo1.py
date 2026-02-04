import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from copy import deepcopy # Important for copying config

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ascp_plot_utils import plot_ewh_per_pixel, plot_spad_sensor_grid
from visionsim.emulate.aspc.histogrammers import HistConfig, Histogrammer
from ruamel.yaml import YAML
from visionsim.emulate.aspc.sensors import SPADSensor
from visionsim.emulate.aspc.sources import PulsedLaser, Sun, get_light_conditions_from_string
from visionsim.emulate.aspc.utils import (
    eval_constructor,
    irradiance_photons,
    preproc_albedo_intensity_depth_frames,
    tof2depth,
    ureg,
    ureg_constructor,
)

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
                y_min_px / H, # y_min
                y_max_px / H, # y_max
                x_min_px / W, # x_min
                x_max_px / W  # x_max
            ]
            fov_list.append(norm_box)
            
    print(f"Generated {len(fov_list)} FOV regions.")
    return fov_list

def run_simulation_scenario(base_config, scenario_name, output_dir, device="cpu"):
    """
    Runs the simulation with the specific config object and saves results.
    """
    print(f"\n--- Running Scenario: {scenario_name} ---")
    
    # 1. Setup Data
    root = Path("examples/renders/scene1/")
    albedo_frames, intensity_frames, depth_frames = preproc_albedo_intensity_depth_frames(
        root, device, base_config, 0, num_frames=1, requires_grad=False
    )

    # 2. Configure Sources
    # --- FIX: Pop 'enabled' key before passing config to class ---
    active_config = base_config["active_source"]["pulsed_laser"]
    if "enabled" in active_config:
        active_config.pop("enabled") # Remove the key causing the error
    active_source = PulsedLaser(**active_config)

    # Ambient
    ambient_config = base_config["ambient_source"]["sun"]
    if "enabled" in ambient_config:
        ambient_config.pop("enabled") # Remove here too
        
    # Handle string conversion for light conditions if needed
    if isinstance(ambient_config.get("light_conditions"), str):
        ambient_config["light_conditions"] = get_light_conditions_from_string(ambient_config["light_conditions"])
    ambient_source = Sun(**ambient_config)

    # 3. Configure Histogrammer & FOV
    hist_config_dict = base_config["histogrammer"]
    
    # --- AUTOMATIC FOV GENERATION ---
    sensor_h, sensor_w = base_config["sensor"]["size"]
    k_h, k_w = 4, 4
    s_h, s_w = 2, 2
    generated_fovs = generate_sliding_window_fovs((sensor_h, sensor_w), (k_h, k_w), (s_h, s_w))
    grid_rows = len(range(0, sensor_h - k_h + 1, s_h))
    grid_cols = len(range(0, sensor_w - k_w + 1, s_w))
    spad_grid_shape = (grid_rows, grid_cols)
    hist_config_dict["pixel_fov_list"] = generated_fovs
    # -------------------------------------------------

    vignette = hist_config_dict["vignette"]
    hist_obj_config = HistConfig(**hist_config_dict)
    histogrammer = Histogrammer(hist_obj_config)

    # FOV Masks
    _, img_rows, img_cols = depth_frames.shape
    empty_mask = torch.zeros((img_rows, img_cols), dtype=bool, device=device)
    fov_masks = histogrammer.get_perpixel_fov_masks(empty_mask, hist_obj_config.pixel_fov_list, device=device, vignette=vignette)

    # Sensor
    sensor = SPADSensor(**base_config["sensor"])

    # 4. Physics Simulation
    num_pixels = base_config["sensor"]["size"][0] * base_config["sensor"]["size"][1]
    
    # Active Radiance
    albedo_quantity = albedo_frames * ureg.dimensionless
    depth_quantity = depth_frames
    radiance = active_source.get_scene_radiance(albedo_quantity, depth_quantity, num_pixels, sensor.omega)    
    irradiance = (radiance * torch.pi / 4 * (1 / sensor.f_number) ** 2).to(irradiance_photons) * (sensor.pixel_pitch.to(ureg.meter)) ** 2
    irradiance_tensor = irradiance.magnitude

    # Ambient Radiance
    ambient_radiance = ambient_source.get_scene_radiance(sensor.omega, albedo_quantity, active_source.frequency)
    ambient_irradiance = (ambient_radiance * torch.pi / 4 * (1 / sensor.f_number) ** 2).to(irradiance_photons) * (sensor.pixel_pitch.to(ureg.meter)) ** 2
    offset = ambient_irradiance.magnitude

    # Transients
    transients, ambient_offsets = histogrammer.calculate_transients(
        irradiance_tensor, depth_frames.magnitude, offset, fov_masks, histogrammer.n_bins, active_source.max_resolvable_depth.magnitude,
    )

    # Arrival Rates & IRF
    bin_width = 2 * tof2depth(1 / active_source.frequency) / histogrammer.n_bins
    _, irf = active_source.get_kernel(bin_width)
    irf_tensor = torch.tensor(irf, dtype=float, device=device)
    arrival_rates = histogrammer.calculate_arrival_rates(irf_tensor, transients, ambient_offsets, histogrammer.n_bins)

    # EWH Simulation
    dead_time_bins = int(histogrammer.dead_time_s * histogrammer.n_bins * active_source.frequency)
    
    ewh_list = histogrammer.simulate_ewh(
        arrival_rates, histogrammer.n_pulses, histogrammer.n_bins, histogrammer.free_running, dead_time_bins,
    )

    # 5. Save Plot
    output_path = os.path.join(output_dir, f"{scenario_name}_reconstruction.png")
    plot_spad_sensor_grid(
        histogrammer,
        fov_masks,
        spad_grid_shape,
        albedo_frames[0],
        depth_frames[0].magnitude,
        transients,
        arrival_rates,
        ewh_list,
        save_path=output_path  # Save instead of show
    )


if __name__ == "__main__":
    # Load Base Config
    config_path = "visionsim/emulate/aspc/examples/active_spc_demo.yaml"
    yaml = YAML()
    safe_builtins = {"__builtins__": {"list": list, "dict": dict, "tuple": tuple}, "np": np, "math": math}
    yaml.Constructor.add_constructor(tag="!Quantity", constructor=ureg_constructor(ureg.Quantity))
    yaml.Constructor.add_constructor(tag="!expr", constructor=eval_constructor(eval, safe_builtins))
    
    base_config = yaml.load(open(config_path))
    output_dir = "results_comparison"
    os.makedirs(output_dir, exist_ok=True)

    # --- DEFINE SCENARIOS ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Scenario 1: High Fidelity (Your current "nice" settings)
    # Good laser power, average sun
    cfg_1 = deepcopy(base_config)
    cfg_1["active_source"]["pulsed_laser"]["avg_watts"] = 0.05 * ureg.watt
    cfg_1["histogrammer"]["n_pulses"] = 10000
    run_simulation_scenario(cfg_1, "1_High_Fidelity", output_dir, device)

    # Scenario 2: Photon Starved (Low SNR)
    # Reduced laser power significantly. Depth estimation should become noisy/patchy.
    # Traditional simulators would still show perfect depth here!
    cfg_2 = deepcopy(base_config)
    cfg_2["active_source"]["pulsed_laser"]["avg_watts"] = 0.0005 * ureg.watt # 100x weaker
    cfg_2["histogrammer"]["n_pulses"] = 2000 # Fewer pulses
    run_simulation_scenario(cfg_2, "2_Photon_Starved", output_dir, device)

    # Scenario 3: High Ambient Interference
    # Standard laser power, but very intense background light.
    # This floods the histogram with noise, burying the signal peak.
    cfg_3 = deepcopy(base_config)
    cfg_3["active_source"]["pulsed_laser"]["avg_watts"] = 0.01 * ureg.watt
    cfg_3["ambient_source"]["sun"]["intensity"] = 1.0e28 * ureg.watt / ureg.meter**2 # Massive sun scaling
    # Alternatively, keep intensity same but increase aperture or exposure if sun scaling is weird physically
    run_simulation_scenario(cfg_3, "3_High_Ambient", output_dir, device)

    print("All scenarios completed. Check the 'results_comparison' folder.")
