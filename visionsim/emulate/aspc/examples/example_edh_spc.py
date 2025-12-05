import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from ruamel.yaml import YAML

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from local modules
from ascp_plot_utils import plot_edh_per_pixel
from histogrammers import HistConfig, HistogrammerEDH
from sensors import SPADSensor
from sources import PulsedLaser, Sun, get_light_conditions_from_string
from utils import (
    eval_constructor,
    preproc_albedo_intensity_depth_frames,
    tof2depth,
    ureg,
    ureg_constructor,
)

if __name__ == "__main__":
    ## Setting simulation parameters

    root = Path("examples/renders/scene1/")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    requires_grad = True
    start_idx = 0
    num_frames = 1
    data_dir = "data"
    config_path = "visionsim/emulate/aspc/examples/config_edh.yaml"

    # Load config
    yaml = YAML()
    safe_builtins = {"__builtins__": {"list": list, "dict": dict, "tuple": tuple}, "np": np, "math": math}
    yaml.Constructor.add_constructor(tag="!Quantity", constructor=ureg_constructor(ureg.Quantity))
    yaml.Constructor.add_constructor(tag="!expr", constructor=eval_constructor(eval, safe_builtins))
    config = yaml.load(open(config_path))

    albedo_frames, intensity_frames, depth_frames = preproc_albedo_intensity_depth_frames(
        root, device, config, start_idx, num_frames=num_frames, requires_grad=requires_grad
    )

    # Active source
    active_config = config["active_source"]["pulsed_laser"]
    active_enable = active_config.pop("enabled")
    if active_enable:
        active_source = PulsedLaser(**active_config)

    # Ambient source
    ambient_config = config["ambient_source"]["sun"]
    ambient_config["light_conditions"] = get_light_conditions_from_string(ambient_config["light_conditions"])
    ambient_enable = ambient_config.pop("enabled")
    if ambient_enable:
        ambient_source = Sun(**ambient_config)

    # Histogrammer
    hist_config = HistConfig(**config["histogrammer"])
    histogrammer = HistogrammerEDH(hist_config)

    # FOV masks
    _, img_rows, img_cols = depth_frames.shape
    empty_mask = torch.zeros((img_rows, img_cols), dtype=bool, device=device)
    fov_masks = histogrammer.get_perpixel_fov_masks(empty_mask, histogrammer.pixel_fov_list, device=device)

    # Sensor
    sensor_config = config["sensor"]
    sensor = SPADSensor(**sensor_config)

    # Get transients
    num_pixels = sensor.num_pixels

    radiance = active_source.get_scene_radiance(albedo_frames, depth_frames, num_pixels, sensor.omega)
    irradiance = (radiance * torch.pi / 4 * (1 / sensor.f_number) ** 2) * sensor.pixel_pitch**2
    irradiance_tensor = irradiance.magnitude

    # Get ambient offset
    ambient_radiance = ambient_source.get_scene_radiance(sensor.omega, albedo_frames, active_source.frequency)
    ambient_irradiance = (ambient_radiance * torch.pi / 4 * (1 / sensor.f_number) ** 2) * sensor.pixel_pitch**2
    offset = ambient_irradiance.magnitude
    transients, ambient_offsets = histogrammer.calculate_transients(
        irradiance_tensor,
        depth_frames,
        offset,
        fov_masks,
        histogrammer.n_bins,
        active_source.max_resolvable_depth.magnitude,
    )

    # Calculate arrival rates
    bin_width = 2 * tof2depth(1 / active_source.frequency) / histogrammer.n_bins
    _, irf = active_source.get_kernel(bin_width)
    irf_tensor = torch.tensor(irf, dtype=float, device=device)
    arrival_rates = histogrammer.calculate_arrival_rates(irf_tensor, transients, ambient_offsets, histogrammer.n_bins)

    active_source.plot_kernel(bin_width)

    # Simulate EDH with dead time
    assert histogrammer.type == "edh", "Incorrect SPC type mentioned in config file"

    photon_hist_list, edh_list = histogrammer.simulate_edh(
        arrival_rates,
        histogrammer.n_pulses,
        histogrammer.n_bins,
        histogrammer.free_running,
        histogrammer.dead_time_s,
    )

    plot_edh_per_pixel(
        histogrammer, fov_masks, albedo_frames[0], depth_frames[0], transients, arrival_rates, photon_hist_list, edh_list
    )
