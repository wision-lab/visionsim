import math
from pathlib import Path

import numpy as np
import torch

from visionsim.emulate.aspc.examples.ascp_plot_utils import plot_ewh_per_pixel
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

if __name__ == "__main__":
    ## Setting simulation parameters

    root = Path("examples/renders/scene1/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    requires_grad = True
    start_idx = 0
    num_frames = 1
    data_dir = "data"
    config_path = "visionsim/emulate/aspc/examples/config_ewh.yaml"

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
    hist_config = config["histogrammer"]
    hist_config = HistConfig(**hist_config)
    histogrammer = Histogrammer(hist_config)

    # FOV masks
    _, img_rows, img_cols = depth_frames.shape
    empty_mask = torch.zeros((img_rows, img_cols), dtype=bool, device=device)
    fov_masks = histogrammer.get_perpixel_fov_masks(empty_mask, hist_config.pixel_fov_list, device=device)

    # Sensor
    sensor_config = config["sensor"]
    sensor = SPADSensor(**sensor_config)

    # Get transients
    sensor_config = config["sensor"]
    num_pixels = sensor_config["size"][0] * sensor_config["size"][1]

    # Convert tensors to Pint quantities for active source
    albedo_quantity = albedo_frames * ureg.dimensionless
    depth_quantity = depth_frames

    radiance = active_source.get_scene_radiance(albedo_quantity, depth_quantity, num_pixels, sensor.omega)
    irradiance = (radiance * torch.pi / 4 * (1 / sensor.f_number) ** 2).to(irradiance_photons) * (
        sensor.pixel_pitch.to(ureg.meter)
    ) ** 2
    irradiance_tensor = irradiance.magnitude

    # Get ambient offset
    ambient_radiance = ambient_source.get_scene_radiance(sensor.omega, albedo_quantity, active_source.frequency)
    ambient_irradiance = (ambient_radiance * torch.pi / 4 * (1 / sensor.f_number) ** 2).to(irradiance_photons) * (
        sensor.pixel_pitch.to(ureg.meter)
    ) ** 2
    offset = ambient_irradiance.magnitude
    transients, ambient_offsets = histogrammer.calculate_transients(
        irradiance_tensor,
        depth_frames.magnitude,
        offset,
        fov_masks,
        histogrammer.n_bins,
        active_source.max_resolvable_depth.magnitude,
    )
    print("transients.shape: ", transients.shape)

    # Calculate arrival rates
    bin_width = 2 * tof2depth(1 / active_source.frequency) / histogrammer.n_bins
    _, irf = active_source.get_kernel(bin_width)
    irf_tensor = torch.tensor(irf, dtype=float, device=device)
    arrival_rates = histogrammer.calculate_arrival_rates(irf_tensor, transients, ambient_offsets, histogrammer.n_bins)

    active_source.plot_kernel(bin_width)

    # Simulate EWH with dead time
    ewh_list = histogrammer.simulate_ewh(
        arrival_rates,
        histogrammer.n_pulses,
        histogrammer.n_bins,
        histogrammer.free_running,
        histogrammer.dead_time_s.magnitude,
    )

    plot_ewh_per_pixel(
        histogrammer,
        fov_masks,
        albedo_frames[0],
        depth_frames[0].magnitude,
        transients,
        arrival_rates,
        ewh_list,
    )
