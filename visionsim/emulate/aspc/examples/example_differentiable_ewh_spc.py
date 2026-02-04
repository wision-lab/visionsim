import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ruamel.yaml import YAML

from visionsim.emulate.aspc.histogrammers import HistConfig, Histogrammer
from visionsim.emulate.aspc.main import get_light_conditions_from_string
from visionsim.emulate.aspc.sensors import SPADSensor
from visionsim.emulate.aspc.sources import PulsedLaser, Sun
from visionsim.emulate.aspc.utils import (
    eval_constructor,
    irradiance_photons,
    preproc_albedo_intensity_depth_frames,
    tof2depth,
    ureg,
    ureg_constructor,
)

# Import aspc package first to enable aspc.* imports
import visionsim.emulate.aspc  # noqa: F401


def forward_pass_ewh_diff(
    albedo_frames, intensity_frames, depth_frames, active_source, ambient_source, fov_masks, config, irf_tensor
):
    sensor_config = config["sensor"]
    hist_config = HistConfig(**config["histogrammer"])
    histogrammer = Histogrammer(hist_config)
    num_pixels = sensor_config["size"][0] * sensor_config["size"][1]
    sensor = SPADSensor(**sensor_config)

    # Get device from input tensors
    # device = albedo_frames.device

    assert histogrammer.dead_time_s == 0, "Current differentiable EWH does not support non-zero dead time"

    # Get transients
    # Convert tensors to Pint quantities for active source
    albedo_quantity = albedo_frames * ureg.dimensionless
    depth_quantity = depth_frames

    radiance = active_source.get_scene_radiance(albedo_quantity, depth_quantity, num_pixels, sensor.omega)
    irradiance = (radiance * torch.pi / 4 * (1 / sensor.f_number) ** 2).to(irradiance_photons) * (
        sensor.pixel_pitch.to(ureg.meter)
    ) ** 2
    irradiance_tensor = irradiance.magnitude

    ambient_radiance = ambient_source.get_scene_radiance(sensor.omega, albedo_quantity, active_source.frequency)
    ambient_irradiance = (ambient_radiance * torch.pi / 4 * (1 / sensor.f_number) ** 2).to(irradiance_photons) * (
        sensor.pixel_pitch.to(ureg.meter)
    ) ** 2
    offsets = ambient_irradiance.magnitude
    transients, ambient_offsets = histogrammer.calculate_transients(
        irradiance_tensor,
        depth_frames,
        offsets,
        fov_masks,
        histogrammer.n_bins,
        active_source.max_resolvable_depth.magnitude,
    )

    # Calculate arrival rates
    arrival_rates = histogrammer.calculate_arrival_rates(irf_tensor, transients, ambient_offsets, histogrammer.n_bins)

    ewh_list = histogrammer.simulate_ewh_diff(
        arrival_rates,
        histogrammer.n_pulses,
        histogrammer.n_bins,
        histogrammer.free_running,
        histogrammer.dead_time_s,
    )

    return transients, arrival_rates, ewh_list


def compute_rmse(pred, gt):
    rmse = torch.mean(torch.mean((pred - gt) ** 2, axis=-1) ** 0.5)
    return rmse


if __name__ == "__main__":
    ## Setting simulation parameters

    root = Path("examples/renders/scene1/")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    requires_grad = False
    start_idx = 0
    num_frames = 1
    data_dir = "data"
    config_path = "visionsim/emulate/aspc/examples/config_diff_ewh.yaml"

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
    fov_masks = histogrammer.get_perpixel_fov_masks(empty_mask, histogrammer.pixel_fov_list, device=device)

    bin_width = 2 * tof2depth(1 / active_source.frequency) / histogrammer.n_bins
    _, irf = active_source.get_kernel(bin_width)
    irf_tensor_gt = torch.tensor(irf, dtype=float, device=device)
    active_source.plot_kernel(bin_width)

    transients, arrival_rates, ewh_list_gt = forward_pass_ewh_diff(
        albedo_frames, intensity_frames, depth_frames, active_source, ambient_source, fov_masks, config, irf_tensor_gt
    )

    ewh_list_measurement = ewh_list_gt
    ########################################################################
    requires_grad = True

    albedo_frames2, intensity_frames2, depth_frames2 = preproc_albedo_intensity_depth_frames(
        root, device, config, start_idx, num_frames=num_frames, requires_grad=requires_grad
    )

    # Active source
    if active_enable:
        active_source2 = PulsedLaser(**active_config)

    bin_width = 2 * tof2depth(1 / active_source2.frequency) / histogrammer.n_bins
    _, irf = active_source2.get_kernel(bin_width)
    irf_tensor_gt = torch.tensor(irf, dtype=float, device=device)

    if ambient_enable:
        ambient_source2 = Sun(**ambient_config)

    # FOV masks
    _, img_rows, img_cols = depth_frames.shape
    empty_mask = torch.zeros((img_rows, img_cols), dtype=bool, device=device)
    fov_masks2 = histogrammer.get_perpixel_fov_masks(empty_mask, histogrammer.pixel_fov_list, device=device)

    irf_init = [0.1, 0.1, 0.1, 0.4, 0.8, 0.9, 0.9, 0.9, 0.8, 0.4, 0.1, 0.1, 0.1]

    irf_tensor_estim = nn.Parameter(
        torch.tensor(irf_init, device=irf_tensor_gt.device, dtype=irf_tensor_gt.dtype), requires_grad=True
    )

    optimizer = optim.Adam([irf_tensor_estim], lr=0.1)

    for epoch in range(100):
        optimizer.zero_grad()
        transients_pred, arrival_rates_pred, ewh_list_pred = forward_pass_ewh_diff(
            albedo_frames2,
            intensity_frames2,
            depth_frames2,
            active_source2,
            ambient_source2,
            fov_masks2,
            config,
            F.relu(irf_tensor_estim),
        )

        err1 = compute_rmse(ewh_list_pred, ewh_list_measurement)
        err1.backward(retain_graph=True)
        optimizer.step()
        print("Error :", err1)
        print("Gradient:", irf_tensor_estim.grad.abs().mean(), irf_tensor_estim.grad.abs().max())

    plt.plot(irf_tensor_estim.detach().cpu().numpy())
    plt.show()
