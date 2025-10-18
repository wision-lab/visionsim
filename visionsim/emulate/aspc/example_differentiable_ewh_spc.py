import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from histogrammers import (
    calculate_arrival_rates,
    calculate_transients,
    get_perpixel_fov_masks,
    simulate_ewh_diff,
)
from ruamel.yaml import YAML

# Import from local modules
from sources import PulsedLaser, Sun, get_light_conditions_from_string

# from utils import tof2depth, ureg, preproc_albedo_intensity_depth_frames
from utils import eval_constructor, preproc_albedo_intensity_depth_frames, tof2depth, ureg, ureg_constructor


def forward_pass_ewh_diff(
    albedo_frames, intensity_frames, depth_frames, active_source, ambient_source, fov_masks, config, irf_tensor
):
    sensor_config = config["sensor"]
    hist_config = config["histogrammer"]
    num_pixels = sensor_config["size"][0] * sensor_config["size"][1]  # changed by BHUYASHI

    assert hist_config["dead_time_s"] == 0, "Current differentiable EWH does not support non-zero dead time"

    # Get transients
    # Convert tensors to Pint quantities for active source
    albedo_quantity = albedo_frames * ureg.dimensionless
    # depth_quantity = depth_frames * ureg.meter
    depth_quantity = depth_frames  # added by BHUYASHI

    radiance = active_source.get_scene_radiance(
        albedo_quantity, depth_quantity, num_pixels, float(sensor_config["omega"]) * ureg.steradian
    )
    irradiance = (radiance * torch.pi / 4 * (1 / sensor_config["f_number"]) ** 2) * float(
        sensor_config["pixel_pitch"] ** 2
    )
    irradiance_tensor = irradiance.magnitude

    # print("Number of photons per cycle: ", active_source.num_photons_per_cycle)
    # exit(-1)

    # Get ambient offset
    ambient_radiance = ambient_source.get_scene_radiance(
        float(sensor_config["omega"]) * ureg.steradian, albedo_quantity, active_source.frequency
    )
    ambient_irradiance = (ambient_radiance * torch.pi / 4 * (1 / sensor_config["f_number"]) ** 2) * float(
        sensor_config["pixel_pitch"] ** 2
    )
    offset = ambient_irradiance.magnitude
    transients = calculate_transients(
        irradiance_tensor, depth_frames, fov_masks, hist_config["n_bins"], active_source.max_resolvable_depth.magnitude
    )

    # Calculate arrival rates
    # bin_width = (2 * tof2depth(1 / active_source.frequency) / hist_config['n_bins'])
    # _, irf = active_source.get_kernel(bin_width)
    # print("irf",irf)
    # irf_tensor = torch.tensor(irf, dtype=float, device=device)
    arrival_rates = calculate_arrival_rates(irf_tensor, transients, offset, hist_config["n_bins"])

    ewh_list = simulate_ewh_diff(
        arrival_rates,
        hist_config["n_pulses"],
        hist_config["n_bins"],
        hist_config["free_running"],
        float(hist_config["dead_time_s"]),
    )

    return transients, arrival_rates, ewh_list


def compute_rmse(pred, gt):
    rmse = torch.mean(torch.mean((pred - gt) ** 2, axis=-1) ** 0.5)
    return rmse


if __name__ == "__main__":
    ## Setting simulation parameters

    root = Path("examples/renders/scene1/")  # changed by BHUYASHI
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    requires_grad = False
    start_idx = 0
    num_frames = 1
    data_dir = "data"
    config_path = "visionsim/emulate/aspc/config_diff_ewh_bh.yaml"  # changed by BHUYASHI

    # Load config
    # with open(config_path, "r") as f:
    #     config = yaml.safe_load(f)
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
    # active_source = PulsedLaser(
    #     wavelength=float(active_config['wavelength']) * ureg.nanometer,
    #     frequency=float(active_config['frequency']) * ureg.hertz,
    #     pulse_width=float(active_config['pulse_width']) * ureg.second,
    #     avg_watts=float(active_config['avg_watts']) * ureg.watt,
    #     pulse_shape=active_config['pulse_shape'],
    #     pulse_shape_custom=active_config['pulse_shape_custom']
    # )
    # added by BHUYASHI
    active_enable = active_config.pop("enabled")
    if active_enable:
        active_source = PulsedLaser(**active_config)

    bin_width = 2 * tof2depth(1 / active_source.frequency) / config["histogrammer"]["n_bins"]
    _, irf = active_source.get_kernel(bin_width)
    print("irf", irf)
    irf_tensor_gt = torch.tensor(irf, dtype=float, device=device)

    active_source.plot_kernel(bin_width)

    # Ambient source
    ambient_config = config["ambient_source"]["sun"]
    # ambient_source = Sun(
    #     intensity=float(ambient_config['intensity']) * ureg.watt / ureg.meter**2,
    #     stability_factor=float(ambient_config['stability_factor']) * ureg.dimensionless,
    #     temperature=float(ambient_config['temperature']) * ureg.kelvin,
    #     lambda_pass=float(ambient_config['lambda_pass']) * ureg.nanometer,
    #     delta_lambda=float(ambient_config['delta_lambda']) * ureg.nanometer,
    #     light_conditions=get_light_conditions_from_string(ambient_config['light_conditions'])
    # )
    # added by BHUYASHI
    ambient_config["light_conditions"] = get_light_conditions_from_string(ambient_config["light_conditions"])
    ambient_enable = ambient_config.pop("enabled")
    if ambient_enable:
        ambient_source = Sun(**ambient_config)

    # FOV masks
    _, img_rows, img_cols = depth_frames.shape
    empty_mask = torch.zeros((img_rows, img_cols), dtype=bool)
    fov_masks = get_perpixel_fov_masks(empty_mask, config["histogrammer"]["pixel_fov_list"], device=device)
    # print("fov_masks", fov_masks.shape)

    transients, arrival_rates, ewh_list_gt = forward_pass_ewh_diff(
        albedo_frames, intensity_frames, depth_frames, active_source, ambient_source, fov_masks, config, irf_tensor_gt
    )

    # plot_ewh_per_pixel(config,
    #                 fov_masks,
    #                 albedo_frames[0],
    #                 depth_frames[0],
    #                 transients,
    #                 arrival_rates,
    #                 ewh_list_gt)

    # ewh_list_measurement = [torch.tensor(ewh1.detach().cpu().numpy(), device=device, requires_grad=True) for ewh1 in ewh_list_gt]
    ewh_list_measurement = ewh_list_gt
    ########################################################################
    requires_grad = True

    albedo_frames2, intensity_frames2, depth_frames2 = preproc_albedo_intensity_depth_frames(
        root, device, config, start_idx, num_frames=num_frames, requires_grad=requires_grad
    )

    # Active source
    active_config = config["active_source"]["pulsed_laser"]
    active_source2 = PulsedLaser(
        wavelength=float(active_config["wavelength"]) * ureg.nanometer,
        frequency=float(active_config["frequency"]) * ureg.hertz,
        pulse_width=float(active_config["pulse_width"]) * ureg.second,
        avg_watts=float(active_config["avg_watts"]) * ureg.watt,
        pulse_shape=active_config["pulse_shape"],
        pulse_shape_custom=active_config["pulse_shape_custom"],
    )

    bin_width = 2 * tof2depth(1 / active_source2.frequency) / config["histogrammer"]["n_bins"]
    _, irf = active_source2.get_kernel(bin_width)
    print("irf", irf)
    irf_tensor_gt = torch.tensor(irf, dtype=float, device=device)

    # Ambient source
    ambient_config = config["ambient_source"]["sun"]
    ambient_source2 = Sun(
        intensity=float(ambient_config["intensity"]) * ureg.watt / ureg.meter**2,
        stability_factor=float(ambient_config["stability_factor"]) * ureg.dimensionless,
        temperature=float(ambient_config["temperature"]) * ureg.kelvin,
        lambda_pass=float(ambient_config["lambda_pass"]) * ureg.nanometer,
        delta_lambda=float(ambient_config["delta_lambda"]) * ureg.nanometer,
        light_conditions=get_light_conditions_from_string(ambient_config["light_conditions"]),
    )

    # FOV masks
    _, img_rows, img_cols = depth_frames.shape
    empty_mask = torch.zeros((img_rows, img_cols), dtype=bool)
    fov_masks2 = get_perpixel_fov_masks(empty_mask, config["histogrammer"]["pixel_fov_list"], device=device)
    # print("fov_masks", fov_masks.shape)

    irf_init = [0.1, 0.1, 0.1, 0.4, 0.8, 0.9, 0.9, 0.9, 0.8, 0.4, 0.1, 0.1, 0.1]

    irf_tensor_estim = nn.Parameter(
        torch.tensor(irf_init, device=irf_tensor_gt.device, dtype=irf_tensor_gt.dtype), requires_grad=True
    )

    # requires_grad = True

    optimizer = optim.Adam([irf_tensor_estim], lr=0.1)

    # plt.plot(irf_tensor_estim.detach().cpu().numpy())
    # plt.show()

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
