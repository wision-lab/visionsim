from pathlib import Path

import torch
import yaml
from ascp_plot_utils import plot_edh_per_pixel
from histogrammers import calculate_arrival_rates, calculate_transients, get_perpixel_fov_masks, simulate_edh

# Import from local modules
from sources import PulsedLaser, Sun, get_light_conditions_from_string
from utils import preproc_albedo_intensity_depth_frames, tof2depth, ureg

if __name__ == "__main__":
    ## Setting simulation parameters

    root = Path("../../../examples/renders/scene1/")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    requires_grad = True
    start_idx = 0
    num_frames = 1
    data_dir = "data"
    config_path = "config_edh.yaml"

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    albedo_frames, intensity_frames, depth_frames = preproc_albedo_intensity_depth_frames(
        root, device, config, start_idx, num_frames=num_frames, requires_grad=requires_grad
    )

    # Active source
    active_config = config["active_source"]["pulsed_laser"]
    active_source = PulsedLaser(
        wavelength=float(active_config["wavelength"]) * ureg.nanometer,
        frequency=float(active_config["frequency"]) * ureg.hertz,
        pulse_width=float(active_config["pulse_width"]) * ureg.second,
        avg_watts=float(active_config["avg_watts"]) * ureg.watt,
        pulse_shape=active_config["pulse_shape"],
        pulse_shape_custom=active_config["pulse_shape_custom"],
    )

    # Ambient source
    ambient_config = config["ambient_source"]["sun"]
    ambient_source = Sun(
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
    fov_masks = get_perpixel_fov_masks(empty_mask, config["histogrammer"]["pixel_fov_list"], device=device)
    # print("fov_masks", fov_masks.shape)

    # Get transients
    sensor_config = config["sensor"]
    hist_config = config["histogrammer"]
    num_pixels = sensor_config["resolution"][0] * sensor_config["resolution"][1]

    # Convert tensors to Pint quantities for active source
    albedo_quantity = albedo_frames * ureg.dimensionless
    depth_quantity = depth_frames * ureg.meter

    # print("albedo_quantity", albedo_quantity.detach().cpu().min(), albedo_quantity.detach().cpu().max())
    # print("depth_quantity", depth_quantity.detach().cpu().min(), depth_quantity.detach().cpu().max())

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
    bin_width = 2 * tof2depth(1 / active_source.frequency) / hist_config["n_bins"]
    _, irf = active_source.get_kernel(bin_width)
    irf_tensor = torch.tensor(irf, dtype=float, device=device)
    arrival_rates = calculate_arrival_rates(irf_tensor, transients, offset, hist_config["n_bins"])

    active_source.plot_kernel(bin_width)

    # Simulate EDH with dead time
    assert hist_config["type"] == "edh", "Incorrect SPC type mentioned in config file"

    photon_hist_list, edh_list = simulate_edh(
        arrival_rates,
        hist_config["n_pulses"],
        hist_config["n_hist_bins"],
        hist_config["free_running"],
        float(hist_config["dead_time_s"]),
    )

    plot_edh_per_pixel(
        config, fov_masks, albedo_frames[0], depth_frames[0], transients, arrival_rates, photon_hist_list, edh_list
    )
