import torch
import yaml
from pathlib import Path
from ruamel.yaml import YAML
import numpy as np
import math

# Import from local modules
from sources import (
    PulsedLaser, 
    Sun, 
    get_light_conditions_from_string)

from histogrammers import (
    get_perpixel_fov_masks, 
    calculate_transients, 
    calculate_arrival_rates, 
    simulate_ewh,
    simulate_ewh_diff
)

from sensors import SPADSensor

from utils import tof2depth, ureg, preproc_albedo_intensity_depth_frames, ureg_constructor, eval_constructor, irradiance_photons

from ascp_plot_utils import plot_ewh_per_pixel


if __name__ == "__main__":
    ## Setting simulation parameters 

    root = Path("examples/renders/scene1/")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    requires_grad = True
    start_idx = 0
    num_frames = 1
    data_dir = "data"
    config_path = "visionsim/emulate/aspc/config.yaml"
    
    # Load config
    yaml = YAML()
    safe_builtins = {'__builtins__': {'list': list, 'dict': dict, 'tuple': tuple}, 'np': np, 'math': math}
    yaml.Constructor.add_constructor(tag="!Quantity", constructor=ureg_constructor(ureg.Quantity))
    yaml.Constructor.add_constructor(tag="!expr", constructor=eval_constructor(eval, safe_builtins))
    config = yaml.load(open(config_path))
    
    albedo_frames, intensity_frames, depth_frames = preproc_albedo_intensity_depth_frames(
        root,
        device,
        config, 
        start_idx,
        num_frames = num_frames, 
        requires_grad = requires_grad)

    # Active source
    active_config = config['active_source']['pulsed_laser']
    # active_source = PulsedLaser(
    #     wavelength=float(active_config['wavelength']) * ureg.nanometer,
    #     frequency=float(active_config['frequency']) * ureg.hertz,
    #     pulse_width=float(active_config['pulse_width']) * ureg.second,
    #     avg_watts=float(active_config['avg_watts']) * ureg.watt,
    #     pulse_shape=active_config['pulse_shape'],
    #     pulse_shape_custom=active_config['pulse_shape_custom']
    # )
    active_enable = active_config.pop('enabled')
    if active_enable:
        active_source = PulsedLaser(**active_config)

    # Ambient source
    ambient_config = config['ambient_source']['sun']
    # ambient_source = Sun(
    #     intensity=float(ambient_config['intensity']) * ureg.watt / ureg.meter**2,
    #     stability_factor=float(ambient_config['stability_factor']) * ureg.dimensionless,
    #     temperature=float(ambient_config['temperature']) * ureg.kelvin,
    #     lambda_pass=float(ambient_config['lambda_pass']) * ureg.nanometer,
    #     delta_lambda=float(ambient_config['delta_lambda']) * ureg.nanometer,
    #     light_conditions=get_light_conditions_from_string(ambient_config['light_conditions'])
    # )
    ambient_config['light_conditions'] = get_light_conditions_from_string(ambient_config['light_conditions'])
    ambient_enable = ambient_config.pop('enabled')
    if ambient_enable:
        ambient_source = Sun(**ambient_config)

    # FOV masks
    _, img_rows, img_cols = depth_frames.shape
    empty_mask = torch.zeros((img_rows, img_cols), dtype=bool)
    fov_masks = get_perpixel_fov_masks(
        empty_mask, 
        config['histogrammer']['pixel_fov_list'], 
        device=device)
    # print("fov_masks", fov_masks.shape)

    # Histogrammer
    hist_config = config['histogrammer']

    # Sensor
    sensor_config = config['sensor']
    sensor = SPADSensor(**sensor_config)


    # Get transients
    sensor_config = config['sensor']
    hist_config = config['histogrammer']
    num_pixels = sensor_config['size'][0] * sensor_config['size'][1]
    
    # Convert tensors to Pint quantities for active source
    albedo_quantity = albedo_frames * ureg.dimensionless
    depth_quantity = depth_frames * ureg.meter

    # print("albedo_quantity", albedo_quantity.detach().cpu().min(), albedo_quantity.detach().cpu().max())
    # print("depth_quantity", depth_quantity.detach().cpu().min(), depth_quantity.detach().cpu().max())

    radiance = active_source.get_scene_radiance(
        albedo_quantity, 
        depth_quantity, 
        num_pixels, 
        sensor.omega)
    # irradiance = (radiance * torch.pi / 4 * (1 / sensor_config['f_number']) ** 2) * float(sensor_config['pixel_pitch']**2)
    irradiance = (radiance * torch.pi / 4 * (1 / sensor.f_number) ** 2).to(irradiance_photons) * (sensor.pixel_pitch.to(ureg.meter))**2
    irradiance_tensor = irradiance.magnitude

    # print("Number of photons per cycle: ", active_source.num_photons_per_cycle)
    # exit(-1)

    # Get ambient offset
    ambient_radiance = ambient_source.get_scene_radiance(sensor.omega, albedo_quantity, active_source.frequency)
    ambient_irradiance = (ambient_radiance * torch.pi / 4 * (1 / sensor.f_number) ** 2).to(irradiance_photons) * (sensor.pixel_pitch.to(ureg.meter))**2
    offset = ambient_irradiance.magnitude
    transients = calculate_transients(
        irradiance_tensor, 
        depth_frames, 
        offset,
        fov_masks, 
        hist_config['n_bins'], 
        active_source.max_resolvable_depth.magnitude)
    print("transients.shape: ", transients.shape)

    # Calculate arrival rates
    bin_width = (2 * tof2depth(1 / active_source.frequency) / hist_config['n_bins'])
    _, irf = active_source.get_kernel(bin_width)
    irf_tensor = torch.tensor(irf, dtype=float, device=device)
    arrival_rates = calculate_arrival_rates(irf_tensor, transients, offset, hist_config['n_bins'])
    
    active_source.plot_kernel(bin_width)

    # Simulate EWH with dead time
    ewh_list = simulate_ewh(arrival_rates, hist_config['n_pulses'], hist_config['n_bins'], hist_config['free_running'], float(hist_config['dead_time_s']))

    plot_ewh_per_pixel(config,
                    fov_masks,
                    albedo_frames[0],
                    depth_frames[0],
                    transients,
                    arrival_rates,
                    ewh_list)

    # Simulate Differentiable EWH (Does not support dead time)
    hist_config['dead_time_s'] = 0
    ewh_list_gt = simulate_ewh_diff(arrival_rates, hist_config['n_pulses'], hist_config['n_bins'], hist_config['free_running'], float(hist_config['dead_time_s']))

    plot_ewh_per_pixel(config,
                    fov_masks,
                    albedo_frames[0],
                    depth_frames[0],
                    transients,
                    arrival_rates,
                    ewh_list_gt)
    



