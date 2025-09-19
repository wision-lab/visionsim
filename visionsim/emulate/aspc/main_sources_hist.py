import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from ruamel.yaml import YAML

# Import from local modules
from sources import (
    PulsedLaser, Sun, LightConditions
)
from histogrammers_temp_new import (
    get_albedo_intensity_depth_frames, get_perpixel_fov_masks, 
    calculate_transients, calculate_arrival_rates, simulate_ewh
)
from sensors import SPADSensor
from utils import tof2depth, ureg, load_config, from_yaml_constructor, get_masked_fov, irradiance_photons    

def get_light_conditions_from_string(condition_str: str) -> LightConditions:
    """Convert string to LightConditions enum value."""
    return getattr(LightConditions, condition_str)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = "./data"
    config_path = "./config.yaml"

    # Load config
    yaml = YAML()
    safe_builtins = {'__builtins__': {'list': list, 'dict': dict, 'tuple': tuple}, 'np': np, 'math': math}
    yaml.Constructor.add_constructor(tag="!Quantity", constructor=from_yaml_constructor(ureg.Quantity))
    yaml.Constructor.add_constructor(tag="!expr", constructor=from_yaml_constructor(eval, safe_builtins))
    config = yaml.load(open("sample_config.yaml"))

    # Load data
    albedo_frames, intensity_frames, depth_frames = get_albedo_intensity_depth_frames(data_dir, device=device)

    # Active source
    active_config = config['active_source']['pulsed_laser']
    active_enable = active_config.pop('enabled')
    if active_enable:
        active_source = PulsedLaser(**active_config)

    # Ambient source
    ambient_config = config['ambient_source']['sun']
    ambient_config['light_conditions'] = get_light_conditions_from_string(ambient_config['light_conditions'])
    ambient_enable = ambient_config.pop('enabled')
    if ambient_enable:
        ambient_source = Sun(**ambient_config)

    # FOV masks
    _, img_rows, img_cols = depth_frames.shape
    empty_mask = np.zeros((img_rows, img_cols), dtype=float)
    fov_masks = get_perpixel_fov_masks(empty_mask, config['histogrammer']['pixel_fov_list'], device=device)    

    # Histogrammer
    hist_config = config['histogrammer']

    # Sensor
    sensor_config = config['sensor']
    sensor_config['fov'] = get_masked_fov(sensor_config, hist_config['pixel_fov_list'])
    sensor = SPADSensor(**sensor_config)

    # Get transients
    num_pixels = sensor.w * sensor.h
    # Convert tensors to Pint quantities for active source
    albedo_quantity = albedo_frames.cpu().numpy() * ureg.dimensionless
    depth_quantity = depth_frames.cpu().numpy() * ureg.meter 
    radiance = active_source.get_scene_radiance(albedo_quantity, depth_quantity, num_pixels, sensor.omega)
    irradiance = (radiance * np.pi / 4 * (1 / sensor.f_number) ** 2).to(irradiance_photons) * (sensor.pixel_pitch.to(ureg.meter))**2
    print(f"pixel_pitch: {sensor.pixel_pitch}")
    print(f"radiance[0][0][0]: {radiance[0][0][0]}")
    print(f"irradiance[0][0][0]: {irradiance[0][0][0]}")
    irradiance_tensor = torch.tensor(np.array(irradiance.magnitude), dtype=torch.float32, device=device)
    # print(f"irradiance_tensor[0]: {irradiance_tensor[0][0][0]}")
    # print(f"irradiance_tensor[0].sum(): {irradiance_tensor[0].sum()}")
    # Get ambient offset
    ambient_radiance = ambient_source.get_scene_radiance(sensor.omega, albedo_quantity, active_source.frequency)
    ambient_irradiance = (ambient_radiance * np.pi / 4 * (1 / sensor.f_number) ** 2).to(irradiance_photons) * (sensor.pixel_pitch.to(ureg.meter))**2
    offsets = torch.tensor(np.array(ambient_irradiance.magnitude), dtype=torch.float32, device=device)
    transients, masked_offsets = calculate_transients(irradiance_tensor, depth_frames, offsets, fov_masks, hist_config['n_bins'], active_source.max_resolvable_depth.magnitude)
    # print(f"transients[0].sum(): {transients[0].sum()}")

    # Calculate arrival rates
    # bin_width = hist_config['bin_width'] * ureg.meter
    bin_width = (2 * tof2depth(1 / active_source.frequency) / hist_config['n_bins'])
    print(f"bin_width: {bin_width}")
    _, irf = active_source.get_kernel(bin_width, None)
    irf_tensor = torch.tensor(irf, dtype=torch.float32, device=device)
    arrival_rates = calculate_arrival_rates(irf_tensor, transients, masked_offsets, hist_config['n_bins'])
    active_source.plot_kernel(bin_width)

    # Simulate EWH
    # ewh_list = simulate_ewh(arrival_rates, hist_config['n_pulses'], hist_config['n_bins'], hist_config['free_running'], float(hist_config['dead_time_s']))

    # Plots
    num_fovs = len(config['histogrammer']['pixel_fov_list'])

    # # FOV Masks
    # fig1, ax1 = plt.subplots(1, num_fovs, figsize=(3 * num_fovs, 3))
    # fig1.suptitle("FOV Masks", fontsize=16)
    # for i in range(num_fovs):
    #     current_ax = ax1 if num_fovs == 1 else ax1[i]
    #     current_ax.imshow(fov_masks[i].cpu().numpy(), cmap="gray")
    #     current_ax.set_title(f"FOV {i+1}")
    #     current_ax.axis('off')
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap
    # plt.show()

    # # Albedo values for the first frame
    # fig2, ax2 = plt.subplots(1, num_fovs, figsize=(3 * num_fovs, 3))
    # fig2.suptitle("Albedo Values (First Frame)", fontsize=16)
    # for i in range(num_fovs):
    #     current_ax = ax2 if num_fovs == 1 else ax2[i]
    #     current_ax.imshow(albedo_frames[0].cpu().numpy() * fov_masks[i].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    #     current_ax.set_title(f"FOV {i+1}")
    #     current_ax.axis('off')
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

    # # Depth values for the first frame
    # fig3, ax3 = plt.subplots(1, num_fovs, figsize=(3 * num_fovs, 3))
    # fig3.suptitle("Depth Values (First Frame)", fontsize=16)
    # for i in range(num_fovs):
    #     current_ax = ax3 if num_fovs == 1 else ax3[i]
    #     current_ax.imshow(depth_frames[0].cpu().numpy() * fov_masks[i].cpu().numpy(), cmap="viridis", vmin=0, vmax=10) # Assuming max depth of 10m based on 10.0/255.0 scaling
    #     current_ax.set_title(f"FOV {i+1}")
    #     current_ax.axis('off')
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

    # Transients
    fig4, ax4 = plt.subplots(num_fovs, 1, figsize=(8, 2.5 * num_fovs))
    fig4.suptitle("Transients", fontsize=16)
    for i in range(num_fovs):
        current_ax = ax4 if num_fovs == 1 else ax4[i]
        current_ax.plot(transients[i].cpu().numpy())
        current_ax.set_title(f"FOV {i+1}")
        current_ax.set_xlabel("Time Bins")
        current_ax.set_ylabel("Normalized Amplitude")
        current_ax.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Arrival Rates
    fig5, ax5 = plt.subplots(num_fovs, 1, figsize=(8, 2.5 * num_fovs))
    fig5.suptitle(r'Photon Arrival Rates ($\overline{\Phi}$)', fontsize=16)
    for i in range(num_fovs):
        current_ax = ax5 if num_fovs == 1 else ax5[i]
        current_ax.plot(arrival_rates[i].cpu().numpy())
        current_ax.set_ylim(bottom=0) # Ensure y-axis starts at 0
        current_ax.set_title(f"FOV {i+1}")
        current_ax.set_xlabel("Time Bins")
        current_ax.set_ylabel("Rate (photons/bin)")
        current_ax.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Time Stamp Histograms (EWH)
    # fig6, ax6 = plt.subplots(num_fovs, 1, figsize=(8, 2.5 * num_fovs))
    # fig6.suptitle("Simulated Time Stamp Histograms (EWH)", fontsize=16)
    # for i in range(num_fovs):
    #     current_ax = ax6 if num_fovs == 1 else ax6[i]
    #     current_ax.plot(ewh_list[i].cpu().numpy())
    #     current_ax.set_ylim(bottom=0) # Ensure y-axis starts at 0
    #     current_ax.set_title(f"FOV {i+1}")
    #     current_ax.set_xlabel("Time Bins")
    #     current_ax.set_ylabel("Photon Counts")
    #     current_ax.grid(True)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()
    