import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
from pathlib import Path

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.constants import c
from SPCSim.data_loaders.transient_loaders import TransientGenerator
from SPCSim.postproc.ewh_postproc import PostProcEWH
from SPCSim.sensors import BaseEWHSPC
from SPCSim.utils.plot_utils import plot_ewh, plot_transient
from torch import Tensor
from typing import List, Optional
from visionsim.dataset import Dataset, ImgDataset, NpyDataset


# Import from local modules
from sources import (
    PulsedLaser, Sun, LightConditions
)
from histogrammers_temp_new import (
    get_albedo_intensity_depth_frames, get_perpixel_fov_masks, 
    calculate_transients, calculate_arrival_rates, simulate_ewh
)
from utils import tof2depth, ureg    

def get_light_conditions_from_string(condition_str: str) -> LightConditions:
    """Convert string to LightConditions enum value."""
    return getattr(LightConditions, condition_str)

def preproc_albedo_intensity_depth_frames(frames_data: ImgDataset, 
                                          depths_data: NpyDataset,
                                          num_frames: int,
                                          config: dict,
                                          requires_grad: Optional[bool] = False):# -> List[Tensor]:
    """Function to convert the rgb and depth frames read using data loaders to tensors"

    Args:
        frames_data (ImgDataset): Dataset object to load RGB frames 
        depths_data (NpyDataset): Dataset object to load depth frames
        num_frames (int): Number of rgb-d frames used per single exposure time

    Returns:
        List[Tensor]: Tensors corresponding to albedo, intensity and depth frames
    """

    # Get config parameters
    Nr,Nc = config["sensor"]["resolution"]
    tmax = float(1.0/float(config['active_source']['pulsed_laser']['frequency']))*ureg.second
    n_bins = int(config["histogrammer"]["n_bins"])

    # Compute max depth
    max_depth = ((tmax*((c)*ureg.meter/ureg.second)/2)*1000)/ureg.meter

    print("Max depth in meters",max_depth)
    print("c",c)
    print("tmax", tmax)

    # tmax = 100  # Laser period in nano seconds
    
    rgb_img_list = list(frames[:num_frames][1])
    depth_img_list = list(depths[:num_frames][1])

    print("Inside rgb_imgs.shape",len(rgb_img_list))
    print("Inside depth_imgs.shape",len(depth_img_list))

    albedo_frames_list = []
    intensity_frames_list = []
    depth_frames_list = []

    for idx in range(len(rgb_img_list)):
        rgb_img = rgb_img_list[idx]
        depth_img = depth_img_list[idx][:,:,0]

        depth_img = cv2.inpaint(depth_img, (depth_img > max_depth).astype(np.uint8), 3, cv2.INPAINT_TELEA)

        # Resize and transform to tensor, scale RGB to [0-1] range
        rgb_img = cv2.resize(rgb_img, (Nc, Nr))
        rgb = torch.tensor(rgb_img.astype(float), device=device, requires_grad = requires_grad) / 255.0
        depth_img = cv2.resize(depth_img, (Nc, Nr))
        
        # Filter out depths that might be out-of-range
        depth = torch.tensor(depth_img, device=device, requires_grad = requires_grad).unsqueeze(0)

        # Using the red channel as albedo and intensity
        albedo = intensity = rgb[..., 0].unsqueeze(0)

        # print("Idx: ",idx, "depth: ", depth.shape, depth.min(), depth.max(), depth.mean())
        # print("Idx: ",idx, "albedo: ", albedo.shape, albedo.min(), albedo.max(), albedo.mean())
        # print("Idx: ",idx, "intensity: ", intensity.shape, intensity.min(), intensity.max(), intensity.mean())

        albedo_frames_list.append(albedo)
        intensity_frames_list.append(intensity)
        depth_frames_list.append(depth)

    albedo_frames = torch.concat(albedo_frames_list)
    intensity_frames = torch.concat(intensity_frames_list)
    depth_frames = torch.concat(depth_frames_list) 

    return albedo_frames, intensity_frames, depth_frames
    




if __name__ == "__main__":
    
    root = Path("../../../examples/renders/scene1/")
    frames = Dataset.from_path(root / "frames")
    depths = Dataset.from_path(root / "depths")
    assert len(depths) == len(frames), "Different number of depth and RGB frames"
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    requires_grad = False
    num_frames = 1

    data_dir = "data"
    config_path = "config.yaml"

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    albedo_frames, intensity_frames, depth_frames = preproc_albedo_intensity_depth_frames(frames, depths, num_frames, config, requires_grad)

    # Active source
    active_config = config['active_source']['pulsed_laser']
    active_source = PulsedLaser(
        wavelength=float(active_config['wavelength']) * ureg.nanometer,
        frequency=float(active_config['frequency']) * ureg.hertz,
        pulse_width=float(active_config['pulse_width']) * ureg.second,
        avg_watts=float(active_config['avg_watts']) * ureg.watt,
        pulse_shape=active_config['pulse_shape'],
        pulse_shape_custom=active_config['pulse_shape_custom']
    )

    # Ambient source
    ambient_config = config['ambient_source']['sun']
    ambient_source = Sun(
        intensity=float(ambient_config['intensity']) * ureg.watt / ureg.meter**2,
        stability_factor=float(ambient_config['stability_factor']) * ureg.dimensionless,
        temperature=float(ambient_config['temperature']) * ureg.kelvin,
        lambda_pass=float(ambient_config['lambda_pass']) * ureg.nanometer,
        delta_lambda=float(ambient_config['delta_lambda']) * ureg.nanometer,
        light_conditions=get_light_conditions_from_string(ambient_config['light_conditions'])
    )

    # FOV masks
    _, img_rows, img_cols = depth_frames.shape
    empty_mask = torch.zeros((img_rows, img_cols), dtype=bool)
    fov_masks = get_perpixel_fov_masks(empty_mask, config['histogrammer']['pixel_fov_list'], device=device)
    print("fov_masks", fov_masks.shape)

    # Get transients
    sensor_config = config['sensor']
    hist_config = config['histogrammer']
    num_pixels = sensor_config['resolution'][0] * sensor_config['resolution'][1]
    # Convert tensors to Pint quantities for active source
    albedo_quantity = albedo_frames * ureg.dimensionless
    depth_quantity = depth_frames * ureg.meter 
    radiance = active_source.get_scene_radiance(albedo_quantity, depth_quantity, num_pixels, float(sensor_config['omega']) * ureg.steradian)
    irradiance = (radiance * torch.pi / 4 * (1 / sensor_config['f_number']) ** 2) * float(sensor_config['pixel_pitch']**2)
    irradiance_tensor = irradiance.magnitude

    # Get ambient offset
    ambient_radiance = ambient_source.get_scene_radiance(float(sensor_config['omega']) * ureg.steradian, albedo_quantity, active_source.frequency)
    ambient_irradiance = (ambient_radiance * torch.pi / 4 * (1 / sensor_config['f_number']) ** 2) * float(sensor_config['pixel_pitch']**2)
    offset = ambient_irradiance.magnitude
    transients = calculate_transients(irradiance_tensor, depth_frames, fov_masks, hist_config['n_bins'], active_source.max_resolvable_depth.magnitude)

    # Calculate arrival rates
    # bin_width = hist_config['bin_width'] * ureg.meter
    bin_width = (2 * tof2depth(1 / active_source.frequency) / hist_config['n_bins'])
    _, irf = active_source.get_kernel(bin_width)
    irf_tensor = torch.tensor(irf, dtype=float, device=device)
    arrival_rates = calculate_arrival_rates(irf_tensor, transients, offset, hist_config['n_bins'])
    
    active_source.plot_kernel(bin_width)

    # Simulate EWH
    ewh_list = simulate_ewh(arrival_rates, hist_config['n_pulses'], hist_config['n_bins'], hist_config['free_running'], float(hist_config['dead_time_s']))

    # Plots
    num_fovs = len(config['histogrammer']['pixel_fov_list'])
    # FOV Masks
    fig1, ax1 = plt.subplots(1, num_fovs, figsize=(3 * num_fovs, 3))
    fig1.suptitle("FOV Masks", fontsize=16)
    for i in range(num_fovs):
        current_ax = ax1 if num_fovs == 1 else ax1[i]
        current_ax.imshow(fov_masks[i].cpu().numpy(), cmap="gray")
        current_ax.set_title(f"FOV {i+1}")
        current_ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap
    plt.show()

    # Albedo values for the first frame
    fig2, ax2 = plt.subplots(1, num_fovs, figsize=(3 * num_fovs, 3))
    fig2.suptitle("Albedo Values (First Frame)", fontsize=16)
    for i in range(num_fovs):
        current_ax = ax2 if num_fovs == 1 else ax2[i]
        current_ax.imshow(albedo_frames[0].cpu().numpy() * fov_masks[i].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        current_ax.set_title(f"FOV {i+1}")
        current_ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Depth values for the first frame
    fig3, ax3 = plt.subplots(1, num_fovs, figsize=(3 * num_fovs, 3))
    fig3.suptitle("Depth Values (First Frame)", fontsize=16)
    for i in range(num_fovs):
        current_ax = ax3 if num_fovs == 1 else ax3[i]
        current_ax.imshow(depth_frames[0].cpu().numpy() * fov_masks[i].cpu().numpy(), cmap="viridis", vmin=0, vmax=10) # Assuming max depth of 10m based on 10.0/255.0 scaling
        current_ax.set_title(f"FOV {i+1}")
        current_ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

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
    fig6, ax6 = plt.subplots(num_fovs, 1, figsize=(8, 2.5 * num_fovs))
    fig6.suptitle("Simulated Time Stamp Histograms (EWH)", fontsize=16)
    for i in range(num_fovs):
        current_ax = ax6 if num_fovs == 1 else ax6[i]
        current_ax.plot(ewh_list[i].cpu().numpy())
        current_ax.set_ylim(bottom=0) # Ensure y-axis starts at 0
        current_ax.set_title(f"FOV {i+1}")
        current_ax.set_xlabel("Time Bins")
        current_ax.set_ylabel("Photon Counts")
        current_ax.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    

