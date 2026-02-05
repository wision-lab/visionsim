import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from ruamel.yaml import YAML

from visionsim.emulate.aspc.histogrammers import HistConfig, Histogrammer
from visionsim.emulate.aspc.sensors import SPADSensor
from visionsim.emulate.aspc.sources import LightConditions, PulsedLaser, Sun
from visionsim.emulate.aspc.utils import (
    irradiance_photons,
    preproc_albedo_intensity_depth_frames,
    tof2depth,
    ureg,
    yaml_constructor,
)


class Camera:
    """Camera class for ASPC simulation"""
    def __init__(self, data_path, config_path, device):
        """Initialize Camera"""
        self.device = device
        self.config = self._load_config(config_path)
        self.albedo_frames, self.intensity_frames, self.depth_frames = self._load_data(data_path)

        # Active source
        active_config = self.config["active_source"]["pulsed_laser"]
        active_enable = active_config.pop("enabled")
        if active_enable:
            self.active_source = PulsedLaser(**active_config)

        # Ambient source
        ambient_config = self.config["ambient_source"]["sun"]
        ambient_config["light_conditions"] = self._get_light_conditions_from_string(ambient_config["light_conditions"])
        ambient_enable = ambient_config.pop("enabled")
        if ambient_enable:
            self.ambient_source = Sun(**ambient_config)

        # Histogrammer
        hist_config = HistConfig(**self.config["histogrammer"])
        self.histogrammer = Histogrammer(hist_config)

        # Sensor
        sensor_config = self.config["sensor"]
        self.sensor = SPADSensor(**sensor_config)

    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        yaml = YAML()
        safe_builtins = {"__builtins__": {"list": list, "dict": dict, "tuple": tuple}, "np": np, "math": math}
        yaml.Constructor.add_constructor(tag="!Quantity", constructor=yaml_constructor(ureg.Quantity))
        yaml.Constructor.add_constructor(tag="!expr", constructor=yaml_constructor(eval, safe_builtins))
        yaml.Constructor.add_constructor(tag="!file", constructor=yaml_constructor(config_path))
        return yaml.load(open(config_path))

    def _load_data(self, data_path):
        """Load data from directory"""
        return preproc_albedo_intensity_depth_frames(root=data_path, device=self.device, config=self.config, start_idx=0, num_frames=1, requires_grad=False)

    def _get_light_conditions_from_string(self, condition_str):
        """Convert string to LightConditions enum value."""
        return getattr(LightConditions, condition_str)

    def _get_fov_masks(self):
        """Get FOV masks"""
        _, img_rows, img_cols = self.depth_frames.shape
        empty_mask = torch.zeros((img_rows, img_cols), dtype=torch.float32, device=self.device)
        fov_masks = self.histogrammer.get_perpixel_fov_masks(empty_mask, self.histogrammer.pixel_fov_list, device=self.device)
        return fov_masks

    def _get_signal(self):
        """Get signal from active source"""
        num_pixels = self.sensor.w * self.sensor.h
        radiance = self.active_source.get_scene_radiance(self.albedo_frames, self.depth_frames, num_pixels, self.sensor.omega)
        irradiance = (radiance * torch.pi / 4 * (1 / self.sensor.f_number) ** 2).to(irradiance_photons) * (
            self.sensor.pixel_pitch.to(ureg.meter)
        ) ** 2
        irradiance = torch.tensor(irradiance.magnitude, dtype=torch.float32, device=self.device)
        return irradiance

    def _get_ambient_offset(self):
        """Get ambient offset from ambient source"""
        ambient_radiance = self.ambient_source.get_scene_radiance(self.sensor.omega, self.albedo_frames, self.active_source.frequency)
        ambient_irradiance = (ambient_radiance * torch.pi / 4 * (1 / self.sensor.f_number) ** 2).to(irradiance_photons) * (
            self.sensor.pixel_pitch.to(ureg.meter)
        ) ** 2
        offsets = torch.tensor(ambient_irradiance.magnitude, dtype=torch.float32, device=self.device)
        return offsets

    def get_transients(self):
        """Get transient data from histogrammer"""
        irradiance = self._get_signal()
        offsets = self._get_ambient_offset()
        fov_masks = self._get_fov_masks()
        return self.histogrammer.calculate_transients(
            irradiance,
            self.depth_frames,
            offsets,
            fov_masks,
            self.histogrammer.n_bins,
            self.active_source.max_resolvable_depth.magnitude,
            self.sensor.fov,
            self.histogrammer.pixel_fov_list,
            self.sensor.w,
            self.sensor.h,
            self.sensor.omega,
        )

    def get_arrival_rates(self):
        """Get arrival rates from histogrammer"""
        transients, ambient_offsets = self.get_transients()
        bin_width = 2 * tof2depth(1 / self.active_source.frequency) / self.histogrammer.n_bins
        _, irf = self.active_source.get_kernel(bin_width, None)
        irf_tensor = torch.tensor(irf, dtype=torch.float32, device=self.device)
        arrival_rates = self.histogrammer.calculate_arrival_rates(irf_tensor, transients, ambient_offsets, self.histogrammer.n_bins)
        # self.active_source.plot_kernel(bin_width)
        return arrival_rates

    def get_ewh(self):
        """Get EWH from histogrammer"""
        arrival_rates = self.get_arrival_rates()
        dead_time_bins = int(self.histogrammer.dead_time_s * self.histogrammer.n_bins * self.active_source.frequency)
        ewh_list = self.histogrammer.simulate_ewh(
            arrival_rates,
            self.histogrammer.n_pulses,
            self.histogrammer.n_bins,
            self.histogrammer.free_running,
            dead_time_bins,
        )
        return ewh_list

    ## Plotting

    def _plot_fov_masks(self, num_fovs, fov_masks):
        """Plot FOV masks"""
        fig, ax = plt.subplots(1, num_fovs, figsize=(3 * num_fovs, 3))
        fig.suptitle("FOV Masks", fontsize=16)
        for i in range(num_fovs):
            ax[i].imshow(fov_masks[i].cpu().numpy(), cmap="gray")
            ax[i].set_title(f"FOV {i + 1}")
            ax[i].axis("off")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def _plot_albedo_frames(self, num_fovs, fov_masks):
        """Plot albedo frames"""
        fig, ax = plt.subplots(1, num_fovs, figsize=(3 * num_fovs, 3))
        fig.suptitle("Albedo Values (First Frame)", fontsize=16)
        for i in range(num_fovs):
            ax[i].imshow(self.albedo_frames[0].cpu().numpy() * fov_masks[i].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax[i].set_title(f"FOV {i + 1}")
            ax[i].axis("off")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def _plot_depth_frames(self, num_fovs, fov_masks):
        """Plot depth frames"""
        fig, ax = plt.subplots(1, num_fovs, figsize=(3 * num_fovs, 3))
        fig.suptitle("Depth Values (First Frame)", fontsize=16)
        for i in range(num_fovs):
            ax[i].imshow(self.depth_frames[0].cpu().numpy() * fov_masks[i].cpu().numpy(), cmap="viridis", vmin=0, vmax=10)
            ax[i].set_title(f"FOV {i + 1}")
            ax[i].axis("off")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def _plot_transients(self, num_fovs, transients):
        """Plot transients"""
        fig, ax = plt.subplots(num_fovs, 1, figsize=(8, 2.5 * num_fovs))
        fig.suptitle("Transients", fontsize=16)
        for i in range(num_fovs):
            ax[i].plot(transients[i].detach().cpu().numpy())
            ax[i].set_title(f"FOV {i + 1}")
            ax[i].set_xlabel("Time Bins")
            ax[i].set_ylabel("Normalized Amplitude")
            ax[i].grid(True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def _plot_arrival_rates(self, num_fovs, arrival_rates):
        """Plot arrival rates"""
        fig, ax = plt.subplots(num_fovs, 1, figsize=(8, 2.5 * num_fovs))
        fig.suptitle(r"Photon Arrival Rates ($\overline{\Phi}$)", fontsize=16)
        for i in range(num_fovs):
            ax[i].plot(arrival_rates[i].detach().cpu().numpy())
            ax[i].set_ylim(bottom=0)
            ax[i].set_title(f"FOV {i + 1}")
            ax[i].set_xlabel("Time Bins")
            ax[i].set_ylabel("Rate (photons/bin)")
            ax[i].grid(True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def _plot_ewh(self, num_fovs, ewh_list):
        """Plot EWH"""
        fig, ax = plt.subplots(num_fovs, 1, figsize=(8, 2.5 * num_fovs))
        fig.suptitle("Simulated Time Stamp Histograms (EWH)", fontsize=16)
        for i in range(num_fovs):
            ax[i].plot(ewh_list[i].cpu().numpy())
            ax[i].set_ylim(bottom=0)
            ax[i].set_title(f"FOV {i + 1}")
            ax[i].set_xlabel("Time Bins")
            ax[i].set_ylabel("Photon Counts")
            ax[i].grid(True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_all(self):
        """Plot all"""
        transients, _ = self.get_transients()
        arrival_rates = self.get_arrival_rates()
        ewh_list = self.get_ewh()
        num_fovs = len(self.histogrammer.pixel_fov_list)
        fov_masks = self._get_fov_masks()
        self._plot_fov_masks(num_fovs, fov_masks)
        self._plot_albedo_frames(num_fovs, fov_masks)
        self._plot_depth_frames(num_fovs, fov_masks)
        self._plot_transients(num_fovs, transients)
        self._plot_arrival_rates(num_fovs, arrival_rates)
        self._plot_ewh(num_fovs, ewh_list)