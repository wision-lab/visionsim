import math
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from pint import Quantity
from ruamel.yaml import YAML

from visionsim.emulate.aspc.histogrammers import HistConfig, Histogrammer, HistogrammerEDH
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

    def __init__(self, data_path, config_path, device, requires_grad=False):
        """Initialize Camera"""
        self.device = device
        self.config = self._load_config(config_path)
        self.validate_config(self.config)
        self.requires_grad = requires_grad
        self.albedo_frames, self.intensity_frames, self.depth_frames = self._load_data(data_path, requires_grad)
        self._init_components_from_config(self.config)
        self.transients = None
        self.arrival_rates = None
        self.ambient_offsets = None
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        yaml = YAML()
        safe_builtins = {"__builtins__": {"list": list, "dict": dict, "tuple": tuple}, "np": np, "math": math}
        yaml.Constructor.add_constructor(tag="!Quantity", constructor=yaml_constructor(ureg.Quantity))
        yaml.Constructor.add_constructor(tag="!expr", constructor=yaml_constructor(eval, safe_builtins))
        yaml.Constructor.add_constructor(tag="!file", constructor=yaml_constructor(config_path))
        return yaml.load(open(config_path))

    def _merge_config(self, base_config: dict, overrides: dict | None) -> dict:
        """Deep-merge overrides into base_config without mutating either input."""
        if not overrides:
            return base_config

        def _deep_merge(dst, src):
            for k, v in src.items():
                if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
                    dst[k] = _deep_merge(deepcopy(dst[k]), v)
                else:
                    dst[k] = deepcopy(v)
            return dst

        return _deep_merge(deepcopy(base_config), overrides)

    def _init_components_from_config(self, config: dict):
        """Initialize sources, histogrammer, and sensor from the provided config."""
        # Active source
        active_config = config["active_source"]["pulsed_laser"]
        active_enable = active_config.pop("enabled")
        if active_enable:
            self.active_source = PulsedLaser(**active_config)

        # Ambient source
        ambient_config = config["ambient_source"]["sun"]
        ambient_config["light_conditions"] = self._get_light_conditions_from_string(ambient_config["light_conditions"])
        ambient_enable = ambient_config.pop("enabled")
        if ambient_enable:
            self.ambient_source = Sun(**ambient_config)

        # Histogrammer
        if "type" in config["histogrammer"] and config["histogrammer"]["type"] == "edh":
            self.histogrammer = HistogrammerEDH(HistConfig(**config["histogrammer"]))
        else:
            self.histogrammer = Histogrammer(HistConfig(**config["histogrammer"]))

        # Sensor
        self.sensor = SPADSensor(**config["sensor"])

    def reconfigure(self, config_overrides: dict):
        """Update camera configuration (deep merge) and rebuild sources/sensor/histogrammer."""
        self.config = self._merge_config(self.config, config_overrides)
        self._init_components_from_config(self.config)
        self.validate_config(self.config)
        return self

    def _load_data(self, data_path, requires_grad=False):
        """Load data from directory"""
        return preproc_albedo_intensity_depth_frames(
            root=data_path,
            device=self.device,
            config=self.config,
            start_idx=0,
            num_frames=1,
            requires_grad=requires_grad,
        )

    def validate_config(self, config):
        """Validate configuration"""
        max_resolvable_depth = tof2depth(1 / config["active_source"]["pulsed_laser"]["frequency"])
        # check max depth
        # Compare as Pint Quantities so units are handled correctly
        if config["histogrammer"]["max_depth"] > max_resolvable_depth:
            raise ValueError(
                f"Max depth in config {config['histogrammer']['max_depth']} is more than maximum resolvable depth {max_resolvable_depth}"
            )
        if config["histogrammer"]["n_bins"] * config["histogrammer"]["bin_width"] > max_resolvable_depth:
            raise ValueError(
                f"Bin width {config['histogrammer']['bin_width']} x number of bins {config['histogrammer']['n_bins']} is "
                f"{config['histogrammer']['bin_width'] * config['histogrammer']['n_bins']}, which is greater than max resolvable depth {max_resolvable_depth}"
            )
        # if config["histogrammer"]["fast_sim"] and not (config["histogrammer"]["free_running"]):
        #     raise ValueError("histogrammer.fast_sim = True is only supported for histogrammer.free_running = True mode")

    def _get_light_conditions_from_string(self, condition_str):
        """Convert string to LightConditions enum value."""
        return getattr(LightConditions, condition_str)

    def get_fov_masks(self, pixel_fov_list: list = None):
        """Get FOV masks"""
        _, img_rows, img_cols = self.depth_frames.shape
        empty_mask = torch.zeros((img_rows, img_cols), dtype=torch.float32, device=self.device)
        if pixel_fov_list is None:
            pixel_fov_list = self.histogrammer.pixel_fov_list
        fov_masks = self.histogrammer.get_perpixel_fov_masks(
            empty_mask, pixel_fov_list, device=self.device, vignette=self.histogrammer.vignette
        )
        return fov_masks

    def build_pixel_fov_list(
        self,
        per_pixel_fov: Quantity | tuple[Quantity, Quantity],
        output_path: str | None = None,
    ) -> list:
        """
        Build a pixel_fov_list for every pixel location given a per-pixel FOV size.

        Args:
            per_pixel_fov (float | tuple[float, float]): Per-pixel FOV in degrees.
                If a single value is provided, a square FOV is assumed.
            output_path (str | None): Optional file path to save the list
                (space-delimited text via `numpy.savetxt`).

        Returns:
            list: List of normalized rectangles in [row_min, row_max, col_min, col_max] order.
        """
        if isinstance(per_pixel_fov, (Quantity)):
            fov_h = per_pixel_fov
            fov_w = per_pixel_fov
        else:
            fov_h, fov_w = per_pixel_fov[0], per_pixel_fov[1]

        fov_w_ratio = (fov_w / self.sensor.fov_x).to(ureg.dimensionless).magnitude
        fov_h_ratio = (fov_h / self.sensor.fov_y).to(ureg.dimensionless).magnitude

        pixel_fov_list = []
        for row in range(self.sensor.h):
            cy = row / self.sensor.h
            for col in range(self.sensor.w):
                cx = col / self.sensor.w
                r1 = max(0.0, cy - fov_h_ratio / 2.0)
                r2 = min(1.0, cy + fov_h_ratio / 2.0)
                c1 = max(0.0, cx - fov_w_ratio / 2.0)
                c2 = min(1.0, cx + fov_w_ratio / 2.0)
                pixel_fov_list.append([r1, r2, c1, c2])

        if output_path:
            # Save the pixel FOV list in a space-delimited plain text file, one FOV per line as [r1 r2 c1 c2]
            np.savetxt(output_path, np.array(pixel_fov_list, dtype=np.float32), fmt="%.6f", delimiter=" ")
            print(f"Saved pixel_fov_list to: {output_path}")

        return pixel_fov_list

    def _get_signal(self):
        """Get signal from active source"""
        num_pixels = self.sensor.w * self.sensor.h
        radiance = self.active_source.get_scene_radiance(
            self.albedo_frames, self.depth_frames, num_pixels, self.sensor.omega
        )
        irradiance = (radiance * torch.pi / 4 * (1 / self.sensor.f_number) ** 2).to(irradiance_photons) * (
            self.sensor.pixel_pitch.to(ureg.meter)
        ) ** 2
        irradiance = torch.as_tensor(irradiance.magnitude, dtype=torch.float32, device=self.device)
        return irradiance

    def _get_ambient_offset(self):
        """Get ambient offset from ambient source"""
        ambient_radiance = self.ambient_source.get_scene_radiance(
            self.sensor.omega, self.albedo_frames, self.active_source.frequency
        )
        ambient_irradiance = (ambient_radiance * torch.pi / 4 * (1 / self.sensor.f_number) ** 2).to(
            irradiance_photons
        ) * (self.sensor.pixel_pitch.to(ureg.meter)) ** 2
        offsets = torch.as_tensor(ambient_irradiance.magnitude, dtype=torch.float32, device=self.device)
        return offsets

    def get_transients(self):
        """Get transient data from histogrammer"""
        irradiance = self._get_signal()
        offsets = self._get_ambient_offset()
        fov_masks = self.get_fov_masks()
        transients, ambient_offsets = self.histogrammer.calculate_transients(
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
        self.transients = transients
        self.ambient_offsets = ambient_offsets
        return transients, ambient_offsets

    def get_arrival_rates(self):
        """Get arrival rates from histogrammer"""
    
        bin_width = 2 * tof2depth(1 / self.active_source.frequency) / self.histogrammer.n_bins
        _, irf = self.active_source.get_kernel(bin_width, None)
        irf_tensor = torch.tensor(irf, dtype=torch.float32, device=self.device)
        arrival_rates = self.histogrammer.calculate_arrival_rates(
            irf_tensor, self.transients, self.ambient_offsets, self.histogrammer.n_bins
        )
        # self.active_source.plot_kernel(bin_width)
        self.arrival_rates = arrival_rates
        return arrival_rates

    def get_ewh(self):
        """Get EWH from histogrammer"""
       
        dead_time_bins = int(self.histogrammer.dead_time_s * self.histogrammer.n_bins*self.active_source.frequency)
        
        ewh_list = self.histogrammer.simulate_ewh(
            self.arrival_rates,
            self.histogrammer.n_pulses,
            self.histogrammer.n_bins,
            self.histogrammer.free_running,
            dead_time_bins,
            self.histogrammer.fast_sim
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
            ax[i].imshow(
                self.depth_frames[0].cpu().numpy() * (fov_masks[i].detach().cpu().numpy() > 0),
                cmap="viridis",
                vmin=0,
                vmax=10,
            )
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

    def plot_ewh(self, num_fovs, ewh_list):
        """Plot EWH"""
        fig, ax = plt.subplots(num_fovs, 1, figsize=(8, 2.5 * num_fovs))
        fig.suptitle("Simulated Time Stamp Histograms (EWH)", fontsize=16)
        ax = np.atleast_1d(ax)
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
        fov_masks = self.get_fov_masks()
        self._plot_fov_masks(num_fovs, fov_masks)
        self._plot_albedo_frames(num_fovs, fov_masks)
        self._plot_depth_frames(num_fovs, fov_masks)
        self._plot_transients(num_fovs, transients)
        self._plot_arrival_rates(num_fovs, arrival_rates)
        self.plot_ewh(num_fovs, ewh_list)
