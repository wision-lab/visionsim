from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from pint import Quantity

from visionsim.emulate.aspc.ambient import Sun, get_light_conditions_from_string
from visionsim.emulate.aspc.detector import Detector
from visionsim.emulate.aspc.emitter import PulsedLaser
from visionsim.emulate.aspc.histogrammers import (
    HistConfig,
    Histogrammer,
    HistogrammerEDH,
)
from visionsim.emulate.aspc.sensors import SPADSensor


def _build_lidar(
    config: dict,
    device: str | torch.device = "cpu",
) -> tuple[Detector, PulsedLaser | None, Sun | None]:
    """Parses configuration dictionary and instantiates Detector, PulsedLaser emitter, and Sun."""
    cfg = deepcopy(config)

    # 1. Active Emitter Source (PulsedLaser)
    active_cfg = cfg.get("active_source", {}).get("pulsed_laser", {})
    active_enabled = active_cfg.pop("enabled", True)
    emitter = PulsedLaser(**active_cfg) if active_enabled else None

    # 2. Ambient Scene Source (Sun)
    ambient_cfg = cfg.get("ambient_source", {}).get("sun", {})
    ambient_enabled = ambient_cfg.pop("enabled", True)

    if "light_conditions" in ambient_cfg and isinstance(ambient_cfg["light_conditions"], str):
        ambient_cfg["light_conditions"] = get_light_conditions_from_string(ambient_cfg["light_conditions"])

    scene = Sun(**ambient_cfg) if ambient_enabled else None

    # 3. Sensor & Histogrammer Configuration
    sensor_cfg = cfg.get("sensor", {})
    sensor = SPADSensor(**sensor_cfg)

    hist_cfg_dict = cfg.get("histogrammer", {})
    hist_type = hist_cfg_dict.pop("type", "ewh")
    pixel_fov_list = hist_cfg_dict.pop("pixel_fov_list", None)

    hist_config_obj = HistConfig(**hist_cfg_dict)

    if hist_type == "edh":
        histogrammer = HistogrammerEDH(config=hist_config_obj)
    else:
        histogrammer = Histogrammer(config=hist_config_obj)

    detector = Detector(
        sensor=sensor,
        histogrammer=histogrammer,
        device=device,
        pixel_fov_list=pixel_fov_list,
    )

    return detector, emitter, scene


class ASPCEmulator:
    """High-level Orchestrator for Single-Photon LiDAR Simulation."""

    def __init__(
        self,
        config: dict,
        device: str | torch.device = "cpu",
    ):
        self.config = config
        self.device = torch.device(device) if isinstance(device, str) else device
        self.detector, self.emitter, self.scene = _build_lidar(config, device=self.device)

    def process_frame(
        self,
        depth_frame: Quantity | torch.Tensor,
        albedo_frame: Quantity | torch.Tensor,
        ambient_frame: Quantity | torch.Tensor | None = None,
    ) -> np.ndarray:
        """Executes complete forward simulation for input depth, albedo, and optional ambient frame."""
        if self.emitter is None:
            raise RuntimeError("ASPCEmulator requires an active emitter (PulsedLaser) to process frames.")

        _transients, _ = self.detector.get_transients(
            depth_frame=depth_frame,
            albedo_frame=albedo_frame,
            ambient_frame=ambient_frame,
            emitter=self.emitter,
            scene=self.scene,
        )
        _ = self.detector.get_arrival_rates(emitter=self.emitter)
        ewh_list = self.detector.get_ewh(emitter=self.emitter)

        return torch.stack(ewh_list).detach().cpu().numpy()
