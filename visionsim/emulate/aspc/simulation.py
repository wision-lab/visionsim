from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from pint import Quantity

from visionsim.emulate.aspc.detector import Detector
from visionsim.emulate.aspc.histogrammers import (
    Histogrammer,
)
from visionsim.emulate.aspc.sensors import SPADSensor

def _build_lidar(
    device: str | torch.device = "cpu",
):
    sensor = SPADSensor()
    histogrammer = Histogrammer()
    detector = Detector(
        histogrammer=histogrammer,
        device=device,
    )
    return detector


class ASPCEmulator:
    """High-level Orchestrator for Single-Photon LiDAR Simulation."""

    def __init__(
        self,

        device: str | torch.device = "cpu",
    ):

        self.device = torch.device(device) if isinstance(device, str) else device
        self.detector = _build_lidar()

    def process_frame(
        self,
        depth_frame: Quantity | torch.Tensor,
    ) -> np.ndarray:

        _transients, _ = self.detector.get_transients(
            depth_frame=depth_frame,
        )
        _ = self.detector.get_arrival_rates()
        ewh_list = self.detector.get_ewh()

        return torch.stack(ewh_list).detach().cpu().numpy()
