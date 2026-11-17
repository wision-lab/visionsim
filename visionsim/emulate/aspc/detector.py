from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from visionsim.emulate.aspc.histogrammers import HistogrammerBase
from visionsim.emulate.aspc.sensors import SPADSensor


class Detector:
    """
    Unified Detector module combining physical camera optics (SPADSensor),
    spatial pixel FOV masking, and readout processing (Histogrammer).
    """

    def __init__(
        self,
        histogrammer: HistogrammerBase,
        sensor: SPADSensor,
        device: str | torch.device = "cpu",
    ):

        self.histogrammer: Any = histogrammer
        self.device = torch.device(device) if isinstance(device, str) else device


        # Cached simulation state
        self.transients: torch.Tensor | None = None
        self.arrival_rates: torch.Tensor | None = None
        self.ambient_offsets: list[torch.Tensor] | torch.Tensor | None = None


    def get_transients(
        self,
        depth_frame,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:

        transients, ambient_offsets = self.histogrammer.calculate_transients(
            depth_frames=depth_frame,
            gt_ntime_bins=self.histogrammer.n_bins,
            sensor_fov=self.sensor.fov,
            w=self.sensor.w,
            h=self.sensor.h,
        )
        self.transients = transients
        self.ambient_offsets = ambient_offsets
        return transients, ambient_offsets

    def get_arrival_rates(self) -> torch.Tensor:
        """Convolves transients with IRF pulse kernel and adds background offset noise."""

        bin_width = 10 ## NanoSeconds
        
        ### IRF FIXED
        bin_width_m = 3
        pulse_width_m = 1
        half_num = max(3 * self.histogrammer.n_bins, 5)
        x = np.arange(-half_num, half_num + 1) * bin_width_m
        sigma =  pulse_width_m / 2.35482
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        irf = kernel / np.sum(kernel)
        irf_tensor = torch.tensor(irf, dtype=torch.float32, device=self.device)

        arrival_rates = self.histogrammer.calculate_arrival_rates(
            irf_tensor, self.transients, self.histogrammer.n_bins
        )

        self.arrival_rates = arrival_rates
        return arrival_rates

    def get_ewh(self) -> list[torch.Tensor]:
        """Simulates SPAD Equi-Width Histogram (EWH) output."""

        dead_time_bins = 10

        ewh_list = self.histogrammer.simulate_ewh(
            arrival_rates=self.arrival_rates,
            n_pulses=self.histogrammer.n_pulses,
            n_bins=self.histogrammer.n_bins,
            free_running=self.histogrammer.free_running,
            dead_time_bins=dead_time_bins,
        )
        return ewh_list