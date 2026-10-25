from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from pint import Quantity

from visionsim.emulate.aspc.ambient import Sun
from visionsim.emulate.aspc.emitter import PulsedLaser
from visionsim.emulate.aspc.histogrammers import HistogrammerBase
from visionsim.emulate.aspc.sensors import SPADSensor
from visionsim.emulate.aspc.utils import (
    irradiance_photons,
    ureg,
)


class Detector:
    """
    Unified Detector module combining physical camera optics (SPADSensor),
    spatial pixel FOV masking, and readout processing (Histogrammer).
    """

    def __init__(
        self,
        sensor: SPADSensor,
        histogrammer: HistogrammerBase,
        device: str | torch.device = "cpu",
        pixel_fov_list: list[list[float]] | str | Path | None = None,
    ):
        self.sensor = sensor
        self.histogrammer: Any = histogrammer
        self.device = torch.device(device) if isinstance(device, str) else device

        # Load or set registered FOV coordinates
        parsed_fov_list: list[list[float]]
        if isinstance(pixel_fov_list, (str, Path)):
            path = Path(pixel_fov_list)
            if path.exists():
                raw_data = np.loadtxt(path, dtype=np.float32).tolist()
                if isinstance(raw_data, list) and len(raw_data) > 0 and isinstance(raw_data[0], (float, int)):
                    parsed_fov_list = [raw_data]
                else:
                    parsed_fov_list = raw_data
            else:
                parsed_fov_list = [[0.0, 1.0, 0.0, 1.0]]
        elif pixel_fov_list is None:
            parsed_fov_list = [[0.0, 1.0, 0.0, 1.0]]
        else:
            parsed_fov_list = pixel_fov_list

        self.pixel_fov_list: list[list[float]] = parsed_fov_list

        # Cached simulation state
        self.transients: torch.Tensor | None = None
        self.arrival_rates: torch.Tensor | None = None
        self.ambient_offsets: list[torch.Tensor] | torch.Tensor | None = None

    def build_pixel_fov_list(
        self,
        per_pixel_fov: Quantity | tuple[Quantity, Quantity],
        output_path: str | None = None,
    ) -> list:
        """Builds normalized FOV bounds [r1, r2, c1, c2] for every pixel in sensor array."""
        if isinstance(per_pixel_fov, Quantity):
            fov_h = per_pixel_fov
            fov_w = per_pixel_fov
        else:
            fov_h, fov_w = per_pixel_fov[0], per_pixel_fov[1]

        fov_w_ratio = (fov_w / self.sensor.fov_x).to(ureg.dimensionless).magnitude
        fov_h_ratio = (fov_h / self.sensor.fov_y).to(ureg.dimensionless).magnitude

        pixel_fov_list = []
        for row in range(self.sensor.h):
            cy = (row + 0.5) / self.sensor.h
            for col in range(self.sensor.w):
                cx = (col + 0.5) / self.sensor.w
                r1 = max(0.0, cy - fov_h_ratio / 2.0)
                r2 = min(1.0, cy + fov_h_ratio / 2.0)
                c1 = max(0.0, cx - fov_w_ratio / 2.0)
                c2 = min(1.0, cx + fov_w_ratio / 2.0)
                pixel_fov_list.append([r1, r2, c1, c2])

        if output_path:
            np.savetxt(output_path, np.array(pixel_fov_list, dtype=np.float32), fmt="%.6f", delimiter=" ")

        self.pixel_fov_list = pixel_fov_list
        return pixel_fov_list

    def get_pixel_fov_mask(
        self,
        empty_mask: torch.Tensor,
        row1: float,
        row2: float,
        col1: float,
        col2: float,
        vignette: bool = True,
    ) -> torch.Tensor:
        """Generates a 2D spatial FOV mask with optional smooth vignetting."""
        img_rows, img_cols = empty_mask.shape

        r1 = max(0, min(img_rows, round(row1 * img_rows)))
        r2 = max(0, min(img_rows, round(row2 * img_rows)))
        c1 = max(0, min(img_cols, round(col1 * img_cols)))
        c2 = max(0, min(img_cols, round(col2 * img_cols)))

        mask = torch.zeros((img_rows, img_cols), dtype=torch.float32, device=self.device)

        if r1 == r2 or c1 == c2:
            return mask

        if vignette:
            rr = torch.arange(r1, r2, device=self.device)
            cc = torch.arange(c1, c2, device=self.device)
            yy, xx = torch.meshgrid(rr, cc, indexing="ij")

            cy = (r1 + r2 - 1) / 2.0
            cx = (c1 + c2 - 1) / 2.0
            half_h = max((r2 - r1) / 2.0, 1e-6)
            half_w = max((c2 - c1) / 2.0, 1e-6)

            dy = (yy - cy) / half_h
            dx = (xx - cx) / half_w
            radial = torch.sqrt(dx * dx + dy * dy)

            vignette_weights = torch.clamp(1.0 - torch.pow(radial, 2.0), 0.0, 1.0)
            mask[r1:r2, c1:c2] = vignette_weights
        else:
            mask[r1:r2, c1:c2] = 1.0

        return mask

    def get_perpixel_fov_masks(self, empty_mask: torch.Tensor, vignette: bool = True) -> torch.Tensor:
        """Generates stacked FOV mask tensors for all registered pixel FOVs."""
        mask_list = []
        for r1, r2, c1, c2 in self.pixel_fov_list:
            mask = self.get_pixel_fov_mask(empty_mask, r1, r2, c1, c2, vignette=vignette)
            mask_list.append(mask.unsqueeze(0))

        return torch.concat(mask_list)

    def _get_signal(self, depth_frame: torch.Tensor, albedo_frame: torch.Tensor, emitter: PulsedLaser) -> torch.Tensor:
        """Converts active laser radiance to focal plane irradiance (#photons)."""
        num_pixels = self.sensor.w * self.sensor.h
        radiance = emitter.get_scene_radiance(albedo_frame, depth_frame, num_pixels, self.sensor.omega)
        irradiance = (radiance * torch.pi / 4 * (1 / self.sensor.f_number) ** 2).to(irradiance_photons) * (
            self.sensor.pixel_pitch.to(ureg.meter)
        ) ** 2

        return torch.as_tensor(irradiance.magnitude, dtype=torch.float32, device=self.device)

    def _get_ambient_offset(
        self,
        albedo_frame: torch.Tensor | Quantity,
        emitter: PulsedLaser,
        scene: Sun | None = None,
        ambient_frame: torch.Tensor | Quantity | None = None,
    ) -> torch.Tensor:
        """Converts ambient background radiance to focal plane offset irradiance (#photons)."""
        # 1. Primary path: Use rendered ambient spatial frame if provided
        if ambient_frame is not None:
            if hasattr(ambient_frame, "units"):
                try:
                    ambient_irradiance = (ambient_frame * torch.pi / 4 * (1 / self.sensor.f_number) ** 2).to(
                        irradiance_photons
                    ) * (self.sensor.pixel_pitch.to(ureg.meter)) ** 2
                    return torch.as_tensor(ambient_irradiance.magnitude, dtype=torch.float32, device=self.device)
                except (AttributeError, TypeError, ValueError):
                    raw_ambient = ambient_frame.magnitude
                    return torch.as_tensor(raw_ambient, dtype=torch.float32, device=self.device)
            else:
                return torch.as_tensor(ambient_frame, dtype=torch.float32, device=self.device)

        # 2. Secondary path: Fall back to analytical Sun model
        elif scene is not None:
            ambient_radiance = scene.get_scene_radiance(albedo_frame, emitter.pulse_repetition)
            ambient_irradiance = (ambient_radiance * torch.pi / 4 * (1 / self.sensor.f_number) ** 2).to(
                irradiance_photons
            ) * (self.sensor.pixel_pitch.to(ureg.meter)) ** 2

            return torch.as_tensor(ambient_irradiance.magnitude, dtype=torch.float32, device=self.device)

        # 3. Default path: No ambient source
        else:
            raw_ref = albedo_frame.magnitude if hasattr(albedo_frame, "magnitude") else albedo_frame
            return torch.zeros(raw_ref.shape, dtype=torch.float32, device=self.device)

    def get_transients(
        self,
        depth_frame: torch.Tensor | Quantity,
        albedo_frame: torch.Tensor | Quantity,
        emitter: PulsedLaser,
        scene: Sun | None = None,
        ambient_frame: torch.Tensor | Quantity | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Calculates spatial-temporal transients over laser ToF bins."""
        irradiance = self._get_signal(depth_frame, albedo_frame, emitter)
        offsets = self._get_ambient_offset(
            albedo_frame=albedo_frame,
            emitter=emitter,
            scene=scene,
            ambient_frame=ambient_frame,
        )

        empty_ref = depth_frame.magnitude if hasattr(depth_frame, "magnitude") else depth_frame
        if empty_ref.dim() == 3:
            empty_ref = empty_ref[0]

        vignette_flag = getattr(self.histogrammer, "vignette", True)
        fov_masks = self.get_perpixel_fov_masks(empty_ref, vignette=vignette_flag)

        transients, ambient_offsets = self.histogrammer.calculate_transients(
            irradiance_frames=irradiance,
            depth_frames=depth_frame,
            offsets=offsets,
            fov_masks=fov_masks,
            gt_ntime_bins=self.histogrammer.n_bins,
            max_depth=emitter.max_resolvable_depth.magnitude,
            sensor_fov=self.sensor.fov,
            pixel_fov_list=self.pixel_fov_list,
            w=self.sensor.w,
            h=self.sensor.h,
            omega=self.sensor.omega,
        )

        self.transients = transients
        self.ambient_offsets = ambient_offsets
        return transients, ambient_offsets

    def get_arrival_rates(self, emitter: PulsedLaser) -> torch.Tensor:
        """Convolves transients with IRF pulse kernel and adds background offset noise."""
        if self.transients is None:
            raise RuntimeError("Transients not computed. Call get_transients() first.")

        bin_width = emitter.max_resolvable_depth / self.histogrammer.n_bins

        configured = getattr(self.histogrammer, "bin_width", None)
        if configured is not None:
            ratio = (configured / bin_width).to(ureg.dimensionless).magnitude
            if not np.isclose(ratio, 1.0, rtol=5e-2):
                raise ValueError(
                    f"Configured histogrammer.bin_width ({configured}) disagrees with "
                    f"implied bin width ({bin_width}) from laser pulse_repetition and n_bins."
                )

        _, irf = emitter.get_kernel(bin_width, normalize="sum")
        irf_tensor = torch.tensor(irf, dtype=torch.float32, device=self.device)

        arrival_rates = self.histogrammer.calculate_arrival_rates(
            irf_tensor, self.transients, self.ambient_offsets, self.histogrammer.n_bins
        )

        self.arrival_rates = arrival_rates
        return arrival_rates

    def get_ewh(self, emitter: PulsedLaser) -> list[torch.Tensor]:
        """Simulates SPAD Equi-Width Histogram (EWH) output."""
        if self.arrival_rates is None:
            raise RuntimeError("Arrival rates not computed. Call get_arrival_rates() first.")

        dead_time_bins = int(
            np.round(
                self.histogrammer.dead_time.to(ureg.second).magnitude
                * self.histogrammer.n_bins
                * emitter.pulse_repetition.to(ureg.hertz).magnitude
            )
        )

        ewh_list = self.histogrammer.simulate_ewh(
            arrival_rates=self.arrival_rates,
            n_pulses=self.histogrammer.n_pulses,
            n_bins=self.histogrammer.n_bins,
            free_running=self.histogrammer.free_running,
            dead_time_bins=dead_time_bins,
            fast_sim=self.histogrammer.fast_sim,
        )
        return ewh_list
