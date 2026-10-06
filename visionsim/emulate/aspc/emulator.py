from pathlib import Path

import torch

from .camera import Camera


class ASPCEmulator:
    def __init__(
        self,
        base_cfg: Path | dict,
        config_overrides: dict | None = None,
        device: str | torch.device = "cpu",
    ):
        self.base_cfg = base_cfg
        self.config_overrides = config_overrides or {}

        # Instantiate Camera passing base config and CLI overrides
        self.camera = Camera(
            config_path=self.base_cfg,
            config_overrides=self.config_overrides,
            device=device,
        )

    def process_frame(self, depth_frame, albedo_frame):
        transients, _ = self.camera.get_transients(depth_frame,albedo_frame)
        _ = self.camera.get_arrival_rates()
        ewh_list = self.camera.get_ewh()
        return torch.stack(ewh_list).detach().cpu().numpy()