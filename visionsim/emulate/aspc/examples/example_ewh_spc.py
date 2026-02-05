from pathlib import Path

import torch

from visionsim.emulate.aspc.camera import Camera
from visionsim.emulate.aspc.examples.ascp_plot_utils import plot_ewh_per_pixel

if __name__ == "__main__":

    data_dir = Path("examples/renders/scene1/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = "visionsim/emulate/aspc/examples/config_ewh.yaml"
    requires_grad = True

    camera = Camera(data_dir, config_path, device, requires_grad)

    transients, _ = camera.get_transients()
    arrival_rates = camera.get_arrival_rates()
    ewh_list = camera.get_ewh()

    plot_ewh_per_pixel(
        camera.histogrammer,
        camera.get_fov_masks(),
        camera.albedo_frames[0],
        camera.depth_frames[0].magnitude,
        transients,
        arrival_rates,
        ewh_list,
    )
