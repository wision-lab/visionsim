from pathlib import Path

import torch

from visionsim.emulate.aspc.camera import Camera

# Import from local modules
from visionsim.emulate.aspc.examples.ascp_plot_utils import plot_edh_per_pixel

if __name__ == "__main__":
    ## Setting simulation parameters

    data_dir = Path("examples/renders/scene1/")
    config_path = "visionsim/emulate/aspc/examples/config_edh.yaml"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    requires_grad = True

    camera = Camera(data_dir, config_path, device, requires_grad)
    transients, _ = camera.get_transients()
    arrival_rates = camera.get_arrival_rates()

    # Simulate EDH with dead time
    assert camera.histogrammer.type == "edh", "Incorrect SPC type mentioned in config file"

    photon_hist_list, edh_list = camera.histogrammer.simulate_edh(
        arrival_rates,
        camera.histogrammer.n_pulses,
        camera.histogrammer.n_bins,
        camera.histogrammer.free_running,
        camera.histogrammer.dead_time_s.magnitude,
    )

    plot_edh_per_pixel(
        camera.histogrammer,
        camera.get_fov_masks(),
        camera.albedo_frames[0],
        camera.depth_frames[0],
        transients,
        arrival_rates,
        photon_hist_list,
        edh_list,
    )
