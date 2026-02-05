from pathlib import Path

import torch

from visionsim.emulate.aspc.camera import Camera

if __name__ == "__main__":
    data_dir = Path("examples/renders/scene1/")
    config_path = "visionsim/emulate/aspc/config.yaml"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    camera = Camera(data_dir, config_path, device)

    transients = camera.get_transients()
    arrival_rates = camera.get_arrival_rates()
    ewh_list = camera.get_ewh()

    # Plots
    camera.plot_all()
