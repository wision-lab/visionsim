import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Import aspc package first to enable aspc.* imports
import visionsim.emulate.aspc  # noqa: F401
from visionsim.emulate.aspc.camera import Camera
from visionsim.emulate.aspc.utils import tof2depth


def forward_pass_ewh_diff(
    histogrammer,
    transients,
    ambient_offsets,
    irf_tensor,
):
    assert histogrammer.dead_time_s == 0, "Current differentiable EWH does not support non-zero dead time"

    arrival_rates = histogrammer.calculate_arrival_rates(irf_tensor, transients, ambient_offsets, histogrammer.n_bins)
    ewh_list = histogrammer.simulate_ewh_diff(
        arrival_rates,
        histogrammer.n_pulses,
        histogrammer.n_bins,
        histogrammer.free_running,
        histogrammer.dead_time_s,
    )

    return ewh_list

def compute_rmse(pred, gt):
    rmse = torch.mean(torch.mean((pred - gt) ** 2, axis=-1) ** 0.5)
    return rmse


if __name__ == "__main__":
    ## Setting simulation parameters

    data_dir = Path("examples/renders/scene1/")
    config_path = "visionsim/emulate/aspc/examples/config_diff_ewh.yaml"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requires_grad = True

    camera = Camera(data_dir, config_path, device, requires_grad)
    transients, ambient_offsets = camera.get_transients()

    bin_width = 2 * tof2depth(1 / camera.active_source.frequency) / camera.histogrammer.n_bins
    _, irf = camera.active_source.get_kernel(bin_width)
    irf_tensor_gt = torch.tensor(irf, dtype=torch.float32, device=device)
    camera.active_source.plot_kernel(bin_width)

    ewh_list_gt = forward_pass_ewh_diff(
        camera.histogrammer,
        transients,
        ambient_offsets,
        irf_tensor_gt,
    )

    ewh_list_measurement = ewh_list_gt
    print("Part 1: Forward pass complete")
    ########################################################################
    irf_init = [0.1, 0.1, 0.1, 0.4, 0.8, 0.9, 0.9, 0.9, 0.8, 0.4, 0.1, 0.1, 0.1]

    irf_tensor_estim = nn.Parameter(
        torch.tensor(irf_init, device=irf_tensor_gt.device, dtype=irf_tensor_gt.dtype), requires_grad=True
    )

    optimizer = optim.Adam([irf_tensor_estim], lr=0.1)

    for epoch in range(100):
        optimizer.zero_grad()
        transients_pred, ambient_offsets_pred = camera.get_transients()
        ewh_list_pred = forward_pass_ewh_diff(
            camera.histogrammer,
            transients_pred,
            ambient_offsets_pred,
            F.relu(irf_tensor_estim),
        )

        err1 = compute_rmse(ewh_list_pred, ewh_list_measurement)
        err1.backward(retain_graph=True)
        optimizer.step()
        print("Error :", err1)
        print("Gradient:", irf_tensor_estim.grad.abs().mean(), irf_tensor_estim.grad.abs().max())

    plt.plot(irf_tensor_estim.detach().cpu().numpy())
    plt.show()
