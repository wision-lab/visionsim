from pathlib import Path
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

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


def KL_divergence_loss(pred_hist, target_hist):
    p = target_hist / (target_hist.sum(dim=-1, keepdim=True) + 1e-10)
    q = pred_hist / (pred_hist.sum(dim=-1, keepdim=True) + 1e-10)
    # p = target_hist
    # q = pred_hist
    return F.kl_div(q.log(), p, reduction="batchmean")
    # return F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction="batchmean")


def wasserstein_loss(pred_hist, target_hist):
    p = target_hist / (target_hist.sum(dim=-1, keepdim=True) + 1e-10)
    q = pred_hist / (pred_hist.sum(dim=-1, keepdim=True) + 1e-10)
    # p = target_hist
    # q = pred_hist
    cdf_p = torch.cumsum(p, dim=-1)
    cdf_q = torch.cumsum(q, dim=-1)
    return torch.mean(torch.square(cdf_p - cdf_q))
    # return torch.mean(torch.abs(cdf_p - cdf_q))


if __name__ == "__main__":
    data_dir = Path("examples/renders/scene1/")
    config_path = "visionsim/emulate/aspc/examples/config_diff_ewh.yaml"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requires_grad = True
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    camera = Camera(data_dir, config_path, device, requires_grad)

    bin_width = 2 * tof2depth(1 / camera.active_source.frequency) / camera.histogrammer.n_bins
    _, irf = camera.active_source.get_kernel(bin_width)
    irf_tensor_gt = torch.tensor(irf, dtype=torch.float32, device=device)
    # print("irf tensor length: ", len(irf_tensor_gt))
    # camera.active_source.plot_kernel(bin_width)
    transients, ambient_offsets = camera.get_transients()

    # ewh_list_gt = forward_pass_ewh_diff(
    #     camera.histogrammer,
    #     transients,
    #     ambient_offsets,
    #     irf_tensor_gt,
    # )
    # camera.plot_ewh(1, ewh_list_gt.detach())
    ewh_list_gt = camera.get_ewh()
    # camera.plot_ewh(1, ewh_list_gt)

    if isinstance(ewh_list_gt, torch.Tensor):
        ewh_list_measurement = ewh_list_gt.to(device=device, dtype=torch.float32)
    else:
        ewh_list_measurement = torch.stack(ewh_list_gt).to(device=device, dtype=torch.float32)
    # camera.plot_ewh(1, ewh_list_measurement.detach())
    print("Part 1: Forward pass complete")
    ########################################################################
    # irf_init = [random.uniform(0, 1)*0.5 for _ in range(51)] 
    irf_init = [0.5]*51
    # irf_init = [0.1]*20 + [0.1, 0.1, 0.1, 0.4, 0.8, 0.9, 0.9, 0.9, 0.8, 0.4, 0.1, 0.1, 0.1] + [0.1]*20
    # irf_init = [0.001]*25 + [0.5] + [0.001]*25

    irf_tensor_estim = nn.Parameter(
        torch.tensor(irf_init, device=irf_tensor_gt.device, dtype=irf_tensor_gt.dtype), requires_grad=True
    )

    # optimizer = optim.Adam([irf_tensor_estim], lr=0.01)
    optimizer = optim.Adam([irf_tensor_estim], lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    # optimizer = optim.AdamW([irf_tensor_estim], lr=0.1, weight_decay=1e-4)
    total_epochs = 7000
    # warmup_epochs = 10
    # warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    # cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=total_epochs - warmup_epochs,
    # )
    # scheduler = optim.lr_scheduler.SequentialLR(
    #     optimizer,
    #     schedulers=[warmup_scheduler, cosine_scheduler],
    #     milestones=[warmup_epochs],
    # )
    loss_history = []
    # window = torch.hann_window(len(irf_init), periodic=False, device=device)

    for epoch in tqdm(range(total_epochs)):
        optimizer.zero_grad()
        # irf_smooth = F.relu(irf_tensor_estim)
        irf_smooth = irf_tensor_estim
        transients_pred, ambient_offsets_pred = camera.get_transients()
        ewh_list_pred = forward_pass_ewh_diff(
            camera.histogrammer,
            transients_pred,
            ambient_offsets_pred,
            irf_smooth,
        )
        

        rmse_loss = compute_rmse(ewh_list_pred, ewh_list_measurement)
        mse_loss = F.mse_loss(ewh_list_pred, ewh_list_measurement)
        kl_loss = KL_divergence_loss(ewh_list_pred, ewh_list_measurement)
        w_loss = wasserstein_loss(ewh_list_pred, ewh_list_measurement)
        boundary_penalty = (irf_smooth[:10].mean() + irf_smooth[-10:].mean())
        tv_penalty = torch.mean(torch.abs(irf_smooth[1:] - irf_smooth[:-1]))
        smoothness_loss = torch.mean(torch.diff(irf_smooth, n=2) ** 2)  # penalize curvature
        norm_loss = (irf_smooth.sum() - 1.0) ** 2
        # total_loss = w_loss + 0.1*norm_loss + 0.05*smoothness_loss    # with delta
        total_loss = rmse_loss + 0.5*norm_loss
        # total_loss = w_loss + 0.05*norm_loss + 0.05*boundary_penalty    # with const
        loss_history.append(total_loss.detach().cpu().item())
        total_loss.backward(retain_graph=True)
        optimizer.step()
        # scheduler.step()
        # print("Error :", err1)
        # print("Gradient:", irf_tensor_estim.grad.abs().mean(), irf_tensor_estim.grad.abs().max())
        # if epoch % 500 == 0:
        # #     camera.plot_ewh(1, ewh_list_pred.detach())
            # print("w_loss: ", w_loss.detach().cpu().item(), "kl_loss: ", kl_loss.detach().cpu().item(), "rmse_loss: ", rmse_loss.detach().cpu().item())
        #     print("w_loss: ", w_loss.detach().cpu().item(), "norm_loss: ", norm_loss.detach().cpu().item(), "smoothness_loss: ", smoothness_loss.detach().cpu().item())
            # plt.plot(irf_tensor_estim.detach().cpu().numpy())
            # plt.show()

    pad = len(irf_tensor_estim) - len(irf_tensor_gt)
    pad_left = pad // 2
    pad_right = pad - pad_left
    irf_gt_centered = F.pad(
        irf_tensor_gt.view(1, 1, -1),
        (pad_left, pad_right),
        mode="replicate",
    ).view(-1)
    plt.plot(irf_tensor_estim.detach().cpu().numpy())
    # plt.plot(irf_smooth.detach().cpu().numpy())
    plt.plot(irf_gt_centered.detach().cpu().numpy())
    plt.show()
    plt.plot(loss_history)
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
