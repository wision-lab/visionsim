"""Active-SPC φ pipeline: synthetic-scene verification (no data loader required).

Builds albedo/depth maps analytically instead of loading renders, so the whole
``scene -> transient -> arrival rate (φ)`` path can be exercised and inspected with
zero external data. Every scene has a closed-form expected answer, so each figure
comes with a printed PASS/FAIL check rather than asking you to eyeball it.

φ is the photon *arrival* rate. Gating and dead time decide which arrivals become
detections, and that is entirely downstream — nothing here depends on gated vs
free-running operation.


================================================================================
THE LOADER CONTRACT
================================================================================
This example is also the **specification of the scene-loader boundary**. The data
loader is deliberately not exercised here; instead the scene builders below are the
reference implementation of what any loader must produce. Satisfy this contract and
everything downstream of it — through to φ — is verified by the checks in this file.

Violations are **silent**. Nothing validates these at runtime, and because
out-of-range depths now alias (see `depth` below) a wrong value usually produces a
plausible-looking wrong answer rather than an error. Read this carefully.

``albedo`` : float tensor, shape (n_frames, H, W), values in **[0, 1]**
    Lambertian diffuse reflectance. The pipeline multiplies laser irradiance by this
    directly.
    *If you supply shaded RGB instead of true albedo* — e.g. a render's red channel —
    the renderer's own illumination is double-counted, because the laser term
    multiplies by it again. That is the approximation the current in-tree loader
    makes; it is accepted and documented, not correct. A loader that can supply a
    true albedo pass should.

``depth`` : float tensor **or** pint Quantity, shape (n_frames, H, W), in **metres**
    One-way distance from camera to surface (not round-trip). Both a plain tensor and
    a `Quantity` are accepted; metres is assumed for a bare tensor.
      * **Valid**   : finite and > 0.
      * **Invalid** : 0, negative, NaN or inf. These mean "no surface" (sky, dropout)
                      and are dropped — they contribute nothing and produce no NaN.
      * **Beyond range**: NOT invalid. A return from past `max_resolvable_depth`
                      genuinely arrives during a later laser cycle, so it aliases to
                      `d mod max_resolvable_depth`. This is correct physics and is
                      independent of gated vs free-running operation.

    ⚠ **Never use an out-of-band numeric sentinel for missing depth.** Such a value
    *aliases into a pseudo-random in-range bin* and reads as a real surface. Use 0 or
    NaN. Note the hazard is worst for sentinels near the valid range: at 25 m it lands
    at a few percent of the true peak — a plausible weak return. A very large sentinel
    (65535) is partly self-suppressing because inverse-square falloff crushes its
    radiance by ~1e8, but it is still a fabricated return in the wrong bin. See
    figure 7(b).

    ⚠ **Never inpaint or interpolate across depth discontinuities.** Bilinear resize
    and hole-filling both invent surfaces at object edges, which a ToF simulator
    faithfully converts into returns from empty space. Resample depth with
    nearest-neighbour.

``offsets`` : float tensor, shape (n_frames, H, W)
    Per-pixel ambient photons per cycle. Same shape as albedo/depth.

Shape, resolution and device
    All three arrays share (n_frames, H, W); FOV masks are (n_masks, H, W). float32 is
    expected, and all tensors must be on the same device.
    **Render resolution is free.** The pipeline averages over each FOV, so H×W is a
    sampling choice that does not change collected photon counts — a loader need not
    resize to `sensor.size`. (This was not true before the F3 fix; doubling resolution
    used to collect 4× the photons.) Verified in figure 5(a).

Not part of the contract
    The in-tree loader also returns an `intensity_frames` array. Nothing downstream
    consumes it, and it is currently just a copy of albedo. A new loader need not
    produce it.
================================================================================

Run from the repo root (add PYTHONPATH=. if visionsim is not pip-installed):
    PYTHONPATH=. python examples/sensors/aspc_transients_synthetic.py [--outdir DIR] [--show]

Exits non-zero if any check fails, so it doubles as a smoke test.

Figures written to ``--outdir``, which defaults to ``examples/sensors/aspc/figures``
relative to this file, so the output lands in the same place from any working directory:
    1_scenes.png       synthetic albedo/depth maps
    2_pipeline.png     transient -> IRF -> φ, stage by stage
    3_depth_binning.png transient spikes vs analytically predicted bins
    4_depth_recovery.png round-trip: true depth vs depth recovered from φ
    5_invariances.png  independence from render resolution, FOV size; vignette
    6_guards.png       regression guards for the fixes in this layer
    7_loader_contract.png  what the loader contract buys, and how violations fail
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from visionsim.emulate.aspc.histogrammers import HistConfig, Histogrammer
from visionsim.emulate.aspc.sensors import SPADSensor
from visionsim.emulate.aspc.sources import PulsedLaser
from visionsim.emulate.aspc.utils import irradiance_photons, tof2depth, ureg

# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #
N_BINS = 200
GRID = 64  # render resolution (H = W)
FREQ = 10e6 * ureg.hertz
PULSE_WIDTH = 2 * ureg.nanosecond
AMBIENT_PER_BIN = 0.0  # set > 0 to see the ambient floor in figure 2

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))


# --------------------------------------------------------------------------- #
# Pipeline pieces                                                              #
# --------------------------------------------------------------------------- #
def make_laser() -> PulsedLaser:
    return PulsedLaser(
        wavelength=550 * ureg.nanometer,
        frequency=FREQ,
        pulse_width=PULSE_WIDTH,
        avg_watts=1 * ureg.milliwatt,
        pulse_shape="gaussian",
    )


def scene_irradiance(laser: PulsedLaser, albedo: torch.Tensor, depth_m: torch.Tensor) -> torch.Tensor:
    """Photons per SPAD pixel per cycle. Mirrors ``Camera._get_signal`` exactly, but
    without requiring a Camera (which needs a dataset path to construct)."""
    h, w = albedo.shape
    sensor = SPADSensor(size=(h, w), fov=66 * ureg.degree)
    radiance = laser.get_scene_radiance(albedo, depth_m * ureg.meter, sensor.w * sensor.h, sensor.omega)
    irr = (radiance * np.pi / 4 * (1 / sensor.f_number) ** 2).to(irradiance_photons) * (
        sensor.pixel_pitch.to(ureg.meter)
    ) ** 2
    return torch.as_tensor(irr.magnitude, dtype=torch.float32)


def bin_width_of(laser: PulsedLaser, n_bins: int = N_BINS):
    """Round-trip distance per bin — the convention ``get_kernel`` expects."""
    return 2 * tof2depth(1 / laser.frequency) / n_bins


def max_depth_of(laser: PulsedLaser) -> float:
    """One-way unambiguous range. Returns beyond this alias rather than vanishing."""
    return float(laser.max_resolvable_depth.to(ureg.meter).magnitude)


def run_pipeline(laser, hist, albedo, depth_m, masks, ambient=AMBIENT_PER_BIN, n_bins=N_BINS):
    """scene -> transient -> φ, using the real library functions throughout."""
    irr = scene_irradiance(laser, albedo, depth_m).unsqueeze(0)
    dep = depth_m.unsqueeze(0)
    off = torch.full_like(dep, float(ambient), dtype=torch.float32)

    transients, ambient_offsets = hist.calculate_transients(
        irr, dep, off, masks, n_bins, max_depth_of(laser)
    )
    _, irf = laser.get_kernel(bin_width_of(laser, n_bins), "sum")
    irf_t = torch.as_tensor(np.asarray(irf, dtype=np.float32))
    phi = hist.calculate_arrival_rates(irf_t, transients, ambient_offsets, n_bins)
    # ambient_offsets is the per-bin ambient rate: the FOV sum divided by both the
    # pixel count and the bin count, so the ambient contribution to Σφ is n_bins × it.
    return transients, irf_t, phi, ambient_offsets


def depth_to_bin(depth_m: float, laser: PulsedLaser, n_bins: int = N_BINS) -> int:
    """The analytic oracle, including aliasing of beyond-range returns."""
    return int(np.floor(depth_m * n_bins / max_depth_of(laser))) % n_bins


def bin_to_depth(idx: int, laser: PulsedLaser, n_bins: int = N_BINS) -> float:
    return (idx + 0.5) * max_depth_of(laser) / n_bins


# --------------------------------------------------------------------------- #
# Synthetic scenes                                                             #
# --------------------------------------------------------------------------- #
def flat_wall(depth, n=GRID, albedo=0.8):
    return torch.full((n, n), float(albedo)), torch.full((n, n), float(depth))


def two_planes(d_near, d_far, n=GRID):
    a = torch.full((n, n), 0.8)
    d = torch.empty(n, n)
    d[: n // 2] = d_near
    d[n // 2 :] = d_far
    return a, d


def tilted_plane(d_min, d_max, n=GRID):
    a = torch.full((n, n), 0.8)
    d = torch.linspace(d_min, d_max, n).view(n, 1).expand(n, n).contiguous()
    return a, d


def sphere_on_backdrop(n=GRID, d_back=11.0, d_sphere=4.0, radius=0.32):
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij")
    r = torch.sqrt(xx**2 + yy**2)
    inside = r < radius
    d = torch.where(inside, d_sphere - 1.5 * torch.sqrt(torch.clamp(radius**2 - r**2, min=0)), torch.tensor(d_back))
    a = torch.where(inside, torch.tensor(0.9), torch.tensor(0.35))
    return a, d


def scene_with_invalid(n=GRID, depth=5.0):
    """Sky / no-return pixels. Depth <= 0 and non-finite mean 'no surface'."""
    a, d = flat_wall(depth, n)
    d = d.clone()
    d[: n // 4, :] = 0.0  # sky band
    d[:, : n // 8] = float("nan")  # sensor dropout
    return a, d


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #
def fig_scenes(outdir, scenes):
    fig, axes = plt.subplots(2, len(scenes), figsize=(3.1 * len(scenes), 6.2))
    for j, (name, (a, d)) in enumerate(scenes.items()):
        im0 = axes[0, j].imshow(a.numpy(), cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(name, fontsize=10)
        plt.colorbar(im0, ax=axes[0, j], fraction=0.046)
        im1 = axes[1, j].imshow(np.ma.masked_invalid(d.numpy()), cmap="viridis")
        plt.colorbar(im1, ax=axes[1, j], fraction=0.046, label="m")
        for ax in (axes[0, j], axes[1, j]):
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("albedo", fontsize=11)
    axes[1, 0].set_ylabel("depth", fontsize=11)
    fig.suptitle("Figure 1 — synthetic scenes (no data loader)", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "1_scenes.png", dpi=130)
    plt.close(fig)


def fig_pipeline(outdir, laser, hist):
    a, d = sphere_on_backdrop()
    masks = torch.ones(1, GRID, GRID)
    transients, irf, phi, amb = run_pipeline(laser, hist, a, d, masks, ambient=2e-4)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5))
    axes[0].bar(np.arange(N_BINS), transients[0].numpy(), width=1.0, color="#3b7dd8")
    axes[0].set_title("Stage 1 — transient: scene radiance binned by depth", fontsize=11)
    axes[0].set_ylabel("photons / bin")

    centre = len(irf) // 2
    axes[1].plot(np.arange(len(irf)) - centre, irf.numpy(), color="#d1495b", lw=1.8)
    axes[1].set_title(
        f"Stage 2 — laser IRF, sum = {float(irf.sum()):.6f} (must be 1: normalize='sum')", fontsize=11
    )
    axes[1].set_xlabel("bins from pulse centre")

    axes[2].bar(np.arange(N_BINS), phi[0].numpy(), width=1.0, color="#2a9d8f")
    axes[2].set_title("Stage 3 — φ = (transient ∗ IRF) + ambient", fontsize=11)
    axes[2].set_xlabel("time bin")
    axes[2].set_ylabel("arrival rate")

    ambient_total = N_BINS * float(amb[0])
    signal_total = float(phi[0].sum()) - ambient_total
    conserved = np.isclose(signal_total, float(transients[0].sum()), rtol=1e-3)
    check("Convolution conserves photons", conserved,
          f"Σφ−ambient={signal_total:.6g} vs Σtransient={float(transients[0].sum()):.6g}")
    check("IRF is energy-normalised (T7)", np.isclose(float(irf.sum()), 1.0, rtol=1e-6), f"Σirf={float(irf.sum()):.6f}")

    fig.suptitle("Figure 2 — φ pipeline, stage by stage", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "2_pipeline.png", dpi=130)
    plt.close(fig)


def fig_depth_binning(outdir, laser, hist):
    cases = {
        "flat wall @ 4.0 m": flat_wall(4.0),
        "two planes @ 3 / 9 m": two_planes(3.0, 9.0),
        "tilted plane 2→13 m": tilted_plane(2.0, 13.0),
    }
    masks = torch.ones(1, GRID, GRID)
    fig, axes = plt.subplots(len(cases), 1, figsize=(10, 8.5))
    for ax, (name, (a, d)) in zip(axes, cases.items()):
        transients, _, _, _ = run_pipeline(laser, hist, a, d, masks)
        ax.bar(np.arange(N_BINS), transients[0].numpy(), width=1.0, color="#3b7dd8", label="transient")
        predicted = sorted({depth_to_bin(float(x), laser) for x in torch.unique(d)})
        for k, p in enumerate(predicted):
            ax.axvline(p, color="#e76f51", ls="--", lw=1.0, alpha=0.9, label="predicted bin" if k == 0 else None)
        occupied = sorted(torch.nonzero(transients[0]).flatten().tolist())
        check(f"Depth→bin exact: {name}", occupied == predicted, f"got {occupied[:6]}… vs {predicted[:6]}…")
        ax.set_title(f"{name}  —  occupied bins match prediction: {occupied == predicted}", fontsize=11)
        ax.set_ylabel("photons")
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("time bin")
    fig.suptitle("Figure 3 — depth → bin mapping vs closed-form prediction", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "3_depth_binning.png", dpi=130)
    plt.close(fig)


def fig_depth_recovery(outdir, laser, hist):
    """The end-to-end check: put a wall at depth d, read φ, get d back."""
    masks = torch.ones(1, GRID, GRID)
    md = max_depth_of(laser)
    truths = np.linspace(0.6, md * 0.97, 40)
    recovered = []
    for d0 in truths:
        a, d = flat_wall(float(d0))
        _, _, phi, _ = run_pipeline(laser, hist, a, d, masks)
        recovered.append(bin_to_depth(int(phi[0].argmax()), laser))
    recovered = np.array(recovered)
    resid_bins = (recovered - truths) / (md / N_BINS)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot([0, md], [0, md], color="#adb5bd", ls="--", lw=1.2, label="ideal y = x")
    axes[0].scatter(truths, recovered, s=26, color="#2a9d8f", zorder=3, label="recovered from φ")
    axes[0].set_xlabel("true depth (m)")
    axes[0].set_ylabel("recovered depth (m)")
    axes[0].set_title("Depth round-trip through the full φ pipeline", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].axhspan(-0.5, 0.5, color="#2a9d8f", alpha=0.15, label="±½ bin (quantisation)")
    axes[1].scatter(truths, resid_bins, s=26, color="#e76f51", zorder=3)
    axes[1].axhline(0, color="#adb5bd", lw=1)
    axes[1].set_xlabel("true depth (m)")
    axes[1].set_ylabel("error (bins)")
    axes[1].set_title(f"Residual — max |err| = {np.abs(resid_bins).max():.3f} bins", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    check(
        "Depth recovered within ½ bin (no systematic bias)",
        np.abs(resid_bins).max() <= 0.5 + 1e-6,
        f"max |err| = {np.abs(resid_bins).max():.4f} bins",
    )
    fig.suptitle("Figure 4 — end-to-end depth recovery", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "4_depth_recovery.png", dpi=130)
    plt.close(fig)


def fig_invariances(outdir, laser, hist):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # (a) render resolution — the F3 fix
    totals_res = []
    # Distinct linestyles and decreasing widths so four *coincident* curves are visibly
    # four curves -- otherwise a perfect overlay is indistinguishable from a single plot.
    styles = [("-", 6.0), ("--", 4.0), (":", 2.5), ("-.", 1.2)]
    for (n, (ls, lw)) in zip([16, 32, 64, 128], styles):
        a, d = flat_wall(6.0, n=n)
        transients, _, _, _ = run_pipeline(laser, hist, a, d, torch.ones(1, n, n))
        totals_res.append(float(transients.sum()))
        axes[0].plot(transients[0].numpy(), ls=ls, lw=lw, alpha=0.85, label=f"{n}×{n}")
    axes[0].set_title("(a) independent of render resolution (F3)", fontsize=11)
    axes[0].set_xlabel("time bin")
    axes[0].set_ylabel("photons")
    axes[0].set_xlim(depth_to_bin(6.0, laser) - 12, depth_to_bin(6.0, laser) + 12)
    axes[0].legend(fontsize=8)
    spread_res = (max(totals_res) - min(totals_res)) / max(totals_res)
    check("Transient independent of render resolution (F3)", spread_res < 1e-3, f"spread {spread_res:.2e}")

    # (b) FOV size — a SPAD averages over its FOV
    fovs = [[0, 1, 0, 1], [0, 0.5, 0, 0.5], [0, 0.25, 0, 0.25], [0, 0.5, 0, 1]]
    a, d = flat_wall(6.0)
    totals_fov = []
    for fov in fovs:
        masks = hist.get_perpixel_fov_masks(torch.zeros(GRID, GRID), [fov], vignette=False)
        transients, _, _, _ = run_pipeline(laser, hist, a, d, masks)
        totals_fov.append(float(transients.sum()))
    axes[1].bar([str(f) for f in fovs], totals_fov, color="#3b7dd8")
    axes[1].set_title("(b) independent of FOV size", fontsize=11)
    axes[1].set_ylabel("Σ transient")
    axes[1].tick_params(axis="x", rotation=25, labelsize=7)
    spread_fov = (max(totals_fov) - min(totals_fov)) / max(totals_fov)
    check("Transient independent of FOV size (F3)", spread_fov < 1e-3, f"spread {spread_fov:.2e}")

    # (c) vignette DOES attenuate — it is signal loss, not a normalisation
    masks_flat = hist.get_perpixel_fov_masks(torch.zeros(GRID, GRID), [[0, 1, 0, 1]], vignette=False)
    masks_vig = hist.get_perpixel_fov_masks(torch.zeros(GRID, GRID), [[0, 1, 0, 1]], vignette=True)
    t_flat, _, _, _ = run_pipeline(laser, hist, a, d, masks_flat)
    t_vig, _, _, _ = run_pipeline(laser, hist, a, d, masks_vig)
    expected_ratio = float(masks_vig[0].sum() / (masks_vig[0] > 0).sum())
    got_ratio = float(t_vig.sum() / t_flat.sum())
    axes[2].bar(["no vignette", "vignette"], [float(t_flat.sum()), float(t_vig.sum())], color=["#3b7dd8", "#e76f51"])
    axes[2].set_title(f"(c) vignette attenuates: ×{got_ratio:.4f}\n(expected ×{expected_ratio:.4f})", fontsize=11)
    axes[2].set_ylabel("Σ transient")
    check("Vignette applied with the right weight (T1)", np.isclose(got_ratio, expected_ratio, rtol=1e-3),
          f"got ×{got_ratio:.5f}, expected ×{expected_ratio:.5f}")

    fig.suptitle("Figure 5 — invariances that must hold", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "5_invariances.png", dpi=130)
    plt.close(fig)


def fig_guards(outdir, laser, hist):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    masks = torch.ones(1, GRID, GRID)
    bw = bin_width_of(laser)

    # (a) IRF normalisation — T7
    _, irf_sum = laser.get_kernel(bw, "sum")
    _, irf_raw = laser.get_kernel(bw, None)
    x = np.arange(len(irf_sum)) - len(irf_sum) // 2
    axes[0, 0].plot(x, np.asarray(irf_raw), color="#e76f51", lw=1.6, label=f"normalize=None (Σ={np.sum(irf_raw):.2f})")
    axes[0, 0].plot(x, np.asarray(irf_sum), color="#2a9d8f", lw=1.6, label=f"normalize='sum' (Σ={np.sum(irf_sum):.4f})")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("(a) T7 — IRF must integrate to 1", fontsize=11)
    axes[0, 0].set_xlabel("bins from centre")
    axes[0, 0].legend(fontsize=8)

    # (b) convolution orientation — T4
    irf_asym = np.array([0, 0, 0, 0, 0.1, 1.0, 0, 0, 0], dtype=np.float32)
    spike = torch.zeros(1, N_BINS)
    spike[0, 60] = 1.0
    got = hist.calculate_arrival_rates(torch.tensor(irf_asym), spike, 0.0, N_BINS)[0].numpy()
    true_conv = np.convolve(spike[0].numpy(), irf_asym, mode="same")
    mirrored = np.convolve(spike[0].numpy(), irf_asym[::-1], mode="same")
    sl = slice(54, 68)
    axes[0, 1].plot(np.arange(N_BINS)[sl], true_conv[sl], "o-", color="#2a9d8f", lw=2, label="true convolution")
    axes[0, 1].plot(np.arange(N_BINS)[sl], mirrored[sl], "s--", color="#e76f51", lw=1.4, label="mirrored (old bug)")
    axes[0, 1].plot(np.arange(N_BINS)[sl], got[sl], "x", color="black", ms=9, label="pipeline output")
    axes[0, 1].set_title("(b) T4 — asymmetric IRF is not time-reversed", fontsize=11)
    axes[0, 1].set_xlabel("time bin")
    axes[0, 1].legend(fontsize=8)
    check("Convolution not mirrored (T4)", np.allclose(got, true_conv, atol=1e-6), "matches np.convolve")

    # (c) aliasing vs clamping — T3
    beyond = max_depth_of(laser) * 1.4
    a, d = two_planes(4.0, beyond)
    transients, _, _, _ = run_pipeline(laser, hist, a, d, masks)
    axes[1, 0].bar(np.arange(N_BINS), transients[0].numpy(), width=1.0, color="#3b7dd8")
    axes[1, 0].axvline(depth_to_bin(4.0, laser), color="#2a9d8f", ls="--", label="4 m (in range)")
    axes[1, 0].axvline(depth_to_bin(beyond, laser), color="#e76f51", ls="--", label=f"{beyond:.1f} m → aliased")
    axes[1, 0].axvline(N_BINS - 1, color="black", ls=":", label="last bin (old clamp target)")
    # Log scale: the aliased return is genuinely ~27x fainter than the 4 m one purely
    # from inverse-square falloff ((21/4)^2), so it is invisible on a linear axis.
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylim(bottom=float(transients[0][transients[0] > 0].min()) * 0.5)
    axes[1, 0].set_title("(c) T3 — beyond-range returns alias, not clamp", fontsize=11)
    axes[1, 0].set_xlabel("time bin")
    axes[1, 0].legend(fontsize=8)
    check("Beyond-range aliases, last bin empty (T3)", float(transients[0, -1]) == 0.0,
          f"last bin = {float(transients[0,-1]):.3g}")

    # (d) invalid depths — L1 / T8
    a, d = scene_with_invalid()
    transients, _, phi, _ = run_pipeline(laser, hist, a, d, masks)
    valid_frac = float(torch.isfinite(d) .logical_and(d > 0).float().mean())
    axes[1, 1].bar(np.arange(N_BINS), transients[0].numpy(), width=1.0, color="#3b7dd8")
    axes[1, 1].axvline(depth_to_bin(5.0, laser), color="#2a9d8f", ls="--", label="5 m (only valid surface)")
    axes[1, 1].set_title(
        f"(d) L1/T8 — invalid depths dropped, not inpainted\n{valid_frac*100:.0f}% of pixels valid, "
        f"finite output: {bool(torch.isfinite(phi).all())}",
        fontsize=11,
    )
    axes[1, 1].set_xlabel("time bin")
    axes[1, 1].legend(fontsize=8)
    occupied = torch.nonzero(transients[0]).flatten().tolist()
    check("Invalid depths contribute nothing (L1/T8)", occupied == [depth_to_bin(5.0, laser)], f"bins {occupied}")
    check("No NaN/Inf anywhere in φ", bool(torch.isfinite(phi).all()))
    check(
        "Valid-pixel fraction scales the signal",
        np.isclose(float(transients.sum()), valid_frac * float(transients.sum()) / valid_frac, rtol=1e-6),
    )

    fig.suptitle("Figure 6 — regression guards for this layer's fixes", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "6_guards.png", dpi=130)
    plt.close(fig)


def fig_contract(outdir, laser, hist):
    """Make the loader contract concrete: what conformance buys, and what two
    plausible violations silently produce."""
    masks = torch.ones(1, GRID, GRID)
    md = max_depth_of(laser)
    true_bin = depth_to_bin(5.0, laser)

    def sky_scene(sky_value, depth=5.0, scale=1.0):
        a, d = flat_wall(depth * scale)
        d = d.clone()
        d[: GRID // 3, :] = sky_value
        return a, d

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # (a) conforming: sky marked with 0 -> dropped
    a, d = sky_scene(0.0)
    t_ok, _, _, _ = run_pipeline(laser, hist, a, d, masks)
    axes[0].bar(np.arange(N_BINS), t_ok[0].numpy(), width=1.2, color="#2a9d8f")
    axes[0].axvline(true_bin, color="#264653", ls="--", lw=1, label=f"wall @ 5 m (bin {true_bin})")
    axes[0].set_title("(a) CONFORMING — sky as 0 or NaN\nsky dropped, one true surface", fontsize=10)
    axes[0].legend(fontsize=8)
    ok_bins = torch.nonzero(t_ok[0]).flatten().tolist()
    check("Contract: sky marked 0/NaN is dropped", ok_bins == [true_bin], f"bins {ok_bins}")

    # (b) violation: a near-range numeric sentinel -> aliases into a false surface.
    # 25 m is the dangerous case: a plausible "just past our max range" marker. Its
    # radiance is only (25/5)^2 = 25x down, so the ghost is a few percent of the true
    # peak -- easily read as a real weak return. A huge sentinel (65535) is largely
    # self-suppressing because inverse-square crushes it by ~1e8; that makes the
    # magnitude of the hazard depend on the sentinel, not the rule.
    SENTINEL = 25.0
    a, d = sky_scene(SENTINEL)
    t_bad, _, _, _ = run_pipeline(laser, hist, a, d, masks)
    bad_bins = sorted(torch.nonzero(t_bad[0]).flatten().tolist())
    ghost = [b for b in bad_bins if b != true_bin]
    axes[1].bar(np.arange(N_BINS), t_bad[0].numpy(), width=1.2, color="#e76f51")
    axes[1].set_yscale("log")
    axes[1].set_ylim(bottom=float(t_bad[0][t_bad[0] > 0].min()) * 0.4)
    axes[1].axvline(true_bin, color="#264653", ls="--", lw=1, label=f"wall @ 5 m (bin {true_bin})")
    for g in ghost:
        rel = float(t_bad[0, g] / t_bad[0, true_bin]) * 100
        axes[1].axvline(g, color="#e76f51", ls=":", lw=1.4,
                        label=f"GHOST (bin {g}, {rel:.0f}% of peak)")
    axes[1].set_title(
        f"(b) VIOLATION — sky as {SENTINEL:.0f} m\nsentinel aliases into a false surface (log scale)",
        fontsize=10,
    )
    axes[1].legend(fontsize=8)
    rel = float(t_bad[0, ghost[0]] / t_bad[0, true_bin]) * 100 if ghost else 0.0
    check(
        "Contract: near-range sentinel creates a ghost surface (why it is forbidden)",
        len(ghost) == 1 and rel > 1.0,
        f"{SENTINEL:.0f} m aliases to bin {ghost[0] if ghost else '?'} at {rel:.0f}% of the true peak",
    )

    # (c) violation: depth in millimetres
    a, d = sky_scene(0.0, scale=1000.0)
    t_mm, _, _, _ = run_pipeline(laser, hist, a, d, masks)
    axes[2].bar(np.arange(N_BINS), t_mm[0].numpy(), width=1.2, color="#e76f51")
    axes[2].axvline(true_bin, color="#264653", ls="--", lw=1, label=f"expected bin {true_bin}")
    axes[2].set_title(
        "(c) VIOLATION — depth in millimetres\nno error, wrong answer", fontsize=10
    )
    axes[2].legend(fontsize=8)
    mm_bins = torch.nonzero(t_mm[0]).flatten().tolist()
    check("Contract: wrong depth units fail silently (why metres are required)",
          mm_bins != [true_bin], f"lands in {mm_bins} instead of [{true_bin}]")

    for ax in axes:
        ax.set_xlabel("time bin")
    axes[0].set_ylabel("photons")

    # units clause: a pint Quantity must behave identically to a bare tensor in metres
    a, d = flat_wall(5.0)
    t_q, _ = hist.calculate_transients(
        scene_irradiance(laser, a, d).unsqueeze(0), (d * ureg.meter).unsqueeze(0),
        torch.zeros(1, GRID, GRID), masks, N_BINS, md,
    )
    t_p, _ = hist.calculate_transients(
        scene_irradiance(laser, a, d).unsqueeze(0), d.unsqueeze(0),
        torch.zeros(1, GRID, GRID), masks, N_BINS, md,
    )
    check("Contract: pint Quantity depth == bare tensor in metres",
          bool(torch.allclose(t_q, t_p)), "both accepted, identical result")

    fig.suptitle("Figure 7 — the loader contract, made concrete", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "7_loader_contract.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # Anchored to this file rather than the cwd, so output always lands beside the
    # example regardless of where it is invoked from.
    ap.add_argument("--outdir", type=Path, default=Path(__file__).parent / "aspc" / "figures")
    ap.add_argument("--show", action="store_true", help="also open the figures interactively")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    laser = make_laser()
    hist = Histogrammer(HistConfig(n_bins=N_BINS))

    print(f"laser {FREQ.to(ureg.megahertz):~P}, pulse {PULSE_WIDTH:~P}")
    print(f"unambiguous range {max_depth_of(laser):.3f} m over {N_BINS} bins "
          f"({max_depth_of(laser)/N_BINS*100:.2f} cm/bin)\n")

    scenes = {
        "flat wall 4 m": flat_wall(4.0),
        "two planes 3/9 m": two_planes(3.0, 9.0),
        "tilted 2→13 m": tilted_plane(2.0, 13.0),
        "sphere + backdrop": sphere_on_backdrop(),
        "invalid depths": scene_with_invalid(),
    }

    fig_scenes(args.outdir, scenes)
    fig_pipeline(args.outdir, laser, hist)
    fig_depth_binning(args.outdir, laser, hist)
    fig_depth_recovery(args.outdir, laser, hist)
    fig_invariances(args.outdir, laser, hist)
    fig_guards(args.outdir, laser, hist)
    fig_contract(args.outdir, laser, hist)

    width = max(len(n) for n, _, _ in CHECKS)
    print("=" * (width + 30))
    for name, ok, detail in CHECKS:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    print("=" * (width + 30))
    n_fail = sum(1 for _, ok, _ in CHECKS if not ok)
    print(f"{len(CHECKS) - n_fail}/{len(CHECKS)} checks passed")
    print(f"figures written to {args.outdir.resolve()}")

    if args.show:
        import subprocess

        for p in sorted(args.outdir.glob("*.png")):
            subprocess.run(["xdg-open", str(p)], check=False)

    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
