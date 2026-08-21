"""Differentiability in a real use case: recovering depth by fitting the forward model.

The forward models are differentiable end to end, which makes them usable as the
render step of an inverse problem. This example does the obvious one: take a
*measured* histogram from the Monte-Carlo photon simulator, and recover the scene
parameters that produced it by gradient descent through the closed form.

    depth, signal, background  ->  phi_bar  ->  forward model  ->  predicted shape
                          loss = multinomial NLL against measured counts
                          d(loss)/d(depth) via autograd -> Adam

Why this is worth showing rather than just asserting "the gradient is finite":
pile-up **biases** the naive estimator. Under gated single-hit the first photon
wins, so the measured peak sits *earlier* than the true return; under free-running
the dead time carves a shadow after it. Reading ``argmax`` off the raw histogram
therefore reports a depth that is systematically too short, and no amount of
averaging fixes it -- it is a bias, not noise.

Fitting through the forward model inverts that distortion. The optimiser is
*initialised at the argmax estimate*, so the plots show it walking away from the
biased starting point onto the true depth.

Both modes are exercised:

    free-running       calculate_distorted_transient       (75 ns dead time)
    gated, single-hit  calculate_distorted_transient_sync  (one detection/cycle)

Usage::

    PYTHONPATH=. python examples/sensors/aspc_differentiable_depth_fit.py [--outdir DIR]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.constants import c as C_LIGHT

from visionsim.emulate.aspc.detector import (
    simulate_photon_timestamps,
    timestamps_to_histogram,
)
from visionsim.emulate.aspc.histogrammers import (
    calculate_distorted_transient,
    calculate_distorted_transient_sync,
)

PERIOD_NS = 100.0
N_BINS = 100
BIN_NS = PERIOD_NS / N_BINS
DEAD_TIME_BINS = 75
IRF_SIGMA_BINS = 1.5

TRUE_SIGNAL = 1.0
TRUE_BACKGROUND = 3.0
N_CYCLES = 10_000
N_STEPS = 150
DEPTH_SWEEP = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
DEMO_DEPTH = 5.0
SEED = 20260821

DTYPE = torch.float64
MODES = [(True, "free-running", "free_running"),
         (False, "gated, single-hit", "gated_single_hit")]

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        _failures.append(name)


def bin_to_depth(b) -> float:
    return float(b) * BIN_NS * 1e-9 * C_LIGHT / 2.0


# --------------------------------------------------------------------------- #
# Differentiable scene -> arrival rate                                        #
# --------------------------------------------------------------------------- #
def phi_from_params(depth_m, signal, background):
    """phi_bar as a differentiable function of the scene parameters.

    ``torch.remainder`` wraps the gaussian around the cycle boundary and carries
    a gradient of 1 w.r.t. its input, so d(phi)/d(depth) survives the wrap.
    """
    centre = (2.0 * depth_m / C_LIGHT) * 1e9 / BIN_NS
    bins = torch.arange(N_BINS, dtype=DTYPE)
    delta = torch.remainder(bins - centre + N_BINS / 2.0, N_BINS) - N_BINS / 2.0
    kernel = torch.exp(-0.5 * (delta / IRF_SIGMA_BINS) ** 2)
    kernel = kernel / kernel.sum()
    return signal * kernel + background / N_BINS


def forward(phi, free_running):
    if free_running:
        return calculate_distorted_transient(phi, DEAD_TIME_BINS, N_BINS)
    return calculate_distorted_transient_sync(phi, N_BINS, N_BINS)


# --------------------------------------------------------------------------- #
# Measurement + estimators                                                    #
# --------------------------------------------------------------------------- #
def measure(depth_m, free_running, seed):
    """A 'real' measurement: sampled photon timestamps, binned into counts."""
    phi = phi_from_params(torch.tensor(depth_m, dtype=DTYPE), TRUE_SIGNAL, TRUE_BACKGROUND)
    timestamps = simulate_photon_timestamps(
        phi.detach().float(), N_CYCLES,
        dead_time_bins=DEAD_TIME_BINS if free_running else N_BINS,
        free_running=free_running, paralyzable=False,
        max_detections_per_cycle=None if free_running else 1,
        generator=torch.Generator().manual_seed(seed),
    )
    return timestamps_to_histogram(timestamps, N_BINS, N_BINS).to(DTYPE)


def grid_init(counts, free_running, n_grid=100) -> float:
    """Coarse depth scan by brute force -- forward passes only, no autograd.

    Gradient descent on depth has a *small basin of attraction*: the IRF is ~1.5
    bins wide, so if the predicted return does not overlap the measured one the
    gradient w.r.t. depth is essentially zero and the optimiser cannot find its
    way. Initialising at ``argmax`` is fine until pile-up buries the return --
    under gated single-hit at 11 m the true return survives to only ~220 counts
    while an ambient bin reaches ~300, so ``argmax`` lands on noise and the fit
    never recovers.

    The forward model is cheap enough to just scan, which is the standard remedy:
    exhaustive coarse search to get inside the basin, gradients for the sub-bin
    refinement that a grid can never give you. Two complementary uses of the same
    differentiable model.
    """
    max_depth = bin_to_depth(N_BINS)
    best_loss, best_depth = float("inf"), 0.0
    with torch.no_grad():
        for k in range(n_grid):
            d = (k + 0.5) * max_depth / n_grid
            phi = phi_from_params(torch.tensor(d, dtype=DTYPE), TRUE_SIGNAL, TRUE_BACKGROUND)
            pred = forward(phi, free_running)
            loss = float(-(counts * torch.log(pred.clamp_min(1e-12))).sum())
            if loss < best_loss:
                best_loss, best_depth = loss, d
    return best_depth


def argmax_depth(counts) -> float:
    """The naive estimator: peak bin of the raw histogram."""
    return bin_to_depth(int(torch.argmax(counts)))


def fit_depth(counts, free_running, init_depth, n_steps=N_STEPS, track=False):
    """Recover (depth, signal, background) by gradient descent through the model.

    Loss is the multinomial negative log-likelihood ``-sum_b n_b log p_b`` -- the
    exact likelihood for gated single-hit given the detection total, and a sound
    surrogate for free-running.
    """
    depth = torch.tensor(float(init_depth), dtype=DTYPE, requires_grad=True)
    log_sig = torch.tensor(float(np.log(TRUE_SIGNAL * 2)), dtype=DTYPE, requires_grad=True)
    log_bkg = torch.tensor(float(np.log(TRUE_BACKGROUND * 2)), dtype=DTYPE, requires_grad=True)
    opt = torch.optim.Adam([{"params": [depth], "lr": 0.05},
                            {"params": [log_sig, log_bkg], "lr": 0.08}])

    history = []
    for _ in range(n_steps):
        opt.zero_grad()
        phi = phi_from_params(depth, torch.exp(log_sig), torch.exp(log_bkg))
        pred = forward(phi, free_running)
        loss = -(counts * torch.log(pred.clamp_min(1e-12))).sum()
        loss.backward()
        if track:
            history.append((float(loss), float(depth), float(depth.grad)))
        opt.step()
    return float(depth), float(torch.exp(log_sig)), float(torch.exp(log_bkg)), history


# --------------------------------------------------------------------------- #
# Figures                                                                     #
# --------------------------------------------------------------------------- #
def fig_convergence(free_running, label, slug, outdir: Path):
    counts = measure(DEMO_DEPTH, free_running, SEED)
    init = argmax_depth(counts)
    t0 = time.perf_counter()
    est, sig, bkg, hist = fit_depth(counts, free_running, init, track=True)
    elapsed = time.perf_counter() - t0

    loss = [h[0] for h in hist]
    depths = [h[1] for h in hist]
    grads = [abs(h[2]) for h in hist]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax1.plot(depths, color="#0b6e73", lw=2)
    ax1.axhline(DEMO_DEPTH, color="#b0392b", ls=":", lw=1.2, label=f"true depth {DEMO_DEPTH:.2f} m")
    ax1.axhline(init, color="#e0a458", ls="--", lw=1.2,
                label=f"argmax init {init:.3f} m  (bias {init-DEMO_DEPTH:+.3f} m)")
    ax1.scatter([len(depths) - 1], [est], color="#0b6e73", zorder=5, s=28)
    ax1.annotate(f"  fit {est:.3f} m\n  err {est-DEMO_DEPTH:+.4f} m",
                 (len(depths) - 1, est), fontsize=8.5, va="center", family="monospace")
    ax1.set_xlabel("Adam iteration"); ax1.set_ylabel("depth estimate  [m]")
    ax1.set_title("Gradient descent walks off the biased init", fontsize=10.5)
    ax1.legend(fontsize=8); ax1.grid(alpha=.25, lw=.6)

    ax2.plot(loss, color="#2f6f9f", lw=1.8, label="multinomial NLL")
    ax2.set_xlabel("Adam iteration"); ax2.set_ylabel("loss")
    ax2b = ax2.twinx()
    ax2b.semilogy(grads, color="#9aa4ad", lw=1.1, ls="--", label=r"$|\partial L/\partial d|$")
    ax2b.set_ylabel(r"$|\partial \mathrm{loss}/\partial \mathrm{depth}|$", color="#6b7883")
    ax2.set_title(f"Loss and depth gradient  ({elapsed:.1f} s, {N_STEPS} steps)", fontsize=10.5)
    ax2.grid(alpha=.25, lw=.6)
    h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=8)

    fig.suptitle(f"{label}  |  fitting depth through the differentiable forward model",
                 fontsize=11.5)
    fig.tight_layout()
    path = outdir / f"diffdepth_convergence_{slug}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"    wrote {path.name}  ({elapsed:.1f} s)")

    check(f"{label} — fit beats argmax at {DEMO_DEPTH:.0f} m",
          abs(est - DEMO_DEPTH) < abs(init - DEMO_DEPTH),
          f"|fit err|={abs(est-DEMO_DEPTH):.4f} < |argmax err|={abs(init-DEMO_DEPTH):.4f}")
    check(f"{label} — gradient is finite and non-zero throughout",
          all(np.isfinite(g) for g in grads) and max(grads) > 0)
    return elapsed


def fig_sweep(free_running, label, slug, outdir: Path):
    rows = []
    t0 = time.perf_counter()
    for i, d_true in enumerate(DEPTH_SWEEP):
        counts = measure(d_true, free_running, SEED + i)
        d_arg = argmax_depth(counts)
        d_fit, _, _, _ = fit_depth(counts, free_running, grid_init(counts, free_running))
        if abs(d_arg - d_true) > 1.0:
            print(f"      note: argmax failed at {d_true:.0f} m "
                  f"({d_arg:.2f} m) — return buried by pile-up; grid init recovers it")
        rows.append((d_true, d_arg, d_fit))
    elapsed = time.perf_counter() - t0

    true = np.array([r[0] for r in rows])
    arg = np.array([r[1] for r in rows])
    fit = np.array([r[2] for r in rows])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    lim = [0, max(DEPTH_SWEEP) + 1]
    ax1.plot(lim, lim, color="#9aa4ad", ls=":", lw=1.2, label="ideal  y = x")
    ax1.plot(true, arg, "o--", color="#e0a458", lw=1.4, ms=6, label="argmax of raw histogram")
    ax1.plot(true, fit, "o-", color="#0b6e73", lw=2, ms=6,
             label="grid init + gradient refine, through the model")
    ax1.set_ylabel("estimated depth  [m]"); ax1.set_xlim(*lim); ax1.set_ylim(*lim)
    ax1.legend(fontsize=8.5); ax1.grid(alpha=.25, lw=.6)
    ax1.set_title(f"{label}  |  depth recovery over the unambiguous range", fontsize=11)

    ax2.axhline(0, color="#9aa4ad", lw=1)
    half_bin = bin_to_depth(0.5)
    ax2.axhspan(-half_bin, half_bin, color="#9aa4ad", alpha=.15,
                label=f"argmax quantisation, ±½ bin (±{half_bin*100:.1f} cm)")
    ax2.plot(true, arg - true, "o--", color="#e0a458", lw=1.4, ms=5)
    ax2.plot(true, fit - true, "o-", color="#0b6e73", lw=2, ms=5)
    ax2.set_xlabel("true depth  [m]"); ax2.set_ylabel("error  [m]")
    ax2.legend(fontsize=8); ax2.grid(alpha=.25, lw=.6)

    fig.tight_layout()
    path = outdir / f"diffdepth_sweep_{slug}.png"
    fig.savefig(path, dpi=150); plt.close(fig)

    bias_arg, bias_fit = float(np.mean(arg - true)), float(np.mean(fit - true))
    rms_arg = float(np.sqrt(np.mean((arg - true) ** 2)))
    rms_fit = float(np.sqrt(np.mean((fit - true) ** 2)))
    print(f"    wrote {path.name}  ({elapsed:.1f} s)")
    print(f"      argmax : bias {bias_arg*100:+6.2f} cm   RMS {rms_arg*100:6.2f} cm")
    print(f"      fit    : bias {bias_fit*100:+6.2f} cm   RMS {rms_fit*100:6.2f} cm")

    check(f"{label} — fit RMS beats argmax RMS over the sweep",
          rms_fit < rms_arg, f"{rms_fit*100:.2f} cm < {rms_arg*100:.2f} cm")
    check(f"{label} — fit reaches sub-bin accuracy",
          rms_fit < half_bin, f"RMS {rms_fit*100:.2f} cm < half-bin {half_bin*100:.2f} cm")
    check(f"{label} — argmax bias is systematically negative (pile-up pulls depth short)",
          bias_arg < 0, f"{bias_arg*100:+.2f} cm")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path,
                    default=Path(__file__).parent / "aspc" / "figures")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"\nInverse problem: recover depth from a {N_CYCLES:,}-cycle measurement")
    print(f"  scene      signal {TRUE_SIGNAL:.0f}, background {TRUE_BACKGROUND:.0f} photons/cycle")
    print(f"  detector   {PERIOD_NS:.0f} ns period, {N_BINS} bins, {DEAD_TIME_BINS} bin dead time")
    print(f"  optimiser  Adam, {N_STEPS} steps, fitting (depth, signal, background)")
    print(f"  one bin    = {bin_to_depth(1)*100:.2f} cm of depth\n")

    for free_running, label, slug in MODES:
        print(f"{label}:")
        fig_convergence(free_running, label, slug, args.outdir)
        fig_sweep(free_running, label, slug, args.outdir)
        print()

    if _failures:
        print(f"{len(_failures)} check(s) FAILED: " + ", ".join(_failures))
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
