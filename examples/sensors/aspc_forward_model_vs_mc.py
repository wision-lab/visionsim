"""Single-pixel forward models vs the Monte-Carlo reference.

Both single-pixel closed forms are now exact against the sampler, and this is the
picture of that. For each operating mode and each signal-to-background ratio it
overlays four curves on one axes:

    phi_bar            the arrival rate that goes in (sum-normalised)
    MC, 1000 cycles    sampled truth, visibly noisy
    MC, 10000 cycles   sampled truth, converging
    forward model      the closed form, one pass, no sampling

Six figures: two models x three SBRs.

Modes
-----
free-running   ``calculate_distorted_transient``. Dead time is the real 75 ns:
               the detector re-arms 75 ns after each detection regardless of
               cycle boundaries, so early bins shadow later ones and the
               measured shape is visibly pile-up-distorted.
gated          ``calculate_distorted_transient_sync`` in single-hit mode. One
               detection per cycle and a re-arm at every cycle start, so the
               dead time can never bind -- ``dead_time_bins`` only selects
               single-hit, its value is irrelevant. The script checks that.

Scene: flat surface, albedo 1, 100 ns laser period, 100 bins (1 ns each), 75 ns
SPAD dead time. Depths are swept via ``--depth``; the unambiguous range at this
period is ``c/2f`` = 15 m, so anything beyond that aliases back into the cycle.

Usage::

    PYTHONPATH=. python examples/sensors/aspc_forward_model_vs_mc.py [--outdir DIR] [--depth M ...]
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

# --------------------------------------------------------------------------- #
# Scene / detector configuration                                              #
# --------------------------------------------------------------------------- #
PERIOD_NS = 100.0
N_BINS = 100                      # -> 1.0 ns per bin
BIN_NS = PERIOD_NS / N_BINS
DEAD_TIME_NS = 75.0
DEAD_TIME_BINS = int(round(DEAD_TIME_NS / BIN_NS))

DEPTHS_M = [1.0, 5.0]
ALBEDO = 1.0
IRF_SIGMA_BINS = 1.5              # gaussian instrument response

SBR_CASES = [(1.0, 1.0), (1.0, 3.0), (1.0, 10.0)]
CYCLE_COUNTS = [1_000, 10_000]
SEED = 20260821

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        _failures.append(name)


# --------------------------------------------------------------------------- #
# Arrival rate                                                                #
# --------------------------------------------------------------------------- #
def tof_bin(depth_m: float) -> float:
    """Round-trip time of flight, in bins."""
    return (2.0 * depth_m / C_LIGHT) * 1e9 / BIN_NS


def make_phi(signal: float, background: float, depth_m: float) -> torch.Tensor:
    """Gaussian return on a flat ambient floor.

    ``signal`` is total signal photons per cycle (the pulse integrates to it) and
    ``background`` is total ambient photons per cycle, spread evenly over bins.
    Albedo scales the signal term only -- ambient is already a scene radiance.
    """
    centre = tof_bin(depth_m)
    bins = np.arange(N_BINS)
    # wrap the gaussian so a return near the cycle edge is not clipped
    delta = (bins - centre + N_BINS / 2) % N_BINS - N_BINS / 2
    kernel = np.exp(-0.5 * (delta / IRF_SIGMA_BINS) ** 2)
    kernel /= kernel.sum()
    phi = ALBEDO * signal * kernel + background / N_BINS
    return torch.tensor(phi, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# The two paths                                                               #
# --------------------------------------------------------------------------- #
def timed(fn, repeats=3):
    """Best-of-N wall time, in milliseconds, alongside the result."""
    best = float("inf")
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best * 1e3


def mc_ewh(phi, n_cycles, *, free_running, seed=SEED):
    """Sampled EWH, sum-normalised, plus the raw detection total."""
    cap = None if free_running else 1
    timestamps = simulate_photon_timestamps(
        phi, n_cycles,
        dead_time_bins=DEAD_TIME_BINS if free_running else N_BINS,
        free_running=free_running,
        paralyzable=False,
        max_detections_per_cycle=cap,
        generator=torch.Generator().manual_seed(seed),
    )
    hist = timestamps_to_histogram(timestamps, N_BINS, N_BINS)
    total = float(hist.sum())
    return (hist / total).numpy(), total


def forward_ewh(phi, *, free_running):
    if free_running:
        return calculate_distorted_transient(phi, DEAD_TIME_BINS, N_BINS).numpy()
    # dead_time_bins >= n_hist_bins selects single-hit; the value cannot matter
    return calculate_distorted_transient_sync(phi, N_BINS, N_BINS).numpy()


# --------------------------------------------------------------------------- #
# Plot                                                                        #
# --------------------------------------------------------------------------- #
def figure(phi, signal, background, depth_m, free_running, outdir: Path) -> None:
    mode = "free-running" if free_running else "gated, single-hit"
    slug = "free_running" if free_running else "gated_single_hit"

    phi_norm = (phi / phi.sum()).numpy()
    model, ms_model = timed(lambda: forward_ewh(phi, free_running=free_running), repeats=10)
    curves, ms_mc = {}, {}
    for n in CYCLE_COUNTS:
        curves[n], ms_mc[n] = timed(lambda n=n: mc_ewh(phi, n, free_running=free_running))

    t = np.arange(N_BINS) * BIN_NS
    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    ax.plot(t, phi_norm, color="#9aa4ad", lw=1.6, ls=(0, (5, 2)),
            label=r"$\bar{\varphi}$ arrival rate (sum-normalised)")
    ax.plot(t, curves[1_000][0], color="#e0a458", lw=1.0, alpha=.85,
            label="Monte-Carlo EWH, 1000 cycles")
    ax.plot(t, curves[10_000][0], color="#2f6f9f", lw=1.3,
            label="Monte-Carlo EWH, 10000 cycles")
    ax.plot(t, model, color="#0b6e73", lw=2.2,
            label="Forward model (single pass)")

    ax.axvline(tof_bin(depth_m) * BIN_NS, color="#b0392b", lw=1.0, ls=":",
               label=f"true ToF, {depth_m:.0f} m")

    ax.set_xlabel("time within cycle  [ns]")
    ax.set_ylabel("normalised detections per bin")
    ax.set_title(
        f"{mode}   |   {depth_m:.0f} m   |   signal:background = {signal:.0f}:{background:.0f}"
        f"   |   period {PERIOD_NS:.0f} ns"
        + (f", dead time {DEAD_TIME_NS:.0f} ns" if free_running
           else ", dead time cannot bind"),
        fontsize=11,
    )
    ax.set_xlim(0, PERIOD_NS)
    ax.grid(alpha=.25, lw=.6)
    ax.legend(fontsize=8.5, framealpha=.95)

    total_10k = curves[10_000][1]
    err = float(np.abs(curves[10_000][0] - model).max())
    ax.text(.985, .60,
            f"max|MC$_{{10k}}$ - model| = {err:.4f}\n"
            f"detections = {total_10k:,.0f} / {10_000:,} cycles\n"
            f"MC 1k {ms_mc[1_000]:6.1f} ms | MC 10k {ms_mc[10_000]:6.1f} ms\n"
            f"model  {ms_model:6.2f} ms  ({ms_mc[10_000]/ms_model:.1f}x faster)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            family="monospace", color="#3d4852",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#d3dae1"))

    fig.tight_layout()
    path = outdir / f"fm_vs_mc_{slug}_d{depth_m:.0f}m_sbr_{signal:.0f}_{background:.0f}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    wrote {path.name}   "
          f"[MC 1k {ms_mc[1_000]:.1f} ms, MC 10k {ms_mc[10_000]:.1f} ms, "
          f"model {ms_model:.2f} ms -> {ms_mc[10_000]/ms_model:.1f}x]")

    # the model must sit inside the sampling error of the converged MC run
    q = float(model.max())
    tol = 5.0 * (q * (1 - q) / total_10k) ** 0.5
    check(f"{mode}, {depth_m:.0f} m, SBR {signal:.0f}:{background:.0f} — model matches MC(10k)",
          err < tol, f"err={err:.5f} < 5sigma={tol:.5f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path,
                    default=Path(__file__).parent / "aspc" / "figures")
    ap.add_argument("--depth", type=float, nargs="+", default=DEPTHS_M,
                    metavar="M", help="scene depth(s) in metres")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"\nScene: albedo {ALBEDO:.0f}, period {PERIOD_NS:.0f} ns, "
          f"{N_BINS} bins ({BIN_NS:.1f} ns each), dead time {DEAD_TIME_NS:.0f} ns "
          f"= {DEAD_TIME_BINS} bins")
    print(f"Unambiguous range = c/2f = {C_LIGHT*PERIOD_NS*1e-9/2:.2f} m")
    for d in args.depth:
        print(f"  ToF at {d:>4.0f} m = {2*d/C_LIGHT*1e9:7.3f} ns = bin {tof_bin(d):6.2f}")
    print()

    for depth_m in args.depth:
        for free_running in (True, False):
            print(f"{depth_m:.0f} m, {'free-running' if free_running else 'gated, single-hit'}:")
            for signal, background in SBR_CASES:
                figure(make_phi(signal, background, depth_m), signal, background,
                       depth_m, free_running, args.outdir)
            print()

    # The claim that makes 75 ns meaningless in the gated figures.
    print("gated single-hit — dead time really cannot bind:")
    phi = make_phi(*SBR_CASES[0], args.depth[0])
    ref = None
    for dt in (0, DEAD_TIME_BINS, N_BINS, 2 * N_BINS):
        ts = simulate_photon_timestamps(
            phi, 10_000, dead_time_bins=dt, free_running=False,
            max_detections_per_cycle=1,
            generator=torch.Generator().manual_seed(SEED))
        hist = timestamps_to_histogram(ts, N_BINS, N_BINS)
        ref = hist if ref is None else ref
        check(f"dt={dt:>3d} bins gives an identical histogram",
              bool(torch.equal(hist, ref)), f"{float(hist.sum()):,.0f} detections")

    print()
    if _failures:
        print(f"{len(_failures)} check(s) FAILED: " + ", ".join(_failures))
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
