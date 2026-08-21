"""Active-SPC ground-truth detection model: mode and dead-time verification.

Companion to ``aspc_transients_synthetic.py``. That example ends at φ, the photon
*arrival* rate. This one starts there and covers the next stage: which arrivals
become **detections**, under each operating mode and dead-time model.

φ is constructed analytically here (a Gaussian return on an ambient floor), so no
scene, loader, or render is involved -- the arrival-rate pipeline is verified
separately and is deliberately not re-exercised.

Every figure carries a printed PASS/FAIL check rather than asking you to eyeball
it, and the script exits non-zero if any check fails.


================================================================================
WHAT IS BEING VERIFIED
================================================================================
The simulator emits raw ``(cycle, bin)`` timestamps; the histogram is a reduction
of them. Photon arrivals are drawn as per-bin Poisson **counts**, so a bin can
receive several photons and the "one detection per bin" behaviour of a real
detector emerges from the dead time rather than being assumed.

Modes
    synchronous (gated)
        Detector re-armed at the start of every cycle. Defaults to one detection
        per cycle -- the conventional lidar setup the Coates estimator inverts.
        Dead time cannot cross a cycle boundary, and in single-hit mode it cannot
        bind at all.
    free-running (asynchronous)
        Detector re-arms ``dead_time_bins`` after each detection regardless of
        cycle boundaries. Always multi-hit. Dead time wraps, including across
        more than one full cycle.

Dead-time models
    non-paralyzable
        Only *detections* open the dead window (active quenching).
    paralyzable
        *Every arrival* re-opens it, detected or not (passive quenching). Rate
        peaks and then collapses as flux rises, rather than saturating.

The unit tests in ``tests/test_aspc_detector.py`` pin all of this against an
exhaustive enumeration oracle and against closed-form rate formulas. This script
exists for the other half of the job: seeing the behaviour, and having somewhere
to look when a number comes out wrong.

Usage::

    PYTHONPATH=. python examples/sensors/aspc_detection_groundtruth.py [--outdir DIR]

Figures are written to ``--outdir``, which defaults to
``examples/sensors/aspc/figures`` alongside the φ example's output.
"""

from __future__ import annotations

import argparse
import math
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

# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #
N_BINS = 200
FREQ_HZ = 10e6
N_PULSES = 20_000

PERIOD_S = 1.0 / FREQ_HZ
MAX_DEPTH_M = C_LIGHT * PERIOD_S / 2.0  # round-trip, so half the light-distance
BIN_S = PERIOD_S / N_BINS
DEPTH_PER_BIN_M = MAX_DEPTH_M / N_BINS

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))


# --------------------------------------------------------------------------- #
# Arrival-rate construction (φ)                                                #
# --------------------------------------------------------------------------- #
def make_phi(
    peak_bin: float = 120.0,
    width_bins: float = 4.0,
    signal_total: float = 0.4,
    ambient_per_bin: float = 2e-3,
    n_bins: int = N_BINS,
) -> torch.Tensor:
    """A Gaussian laser return of ``signal_total`` photons/cycle on an ambient floor."""
    b = torch.arange(n_bins, dtype=torch.float32)
    peak = torch.exp(-0.5 * ((b - peak_bin) / width_bins) ** 2)
    peak = peak / peak.sum() * signal_total
    return peak + ambient_per_bin


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def coates(hist: torch.Tensor, n_pulses: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert first-photon-wins pile-up to recover φ from a gated single-hit histogram.

    Cycles still "alive" entering bin b are those that had not yet detected; the
    fraction of those that fire in bin b gives ``1 - exp(-phi_b)`` directly.

    Returns ``(phi_hat, alive_fraction)``. The second value matters: once almost
    every cycle has already fired, the surviving sample is tiny and the estimate
    there is dominated by noise. Coates is exact but not magic -- it cannot
    recover information from bins that were never observed. Callers should mask
    on ``alive_fraction`` rather than trusting the deep tail.
    """
    consumed = torch.cat([torch.zeros(1), torch.cumsum(hist, 0)[:-1]])
    alive = n_pulses - consumed
    # Bins where every surviving cycle fired carry no information; clamp so the
    # log stays finite rather than propagating inf into the plot.
    ratio = torch.clamp((alive - hist) / torch.clamp(alive, min=1.0), min=1e-9)
    return -torch.log(ratio), alive / n_pulses


def depth_axis(n_bins: int = N_BINS) -> np.ndarray:
    return (np.arange(n_bins) + 0.5) * (MAX_DEPTH_M / n_bins)


# --------------------------------------------------------------------------- #
# 10 - Timestamp raster: the raw object the simulator emits                    #
# --------------------------------------------------------------------------- #
def fig_raster(outdir: Path) -> None:
    phi = make_phi(signal_total=0.8, ambient_per_bin=6e-3)
    n_show = 300
    # Long enough (30% of a cycle) that the free-running wrap across cycle
    # boundaries is actually visible; at dt=20 the two gated/free panels differ
    # by well under 1% and look identical.
    dt = 60

    configs = [
        ("synchronous, single-hit\n(conventional; dead time cannot bind)",
         dict(free_running=False), "tab:blue"),
        ("synchronous, multi-hit\n(dead time resets each cycle)",
         dict(free_running=False, max_detections_per_cycle=None), "tab:orange"),
        ("free-running\n(dead time wraps across cycles)",
         dict(free_running=True), "tab:green"),
    ]

    fig, axes = plt.subplots(
        1, 4, figsize=(16, 4.6), sharey=True,
        gridspec_kw=dict(width_ratios=[1, 1, 1, 0.32]),
    )
    per_cycle_max, totals = {}, {}
    for ax, (title, kw, colour) in zip(axes, configs):
        ts = simulate_photon_timestamps(
            phi, n_show, dead_time_bins=dt, generator=gen(1), **kw
        )
        ax.scatter(ts[:, 0].numpy(), ts[:, 1].numpy(), s=3.5, c=colour, alpha=0.55,
                   linewidths=0)
        counts = torch.bincount(ts[:, 0], minlength=n_show)
        per_cycle_max[title] = int(counts.max())
        totals[title] = ts.shape[0]
        ax.set_title(f"{title}\n{ts.shape[0]} detections, max {int(counts.max())}/cycle",
                     fontsize=9)
        ax.set_xlabel("laser cycle")
        ax.set_xlim(0, n_show)
        ax.set_ylim(0, N_BINS)

    axes[0].set_ylabel("time bin within cycle")
    # φ gets its own panel rather than a twin axis: overlaid on the raster it
    # spans the full width and reads as data.
    axes[3].plot(phi.numpy(), np.arange(N_BINS), color="k", lw=1.3)
    axes[3].set_xlabel("φ\n(photons/bin/cycle)", fontsize=8)
    axes[3].set_title("arrival rate", fontsize=9)
    axes[3].tick_params(labelsize=7)

    fig.suptitle(
        f"10 — Raw (cycle, bin) timestamps, dead time {dt} bins. The histogram is a "
        "reduction of this, not the primary output.", fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(outdir / "10_timestamp_raster.png", dpi=130)
    plt.close(fig)

    sync_single = per_cycle_max[configs[0][0]]
    sync_multi = per_cycle_max[configs[1][0]]
    free = per_cycle_max[configs[2][0]]
    check("raster: gated single-hit never exceeds 1 detection/cycle",
          sync_single == 1, f"max = {sync_single}")
    check("raster: gated multi-hit does exceed 1 detection/cycle",
          sync_multi > 1, f"max = {sync_multi}")
    check("raster: free-running is multi-hit",
          free > 1, f"max = {free}")

    # Free-running carries dead time over the cycle boundary, so it must detect
    # strictly fewer photons than gated multi-hit on the same arrivals. The gap is
    # small (only detections within dt of a boundary are affected), so it needs
    # more cycles than the raster shows to resolve.
    n_stat = 20_000
    gated = simulate_photon_timestamps(
        phi, n_stat, dead_time_bins=dt, free_running=False,
        max_detections_per_cycle=None, generator=gen(1),
    ).shape[0]
    freerun = simulate_photon_timestamps(
        phi, n_stat, dead_time_bins=dt, free_running=True, generator=gen(1)
    ).shape[0]
    check("raster: free-running loses detections to cross-boundary dead time",
          freerun < gated,
          f"{freerun} vs {gated} over {n_stat} cycles ({(gated-freerun)/gated*100:.2f}% fewer)")


# --------------------------------------------------------------------------- #
# 11 - Inter-detection gaps: the most diagnostic dead-time plot                #
# --------------------------------------------------------------------------- #
def fig_gaps(outdir: Path) -> None:
    # Deliberately flat φ. A structured φ imprints the laser period on the gap
    # distribution (detections cluster at the return), which would confound the
    # thing this plot exists to show.
    phi = torch.full((N_BINS,), 0.03)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    lam = float(phi[0])
    ok_floor, ok_mode, ok_lam = True, True, True
    recovered = []
    for ax, dt in zip(axes, (1, 10, 40)):
        ts = simulate_photon_timestamps(
            phi, N_PULSES, dead_time_bins=dt, free_running=True, generator=gen(2)
        )
        g = (ts[:, 0] * N_BINS + ts[:, 1]).numpy()
        gaps = np.diff(g)
        counts = np.bincount(gaps)

        # Plot the exact per-gap counts rather than re-binning. `hist` closes its
        # final bin at both ends, which merges two gap values into one bar and
        # produces a spurious spike at the right edge.
        hi = dt + 80
        ax.bar(np.arange(min(hi, len(counts))), counts[:hi], width=1.0,
               color="tab:purple", alpha=0.8)
        ax.axvline(dt, color="crimson", lw=1.6, ls="--", label=f"dead time = {dt} bins")
        ax.set_yscale("log")
        ax.set_xlabel("gap to previous detection (bins)")
        ax.legend(fontsize=8)

        ok_floor &= gaps.min() >= dt
        ok_mode &= int(counts.argmax()) == dt

        # Past the dead window the detector just waits for the next occupied bin,
        # so the tail is geometric with ratio exp(-lambda) per bin. Its decay rate
        # therefore *recovers lambda* -- a sharper check than "looks straight".
        tail = counts[dt : dt + 40].astype(float)
        tail = tail[: int(np.argmax(tail < 50)) or len(tail)]
        slope = np.polyfit(np.arange(len(tail)), np.log(tail), 1)[0]
        recovered.append(-slope)
        ok_lam &= abs(-slope - lam) / lam < 0.10
        ax.set_title(f"dead time = {dt} bins — min gap {gaps.min()}, "
                     f"tail recovers λ={-slope:.4f}", fontsize=9)

    axes[0].set_ylabel("count (log)")
    fig.suptitle(
        "11 — Inter-detection gaps. A hard floor at the dead time, a peak exactly there, "
        "and a geometric tail whose decay rate returns λ.", fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(outdir / "11_deadtime_gaps.png", dpi=130)
    plt.close(fig)

    check("gaps: no detection pair closer than the dead time", ok_floor)
    check("gaps: distribution peaks exactly at the dead time", ok_mode)
    check("gaps: geometric tail recovers the true arrival rate", ok_lam,
          f"λ={lam:g}, recovered " + ", ".join(f"{r:.4f}" for r in recovered))


# --------------------------------------------------------------------------- #
# 12 - Detection rate vs flux against closed-form oracles                      #
# --------------------------------------------------------------------------- #
def fig_rate_vs_flux(outdir: Path) -> None:
    """Both dead-time models against analytic rates derived from renewal theory.

    These formulas share no code with the simulator, so agreement is genuine
    external validation rather than a self-consistency check.
    """
    n_bins, tau = 64, 10
    lams = np.logspace(-2.4, 0.5, 14)

    n_pulses = 20_000
    n_cells = n_pulses * n_bins
    rows = {}
    for par in (False, True):
        mc, analytic = [], []
        for lam in lams:
            phi = torch.full((n_bins,), float(lam))
            ts = simulate_photon_timestamps(
                phi, n_pulses, dead_time_bins=tau, free_running=True,
                paralyzable=par, generator=gen(3),
            )
            mc.append(ts.shape[0] / n_cells)
            p = 1.0 - math.exp(-lam)
            analytic.append(
                p * math.exp(-lam * (tau - 1)) if par else 1.0 / (tau + (1.0 - p) / p)
            )
        rows[par] = (np.array(mc), np.array(analytic))

    # The paralyzable rate falls to ~1e-13 at the top of the sweep, where a finite
    # simulation yields zero detections and the ratio is meaningless. Plot the whole
    # curve -- the collapse is the point -- but only assert where the expected count
    # is large enough for the comparison to mean anything.
    def well_sampled(analytic):
        return analytic * n_cells > 400

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for par, colour, label in ((False, "tab:blue", "non-paralyzable"),
                               (True, "tab:red", "paralyzable")):
        mc, an = rows[par]
        m = well_sampled(an)
        axes[0].plot(lams, an, color=colour, lw=1.6, label=f"{label} (analytic)")
        axes[0].plot(lams, mc, "o", color=colour, ms=5, mfc="none", label=f"{label} (MC)")
        axes[1].plot(lams[m], (mc / an)[m], "o-", color=colour, ms=4, lw=1.2, label=label)

    axes[0].plot(lams, 1.0 - np.exp(-lams), color="gray", lw=1.0, ls=":",
                 label="ideal (no dead time)")
    axes[0].axhline(1.0 / tau, color="k", lw=0.8, ls="--")
    axes[0].annotate("saturation at 1/τ", (lams[0], 1.0 / tau), fontsize=8,
                     va="bottom", xytext=(0, 3), textcoords="offset points")
    # Explains the stray high-λ paralyzable points: one detection in the whole
    # simulation is the smallest non-zero rate representable here.
    floor = 1.0 / n_cells
    axes[0].axhline(floor, color="gray", lw=0.7, ls="-.")
    axes[0].annotate("single-detection noise floor", (lams[0], floor), fontsize=7,
                     color="gray", va="bottom", xytext=(0, 2), textcoords="offset points")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("arrival rate λ (photons/bin/cycle)")
    axes[0].set_ylabel("detection rate (per bin per cycle)")
    axes[0].set_title(f"Detection rate vs flux, dead time = {tau} bins", fontsize=10)
    axes[0].legend(fontsize=8)

    axes[1].axhline(1.0, color="k", lw=0.8)
    axes[1].fill_between(lams, 0.95, 1.05, color="k", alpha=0.07)
    axes[1].set_xscale("log")
    axes[1].set_ylim(0.9, 1.1)
    axes[1].set_xlabel("arrival rate λ")
    axes[1].set_ylabel("simulated / analytic")
    axes[1].set_title("Ratio to closed form, where counts permit\n(shaded band = ±5%)",
                      fontsize=10)
    axes[1].legend(fontsize=8)

    fig.suptitle(
        "12 — Non-paralyzable saturates at 1/τ; paralyzable peaks then collapses. "
        "The two device types are qualitatively different, not just scaled.", fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(outdir / "12_rate_vs_flux.png", dpi=130)
    plt.close(fig)

    for par, label in ((False, "non-paralyzable"), (True, "paralyzable")):
        mc, an = rows[par]
        m = well_sampled(an)
        err = float(np.abs((mc / an)[m] - 1.0).max())
        check(f"rate vs flux: {label} matches closed form", err < 0.05,
              f"max deviation {err*100:.1f}% over {int(m.sum())}/{len(lams)} well-sampled λ")

    par_mc = rows[True][0]
    check("rate vs flux: paralyzable rate collapses at high flux",
          par_mc[-1] < par_mc.max() * 0.5,
          f"peak {par_mc.max():.4f} -> {par_mc[-1]:.4f}")
    check("rate vs flux: non-paralyzable saturates at 1/τ",
          rows[False][0].max() < 1.05 / tau, f"max {rows[False][0].max():.4f} vs {1/tau:.4f}")


# --------------------------------------------------------------------------- #
# 13 - Pile-up distortion and the Coates inversion                             #
# --------------------------------------------------------------------------- #
def fig_pileup(outdir: Path) -> None:
    """A deliberately flat φ, so every feature in the raw histogram is pile-up."""
    n_bins, n_pulses = 40, 60_000
    fluxes = [0.005, 0.02, 0.08, 0.25]

    min_alive = 0.05  # below this the surviving sample is too small to invert

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    worst_raw, worst_coates = 0.0, 0.0
    for lam, colour in zip(fluxes, plt.cm.viridis(np.linspace(0.15, 0.85, len(fluxes)))):
        phi = torch.full((n_bins,), lam)
        ts = simulate_photon_timestamps(
            phi, n_pulses, free_running=False, generator=gen(4)
        )
        hist = timestamps_to_histogram(ts, n_bins)
        raw = (hist / n_pulses).numpy()
        rec_t, alive = coates(hist, n_pulses)
        rec, alive = rec_t.numpy(), alive.numpy()
        valid = alive >= min_alive

        axes[0].plot(raw / raw[0], color=colour, lw=1.5,
                     label=f"λ={lam:g} ({lam*n_bins:.1f} photons/cycle)")
        axes[1].plot(np.where(valid, rec / lam, np.nan), color=colour, lw=1.5,
                     label=f"λ={lam:g}")
        if not valid.all():
            axes[1].plot(np.where(~valid, rec / lam, np.nan), color=colour, lw=1.0,
                         ls=":", alpha=0.6)

        worst_raw = max(worst_raw, abs(raw[-1] / raw[0] - 1.0))
        # Judge the inversion against counting noise, not a flat percentage: at
        # low flux each bin holds only a few hundred detections, so a fixed
        # tolerance would just be measuring the shot noise.
        p = 1.0 - math.exp(-lam)
        se = np.sqrt(p / ((1.0 - p) * np.maximum(alive * n_pulses, 1.0)))
        worst_coates = max(worst_coates, float((np.abs(rec - lam) / se)[valid].max()))

    axes[0].axhline(1.0, color="k", lw=0.9, ls="--", label="true φ (flat)")
    axes[0].set_title("Raw gated single-hit histogram\n(normalised to bin 0)", fontsize=10)
    axes[0].set_xlabel("time bin")
    axes[0].set_ylabel("detections, relative to bin 0")
    axes[0].legend(fontsize=8)

    axes[1].axhline(1.0, color="k", lw=0.9, ls="--", label="true φ")
    axes[1].set_ylim(0.8, 1.2)
    axes[1].set_title(f"After Coates inversion (recovered φ / true φ)\n"
                      f"dotted = fewer than {min_alive:.0%} of cycles still alive",
                      fontsize=10)
    axes[1].set_xlabel("time bin")
    axes[1].set_ylabel("recovered / true")
    axes[1].legend(fontsize=8)

    fig.suptitle(
        "13 — φ is flat, so all curvature on the left is pile-up: early bins consume "
        "cycles that later bins never get to see. Coates inverts it exactly.", fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(outdir / "13_pileup_coates.png", dpi=130)
    plt.close(fig)

    check("pile-up: raw histogram is visibly distorted at high flux",
          worst_raw > 0.5, f"worst last/first bin ratio departs {worst_raw*100:.0f}%")
    check("pile-up: Coates recovers flat φ at every flux",
          worst_coates < 5.0, f"worst bin {worst_coates:.1f}σ of counting noise")


# --------------------------------------------------------------------------- #
# 14 - What each mode does to the same φ                                       #
# --------------------------------------------------------------------------- #
def fig_mode_comparison(outdir: Path) -> None:
    phi = make_phi(signal_total=0.9, ambient_per_bin=8e-3)
    dt = 25

    # Widths are staggered deliberately: gated multi-hit and free-running differ
    # only for detections within dt of a cycle boundary, so at this dead time they
    # very nearly coincide. Drawing the wider one first makes that visible as a
    # halo instead of hiding one curve completely under the other.
    variants = [
        ("no dead time (dt=0, free-running)",
         dict(dead_time_bins=0, free_running=True), "tab:gray", "-", 1.4),
        ("free-running, dt=25", dict(dead_time_bins=dt, free_running=True),
         "tab:green", "-", 3.2),
        ("gated multi-hit, dt=25",
         dict(dead_time_bins=dt, free_running=False, max_detections_per_cycle=None),
         "tab:orange", "-", 1.3),
        ("gated single-hit", dict(free_running=False), "tab:blue", "-", 1.4),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    d = depth_axis()
    results = {}
    for label, kw, colour, ls, lw in variants:
        ts = simulate_photon_timestamps(phi, N_PULSES, generator=gen(5), **kw)
        h = (timestamps_to_histogram(ts, N_BINS) / N_PULSES).numpy()
        results[label] = h
        axes[0].plot(d, h, color=colour, lw=lw, ls=ls, label=label, alpha=0.9)
        axes[1].plot(d, h / h.sum(), color=colour, lw=lw, ls=ls, label=label, alpha=0.9)

    phi_np = phi.numpy()
    axes[0].plot(d, phi_np, color="k", lw=1.2, ls="--", label="true φ")
    axes[1].plot(d, phi_np / phi_np.sum(), color="k", lw=1.2, ls="--", label="true φ")
    for ax, title in zip(axes, ("Absolute detection rate", "Shape only (area-normalised)")):
        ax.set_xlabel("depth (m)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("detections per cycle per bin")

    fig.suptitle(
        "14 — Same φ, four operating modes. Dead time suppresses total rate; gating "
        "additionally biases the shape toward early bins.\nFree-running and gated "
        "multi-hit nearly coincide: they differ only within dt of a cycle boundary.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(outdir / "14_mode_comparison.png", dpi=130)
    plt.close(fig)

    nodt = results["no dead time (dt=0, free-running)"]
    # Counts are Poisson, so per-bin noise is sqrt(phi/n_pulses); comparing against a
    # flat tolerance would just be testing the noise level. 5 sigma over 200 bins.
    sigma = np.sqrt(np.maximum(phi_np, 1e-12) / N_PULSES)
    worst_z = float((np.abs(nodt - phi_np) / sigma).max())
    check("modes: dt=0 free-running reproduces φ itself",
          worst_z < 5.0, f"worst bin {worst_z:.1f}σ (Poisson noise, 200 bins)")
    check("modes: dead time strictly reduces total detections",
          results["free-running, dt=25"].sum() < nodt.sum(),
          f"{results['free-running, dt=25'].sum():.3f} < {nodt.sum():.3f}")
    check("modes: gated single-hit detects at most one photon per cycle",
          results["gated single-hit"].sum() <= 1.0,
          f"total = {results['gated single-hit'].sum():.3f}/cycle")

    # Centre of mass ahead of the true peak is the pile-up signature.
    com_true = float((d * phi_np).sum() / phi_np.sum())
    com_gated = float((d * results["gated single-hit"]).sum()
                      / results["gated single-hit"].sum())
    check("modes: gating biases the return earlier in the cycle",
          com_gated < com_true, f"centre of mass {com_gated:.2f} m vs true {com_true:.2f} m")


# --------------------------------------------------------------------------- #
# 15 - Dead time longer than one laser cycle                                   #
# --------------------------------------------------------------------------- #
def fig_deadtime_over_cycle(outdir: Path) -> None:
    """Regression guard: this case used to saturate at one cycle of dead time."""
    phi = make_phi(signal_total=1.5, ambient_per_bin=0.02)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    ok_floor = True
    rates, labels = [], []
    for dt_cycles, colour in ((0.5, "tab:blue"), (1.0, "tab:orange"),
                              (2.5, "tab:green"), (5.0, "tab:red")):
        dt = int(dt_cycles * N_BINS)
        ts = simulate_photon_timestamps(
            phi, N_PULSES, dead_time_bins=dt, free_running=True, generator=gen(6)
        )
        g = (ts[:, 0] * N_BINS + ts[:, 1]).numpy()
        gaps = np.diff(g)
        ok_floor &= len(gaps) == 0 or gaps.min() >= dt

        h = (timestamps_to_histogram(ts, N_BINS) / N_PULSES).numpy()
        axes[0].plot(depth_axis(), h, color=colour, lw=1.3,
                     label=f"dt = {dt_cycles}× cycle ({dt} bins)")
        rates.append(ts.shape[0] / N_PULSES)
        labels.append(f"{dt_cycles}×")

    axes[0].set_xlabel("depth (m)")
    axes[0].set_ylabel("detections per cycle per bin")
    axes[0].set_title("Histogram flattens as the dead window spans cycles", fontsize=10)
    axes[0].legend(fontsize=8)

    axes[1].bar(labels, rates, color="tab:purple", alpha=0.85)
    for i, r in enumerate(rates):
        axes[1].text(i, r, f"{r:.3f}", ha="center", va="bottom", fontsize=8)
    axes[1].set_xlabel("dead time, in laser cycles")
    axes[1].set_ylabel("detections per cycle")
    axes[1].set_title("Rate keeps falling past one cycle\n(it used to saturate here)",
                      fontsize=10)

    fig.suptitle(
        "15 — Free-running dead time spanning multiple cycles. Arm time is tracked in "
        "absolute bins, so there is no one-cycle lookback limit.", fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(outdir / "15_deadtime_over_cycle.png", dpi=130)
    plt.close(fig)

    check("multi-cycle dead time: gap floor respected beyond one cycle", ok_floor)
    check("multi-cycle dead time: rate keeps decreasing past one cycle",
          rates[2] < rates[1] and rates[3] < rates[2],
          " > ".join(f"{r:.4f}" for r in rates))
    check("multi-cycle dead time: rate stays below the 1/dt bound",
          all(r <= 1.0 / dtc * 1.02 for r, dtc in zip(rates, (0.5, 1.0, 2.5, 5.0))))


# --------------------------------------------------------------------------- #
# 16 - Consequence for the measurement: depth bias vs flux                     #
# --------------------------------------------------------------------------- #
def fig_depth_bias(outdir: Path) -> None:
    """Why any of this matters: pile-up pulls the estimated depth toward the camera.

    Depth is read out as the centroid of the histogram rather than its argmax.
    Argmax is far too noisy to show the effect -- at a few hundred detections it
    jitters by a couple of bins on shot noise alone, which swamps the bias.
    """
    true_bin = 130.0
    n_pulses = 40_000
    scales = np.logspace(-1.3, 0.8, 12)
    d = depth_axis()
    centroid = lambda h: float((d * h).sum() / h.sum())

    def scene(s):
        return make_phi(peak_bin=true_bin, signal_total=0.25 * s, ambient_per_bin=6e-3 * s)

    true_depth = centroid(scene(1.0).numpy())
    raw_depths, cor_depths, win_true, visible, photons = [], [], [], [], []
    for s in scales:
        phi = scene(s)
        ts = simulate_photon_timestamps(
            phi, n_pulses, free_running=False, generator=gen(7)
        )
        hist = timestamps_to_histogram(ts, N_BINS)
        rec, alive = coates(hist, n_pulses)
        # Bins where almost no cycles survived carry no information; a real
        # pipeline would discard them, so the readout does too.
        valid = (alive >= 0.05).numpy()
        rec = torch.where(torch.from_numpy(valid), rec, torch.zeros_like(rec))
        raw_depths.append(centroid(hist.numpy()))
        cor_depths.append(centroid(rec.numpy()))
        # The truth restricted to the window the detector could actually see.
        # Separating this from the full-range truth splits the two distinct
        # failure modes: pile-up *distortion*, which Coates inverts, and window
        # *truncation*, which no post-processing can undo.
        win_true.append(centroid(np.where(valid, scene(s).numpy(), 0.0)))
        visible.append(float(valid.mean()))
        photons.append(float(phi.sum()))

    raw_err = np.array(raw_depths) - true_depth
    cor_err = np.array(cor_depths) - true_depth
    win_err = np.array(cor_depths) - np.array(win_true)
    visible = np.array(visible)
    photons = np.array(photons)
    full_window = visible >= 0.999

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    axes[0].plot(photons, raw_err, "o-", color="tab:red", ms=4, label="raw histogram centroid")
    axes[0].plot(photons, cor_err, "s-", color="tab:blue", ms=4, label="after Coates")
    axes[0].plot(photons, win_err, "^--", color="tab:green", ms=4,
                 label="after Coates, vs truth over\nthe observable window")
    ax_v = axes[0].twinx()
    ax_v.plot(photons, visible * 100, color="gray", lw=0.9, ls=":")
    ax_v.set_ylabel("% of cycle still observable", fontsize=8, color="gray")
    ax_v.tick_params(labelsize=7, colors="gray")
    ax_v.set_ylim(0, 105)
    axes[0].axhline(0, color="k", lw=0.9)
    axes[0].axhspan(-DEPTH_PER_BIN_M, DEPTH_PER_BIN_M, color="k", alpha=0.08)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("total photons per cycle")
    axes[0].set_ylabel("depth error (m)")
    axes[0].set_title("Pile-up pulls the estimate toward the camera\n"
                      "(shaded band = ±1 bin)", fontsize=10)
    axes[0].legend(fontsize=8)

    # Show the hardest case Coates can still fully handle -- the highest flux at
    # which the whole cycle remains observable. Beyond it the return itself falls
    # outside the observable window, which the left panel already makes the point of.
    i_show = int(np.nonzero(full_window)[0][-1])
    phi = scene(scales[i_show])
    ts = simulate_photon_timestamps(phi, n_pulses, free_running=False, generator=gen(7))
    hist = timestamps_to_histogram(ts, N_BINS)
    rec, alive = coates(hist, n_pulses)
    recn = (rec / rec.sum()).numpy()
    valid = (alive >= 0.05).numpy()
    axes[1].plot(d, (hist / hist.sum()).numpy(), color="tab:red", lw=1.3, label="raw")
    axes[1].plot(d, np.where(valid, recn, np.nan), color="tab:blue", lw=1.3, label="Coates")
    axes[1].plot(d, np.where(~valid, recn, np.nan), color="tab:blue", lw=0.8, ls=":",
                 alpha=0.5, label="Coates (<5% of cycles alive)")
    pn = phi.numpy()
    axes[1].plot(d, pn / pn.sum(), color="k", lw=1.1, ls="--", label="true φ")
    axes[1].axvline(true_depth, color="k", lw=0.8, alpha=0.4, label="true centroid")
    axes[1].set_xlabel("depth (m)")
    axes[1].set_ylabel("area-normalised")
    axes[1].set_title(f"Hardest fully-observable case "
                      f"({photons[i_show]:.2f} photons/cycle)", fontsize=10)
    axes[1].legend(fontsize=8)

    fig.suptitle(
        "16 — The measurement consequence: uncorrected gated single-hit ranging is "
        "biased short at high flux, and the bias grows with flux.", fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(outdir / "16_depth_bias.png", dpi=130)
    plt.close(fig)

    check("depth: low-flux estimate is unbiased",
          abs(raw_err[0]) <= 2 * DEPTH_PER_BIN_M,
          f"{raw_err[0]:+.3f} m at {photons[0]:.3f} photons/cycle")
    check("depth: raw estimate is biased short, and worsens monotonically with flux",
          raw_err[-1] < -1.0 and bool(np.all(np.diff(raw_err) < 0.02)),
          f"{raw_err[0]:+.3f} m -> {raw_err[-1]:+.3f} m")
    check("depth: Coates is exact while the whole cycle stays observable",
          float(np.abs(cor_err[full_window]).max()) <= 3 * DEPTH_PER_BIN_M,
          f"worst {cor_err[full_window][np.abs(cor_err[full_window]).argmax()]:+.3f} m "
          f"over {int(full_window.sum())}/{len(scales)} flux levels "
          f"(bin = {DEPTH_PER_BIN_M:.3f} m)")
    check("depth: Coates stays accurate over whatever window remains observable",
          float(np.abs(win_err).max()) <= 3 * DEPTH_PER_BIN_M,
          f"worst {win_err[np.abs(win_err).argmax()]:+.3f} m, "
          f"down to {visible.min()*100:.0f}% of the cycle visible")
    # Where the window starts closing is the operating limit of gated single-hit:
    # past it the far part of the cycle is never sampled, and that is missing data
    # rather than distortion, so no estimator recovers it.
    onset = photons[~full_window][0] if (~full_window).any() else float("inf")
    check("depth: observable window shrinks monotonically with flux",
          bool(np.all(np.diff(visible) <= 1e-9)),
          f"100% -> {visible.min()*100:.0f}% of the cycle")
    check("depth: gated single-hit operating limit identified",
          onset < photons[-1],
          f"whole cycle observable below {onset:.2f} photons/cycle; "
          f"beyond that the tail is truncated, not merely distorted")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=Path(__file__).parent / "aspc" / "figures")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"laser {FREQ_HZ/1e6:.0f} MHz, {N_BINS} bins "
          f"({BIN_S*1e12:.0f} ps/bin, {DEPTH_PER_BIN_M*100:.1f} cm/bin), "
          f"unambiguous range {MAX_DEPTH_M:.2f} m\n")

    fig_raster(args.outdir)
    fig_gaps(args.outdir)
    fig_rate_vs_flux(args.outdir)
    fig_pileup(args.outdir)
    fig_mode_comparison(args.outdir)
    fig_deadtime_over_cycle(args.outdir)
    fig_depth_bias(args.outdir)

    width = max(len(n) for n, _, _ in CHECKS)
    print("=" * (width + 30))
    for name, ok, detail in CHECKS:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    print("=" * (width + 30))
    n_fail = sum(1 for _, ok, _ in CHECKS if not ok)
    print(f"{len(CHECKS) - n_fail}/{len(CHECKS)} checks passed")
    print(f"figures written to {args.outdir.resolve()}")

    # Persist the results so the HTML report quotes this run rather than numbers
    # transcribed by hand, which would silently go stale.
    results = args.outdir.parent / "detection_checks.tsv"
    results.write_text(
        "".join(f"{'PASS' if ok else 'FAIL'}\t{name}\t{detail}\n" for name, ok, detail in CHECKS)
    )
    print(f"check results written to {results.resolve()}")

    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
