from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from visionsim.simulate.heatsim.solver import HeatSimFEM

# ---------------------------------------------------------------------------
# Minimal surrogate objects for gen_params / sim_params
# ---------------------------------------------------------------------------
# HeatSimFEM.__init__ expects two positional objects:
#   gen_params  — accessed for .device, .RHO, .C, .K, .NUM_FRAME_DELTA
#   sim_params  — accessed for .sim_radiation, .sim_convection,
#                 .add_tikhonov_reg, .sim_time, .record_time
#
# perform_gt_heat_simulation derives dt = gen_params.NUM_FRAME_DELTA / 60.0
# and sim_steps = int(sim_params.sim_time / dt).
#
# Setting record_time == sim_time makes record_attimestep = 0, which triggers
# the "record all" branch: the returned array has shape (sim_steps + 1, N).
# ---------------------------------------------------------------------------

_DT = 0.05  # seconds per timestep
_NUM_STEPS = 5
_SIM_TIME = _NUM_STEPS * _DT  # 0.25 s


def _make_params(n: int):
    gen_params = SimpleNamespace(
        device="cpu",
        RHO=1330.0 / 1e9,  # kg/mm^3  (fallback scalar; per-vertex maps override)
        C=880.0,            # J/(kg·K)
        K=0.17,             # mm^2/s
        NUM_FRAME_DELTA=_DT * 60.0,  # → dt = NUM_FRAME_DELTA / 60 = _DT
    )
    sim_params = SimpleNamespace(
        sim_radiation=True,
        sim_convection=False,   # CONVECTION_COEFF is 0 anyway
        add_tikhonov_reg=False,
        sim_time=_SIM_TIME,
        record_time=_SIM_TIME,  # record_attimestep = 0 → record all steps
    )
    return gen_params, sim_params


def test_solver_produces_finite_physical_temperatures():
    """
    Run a 5-step implicit-Euler FEM solve on a tiny random point cloud (POINTS/ROBUST).
    Asserts: no NaN, all temperatures in the physical range [200 K, 2000 K].
    """
    n = 64
    rng = np.random.default_rng(0)
    points = rng.uniform(-10.0, 10.0, size=(n, 3)).astype(np.float64)   # mm
    irradiance = np.full(n, 1e-4, dtype=np.float64)                     # W/mm^2

    # Per-vertex material maps (PVC-like)
    density = np.full(n, 1330.0 / 1e9, dtype=np.float64)    # kg/mm^3
    specific_heat = np.full(n, 880.0, dtype=np.float64)      # J/(kg·K)
    tdiff = np.full(n, 0.17, dtype=np.float64)               # mm^2/s
    emissivity = np.full(n, 0.9, dtype=np.float64)

    # Initial temperature (slightly above ambient)
    u0 = np.full(n, 295.0, dtype=np.float64)

    gen_params, sim_params = _make_params(n)

    fem = HeatSimFEM(
        gen_params,
        sim_params,
        laplacian_domain="POINTS",
        laplacian_backend="ROBUST",
    )

    # perform_gt_heat_simulation real signature:
    #   verts_np, faces_np, boundary_faces_np,
    #   boundary_verts_mask_override=None, u0=None, irradiance_map=None,
    #   thermal_diffusivity_map=None, density_map=None, specific_heat_map=None,
    #   emissivity_map=None, steady_state=False, tol_K_per_s=0.0,
    #   store_only_final=False
    #
    # In POINTS mode, faces_np and boundary_faces_np are unused; pass None.
    history = fem.perform_gt_heat_simulation(
        verts_np=points,
        faces_np=None,
        boundary_faces_np=None,
        u0=u0,
        irradiance_map=irradiance,
        thermal_diffusivity_map=tdiff,
        density_map=density,
        specific_heat_map=specific_heat,
        emissivity_map=emissivity,
    )

    arr = np.asarray(history, dtype=np.float64)

    # Shape: (sim_steps + 1, n)  — initial state + _NUM_STEPS post-step states
    assert arr.ndim == 2, f"Expected 2-D array, got shape {arr.shape}"
    assert arr.shape[1] == n, f"Expected {n} vertices, got {arr.shape[1]}"
    assert not np.isnan(arr).any(), "NaN values found in temperature history"
    assert arr.min() > 200.0, f"Temperature below 200 K: min={arr.min():.2f}"
    assert arr.max() < 2000.0, f"Temperature above 2000 K: max={arr.max():.2f}"


def _run_history(solver_mode: str) -> np.ndarray:
    """Same 5-step solve as above, run through one linear-solver mode."""
    n = 64
    rng = np.random.default_rng(0)
    points = rng.uniform(-10.0, 10.0, size=(n, 3)).astype(np.float64)
    irradiance = np.full(n, 1e-4, dtype=np.float64)

    density = np.full(n, 1330.0 / 1e9, dtype=np.float64)
    specific_heat = np.full(n, 880.0, dtype=np.float64)
    tdiff = np.full(n, 0.17, dtype=np.float64)
    emissivity = np.full(n, 0.9, dtype=np.float64)
    u0 = np.full(n, 295.0, dtype=np.float64)

    gen_params, sim_params = _make_params(n)
    fem = HeatSimFEM(
        gen_params,
        sim_params,
        laplacian_domain="POINTS",
        laplacian_backend="ROBUST",
        solver_mode=solver_mode,
    )
    history = fem.perform_gt_heat_simulation(
        verts_np=points,
        faces_np=None,
        boundary_faces_np=None,
        u0=u0,
        irradiance_map=irradiance,
        thermal_diffusivity_map=tdiff,
        density_map=density,
        specific_heat_map=specific_heat,
        emissivity_map=emissivity,
    )
    return np.asarray(history, dtype=np.float64)


# Agreement bound between the two linear-solver modes, in Kelvin. Both stop on the
# same ``||r||_2 < 1e-5`` criterion but reach it along different iteration paths, and
# the solve runs in float32, where one ulp at 295 K is already 3.1e-5 K. A few dozen
# ulps of accumulated drift is therefore the expected disagreement; anything that
# changes the physics moves temperatures by O(0.1) K or more and still trips this.
_SOLVER_MODE_TOL_K = 5e-3


def test_pcg_jacobi_matches_unpreconditioned_cg():
    """The Jacobi preconditioner must change the iteration count, not the answer.

    Guards the default flipping to ``pcg_jacobi`` from silently changing physics.
    On a 2000-point cloud the preconditioner cuts the solve from 51 matrix-vector
    products per timestep to 4, for a ~3x wall-clock win.
    """
    pcg = _run_history("pcg_jacobi")
    cg = _run_history("cg")

    assert pcg.shape == cg.shape
    assert not np.isnan(pcg).any()
    max_abs = float(np.abs(pcg - cg).max())
    assert max_abs < _SOLVER_MODE_TOL_K, f"pcg_jacobi and cg disagree by {max_abs:.3e} K"
