from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from visionsim.simulate.heatsim import laplacian
from visionsim.simulate.heatsim.physics import AMBIENT_TEMPERATURE_K, STEFAN_BOLTZMANN_MM
from visionsim.simulate.heatsim.solver import HeatSimFEM, scipy_to_torch_sparse

# Minimal surrogates for HeatSimFEM's two positional param objects: gen_params
# (.device, .RHO, .C, .K, .NUM_FRAME_DELTA) and sim_params (.sim_radiation,
# .sim_convection, .add_tikhonov_reg, .sim_time, .record_time).
# record_time == sim_time records every step, so histories come back (sim_steps + 1, N).

_DT = 0.05  # seconds per timestep
_NUM_STEPS = 5
_SIM_TIME = _NUM_STEPS * _DT  # 0.25 s


def _make_params(n: int):
    gen_params = SimpleNamespace(
        device="cpu",
        RHO=1330.0 / 1e9,  # kg/mm^3  (fallback scalar; per-vertex maps override)
        C=880.0,  # J/(kg·K)
        K=0.17,  # mm^2/s
        NUM_FRAME_DELTA=_DT * 60.0,  # → dt = NUM_FRAME_DELTA / 60 = _DT
    )
    sim_params = SimpleNamespace(
        sim_radiation=True,
        sim_convection=False,  # CONVECTION_COEFF is 0 anyway
        add_tikhonov_reg=False,
        sim_time=_SIM_TIME,
        record_time=_SIM_TIME,  # record_attimestep = 0 → record all steps
    )
    return gen_params, sim_params


@pytest.mark.skipif(
    not laplacian.HAS_ROBUST_LAPLACIAN,
    reason="robust_laplacian lives in Blender's Python; it is not a host dependency on Windows",
)
def test_solver_produces_finite_physical_temperatures():
    """
    Run a 5-step implicit-Euler FEM solve on a tiny random point cloud (POINTS/ROBUST).
    Asserts: no NaN, all temperatures in the physical range [200 K, 2000 K].
    """
    n = 64
    rng = np.random.default_rng(0)
    points = rng.uniform(-10.0, 10.0, size=(n, 3)).astype(np.float64)  # mm
    irradiance = np.full(n, 1e-4, dtype=np.float64)  # W/mm^2

    # Per-vertex material maps (PVC-like)
    density = np.full(n, 1330.0 / 1e9, dtype=np.float64)  # kg/mm^3
    specific_heat = np.full(n, 880.0, dtype=np.float64)  # J/(kg·K)
    tdiff = np.full(n, 0.17, dtype=np.float64)  # mm^2/s
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

    # In POINTS mode faces_np and boundary_faces_np are unused; pass None.
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


def test_only_robust_point_cloud_method_is_accepted():
    gen_params, sim_params = _make_params(4)
    with pytest.raises(ValueError, match="POINTS"):
        HeatSimFEM(gen_params, sim_params, laplacian_domain="MESH")
    with pytest.raises(ValueError, match="robust"):
        HeatSimFEM(gen_params, sim_params, laplacian_backend="IGL")


@pytest.mark.parametrize("initial_K", [295.0, 280.0])
def test_small_heating_is_resolved_across_widely_different_mass_scales(initial_K):
    """An absolute-Kelvin residual must not bury weak flux on dense/sparse mixed samples."""
    n = 4
    gen, sim = _make_params(n)
    fem = HeatSimFEM(gen, sim)
    mass = scipy_to_torch_sparse(sp.diags([1e-4, 1.0, 1e4, 1e8]), "cpu")
    lap = scipy_to_torch_sparse(sp.coo_matrix((n, n)), "cpu")
    rho = np.full(n, gen.RHO)
    heat = np.full(n, gen.C)
    eps = np.full(n, 0.9)
    flux = 1e-6  # 1 W/m^2
    steps = 10
    result = fem._simulate_heat_torch(
        np.full(n, initial_K),
        lap,
        mass,
        mass,
        np.ones(n),
        np.full(n, flux),
        None,
        rho,
        heat,
        eps,
        _DT,
        steps,
    )
    ambient = AMBIENT_TEMPERATURE_K
    radiation = 4 * STEFAN_BOLTZMANN_MM * eps[0] * ambient**3 / (rho[0] * heat[0])
    theta = initial_K - ambient
    expected = [initial_K]
    for _ in range(steps):
        theta = (theta + _DT * flux / (rho[0] * heat[0])) / (1 + _DT * radiation)
        expected.append(theta + ambient)
    tolerance = 2e-7 if initial_K == ambient else 2e-5
    np.testing.assert_allclose(
        result, np.broadcast_to(np.asarray(expected)[:, None], result.shape), atol=tolerance, rtol=0
    )


def test_ambient_without_heating_is_preserved_exactly():
    n = 4
    gen, sim = _make_params(n)
    fem = HeatSimFEM(gen, sim)
    mass = scipy_to_torch_sparse(sp.diags([1e-4, 1.0, 1e4, 1e8]), "cpu")
    lap = scipy_to_torch_sparse(sp.coo_matrix((n, n)), "cpu")
    result = fem._simulate_heat_torch(
        np.full(n, AMBIENT_TEMPERATURE_K),
        lap,
        mass,
        mass,
        np.ones(n),
        np.zeros(n),
        None,
        np.full(n, gen.RHO),
        np.full(n, gen.C),
        np.full(n, 0.9),
        _DT,
        10,
    )
    assert np.all(result == AMBIENT_TEMPERATURE_K)
