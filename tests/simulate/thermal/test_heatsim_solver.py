from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from visionsim.simulate.heatsim import laplacian
from visionsim.simulate.heatsim.physics import AMBIENT_TEMPERATURE_K, STEFAN_BOLTZMANN_MM
from visionsim.simulate.heatsim.solver import ConjugateGradientError, HeatSimFEM, pcg_solve, scipy_to_torch_sparse

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


def test_sparse_conversion_preserves_requested_double_precision():
    values = np.array([1.0 + 2.0**-30, 1.0 - 2.0**-30])
    matrix = scipy_to_torch_sparse(sp.diags(values), "cpu", torch.float64)
    np.testing.assert_array_equal(matrix.values().numpy(), values)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pcg_checks_true_residual_on_stiff_system(dtype):
    # Two strongly coupled samples: recursive CG residuals falsely converge in
    # float32, while the actual residual still exceeds the requested tolerance.
    matrix = torch.tensor([[100001.0, -100000.0], [-100000.0, 100001.0]], dtype=dtype)
    rhs = torch.tensor([[1.0], [2.0]], dtype=dtype)
    inverse_diagonal = 1.0 / matrix.diag().unsqueeze(1)
    try:
        solution = pcg_solve(lambda x: matrix @ x, rhs, inverse_diagonal, max_iter=1000, rtol=1e-6)
    except ConjugateGradientError:
        assert dtype == torch.float32
        return
    assert torch.linalg.vector_norm(rhs - matrix @ solution) <= 1e-6 * torch.linalg.vector_norm(rhs)


def _two_node_pose(monkeypatch):
    """An analytic two-node conduction system, independent of point-cloud meshing."""
    from visionsim.simulate.heatsim import solver

    builds = []

    def matrices(points, **kwargs):
        builds.append(points.copy())
        weight = 1.0 / np.linalg.norm(points[0] - points[1])
        return sp.csr_matrix([[weight, -weight], [-weight, weight]]), sp.eye(2, format="csr")

    monkeypatch.setattr(solver, "point_cloud_laplacian_and_mass", matrices)
    gen, sim = _make_params(2)
    gen.K = 1.0
    sim.sim_radiation = False
    fem = HeatSimFEM(gen, sim, dtype=torch.float64)
    args = {
        "verts_np": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "faces_np": None,
        "boundary_verts_mask": np.ones(2),
        "irradiance_map": np.zeros(2),
        "thermal_diffusivity_map": np.ones(2),
        "density_map": np.ones(2),
        "specific_heat_map": np.ones(2),
        "emissivity_map": np.zeros(2),
        "dt": 1.0,
        "num_substeps": 1,
    }
    return fem, args, builds


def test_pose_steps_continue_the_same_temperature_history(monkeypatch):
    fem, args, builds = _two_node_pose(monkeypatch)
    initial = np.array([400.0, 295.0])
    continuous = fem.simulate_for_pose(**{**args, "num_substeps": 5}, u_prev=initial)
    first = fem.simulate_for_pose(**{**args, "num_substeps": 2}, u_prev=initial)
    second = fem.simulate_for_pose(
        **{**args, "num_substeps": 3},
        u_prev=first[-1],
        rebuild_matrices=False,
    )
    np.testing.assert_allclose(np.vstack([first, second]), continuous, atol=1e-10, rtol=0)
    # Closed equal-capacity bodies conserve their mean temperature.
    np.testing.assert_allclose(continuous.mean(axis=1), initial.mean(), atol=1e-10, rtol=0)
    assert len(builds) == 2


def test_pose_reservoir_is_enforced_in_free_node_equations(monkeypatch):
    fem, args, _ = _two_node_pose(monkeypatch)
    pinned = {"dirichlet_indices": np.array([0]), "dirichlet_values": np.array([400.0])}
    states = fem.simulate_for_pose(**{**args, "num_substeps": 2}, u_prev=np.array([295.0, 295.0]), **pinned)
    # Implicit Euler with a genuinely fixed reservoir: T_next=(T_prev+400)/2.
    # Solving both nodes then clamping the reservoir would fail this check.
    np.testing.assert_allclose(states, [[400.0, 347.5], [400.0, 373.75]], atol=1e-10, rtol=0)
    states = fem.simulate_for_pose(
        **args,
        u_prev=states[-1],
        rebuild_matrices=False,
        dirichlet_indices=[0],
        dirichlet_values=[300.0],
    )
    np.testing.assert_allclose(states, [[300.0, 336.875]], atol=1e-10, rtol=0)


def test_pose_refresh_changes_coupling_and_reuse_holds_old_pose(monkeypatch):
    fem, args, builds = _two_node_pose(monkeypatch)
    initial = np.array([400.0, 295.0])
    old = fem.simulate_for_pose(**args, u_prev=initial)
    moved = {**args, "verts_np": np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])}
    reused = fem.simulate_for_pose(**moved, u_prev=initial, rebuild_matrices=False)
    refreshed = fem.simulate_for_pose(**moved, u_prev=initial)
    np.testing.assert_allclose(reused, old, atol=1e-10, rtol=0)
    assert refreshed[0, 1] < old[0, 1]
    assert len(builds) == 2


def test_pose_recovery_reuses_original_pose_coefficients(monkeypatch):
    fem, args, builds = _two_node_pose(monkeypatch)
    fem.dtype = torch.float32
    initial = np.array([400.0, 295.0])
    reference = fem.simulate_for_pose(**args, u_prev=initial)
    advance = fem._simulate_heat_torch

    def fail_single_precision(*a, **kw):
        if fem.dtype == torch.float32:
            raise ConjugateGradientError("injected float32 failure")
        return advance(*a, **kw)

    monkeypatch.setattr(fem, "_simulate_heat_torch", fail_single_precision)
    moved = {**args, "verts_np": np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])}
    recovered = fem.simulate_for_pose(**moved, u_prev=initial, rebuild_matrices=False)
    assert fem.dtype == torch.float64
    assert len(builds) == 1
    np.testing.assert_allclose(recovered, reference, atol=1e-5, rtol=0)


def test_pose_reuse_rejects_missing_or_changed_system(monkeypatch):
    fem, args, _ = _two_node_pose(monkeypatch)
    initial = np.array([400.0, 295.0])
    with pytest.raises(ValueError, match="before reusing"):
        fem.simulate_for_pose(**args, u_prev=initial, rebuild_matrices=False)
    fem.simulate_for_pose(**args, u_prev=initial)
    with pytest.raises(ValueError, match="diffusivity changes"):
        fem.simulate_for_pose(
            **{**args, "thermal_diffusivity_map": np.array([1.0, 2.0])},
            u_prev=initial,
            rebuild_matrices=False,
        )


def test_zero_pose_steps_initialize_cache_without_advancing(monkeypatch):
    fem, args, builds = _two_node_pose(monkeypatch)
    initial = np.array([400.0, 295.0])
    states = fem.simulate_for_pose(**{**args, "num_substeps": 0}, u_prev=initial)
    assert states.shape == (0, 2)
    assert len(builds) == 1
    advanced = fem.simulate_for_pose(**args, u_prev=initial, rebuild_matrices=False)
    assert advanced.shape == (1, 2)
    assert advanced[0, 1] > initial[1]
    assert len(builds) == 1


def test_pcg_solve_preconditioned_residual_bounds_small_mass_rows():
    """A dominant large-mass row must not hide error in a coupled small-mass row."""
    diagonal = np.array([1e-4, 1.0, 1e4, 1e8], dtype=np.float64)
    matrix = np.diag(diagonal)
    for index in range(3):
        matrix[index, index + 1] = matrix[index + 1, index] = -1.0
    rhs = np.array([1e-6, 1.0, 1e4, 1e8], dtype=np.float64)
    expected = np.linalg.solve(matrix, rhs)

    operator = torch.tensor(matrix, dtype=torch.float32)
    b = torch.tensor(rhs[:, None], dtype=torch.float32)
    inverse_diagonal = 1.0 / torch.tensor(diagonal, dtype=torch.float32)[:, None]
    result = pcg_solve(lambda value: operator @ value, b, inverse_diagonal, tol=1e-6, rtol=1e-6, max_iter=1000)

    np.testing.assert_allclose(result.numpy().ravel(), expected, atol=1e-2, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pcg_preconditioned_residual_on_positive_definite_mass_system(dtype):
    """Small-mass rows remain accurate in a system satisfying CG's SPD contract."""
    diagonal = np.array([1e-4, 1.0, 1e4, 1e8])
    matrix = np.diag(diagonal)
    for index in range(3):
        coupling = -0.1 * np.sqrt(diagonal[index] * diagonal[index + 1])
        matrix[index, index + 1] = matrix[index + 1, index] = coupling
    np.linalg.cholesky(matrix)
    rhs = np.array([1e-6, 1.0, 1e4, 1e8])
    expected = np.linalg.solve(matrix, rhs)
    operator = torch.tensor(matrix, dtype=dtype)
    b = torch.tensor(rhs[:, None], dtype=dtype)
    inverse_diagonal = 1.0 / torch.tensor(diagonal[:, None], dtype=dtype)
    try:
        result = pcg_solve(lambda value: operator @ value, b, inverse_diagonal, rtol=1e-6, max_iter=1000)
    except ConjugateGradientError:
        # Reject a result float32 cannot verify; the pose solver retries in float64.
        assert dtype == torch.float32
        return
    np.testing.assert_allclose(result.numpy().ravel(), expected, atol=1e-2, rtol=1e-3)
    residual = b - operator @ result
    assert torch.linalg.vector_norm(residual) <= 1e-6 or torch.linalg.vector_norm(
        inverse_diagonal * residual
    ) <= 1e-6 * torch.linalg.vector_norm(inverse_diagonal * b)
