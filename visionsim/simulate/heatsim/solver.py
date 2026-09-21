"""Transient heat integration on a robust point-cloud Laplacian."""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp
import torch

from visionsim.simulate.heatsim.laplacian import point_cloud_laplacian_and_mass
from visionsim.simulate.heatsim.physics import AMBIENT_TEMPERATURE_K, STEFAN_BOLTZMANN_MM

_log = logging.getLogger("rich")

def scipy_to_torch_sparse(mat, device, dtype=torch.float32):
    """
    Convert a SciPy sparse matrix to a torch.sparse_coo_tensor on a given device.
    """
    mat = mat.tocoo()
    indices = np.vstack([mat.row, mat.col]).astype(np.int64)
    i = torch.from_numpy(indices).to(device)
    v = torch.from_numpy(mat.data.astype(np.float32)).to(device)
    return torch.sparse_coo_tensor(i, v, mat.shape, device=device, dtype=dtype, check_invariants=True).coalesce()


def sparse_diag(A):
    """Extract the diagonal of a coalesced torch sparse COO tensor as a dense (N,) vector."""
    A = A.coalesce()
    idx = A.indices()
    val = A.values()
    n = A.shape[0]
    d = torch.zeros(n, device=val.device, dtype=val.dtype)
    mask = idx[0] == idx[1]
    d[idx[0][mask]] = val[mask]
    return d


@torch.no_grad()
def pcg_solve(mv, b, Minv, x0=None, tol=1e-6, max_iter=200):
    """
    Jacobi (diagonal) preconditioned Conjugate Gradient for SPD systems A x = b,
    where mv(x) computes A @ x and Minv is the (N,1) inverse-diagonal preconditioner.

    """
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()

    r = b - mv(x)
    if torch.dot(r.flatten(), r.flatten()).sqrt() < tol:
        return x

    z = Minv * r
    p = z.clone()
    rz_old = torch.dot(r.flatten(), z.flatten())

    for _ in range(max_iter):
        Ap = mv(p)
        denom = torch.dot(p.flatten(), Ap.flatten())
        if denom.abs() < 1e-20:
            raise RuntimeError("Thermal conjugate-gradient solve broke down before convergence")
        alpha = rz_old / denom

        x = x + alpha * p
        r = r - alpha * Ap

        if torch.dot(r.flatten(), r.flatten()).sqrt() < tol:
            return x

        z = Minv * r
        rz_new = torch.dot(r.flatten(), z.flatten())
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    residual = float(torch.linalg.vector_norm(r))
    raise RuntimeError(f"Thermal conjugate-gradient solve did not converge: residual={residual:.3g}")


class HeatSimFEM:
    """
    Memory-efficient heat simulation using torch sparse and CG.
    Sparse matrices are used and no dense system matrices are formed.
    """

    def __init__(self, gen_params, sim_params, **kwargs) -> None:
        self.gen_params = gen_params
        self.sim_params = sim_params

        if hasattr(gen_params, "device"):
            self.device = torch.device(gen_params.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Mesh + irradiance
        self.verts_np = kwargs.get("verts_np", None)
        self.faces_np = kwargs.get("faces_np", None)
        self.irradiance_map = kwargs.get("irradiance_map", None)
        # Optional spatially varying material fields (per-vertex):
        # - thermal_diffusivity_map: mm^2/s
        # - density_map: kg/mm^3
        # - specific_heat_map: J/(kg*K)
        # - emissivity_map: unitless [0,1] (used for radiation boundary term)
        self.thermal_diffusivity_map = kwargs.get("thermal_diffusivity_map", None)
        self.density_map = kwargs.get("density_map", None)
        self.specific_heat_map = kwargs.get("specific_heat_map", None)
        self.emissivity_map = kwargs.get("emissivity_map", None)

        if kwargs.get("laplacian_domain", "POINTS") != "POINTS":
            raise ValueError("Only POINTS thermal solves are supported")
        if kwargs.get("laplacian_backend", "ROBUST") != "ROBUST":
            raise ValueError("Only the robust Laplacian is supported")
        self.robust_mollify_factor = float(kwargs.get("robust_mollify_factor", 1e-5))
        self.pointcloud_neighbors = int(kwargs.get("pointcloud_neighbors", 30))

        if self.irradiance_map is not None:
            _log.debug("irradiance_map %s", self.irradiance_map.shape)

    # ------------------------------------------------------------------
    # Core sparse setup + simulation
    # ------------------------------------------------------------------

    def _simulate_heat_torch(
        self,
        u0_np,
        L_t,
        M_t,
        M_boundary_t,
        boundary_mask_np,
        irradiance_map_np,
        alpha_np,
        rho_np,
        c_np,
        eps_np,
        dt,
        num_steps,
        steady_state: bool = False,
        tol_K_per_s: float = 0.0,
        store_only_final: bool = False,
    ):
        """
        Main implicit Euler heat simulation using CG, in torch.
        """
        reg_value = 1e-8 if self.sim_params.add_tikhonov_reg else 0.0

        # Initial condition
        u_prev = torch.from_numpy(u0_np.reshape(-1).astype(np.float32)).to(
            self.device
        )
        u_prev = u_prev.unsqueeze(1)  # (N,1)

        # ------------------------------------------------------------------
        # Build constants + (optionally) time-varying lighting source terms
        # ------------------------------------------------------------------

        # Allow per-vertex rho/c (otherwise fall back to gen_params scalars)
        if rho_np is None:
            rho_np = np.full_like(boundary_mask_np, float(self.gen_params.RHO), dtype=np.float64)
        if c_np is None:
            c_np = np.full_like(boundary_mask_np, float(self.gen_params.C), dtype=np.float64)
        if eps_np is None:
            eps_np = np.full_like(boundary_mask_np, 0.9, dtype=np.float64)
        eps_np = np.clip(eps_np, 0.0, 1.0)

        rho_t = torch.from_numpy(rho_np.astype(np.float32)).to(self.device)
        c_t = torch.from_numpy(c_np.astype(np.float32)).to(self.device)
        rc_t = rho_t * c_t
        eps_t = torch.from_numpy(eps_np.astype(np.float32)).to(self.device)

        boundary_mask = torch.from_numpy(boundary_mask_np.astype(np.float32)).to(self.device)

        sigma = STEFAN_BOLTZMANN_MM
        Tamb = AMBIENT_TEMPERATURE_K
        h = 0.0

        vec_rad_A = None
        vec_conv_A = None
        if self.sim_params.sim_radiation:
            vec_rad_A = boundary_mask * dt * 4.0 * sigma * eps_t * (Tamb**3) / rc_t
        if self.sim_params.sim_convection:
            vec_conv_A = boundary_mask * dt * h / rc_t

        vec_rad_rhs = torch.zeros_like(boundary_mask)
        vec_conv_rhs = torch.zeros_like(boundary_mask)
        if self.sim_params.sim_radiation:
            vec_rad_rhs = boundary_mask * dt * 4.0 * sigma * eps_t * (Tamb**4) / rc_t
        if self.sim_params.sim_convection:
            vec_conv_rhs = boundary_mask * dt * h * Tamb / rc_t

        # Constant part of RHS from radiation+convection only
        rhs_const = (vec_rad_rhs + vec_conv_rhs).unsqueeze(1)  # (N,1)
        B_rad_conv_const = torch.sparse.mm(M_boundary_t, rhs_const)  # (N,1)

        # Base irradiance term (constant over time)
        if irradiance_map_np is None:
            irradiance_map_np = np.zeros_like(boundary_mask_np, dtype=np.float32)
        irr_base_t = torch.from_numpy(np.asarray(irradiance_map_np, dtype=np.float32)).to(self.device)
        vec_light_base = (boundary_mask * dt * irr_base_t / rc_t).unsqueeze(1)
        B_light_base = torch.sparse.mm(M_boundary_t, vec_light_base)  # (N,1)

        # Debug ranges
        try:
            _log.debug(
                "DEBUG: B_rad_conv_const range: [%s, %s]",
                f"{B_rad_conv_const.min().item():.10f}",
                f"{B_rad_conv_const.max().item():.10f}",
            )
            _log.debug(
                "DEBUG: B_light_base range: [%s, %s]",
                f"{B_light_base.min().item():.10f}",
                f"{B_light_base.max().item():.10f}",
            )
        except Exception:
            logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)
        # Pre-define matrix-free operator A(u)
        def mv(x):
            # x: (N,1)
            # M @ x
            Mx = torch.sparse.mm(M_t, x)
            # L @ x
            Lx = torch.sparse.mm(L_t, x)
            # Basic diffusion term
            if alpha_np is None:
                # Backward-compatible scalar diffusivity
                K = float(self.gen_params.K)
                out = Mx - K * dt * Lx
            else:
                # Spatially varying diffusivity (mm^2/s) as a per-vertex diagonal scaling.
                # This is an approximation; we build a weighted Laplacian separately in _build_matrices.
                out = Mx - dt * Lx

            # Radiation / convection on A side
            if vec_rad_A is not None:
                tmp = (vec_rad_A.unsqueeze(1) * x)
                out = out + torch.sparse.mm(M_boundary_t, tmp)
            if vec_conv_A is not None:
                tmp = (vec_conv_A.unsqueeze(1) * x)
                out = out + torch.sparse.mm(M_boundary_t, tmp)

            if reg_value > 0.0:
                out = out + reg_value * x

            return out

        # Run simulation
        u0_cpu = u_prev.detach().cpu().numpy().astype(np.float64).reshape(-1)
        us = []
        if not store_only_final:
            us.append(u0_cpu)

        # Steady-state diagnostics
        WARMUP_STEPS = 5
        WINDOW = 10
        PRINT_EVERY = 20
        recent_changes = []
        converged = False
        max_dT = float("inf")
        nonmonotonic_warned = False

        # Constant RHS across the time loop (no time-varying lighting).
        B_step = B_rad_conv_const + B_light_base

        # ------------------------------------------------------------------
        # Jacobi (diagonal) preconditioner for the constant operator A.
        # A = M - K*dt*L (+ radiation/convection boundary diag + Tikhonov reg).
        # Computed ONCE since A does not change across timesteps.
        # ------------------------------------------------------------------
        diagA = sparse_diag(M_t)
        dL = sparse_diag(L_t)
        if alpha_np is None:
            diagA = diagA - float(self.gen_params.K) * dt * dL
        else:
            diagA = diagA - dt * dL
        if vec_rad_A is not None or vec_conv_A is not None:
            dMb = sparse_diag(M_boundary_t)
            if vec_rad_A is not None:
                diagA = diagA + dMb * vec_rad_A
            if vec_conv_A is not None:
                diagA = diagA + dMb * vec_conv_A
        if reg_value > 0.0:
            diagA = diagA + reg_value
        Minv = (1.0 / torch.clamp(diagA, min=1e-12)).unsqueeze(1)

        for step in range(num_steps):
            # b = M @ u_prev + B_step
            b = torch.sparse.mm(M_t, u_prev) + B_step

            # Use previous solution as initial guess for faster convergence.
            u_next = pcg_solve(mv, b, Minv, x0=u_prev, tol=1e-5, max_iter=200)
            if not bool(torch.isfinite(u_next).all()):
                raise RuntimeError(f"Thermal solve produced non-finite temperatures at step {step + 1}")

            max_dT = float((u_next - u_prev).abs().max().item())
            # Discrete approximation to ||dT/dt||_inf in K/s. dt-invariant.
            rate_K_per_s = max_dT / dt if dt > 0.0 else float("inf")
            if step < 3 or (step + 1) % PRINT_EVERY == 0:
                _log.debug(
                    "FEM step %d: u range [%.4f, %.4f] max_dT=%.6f K  rate=%.6f K/s",
                    step,
                    u_next.min().item(),
                    u_next.max().item(),
                    max_dT,
                    rate_K_per_s,
                )

            if not store_only_final:
                us.append(u_next.detach().cpu().numpy().astype(np.float64).reshape(-1))
            u_prev = u_next

            if steady_state and step >= WARMUP_STEPS:
                recent_changes.append(rate_K_per_s)
                if len(recent_changes) > WINDOW:
                    recent_changes.pop(0)
                # Compare the mean over the window so per-step CG/float-point
                # wobble doesn't keep us from declaring convergence once the
                # system has plateaued.
                if len(recent_changes) >= WINDOW:
                    mean_rate = float(sum(recent_changes) / len(recent_changes))
                    if mean_rate < tol_K_per_s:
                        converged = True
                        _log.debug(
                            "[HeatSim:FEM] Steady-state converged at step "
                            "%d (mean rate over last %d steps "
                            "= %.6f K/s < tol=%.6f K/s; "
                            "latest rate=%.6f K/s)",
                            step + 1,
                            WINDOW,
                            mean_rate,
                            tol_K_per_s,
                            rate_K_per_s,
                        )
                        break
                if (
                    not nonmonotonic_warned
                    and len(recent_changes) == WINDOW
                    and recent_changes[-1] > recent_changes[0]
                ):
                    _log.debug(
                        "[HeatSim:FEM] WARNING: convergence rate not decreasing over "
                        "%d steps (latest %.6f K/s, "
                        "%d ago %.6f K/s). "
                        "Consider shrinking timestep_size.",
                        WINDOW,
                        recent_changes[-1],
                        WINDOW,
                        recent_changes[0],
                    )
                    nonmonotonic_warned = True

        if steady_state and not converged:
            final_rate = max_dT / dt if dt > 0.0 else float("inf")
            _log.debug(
                "[HeatSim:FEM] WARNING: did not reach tol=%.6f K/s in "
                "%d steps (final rate=%.6f K/s). "
                "Returning current state.",
                tol_K_per_s,
                num_steps,
                final_rate,
            )

        if store_only_final:
            final_cpu = u_prev.detach().cpu().numpy().astype(np.float64).reshape(-1)
            return np.stack([final_cpu], axis=0)  # (1, N)
        return np.stack(us, axis=0)  # (num_steps_taken+1, N)

    def _apply_vertex_weighted_laplacian(self, L: sp.spmatrix, alpha_vec: np.ndarray) -> sp.spmatrix:
        """
        Approximate variable-coefficient diffusion by scaling off-diagonal Laplacian entries.

        Assumes L is a (negative semidefinite) Laplacian-like matrix with:
        - off-diagonals >= 0
        - diagonal = -row_sum(offdiag)
        """
        alpha_vec = np.asarray(alpha_vec, dtype=np.float64).reshape(-1)
        L_coo = L.tocoo()  # type: ignore[attr-defined]
        rows = L_coo.row
        cols = L_coo.col
        data = L_coo.data.astype(np.float64)

        off = rows != cols
        rows_off = rows[off]
        cols_off = cols[off]
        data_off = data[off]

        # Scale edge weights by average alpha across the edge
        scale = 0.5 * (alpha_vec[rows_off] + alpha_vec[cols_off])
        data_off = data_off * scale

        # Rebuild with recomputed diagonal so rows sum to ~0
        Lw_off = sp.coo_matrix((data_off, (rows_off, cols_off)), shape=L.shape).tocsr()
        diag = -np.array(Lw_off.sum(axis=1)).reshape(-1)
        Lw = Lw_off + sp.diags(diag, format="csr")
        return Lw

    def _build_matrices(self, verts_np, faces_np=None, *, alpha_vec: np.ndarray | None = None):
        """Build the robust point-cloud Laplacian and mass matrix."""
        points = np.asarray(verts_np, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError("Thermal solve requires at least two finite 3D points")
        L_psd, M = point_cloud_laplacian_and_mass(
            points, mollify_factor=self.robust_mollify_factor, n_neighbors=self.pointcloud_neighbors
        )
        L = -L_psd
        if alpha_vec is not None:
            alpha = np.asarray(alpha_vec, dtype=np.float64).reshape(-1)
            if alpha.shape != (len(points),) or not np.isfinite(alpha).all() or np.any(alpha < 0):
                raise ValueError("Thermal diffusivity must be nonnegative and match the point count")
            L = self._apply_vertex_weighted_laplacian(L, alpha)
        return tuple(scipy_to_torch_sparse(matrix, self.device) for matrix in (L, M, M))

    def perform_gt_heat_simulation(
        self,
        verts_np,
        faces_np,
        boundary_faces_np,
        boundary_verts_mask_override=None,
        u0=None,
        irradiance_map=None,
        thermal_diffusivity_map=None,
        density_map=None,
        specific_heat_map=None,
        emissivity_map=None,
        steady_state: bool = False,
        tol_K_per_s: float = 0.0,
        store_only_final: bool = False,
    ):
        verts_np = np.asarray(verts_np, dtype=np.float64)
        valid_verts: np.ndarray = np.ones(len(verts_np), dtype=bool)

        if u0 is None:
            u0 = np.full((verts_np.shape[0],), 295.0, dtype=np.float64)
        else:
            u0 = u0[valid_verts].reshape(-1)

        # Material maps can be passed per-call (override constructor fields)
        if thermal_diffusivity_map is None:
            thermal_diffusivity_map = self.thermal_diffusivity_map
        if density_map is None:
            density_map = self.density_map
        if specific_heat_map is None:
            specific_heat_map = self.specific_heat_map
        if emissivity_map is None:
            emissivity_map = self.emissivity_map

        # Filter material vectors to valid verts (mesh mode) so they match u0/L/M.
        rho_f = c_f = alpha_f = eps_f = None
        if thermal_diffusivity_map is not None:
            alpha_f = np.asarray(thermal_diffusivity_map, dtype=np.float64).reshape(-1)[valid_verts]
        if density_map is not None:
            rho_f = np.asarray(density_map, dtype=np.float64).reshape(-1)[valid_verts]
        if specific_heat_map is not None:
            c_f = np.asarray(specific_heat_map, dtype=np.float64).reshape(-1)[valid_verts]
        if emissivity_map is not None:
            eps_f = np.asarray(emissivity_map, dtype=np.float64).reshape(-1)[valid_verts]

        boundary_verts_mask = (
            np.ones(len(verts_np), dtype=bool)
            if boundary_verts_mask_override is None
            else np.asarray(boundary_verts_mask_override, dtype=bool).reshape(-1)
        )
        if boundary_verts_mask.shape != (len(verts_np),):
            raise ValueError("Boundary mask must match the thermal point count")

        _log.debug(
            "%s %s %s %s %s",
            verts_np.shape,
            faces_np.shape if faces_np is not None else None,
            u0.shape,
            np.unique(u0),
            verts_np.dtype,
        )

        # Build sparse matrices (optionally variable diffusion via alpha_f)
        L_t, M_t, M_boundary_t = self._build_matrices(verts_np, faces_np, alpha_vec=alpha_f)

        # Time stepping info
        dt = self.gen_params.NUM_FRAME_DELTA / 60.0
        record_attimestep = int(
            (self.sim_params.sim_time - self.sim_params.record_time) / dt
        )
        sim_steps = int(self.sim_params.sim_time / dt)

        timesteps = [0, record_attimestep, sim_steps]

        if irradiance_map is None and self.irradiance_map is not None:
            irradiance_map = self.irradiance_map
        if irradiance_map is not None:
            irradiance_map = irradiance_map[valid_verts].astype(np.float32)

        if record_attimestep == 0 and not store_only_final:
            # Use proper 2D array shape (1, N) for consistent concatenation
            u_real_arr = [u0.reshape(1, -1)]
        else:
            u_real_arr = []

        u0_local = u0.copy()

        for i in range(len(timesteps) - 1):
            sim_length = timesteps[i + 1] - timesteps[i]
            _log.debug("sim_length %d", sim_length)
            if sim_length == 0:
                continue

            u_real_np_tmp = self._simulate_heat_torch(
                u0_local,
                L_t,
                M_t,
                M_boundary_t,
                boundary_verts_mask,
                irradiance_map,
                alpha_f,
                rho_f,
                c_f,
                eps_f,
                dt,
                sim_length,
                steady_state=steady_state,
                tol_K_per_s=tol_K_per_s,
                store_only_final=store_only_final,
            )
            _log.debug("%s", u_real_np_tmp.shape)
            u0_local = u_real_np_tmp[-1].copy()
            # Record if this timestep range ends at or after the recording start time
            if timesteps[i + 1] > record_attimestep:
                if store_only_final:
                    # u_real_np_tmp already has shape (1, N) with only the final state.
                    u_real_arr.append(u_real_np_tmp)
                else:
                    # Determine which results to include
                    start_offset = max(0, record_attimestep - timesteps[i])
                    if start_offset == 0:
                        # Include all results except initial condition (already have it)
                        u_real_arr.append(u_real_np_tmp[1:])
                    else:
                        # Skip some initial results
                        u_real_arr.append(u_real_np_tmp[start_offset + 1:])

        if len(u_real_arr) == 0:
            # No timesteps were recorded, just return initial condition
            u_real_arr = np.array([u0.reshape(-1)]).astype(np.float64)
        else:
            u_real_arr = np.concatenate(u_real_arr, axis=0).astype(np.float64)
        tmp_u_real_np = u_real_arr

        # Scatter back to full vertex set
        u_real_np = np.full(
            (tmp_u_real_np.shape[0], valid_verts.shape[0]),
            295.0,
            dtype=np.float64,
        )
        u_real_np[:, valid_verts] = tmp_u_real_np

        return u_real_np
