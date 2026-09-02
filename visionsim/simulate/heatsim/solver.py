# Vendored from heat-sim-blender:addon/lib/heatsim_fem.py @ e5b4afe
import logging
import numpy as np
import scipy.sparse as sp
import warnings

import torch

from visionsim.simulate.heatsim import constants
from visionsim.simulate.heatsim.laplacian import (
    HAS_ROBUST_LAPLACIAN,
    ROBUST_IMPORT_ERROR,
    mesh_laplacian_and_mass,
    point_cloud_laplacian_and_mass,
)

_log = logging.getLogger("rich")

try:
    import igl  # type: ignore

    HAS_IGL = True
    IGL_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    igl = None
    HAS_IGL = False
    IGL_IMPORT_ERROR = str(e)


# Default per-timestep linear-solver mode. "pcg_jacobi" = Jacobi-preconditioned CG
# (faster, same result); "cg" = original unpreconditioned CG. Callers may override
# per-instance via the HeatSimFEM(solver_mode=...) kwarg.
DEFAULT_SOLVER_MODE = "pcg_jacobi"


def scipy_to_torch_sparse(mat, device, dtype=torch.float32):
    """
    Convert a SciPy sparse matrix to a torch.sparse_coo_tensor on a given device.
    """
    mat = mat.tocoo()
    indices = np.vstack([mat.row, mat.col]).astype(np.int64)
    i = torch.from_numpy(indices).to(device)
    v = torch.from_numpy(mat.data.astype(np.float32)).to(device)
    return torch.sparse_coo_tensor(i, v, mat.shape, device=device, dtype=dtype).coalesce()


@torch.no_grad()
def cg_solve(mv, b, x0=None, tol=1e-6, max_iter=200):
    """
    Simple Conjugate Gradient solver for SPD systems in operator form:
        A x = b, where mv(x) computes A @ x.
    """
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()

    r = b - mv(x)
    p = r.clone()
    rs_old = torch.dot(r.flatten(), r.flatten())

    if rs_old.sqrt() < tol:
        return x

    for _ in range(max_iter):
        Ap = mv(p)
        denom = torch.dot(p.flatten(), Ap.flatten())
        # Avoid division by zero
        if denom.abs() < 1e-20:
            break
        alpha = rs_old / denom

        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r.flatten(), r.flatten())

        if rs_new.sqrt() < tol:
            break

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x


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

    Converges to the SAME solution as cg_solve (identical A, b, stopping tolerance on
    ||r||_2) but in far fewer iterations, so results match the unpreconditioned solver
    within the tolerance while running much faster.
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
            break
        alpha = rz_old / denom

        x = x + alpha * p
        r = r - alpha * Ap

        if torch.dot(r.flatten(), r.flatten()).sqrt() < tol:
            break

        z = Minv * r
        rz_new = torch.dot(r.flatten(), z.flatten())
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    return x


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

        # Laplacian/mass backend config
        # - laplacian_domain: "MESH" or "POINTS"
        # - laplacian_backend: "IGL" or "ROBUST"
        self.laplacian_domain = kwargs.get("laplacian_domain", "MESH")
        self.laplacian_backend = kwargs.get("laplacian_backend", "IGL")
        self.robust_mollify_factor = float(kwargs.get("robust_mollify_factor", 1e-5))
        self.pointcloud_neighbors = int(kwargs.get("pointcloud_neighbors", 30))

        # Linear-solver mode for the per-timestep SPD solve:
        #   "pcg_jacobi" (default) -> Jacobi-preconditioned CG (much faster, same result)
        #   "cg"                   -> original unpreconditioned CG (kept for A/B comparison)
        self.solver_mode = str(kwargs.get("solver_mode", DEFAULT_SOLVER_MODE))

        if self.irradiance_map is not None:
            _log.debug("irradiance_map %s", self.irradiance_map.shape)

    # ------------------------------------------------------------------
    # Core sparse setup + simulation
    # ------------------------------------------------------------------

    def _build_boundary_and_sources(
        self,
        M_boundary_t,
        boundary_mask_np,
        irradiance_map_np,
        rho_np,
        c_np,
        eps_np,
        dt,
        dtype=torch.float32,
    ):
        """
        Build torch tensors for boundary mask and precompute RHS constants.
        """
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

        boundary_mask = torch.from_numpy(boundary_mask_np.astype(np.float32)).to(
            self.device
        )

        if irradiance_map_np is None:
            irradiance_map_np = np.zeros_like(boundary_mask_np, dtype=np.float32)
        irr_t = torch.from_numpy(irradiance_map_np.astype(np.float32)).to(self.device)

        # Constants
        sigma = constants.SIGMA
        Tamb = constants.AMBIENT_TEMP
        h = constants.CONVECTION_COEFF

        # A-side diags (multiplied with u inside mv)
        vec_rad_A = None
        vec_conv_A = None
        if self.sim_params.sim_radiation:
            vec_rad_A = (
                boundary_mask
                * dt
                * 4.0
                * sigma
                * eps_t
                * (Tamb**3)
                / rc_t
            )
        if self.sim_params.sim_convection:
            vec_conv_A = (
                boundary_mask
                * dt
                * h
                / rc_t
            )

        # RHS source vectors (constants)
        vec_rad_rhs = torch.zeros_like(boundary_mask)
        vec_conv_rhs = torch.zeros_like(boundary_mask)
        vec_light_rhs = torch.zeros_like(boundary_mask)


        if self.sim_params.sim_radiation:
            vec_rad_rhs = (
                boundary_mask
                * dt
                * 4.0
                * sigma
                * eps_t
                * (Tamb**4)
                / rc_t
            )
        if self.sim_params.sim_convection:
            vec_conv_rhs = boundary_mask * dt * h * Tamb / rc_t
        # Irradiance term
        vec_light_rhs = boundary_mask * dt * irr_t / rc_t

        _log.debug("DEBUG: vec_rad_rhs range: %s %s", vec_rad_rhs.min().item(), vec_rad_rhs.max().item())
        _log.debug("DEBUG: vec_conv_rhs range: %s %s", vec_conv_rhs.min().item(), vec_conv_rhs.max().item())
        _log.debug("DEBUG: vec_light_rhs range: %s %s", vec_light_rhs.min().item(), vec_light_rhs.max().item())

        # B_const = M_boundary @ (sum of these vectors)
        total_rhs_vec = vec_rad_rhs + vec_conv_rhs + vec_light_rhs  # (N,)
        total_rhs_vec = total_rhs_vec.unsqueeze(1)  # (N,1)

        B_const = torch.sparse.mm(M_boundary_t, total_rhs_vec)  # (N,1)

        return boundary_mask, irr_t, vec_rad_A, vec_conv_A, B_const

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

        sigma = constants.SIGMA
        Tamb = constants.AMBIENT_TEMP
        h = constants.CONVECTION_COEFF

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
            pass

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
        Minv = None
        if self.solver_mode == "pcg_jacobi":
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
            if Minv is not None:
                u_next = pcg_solve(mv, b, Minv, x0=u_prev, tol=1e-5, max_iter=200)
            else:
                u_next = cg_solve(mv, b, x0=u_prev, tol=1e-5, max_iter=200)

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
        L_coo = L.tocoo()
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

    def _compute_mass_matrix_manual(self, verts_np, faces_np):
        """
        Compute lumped mass matrix manually (barycentric).
        Each vertex gets 1/3 of the area of each adjacent triangle.
        """
        n_verts = verts_np.shape[0]
        mass_diag = np.zeros(n_verts, dtype=np.float64)

        # Vectorized computation
        v0 = verts_np[faces_np[:, 0]]
        v1 = verts_np[faces_np[:, 1]]
        v2 = verts_np[faces_np[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)

        # Accumulate area/3 to each vertex
        np.add.at(mass_diag, faces_np[:, 0], areas / 3.0)
        np.add.at(mass_diag, faces_np[:, 1], areas / 3.0)
        np.add.at(mass_diag, faces_np[:, 2], areas / 3.0)

        return sp.diags(mass_diag, format='csr')

    def _compute_cotmatrix_manual(self, verts_np, faces_np):
        """
        Fast, vectorized computation of cotangent Laplacian.
        L_ij = -0.5 * (cot(alpha_ij) + cot(beta_ij)) for edge (i,j)
        L_ii = -sum_j(L_ij)
        Only works for triangle meshes (faces_np n x 3).
        """
        n_verts = verts_np.shape[0]
        n_faces = faces_np.shape[0]
        # Indices of triangle corners
        i0 = faces_np[:, 0]
        i1 = faces_np[:, 1]
        i2 = faces_np[:, 2]

        v0 = verts_np[i0]  # (nF, 3)
        v1 = verts_np[i1]
        v2 = verts_np[i2]

        # For each triangle, compute cotangent at each corner
        # Each triangle has 3 corners: at v0, at v1, at v2.
        # For each corner, we want to compute cot(theta):
        # cot(theta_i) = (u·v) / norm(u x v) where u/v are the two adjacent triangle sides at that corner.
        # Layout:
        # Corner at v0 = angle at v0, sides: (v2-v0), (v1-v0) (from v0 towards v2, v0 towards v1)
        # Corner at v1 = angle at v1, sides: (v0-v1), (v2-v1)
        # Corner at v2 = angle at v2, sides: (v1-v2), (v0-v2)
        u0 = v2 - v0  # (nF, 3)
        v0v = v1 - v0
        u1 = v0 - v1
        v1v = v2 - v1
        u2 = v1 - v2
        v2v = v0 - v2

        # Cotangents at corners
        cot0 = np.einsum('ij,ij->i', u0, v0v) / (np.linalg.norm(np.cross(u0, v0v), axis=1) + 1e-16)
        cot1 = np.einsum('ij,ij->i', u1, v1v) / (np.linalg.norm(np.cross(u1, v1v), axis=1) + 1e-16)
        cot2 = np.einsum('ij,ij->i', u2, v2v) / (np.linalg.norm(np.cross(u2, v2v), axis=1) + 1e-16)

        # For each triangle, contribute to edges:
        # (i1,i2): cot at v0
        # (i2,i0): cot at v1
        # (i0,i1): cot at v2
        # Each cotangent belongs to the edge opposite the current corner.
        I = np.concatenate([i1, i2, i2, i0, i0, i1])
        J = np.concatenate([i2, i1, i0, i2, i1, i0])
        V = 0.5 * np.concatenate([cot0, cot0, cot1, cot1, cot2, cot2])

        # Build sparse matrix
        L = sp.coo_matrix((V, (I, J)), shape=(n_verts, n_verts))
        # Set diagonal: L_ii = -sum_j(L_ij)
        L = L.tocsr()
        L = L - sp.diags(np.array(L.sum(axis=1)).flatten(), format='csr')
        return L

    def _build_pointcloud_fallback_matrices(self, points_np, n_neighbors: int):
        """
        Fallback point-cloud Laplacian if robust_laplacian isn't available.

        Builds a simple symmetric kNN graph Laplacian:
            L_psd = D - W  (PSD)
        We return the NEGATIVE of this matrix to match the solver convention.
        Mass matrix is identity (lumped).
        """
        try:
            from scipy.spatial import cKDTree
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Point-cloud FEM requires either robust_laplacian or scipy.spatial.cKDTree. "
                f"Import error: {e}"
            )

        points_np = np.asarray(points_np, dtype=np.float64)
        n = int(points_np.shape[0])
        if n == 0:
            raise ValueError("Cannot build point-cloud Laplacian for empty point set")

        k = int(max(2, min(n_neighbors, n - 1)))
        tree = cKDTree(points_np)
        dists, idxs = tree.query(points_np, k=k + 1)  # include self at [0]
        dists = dists[:, 1:]
        idxs = idxs[:, 1:]

        # Simple distance weights. Symmetrize later.
        eps = 1e-12
        w = 1.0 / (dists + eps)

        rows = np.repeat(np.arange(n, dtype=np.int64), k)
        cols = idxs.reshape(-1).astype(np.int64)
        data = w.reshape(-1).astype(np.float64)

        W = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        # Symmetrize
        W = 0.5 * (W + W.T)
        d = np.array(W.sum(axis=1)).reshape(-1)
        L_psd = sp.diags(d, format="csr") - W

        # Match solver convention: negative semidefinite Laplacian.
        L = -L_psd
        M = sp.eye(n, format="csr")
        return L, M

    def _build_matrices(self, verts_np, faces_np, *, alpha_vec: np.ndarray | None = None):
        """
        Build Laplacian + mass matrices, then convert to torch sparse.

        Important sign convention:
        - This solver expects a NEGATIVE semidefinite Laplacian (like `igl.cotmatrix()`).
        - `robust_laplacian` returns a POSITIVE semidefinite Laplacian, so we flip sign.
        """
        verts_np = np.array(verts_np, dtype=np.float64)

        # --------------------------------------------------------------
        # Point-cloud mode (no faces required)
        # --------------------------------------------------------------
        if str(self.laplacian_domain).upper() == "POINTS":
            if str(self.laplacian_backend).upper() == "ROBUST":
                if not HAS_ROBUST_LAPLACIAN:
                    warnings.warn(
                        "Requested robust_laplacian point-cloud Laplacian, but robust_laplacian "
                        f"is not available ({ROBUST_IMPORT_ERROR}). Falling back to a simple kNN graph Laplacian.",
                        RuntimeWarning,
                    )
                    L, M = self._build_pointcloud_fallback_matrices(
                        verts_np, n_neighbors=self.pointcloud_neighbors
                    )
                    # Put an info messsage on how the Laplacian is built
                    _log.debug(
                        "[DEBUG] Point-cloud Laplacian built using simple kNN graph Laplacian. L shape: %s, M shape: %s",
                        L.shape,
                        M.shape,
                    )
                else:
                    L_psd, M = point_cloud_laplacian_and_mass(
                        verts_np,
                        mollify_factor=self.robust_mollify_factor,
                        n_neighbors=self.pointcloud_neighbors,
                    )
                    L = -L_psd
                    _log.debug(
                        "[DEBUG] Point-cloud Laplacian built using robust_laplacian API. L shape: %s, M shape: %s",
                        L.shape,
                        M.shape,
                    )

            else:
                warnings.warn(
                    "Point-cloud FEM requested with non-ROBUST backend; using a simple kNN graph Laplacian.",
                    RuntimeWarning,
                )
                L, M = self._build_pointcloud_fallback_matrices(
                    verts_np, n_neighbors=self.pointcloud_neighbors
                )
                _log.debug(
                    "[DEBUG] Point-cloud Laplacian built using simple kNN graph Laplacian. L shape: %s, M shape: %s",
                    L.shape,
                    M.shape,
                )

            # Optional: apply spatially varying diffusivity by weighting the Laplacian.
            if alpha_vec is not None:
                alpha_vec = np.asarray(alpha_vec, dtype=np.float64).reshape(-1)
                if alpha_vec.shape[0] == L.shape[0]:
                    L = self._apply_vertex_weighted_laplacian(L, alpha_vec)
                else:
                    warnings.warn(
                        "alpha_vec length doesn't match point-cloud Laplacian size; ignoring variable diffusion.",
                        RuntimeWarning,
                    )

            M_boundary = M.copy()
            L_t = scipy_to_torch_sparse(L, self.device)
            M_t = scipy_to_torch_sparse(M, self.device)
            M_boundary_t = scipy_to_torch_sparse(M_boundary, self.device)
            return L_t, M_t, M_boundary_t

        # --------------------------------------------------------------
        # Mesh mode
        # --------------------------------------------------------------
        faces_np = np.array(faces_np, dtype=np.int32)
        assert faces_np.shape[1] == 3, "Only triangle meshes supported."

        if str(self.laplacian_backend).upper() == "ROBUST":
            if not HAS_ROBUST_LAPLACIAN:
                warnings.warn(
                    "Requested robust_laplacian mesh Laplacian, but robust_laplacian is not available "
                    f"({ROBUST_IMPORT_ERROR}). Falling back to igl cotmatrix/massmatrix.",
                    RuntimeWarning,
                )
            else:
                L_psd, M = mesh_laplacian_and_mass(
                    verts_np,
                    faces_np,
                    mollify_factor=self.robust_mollify_factor,
                )
                L = -L_psd
                _log.debug(
                    "[DEBUG] Mesh Laplacian built using robust_laplacian API. L shape: %s, M shape: %s",
                    L.shape,
                    M.shape,
                )
                M_boundary = M.copy()
                L_t = scipy_to_torch_sparse(L, self.device)
                M_t = scipy_to_torch_sparse(M, self.device)
                M_boundary_t = scipy_to_torch_sparse(M_boundary, self.device)
                return L_t, M_t, M_boundary_t

        if HAS_IGL:
            L = igl.cotmatrix(verts_np, faces_np)
        else:
            warnings.warn(
                f"libigl (igl) not available ({IGL_IMPORT_ERROR}); computing cotan Laplacian manually.",
                RuntimeWarning,
            )
            L = sp.csr_matrix((verts_np.shape[0], verts_np.shape[0]))
            _log.debug("[DEBUG] Mesh Laplacian built using manual computation. L shape: %s", L.shape)

        # Debug: Check Laplacian
        if L.nnz > 0:
            _log.debug(
                "  Laplacian: nnz=%d, data range=[%.6f, %.6f]",
                L.nnz,
                L.data.min(),
                L.data.max(),
            )
        else:
            _log.debug("  WARNING: Laplacian is empty (nnz=0), will compute manually")

        # Try igl mass matrix
        M = None
        if HAS_IGL and L.nnz > 0:
            M = igl.massmatrix(verts_np, faces_np, igl.MASSMATRIX_TYPE_BARYCENTRIC)
            if M.nnz == 0:
                M = igl.massmatrix(verts_np, faces_np, igl.MASSMATRIX_TYPE_VORONOI)
            if M.nnz > 0:
                _log.debug("  Mass matrix from igl: nnz=%d", M.nnz)

        # Fallback to manual computation if igl failed
        if M is None or M.nnz == 0:
            _log.debug("  Computing mass matrix manually (igl failed)...")
            M = self._compute_mass_matrix_manual(verts_np, faces_np)
            _log.debug("  Manual mass matrix: nnz=%d, diag sum=%.6f", M.nnz, M.diagonal().sum())

        if L.nnz == 0:
            _log.debug("  Computing Laplacian manually (igl unavailable or failed)...")
            L = self._compute_cotmatrix_manual(verts_np, faces_np)
            _log.debug("  Manual Laplacian: nnz=%d", L.nnz)

        M_boundary = M.copy()

        _log.debug("[DEBUG] Laplacian shape: %s", L.shape)
        _log.debug("[DEBUG] Mass Matrix shape: %s", M.shape)
        _log.debug("[DEBUG] Boundary Mass Matrix shape: %s", M_boundary.shape)

        # Print the first 10 diagonal elements of the mass matrix
        # M can be a sparse matrix; use .diagonal() method instead of np.diag for sparse
        if hasattr(M, "diagonal"):
            diag_vals = M.diagonal()
        else:
            diag_vals = np.diag(M)
        _log.debug("[DEBUG] Mass Matrix first 10 diagonal elements: %s", diag_vals[:10].tolist())

        # Optional: apply spatially varying diffusivity by weighting the Laplacian.
        if alpha_vec is not None:
            alpha_vec = np.asarray(alpha_vec, dtype=np.float64).reshape(-1)
            if alpha_vec.shape[0] == L.shape[0]:
                L = self._apply_vertex_weighted_laplacian(L, alpha_vec)
            else:
                warnings.warn(
                    "alpha_vec length doesn't match mesh Laplacian size; ignoring variable diffusion.",
                    RuntimeWarning,
                )

        # Convert to torch sparse on device
        L_t = scipy_to_torch_sparse(L, self.device)
        M_t = scipy_to_torch_sparse(M, self.device)
        M_boundary_t = scipy_to_torch_sparse(M_boundary, self.device)

        return L_t, M_t, M_boundary_t

    def simulate_for_pose(
        self,
        verts_np,
        faces_np,
        boundary_verts_mask,
        u_prev,
        irradiance_map,
        thermal_diffusivity_map,
        density_map,
        specific_heat_map,
        emissivity_map,
        num_substeps: int,
        dt: float,
        dirichlet_indices=None,
        dirichlet_values=None,
    ):
        """Build (L, M, M_boundary) for a single pose and advance ``num_substeps``
        implicit-Euler CG solves with ``u_prev`` as the initial state.

        Per-vertex material vectors (alpha, rho, c, eps) and the per-vertex flux
        ``irradiance_map`` are treated as pose-invariant for Phase 1 animated
        geometry. The Laplacian / mass matrices are rebuilt because they depend
        on positions; everything else is a function of constant per-vertex data
        plus the per-pose mass matrix (which is what couples flux into RHS).

        When ``dirichlet_indices`` is given (with matching ``dirichlet_values``),
        those vertex indices are pinned to their target temperature on the
        initial state AND re-pinned after every substep's CG solve -- not just
        once per frame -- since the FEM/Dirichlet coupling weight in ``mv`` would
        otherwise let the pinned nodes drift within a frame as substeps advance
        (this matters most as ``num_substeps`` grows). Callers that don't pass
        these params (e.g. :meth:`perform_gt_heat_simulation`'s static path) get
        byte-identical behavior to before.

        Returns: ``(num_substeps, N)`` float64 array of post-step states.
        """
        N = int(verts_np.shape[0])
        u_prev = np.asarray(u_prev, dtype=np.float64).reshape(-1)
        assert u_prev.shape[0] == N, f"u_prev length {u_prev.shape[0]} != N {N}"

        alpha_np = (
            np.asarray(thermal_diffusivity_map, dtype=np.float64).reshape(-1)
            if thermal_diffusivity_map is not None
            else None
        )
        L_t, M_t, M_boundary_t = self._build_matrices(verts_np, faces_np, alpha_vec=alpha_np)

        rho_np = (
            np.asarray(density_map, dtype=np.float64).reshape(-1)
            if density_map is not None
            else np.full(N, float(self.gen_params.RHO), dtype=np.float64)
        )
        c_np = (
            np.asarray(specific_heat_map, dtype=np.float64).reshape(-1)
            if specific_heat_map is not None
            else np.full(N, float(self.gen_params.C), dtype=np.float64)
        )
        eps_np = (
            np.asarray(emissivity_map, dtype=np.float64).reshape(-1)
            if emissivity_map is not None
            else np.full(N, 0.9, dtype=np.float64)
        )
        eps_np = np.clip(eps_np, 0.0, 1.0)
        boundary_np = np.asarray(boundary_verts_mask).reshape(-1).astype(np.float32)
        irr_np = (
            np.asarray(irradiance_map, dtype=np.float32).reshape(-1)
            if irradiance_map is not None
            else np.zeros(N, dtype=np.float32)
        )

        rho_t = torch.from_numpy(rho_np.astype(np.float32)).to(self.device)
        c_t = torch.from_numpy(c_np.astype(np.float32)).to(self.device)
        rc_t = rho_t * c_t
        eps_t = torch.from_numpy(eps_np.astype(np.float32)).to(self.device)
        boundary_mask = torch.from_numpy(boundary_np).to(self.device)
        irr_t = torch.from_numpy(irr_np).to(self.device)

        sigma = constants.SIGMA
        Tamb = constants.AMBIENT_TEMP
        h = constants.CONVECTION_COEFF

        vec_rad_A = None
        vec_conv_A = None
        if self.sim_params.sim_radiation:
            vec_rad_A = boundary_mask * dt * 4.0 * sigma * eps_t * (Tamb ** 3) / rc_t
        if self.sim_params.sim_convection:
            vec_conv_A = boundary_mask * dt * h / rc_t

        vec_rad_rhs = (
            boundary_mask * dt * 4.0 * sigma * eps_t * (Tamb ** 4) / rc_t
            if self.sim_params.sim_radiation
            else torch.zeros_like(boundary_mask)
        )
        vec_conv_rhs = (
            boundary_mask * dt * h * Tamb / rc_t
            if self.sim_params.sim_convection
            else torch.zeros_like(boundary_mask)
        )
        vec_light_rhs = boundary_mask * dt * irr_t / rc_t
        rhs_total = (vec_rad_rhs + vec_conv_rhs + vec_light_rhs).unsqueeze(1)
        B_step = torch.sparse.mm(M_boundary_t, rhs_total)  # (N, 1)

        reg_value = 1e-8 if self.sim_params.add_tikhonov_reg else 0.0

        def mv(x):
            Mx = torch.sparse.mm(M_t, x)
            Lx = torch.sparse.mm(L_t, x)
            if alpha_np is None:
                K = float(self.gen_params.K)
                out = Mx - K * dt * Lx
            else:
                out = Mx - dt * Lx
            if vec_rad_A is not None:
                out = out + torch.sparse.mm(M_boundary_t, vec_rad_A.unsqueeze(1) * x)
            if vec_conv_A is not None:
                out = out + torch.sparse.mm(M_boundary_t, vec_conv_A.unsqueeze(1) * x)
            if reg_value > 0.0:
                out = out + reg_value * x
            return out

        u_prev_t = torch.from_numpy(u_prev.astype(np.float32).reshape(-1, 1)).to(self.device)

        di_t = None
        if dirichlet_indices is not None and len(dirichlet_indices) > 0:
            di_t = torch.from_numpy(np.asarray(dirichlet_indices, dtype=np.int64)).to(self.device)
            dv_t = torch.from_numpy(np.asarray(dirichlet_values, dtype=np.float32).reshape(-1, 1)).to(self.device)
            u_prev_t[di_t] = dv_t

        out_states = np.zeros((num_substeps, N), dtype=np.float64)
        for s in range(num_substeps):
            b = torch.sparse.mm(M_t, u_prev_t) + B_step
            u_next_t = cg_solve(mv, b, x0=u_prev_t, tol=1e-5, max_iter=200)
            if di_t is not None:
                u_next_t[di_t] = dv_t
            out_states[s, :] = u_next_t.detach().cpu().numpy().astype(np.float64).reshape(-1)
            u_prev_t = u_next_t

        return out_states

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        # In point-cloud mode we keep ALL vertices (no face-based filtering/renumbering).
        if str(self.laplacian_domain).upper() == "POINTS":
            valid_verts = np.ones(verts_np.shape[0], dtype=bool)
            verts_np = np.asarray(verts_np, dtype=np.float64)
        else:
            # Filter to valid verts (same logic as original; drop unreferenced verts)
            faces_np_unique = np.unique(faces_np).astype(int)
            valid_verts = np.zeros(verts_np.shape[0], dtype=bool)
            valid_verts[faces_np_unique] = True
            _log.debug(
                "valid_verts %d Total verts: %d",
                np.count_nonzero(valid_verts),
                verts_np.shape[0],
            )

            renumber_faces = np.full(verts_np.shape[0], -1, dtype=int)
            renumber_faces[valid_verts] = np.arange(np.count_nonzero(valid_verts))
            faces_np = renumber_faces[faces_np]
            boundary_faces_np = renumber_faces[boundary_faces_np]
            verts_np = verts_np[valid_verts].astype(np.float64)

        if u0 is None:
            u0 = np.full((verts_np.shape[0],), constants.AMBIENT_TEMP, dtype=np.float64)
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

        if str(self.laplacian_domain).upper() == "POINTS":
            if boundary_verts_mask_override is not None:
                boundary_verts_mask = (
                    np.asarray(boundary_verts_mask_override).reshape(-1)[valid_verts].astype(bool)
                )
            else:
                # Legacy fallback: in point-cloud mode all points are boundary.
                boundary_verts_mask = np.ones(verts_np.shape[0], dtype=bool)
        else:
            boundary_verts = np.unique(boundary_faces_np)
            boundary_verts_mask = np.zeros(verts_np.shape[0], dtype=bool)
            boundary_verts_mask[boundary_verts] = 1.0

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
            constants.AMBIENT_TEMP,
            dtype=np.float64,
        )
        u_real_np[:, valid_verts] = tmp_u_real_np

        return u_real_np

    def run_heat_simulation(self):
        u0 = np.full((self.verts_np.shape[0],), constants.AMBIENT_TEMP, dtype=np.float64)

        u_real_np = self.perform_gt_heat_simulation(
            self.verts_np.copy(),
            self.faces_np.copy(),
            self.faces_np,
            u0=u0,
            irradiance_map=self.irradiance_map,
        )
        _log.debug("%s", u_real_np[-1])

        if self.sim_params.add_noise_to_sim:
            u_real_np += np.random.normal(
                0, self.sim_params.heat_noise_std, u_real_np.shape
            )

        self.u_real_np = u_real_np
        return u_real_np
