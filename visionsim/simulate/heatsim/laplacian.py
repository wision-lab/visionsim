# Vendored from heat-sim-blender:addon/lib/robust_laplacian_backend.py @ e5b4afe
"""
Optional integration with `robust_laplacian` (Sharp & Crane SGP 2020).

This module is safe to import even when the dependency is missing.
Callers should check `HAS_ROBUST_LAPLACIAN` before using.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp

try:
    import robust_laplacian  # type: ignore

    HAS_ROBUST_LAPLACIAN = True
    ROBUST_IMPORT_ERROR: str | None = None
except Exception as e:  # pragma: no cover
    robust_laplacian = None
    HAS_ROBUST_LAPLACIAN = False
    ROBUST_IMPORT_ERROR = str(e)


def mesh_laplacian_and_mass(
    verts: np.ndarray,
    faces: np.ndarray,
    mollify_factor: float = 1e-5,
):
    """
    Build robust mesh Laplacian + lumped mass matrix.

    Returns:
        (L, M) as SciPy sparse matrices.
    """
    if not HAS_ROBUST_LAPLACIAN:  # pragma: no cover
        raise ImportError(
            "robust_laplacian is not available. "
            f"Import error: {ROBUST_IMPORT_ERROR or 'unknown'}"
        )

    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    L, M = robust_laplacian.mesh_laplacian(verts, faces, mollify_factor=mollify_factor)
    return L, M


def point_cloud_laplacian_and_mass(
    points: np.ndarray,
    mollify_factor: float = 1e-5,
    n_neighbors: int = 30,
):
    """
    Build robust point-cloud Laplacian + diagonal lumped mass matrix.

    Returns:
        (L, M) as SciPy sparse matrices.
    """
    if not HAS_ROBUST_LAPLACIAN:  # pragma: no cover
        raise ImportError(
            "robust_laplacian is not available. "
            f"Import error: {ROBUST_IMPORT_ERROR or 'unknown'}"
        )

    points = np.asarray(points, dtype=np.float64)
    # Clamp n_neighbors to len(points)-1 (defensive, matches the scipy fallback
    # in solver.py: prevents robust_laplacian "k+1 is greater than number of
    # points" crash on small point clouds; no-op on real dense meshes).
    n_neighbors = max(1, min(int(n_neighbors), len(points) - 1))
    L, M = robust_laplacian.point_cloud_laplacian(
        points, mollify_factor=mollify_factor, n_neighbors=int(n_neighbors)
    )

    # Sanitize: robust_laplacian can emit non-finite entries for degenerate local
    # neighbourhoods (coincident or near-coincident points, where the local tangent
    # plane is undefined). They are rare and they are fatal, because the failure is
    # global rather than local: an inf on L's diagonal makes diagA inf, so the Jacobi
    # preconditioner entry becomes 0, A @ p goes non-finite, and PCG's
    # ``alpha = rz_old / (p . Ap)`` -- a SCALAR -- becomes NaN. One bad node therefore
    # poisons the entire field on the first timestep.
    #
    # Observed on visionsim50/bathroom1: exactly 2 of 120,183 nodes carried -inf on
    # diag(L) while rho, c, eps, alpha, diag(M) and the RHS were all finite. The solve
    # returned 99.80% NaN -- every object, every timestep past the initial condition --
    # and the pipeline still exited 0 and rendered 600 frames from it.
    #
    # Zeroing a bad row isolates that node from conduction; it still exchanges with
    # ambient through the boundary terms and still receives its absorbed flux, so it
    # behaves like a thermally disconnected speck rather than destroying the solve.
    # This is strictly better than the alternative of propagating NaN, and it is a
    # no-op on well-conditioned clouds.
    L = L.tocsr()
    bad_data = ~np.isfinite(L.data)
    if bad_data.any():
        rows = np.repeat(np.arange(L.shape[0]), np.diff(L.indptr))[bad_data]
        bad_rows = np.unique(np.concatenate([rows, L.indices[bad_data]]))
        warnings.warn(
            f"point_cloud_laplacian: {int(bad_data.sum())} non-finite Laplacian entries "
            f"across {bad_rows.size} node(s); isolating them. Node indices: "
            f"{bad_rows[:8].tolist()}{'...' if bad_rows.size > 8 else ''}",
            RuntimeWarning,
        )
        L = L.tolil()
        for r in bad_rows:
            L[r, :] = 0.0
            L[:, r] = 0.0
        L = L.tocsr()
        L.eliminate_zeros()

    M = M.tocsr()
    m_diag = M.diagonal()
    bad_mass = ~np.isfinite(m_diag) | (m_diag <= 0.0)
    if bad_mass.any():
        good = m_diag[~bad_mass]
        repl = float(np.median(good)) if good.size else 1.0
        warnings.warn(
            f"point_cloud_laplacian: {int(bad_mass.sum())} non-finite/non-positive mass "
            f"entries; replacing with the median ({repl:.3e}).",
            RuntimeWarning,
        )
        m_diag = np.where(bad_mass, repl, m_diag)
        M = sp.diags(m_diag, format="csr")

    return L, M
