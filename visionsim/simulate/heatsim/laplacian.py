"""Robust point-cloud Laplacian and mass matrix for thermal samples."""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp

try:
    import robust_laplacian  # type: ignore

    HAS_ROBUST_LAPLACIAN = True
    ROBUST_IMPORT_ERROR: str | None = None
except Exception as e:  # noqa: BLE001
    robust_laplacian = None
    HAS_ROBUST_LAPLACIAN = False
    ROBUST_IMPORT_ERROR = str(e)


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
        raise ImportError(f"robust_laplacian is not available. Import error: {ROBUST_IMPORT_ERROR or 'unknown'}")

    points = np.asarray(points, dtype=np.float64)
    # The library requires fewer neighbours than points.
    n_neighbors = max(1, min(int(n_neighbors), len(points) - 1))
    L, M = robust_laplacian.point_cloud_laplacian(points, mollify_factor=mollify_factor, n_neighbors=int(n_neighbors))

    # Sanitize: robust_laplacian can emit non-finite entries for degenerate local
    # neighbourhoods (coincident points, undefined tangent plane). They are fatal
    # globally, not locally: an inf on diag(L) zeroes the Jacobi entry, so PCG's scalar
    # alpha becomes NaN and one bad node poisons every timestep (observed: 2 of 120k
    # nodes on visionsim50/bathroom1 -> a 99.80% NaN solve that still exited 0).
    # Zeroing the bad rows isolates those nodes from conduction -- they still exchange
    # with ambient through the boundary terms and still absorb flux -- and the whole
    # branch is a no-op on well-conditioned clouds.
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
