"""Non-finite Laplacian sanitization (visionsim.simulate.heatsim.laplacian).

Regression cover for a defect that produced a completely NaN temperature field while the
pipeline exited 0 and rendered every frame from it. ``robust_laplacian`` emitted ``-inf``
on the Laplacian diagonal for 2 of 120,183 nodes -- degenerate local neighbourhoods, where
coincident points leave the local tangent plane undefined. The damage was global rather
than local because it propagates through a scalar: an ``inf`` diagonal makes ``diagA``
inf, the Jacobi preconditioner entry becomes 0, ``A @ p`` goes non-finite, and PCG's
``alpha = rz_old / (p . Ap)`` -- one scalar -- becomes NaN, poisoning every node on the
first timestep.

These tests fake ``robust_laplacian`` so they run on the host interpreter, where the real
package is not installed (it lives in Blender's Python).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from visionsim.simulate.heatsim import laplacian as L


def _fake_backend(monkeypatch, make):
    """Point `laplacian` at a stub whose point_cloud_laplacian returns `make(n)`."""

    class _Stub:
        @staticmethod
        def point_cloud_laplacian(points, mollify_factor=1e-5, n_neighbors=30):
            return make(len(points))

    monkeypatch.setattr(L, "robust_laplacian", _Stub)
    monkeypatch.setattr(L, "HAS_ROBUST_LAPLACIAN", True)


def _clean(n):
    lap = sp.diags([2.0] * n).tolil()
    for i in range(n - 1):
        lap[i, i + 1] = -1.0
        lap[i + 1, i] = -1.0
    return lap.tocsr(), sp.diags([1.0] * n, format="csr")


def test_non_finite_diagonal_is_isolated_not_propagated(monkeypatch):
    """An -inf on one row must not survive into the returned operator."""
    bad_row = 3

    def make(n):
        lap, mass = _clean(n)
        lap = lap.tolil()
        lap[bad_row, bad_row] = -np.inf
        return lap.tocsr(), mass

    _fake_backend(monkeypatch, make)
    pts = np.random.default_rng(0).random((8, 3))

    with pytest.warns(RuntimeWarning, match="non-finite"):
        lap, mass = L.point_cloud_laplacian_and_mass(pts)

    assert np.all(np.isfinite(lap.toarray())), "a non-finite entry survived sanitization"
    assert np.all(np.isfinite(mass.diagonal()))
    # The offending node is isolated: it conducts to nobody and nobody conducts to it.
    dense = lap.toarray()
    assert not dense[bad_row, :].any(), "bad row was not zeroed"
    assert not dense[:, bad_row].any(), "bad column was not zeroed"
    # Every other node keeps its coupling -- this is a local repair, not a global reset.
    assert dense[0, 1] != 0.0


def test_warning_names_the_offending_node(monkeypatch):
    """The warning has to say WHICH node, or it cannot be acted on."""

    def make(n):
        lap, mass = _clean(n)
        lap = lap.tolil()
        lap[5, 5] = np.nan
        return lap.tocsr(), mass

    _fake_backend(monkeypatch, make)
    with pytest.warns(RuntimeWarning) as rec:
        L.point_cloud_laplacian_and_mass(np.random.default_rng(1).random((9, 3)))
    assert "5" in str(rec[0].message)


def test_non_positive_mass_is_repaired(monkeypatch):
    """A zero or negative lumped mass divides by ~zero downstream."""

    def make(n):
        lap, mass = _clean(n)
        diag = mass.diagonal().copy()
        diag[2] = 0.0
        diag[4] = -1.0
        return lap, sp.diags(diag, format="csr")

    _fake_backend(monkeypatch, make)
    with pytest.warns(RuntimeWarning, match="mass"):
        _, mass = L.point_cloud_laplacian_and_mass(np.random.default_rng(2).random((7, 3)))
    diag = mass.diagonal()
    assert np.all(np.isfinite(diag))
    assert np.all(diag > 0.0), "a non-positive mass entry survived"


def test_clean_input_is_untouched_and_silent(monkeypatch):
    """Sanitization must be a no-op on a well-conditioned cloud."""
    _fake_backend(monkeypatch, _clean)
    pts = np.random.default_rng(3).random((10, 3))

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here fails the test
        lap, mass = L.point_cloud_laplacian_and_mass(pts)

    expected_lap, expected_mass = _clean(10)
    assert np.allclose(lap.toarray(), expected_lap.toarray())
    assert np.allclose(mass.diagonal(), expected_mass.diagonal())
