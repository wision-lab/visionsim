from __future__ import annotations

import numpy as np
import pytest

from visionsim.utils import colormaps


def test_bundled_palettes_are_normalized_rgb():
    assert colormaps.list_palettes() == sorted(colormaps.PALETTES)
    assert set(colormaps.PALETTES) == {
        "arctic",
        "black-hot",
        "ironbow",
        "outdoor-alert",
        "rainbow",
        "rainbow-hc",
        "white-hot",
    }
    for name, palette in colormaps.PALETTES.items():
        assert palette.shape == (433, 3), name
        assert palette.min() >= 0.0 and palette.max() <= 1.0, name


def test_colormap_interpolates_the_vendored_palette():
    ironbow = colormaps.PALETTES["ironbow"]
    cmap = colormaps.colormap("ironbow")

    # Endpoints and the exact midpoint entry are preserved (0.5 lands on index 216).
    np.testing.assert_allclose(cmap(0.0)[:3], ironbow[0], atol=1e-6)
    np.testing.assert_allclose(cmap(1.0)[:3], ironbow[-1], atol=1e-6)
    np.testing.assert_allclose(cmap(0.5)[:3], ironbow[216], atol=1e-6)


def test_colormap_falls_back_to_matplotlib_registry():
    assert colormaps.colormap("inferno").name == "inferno"

    with pytest.raises(KeyError):
        colormaps.colormap("definitely-not-a-colormap")
