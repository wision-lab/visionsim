"""Tests for the thermal preview colormap: the global temperature range helper
and the inferno-stop compositor node group (see
docs/superpowers/specs/2026-07-07-visionsim-thermal-parity-design.md §9).

Both drive primitives inside a spawned Blender via the heatsim test service.
"""

from __future__ import annotations

from tests.simulate.thermal.heatsim_test_service import SERVICE_PATH, call_service
from visionsim.simulate.blender import BlenderClient


def test_global_temperature_range_is_robust_to_outliers(executable):
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        result = call_service(client, "temperature_range")

    # A realistic scene: the bulk sits near ambient with a modest warm spread, plus a
    # handful of artifact-hot vertices (2000 K, well under 1% of the data) that must
    # NOT dictate the colormap.
    assert abs(result["tmin"] - 295.0) < 1e-6  # floored at default / P1 of the ambient bulk
    assert result["tmax"] < 320.0  # P99 rejects the 2000 K outliers
    assert result["tmax"] > 300.0  # but still covers the genuine warm spread

    # empty history -> (default, default + 1)
    assert result["empty"] == [295.0, 296.0]

    # near-uniform field -> span widened to >= 1 K (no colour collapse)
    assert result["uniform"][0] == 300.0
    assert (result["uniform"][1] - result["uniform"][0]) >= 1.0

    # A texel object at 340 K (400 texels) dominates the pooled range over a 295.5 K
    # vertex object (50 vertices), so P99 must land near it -- TEXEL and VERTEX entries
    # pool identically.
    assert result["pooled"][1] > 330.0, result["pooled"]


def test_thermal_preview_node_group_uses_inferno_stops_and_range(executable):
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        result = call_service(client, "preview_nodegroup")

    # MapRange wired to [tmin, tmax] -> [0, 1]
    assert abs(result["mr_min"] - 295.0) < 1e-6
    assert abs(result["mr_max"] - 297.0) < 1e-6

    # Inferno ramp: 11 stops from near-black to pale yellow (heat-sim _INFERNO_STOPS)
    assert result["n_stops"] == 11
    assert abs(result["lo_pos"] - 0.0) < 1e-6 and abs(result["hi_pos"] - 1.0) < 1e-6
    lo = result["lo_color"]
    hi = result["hi_color"]
    assert lo[0] < 0.02 and lo[2] < 0.02  # near-black low end
    assert hi[0] > 0.9 and hi[1] > 0.9  # pale-yellow high end

    # Regression guard: the old turbo low stop was blue-ish (0.190, 0.072, 0.232)
    assert abs(lo[0] - 0.18995) > 1e-3, "colormap is still turbo, not inferno"
