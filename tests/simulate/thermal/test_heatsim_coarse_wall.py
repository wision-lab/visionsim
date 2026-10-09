"""A sparse wall must gain true thermal samples under localized illumination."""

from __future__ import annotations

from tests.simulate.thermal.heatsim_test_service import SERVICE_PATH, call_service
from visionsim.simulate.blender import BlenderClient


def test_auto_coarse_wall_resolves_localized_heating(executable):
    with BlenderClient.spawn(executable=executable, timeout=120, service=SERVICE_PATH) as client:
        # A plane with a single point light: localized illumination on a sparse wall.
        call_service(client, "build_scene", "plane", "wall", material="plain", point_light=True)
        result = call_service(client, "coarse_wall", [[64, 16], [256, 32]])

    assert all(result["in_plan"]), f"wall missing from an atlas plan: {result['in_plan']}"
    assert all(result["finite"]), f"non-finite field: {result['finite']}"
    # The atlas gives the sparse wall far more samples than its 4 authored vertices.
    assert result["low"] > result["wall_vertices"]
    assert result["ref"] > result["wall_vertices"]
    assert all(p > 0.5 for p in result["ptp"]), f"localized heating lost on the sparse wall: {result['ptp']}"
    assert result["rms"] < 0.15, f"coarse/reference RMS={result['rms']:.4f} K"
