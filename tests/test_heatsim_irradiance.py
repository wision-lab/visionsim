from __future__ import annotations

from tests.heatsim_test_service import SERVICE_PATH, call_service
from visionsim.simulate.blender import BlenderClient


def test_bake_albedo_map_returns_varying_pixels(executable):
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        call_service(client, "build_scene", "grid", "Checker", material="checker", lit=False)
        result = call_service(client, "bake_albedo")

    assert result["baked"] == "ok", result
    assert len(result["shape"]) == 3 and result["shape"][2] == 3, f"bad pixel shape {result['shape']}"
    assert result["std"] > 0.1, f"expected high-contrast checker variation (std>0.1), got std={result['std']}"
    assert 0.0 <= result["mean"] <= 1.0, f"albedo mean out of range: {result['mean']}"


def test_cycles_absorbed_flux_varies_with_albedo(executable):
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        call_service(client, "build_scene", "grid", "ThermalPlane", subdivisions=20, material="checker")
        result = call_service(client, "absorbed_flux")

    assert result["shape"] == [result["vertices"]]
    assert result["finite"], "absorbed flux contains non-finite values"
    assert result["std"] > 0.01, f"absorbed flux did not vary with checker albedo: {result['std']}"


def test_all_zero_attribute_albedo_is_ignored_and_rebaked(executable):
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        call_service(client, "build_scene", "grid", "Checker", material="checker", lit=False)
        result = call_service(client, "bake_vertex_albedo", stale_zeros=True)

    assert result["baked"] == "ok", "albedo absent"
    assert result["std"] > 0.05, f"zeros attribute was served instead of re-baking (std={result['std']})"
