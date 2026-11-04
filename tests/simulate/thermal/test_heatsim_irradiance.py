from __future__ import annotations

import math

import pytest

from tests.simulate.thermal.heatsim_test_service import SERVICE_PATH, call_service
from visionsim.simulate.blender import BlenderClient


def test_repaired_thin_charts_preserve_uniform_incident_light(executable):
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        result = call_service(client, "thin_chart_irradiance")

    assert result["source_preserved"]
    assert result["flux"] == pytest.approx([math.pi] * 8, rel=0.02), result
    assert result["pixel_centers"] == pytest.approx(list(range(12)))


def test_uv_validation_distinguishes_collinearity_noise_from_unmapped_surfaces(executable):
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        invalid_faces = call_service(client, "bake_uv_degenerate_faces")

    assert invalid_faces == [1], "A real unmapped triangle must fail; float32 collinearity noise must not"


def test_warped_face_repair_preserves_render_triangle_edge_lengths(executable):
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        result = call_service(client, "bake_uv_warped_face")

    assert result["invalid_faces"] == 0, result
    assert result["edge_scale_ratio"] == pytest.approx(1, abs=1e-5), result
    assert result["source_preserved"], result


@pytest.mark.parametrize("kind", ["overlap", "collapsed", "stretched", "subpixel", "hidden"])
def test_generated_bake_uvs_are_repaired_and_reused_without_changing_rgb(executable, kind):
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        result = call_service(client, "bake_uv_repair", kind)

    assert result["invalid_before"] > 0, result
    assert result["invalid_after"] == 0, result
    assert "error" not in result, result
    assert result["reused"], result
    assert result["source_preserved"] and result["resources_restored"], result
    assert result["sync_restored"] and result["object_mode"], result
    if kind == "subpixel":
        assert result["raster_missed_subpixel_overlap"], result


def test_unrepairable_uvs_fail_without_publishing_an_invalid_layout(executable):
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        result = call_service(client, "bake_uv_repair", "failure")

    assert "thermal UV repair left" in result["error"], result
    assert result["unpublished"], result
    assert result["source_preserved"] and result["resources_restored"], result
    assert result["sync_restored"] and result["object_mode"], result


def test_bake_uvs_are_reused_until_geometry_changes(executable):
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        call_service(client, "build_scene", "grid", "ThermalPlane", subdivisions=10, material="checker")
        result = call_service(client, "bake_uv_reuse")

    assert result["reused_coordinates"]
    assert result["regenerated_after_geometry_edit"]
    assert result["regenerated_after_scale_edit"]
    assert result["reused_after_translation"]
    assert result["source_uv_unchanged"]


def test_saved_scene_workers_reuse_identical_bake_uv_addresses(executable, tmp_path):
    scene = str(tmp_path / "uv-source.blend")
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        call_service(client, "build_scene", "grid", "Surface", subdivisions=4, lit=False)
        call_service(client, "save_scene", scene)
        first = call_service(client, "cached_bake_uv", scene, "Surface")
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        second = call_service(client, "cached_bake_uv", scene, "Surface", forbid_projection=True)
    assert first["saved"] and second["saved"]
    assert first["rgb_preserved"] and second["rgb_preserved"]
    assert first["signature"] == second["signature"]
    assert first["uv"] == second["uv"]


def test_bake_uvs_use_physical_dimensions_independent_of_rgb_image_aspect(executable):
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        result = call_service(client, "bake_uv_physical_layout")

    assert result["scaled_vs_applied_error"] < 1e-6, result
    assert result["wide_vs_square_error"] < 1e-6, result
    assert result["physical_spacing_ratio"] == pytest.approx(1, abs=1e-5), result
    assert result["source_restored"], "source geometry, transforms, shape keys or RGB UVs changed"
    assert result["failure_restored"], "UV projection failure changed the source or leaked temporary resources"


def test_nonplanar_ngon_has_unique_thermal_uvs_without_changing_source_geometry(executable):
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        result = call_service(client, "bake_uv_folded_ngon")

    assert result["thermal_points"] > 0, result
    assert result["same_triangles"], "thermal triangulation differs from Blender's rendered geometry"
    assert result["rgb_uv_preserved"], "private mesh changed the RGB UV coordinates"
    assert result["source_preserved"], "source polygon was changed"


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
        # Coarse bake: this checks that flux tracks the checker albedo, not bake quality.
        result = call_service(client, "absorbed_flux", bake_samples=32, irradiance_texture_size=64)

    assert result["shape"] == [result["vertices"]]
    assert result["finite"], "absorbed flux contains non-finite values"
    assert result["std"] > 0.01, f"absorbed flux did not vary with checker albedo: {result['std']}"


def test_all_zero_attribute_albedo_is_ignored_and_rebaked(executable):
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        call_service(client, "build_scene", "grid", "Checker", material="checker", lit=False)
        result = call_service(client, "bake_vertex_albedo", stale_zeros=True)

    assert result["baked"] == "ok", "albedo absent"
    assert result["std"] > 0.05, f"zeros attribute was served instead of re-baking (std={result['std']})"


def test_opaque_receiver_measures_black_and_metal_and_preserves_other_surfaces(executable):
    with BlenderClient.spawn(executable=executable, timeout=120, service=SERVICE_PATH) as client:
        result = call_service(client, "opaque_receiver_fixture")

    assert result["eligible"] == ["black", "gray", "metal"]
    for kind, albedo in (("black", 0), ("gray", 0.18), ("metal", 0.6)):
        values = result["patches"][kind]
        assert values["incident"] == pytest.approx(math.pi, rel=0.02), (kind, values)
        assert values["albedo"] == pytest.approx(albedo, abs=0.005), (kind, values)
        absorbed = values["incident"] * (1 - values["albedo"])
        assert absorbed == pytest.approx(math.pi * (1 - albedo), rel=0.02)
    for kind, values in result["patches"].items():
        if kind in result["eligible"]:
            continue
        assert values["incident"] == pytest.approx(values["old_incident"], abs=1e-6), (kind, values)
        assert values["albedo"] == pytest.approx(values["old_albedo"], abs=1e-6), (kind, values)
    assert result["patches"]["black"]["old_incident"] == 0
    assert result["patches"]["metal"]["old_incident"] == 0
    assert result["restored"], "source material graphs, slots, face assignments or UV state changed"
    assert result["failure_restored"], "private material resources leaked after an injected bake failure"


def test_layered_opaque_receiver_measures_incident_light_without_changing_albedo(executable):
    with BlenderClient.spawn(executable=executable, timeout=120, service=SERVICE_PATH) as client:
        result = call_service(client, "opaque_receiver_fixture", layered=True)

    assert result["eligible"] == []  # The albedo path retains its direct-Principled rule.
    assert result["incident_eligible"] == ["layered_add", "layered_mix", "layered_nested", "layered_glossy"]
    for kind, values in result["patches"].items():
        if kind in result["incident_eligible"]:
            assert values["incident"] == pytest.approx(math.pi, rel=0.02), (kind, values)
        else:
            assert values["incident"] == pytest.approx(values["old_incident"], abs=1e-6), (kind, values)
        assert values["albedo"] == pytest.approx(values["old_albedo"], abs=1e-6), (kind, values)
    assert result["restored"], "layered source materials or UV state changed"
    assert result["failure_restored"], "private resources leaked after a failed layered bake"


@pytest.mark.parametrize("layered", [False, True])
def test_opaque_receiver_preserves_same_object_indirect_reflections(executable, layered):
    with BlenderClient.spawn(executable=executable, timeout=120, service=SERVICE_PATH) as client:
        result = call_service(client, "opaque_receiver_concave", layered=layered)

    assert result["measured"] == pytest.approx(result["expected"], rel=0.02), result
    assert result["global_white"] > 1.8 * result["measured"], result


def test_opaque_receiver_ends_after_a_transparent_crossing(executable):
    with BlenderClient.spawn(executable=executable, timeout=120, service=SERVICE_PATH) as client:
        result = call_service(client, "opaque_receiver_transparent_path")

    assert result["False"] == pytest.approx(1, abs=0.01), result
    assert result["True"] == pytest.approx(0, abs=1e-6), result
