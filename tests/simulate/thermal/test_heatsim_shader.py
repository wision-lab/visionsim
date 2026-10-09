from __future__ import annotations

from tests.simulate.thermal.heatsim_test_service import SERVICE_PATH, call_service
from visionsim.simulate.blender import BlenderClient


def test_temperature_aov_registered(executable):
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        result = call_service(client, "temperature_aov")
    assert "temperature" in result["aov_names"], result["aov_names"]


def test_atlas_shader_group_samples_atlas_and_mixes_by_alpha(executable):
    """Temperature and radiance shaders read covered atlas values consistently."""
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        result = call_service(client, "shader_graphs")

    gray = result["gray"]
    assert gray["has_uv_attr"], "missing atlas UV attribute node"
    assert gray["uv_attr_geometry"], "atlas UV attribute node must be GEOMETRY"
    assert gray["has_coverage_attr"], "missing atlas coverage gate attribute node"
    assert gray["coverage_attr_object"], "atlas coverage gate must be OBJECT"
    assert gray["has_sim_temp_attr"] and gray["has_default_temp_attr"]

    assert gray["n_tex_nodes"] >= 1, "missing atlas Image Texture node"
    assert gray["tex_image_name"] == "HeatSim_Temperature_Atlas"
    assert gray["tex_non_color"], "atlas image must use the Non-Color colorspace"
    assert gray["tex_linear"] and gray["tex_clip"]
    assert gray["uv_feeds_vector"], "atlas UV attribute not wired into an Image Texture Vector input"

    assert gray["has_mix"], "missing atlas Mix node"
    assert gray["mix_float"], "temperature atlas Mix node must be FLOAT"
    assert gray["mix_factor_linked"] and gray["mix_ab_linked"]

    # Mix Factor traces back (within hops) to the Image Texture's Alpha output and the
    # object-level gate -- the atlas validity signal must gate the mix.
    assert gray["factor_multiply"], "Mix Factor must be a MULTIPLY of alpha and gate"
    assert gray["factor_from_tex"], "Mix Factor does not trace back to atlas Alpha"
    assert gray["factor_has_gate"], "Mix Factor missing the object-level gate"
    assert gray["mix_feeds_pow4"], "Mix result not wired into the T^4 chain"

    # Divide filtered temperature by filtered coverage at atlas edges.
    assert gray["b_divide"], "Mix B is not a DIVIDE of filtered temperature by coverage"
    assert gray["b_separate_color"], "divide numerator is not a SeparateColor (red extraction)"
    assert gray["tex_feeds_separate_color"], "texture does not feed the SeparateColor node"
    assert gray["alpha_feeds_divide"], "texture Alpha is not wired into the divide divisor"

    # The AOV material is a separate graph built by _append_temperature_aov_nodes, so it
    # must satisfy the same structural checks -- it can regress independently of gray-body.
    aov = result["aov"]
    assert aov["has_uv_attr"] and aov["uv_attr_geometry"]
    assert aov["has_coverage_attr"] and aov["coverage_attr_object"]
    assert aov["has_sim_temp_attr"] and aov["has_default_temp_attr"]
    assert aov["tex_image_name"] == "HeatSim_Temperature_Atlas"
    assert aov["tex_non_color"] and aov["tex_linear"] and aov["tex_clip"]
    assert aov["uv_feeds_vector"], "atlas UV attribute not wired into an Image Texture Vector input"
    assert aov["has_mix"] and aov["mix_float"]
    assert aov["mix_factor_linked"] and aov["mix_ab_linked"]
    assert aov["factor_multiply"] and aov["factor_from_tex"] and aov["factor_has_gate"]
    assert aov["value_linked"], "AOV Value input is not linked"
    assert aov["mix_reaches_aov"], "Mix result does not reach the OutputAOV"


def test_filtered_atlas_edge_preserves_temperature(executable, tmp_path):
    """A partially covered atlas sample must stay at the solved temperature."""
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        result = call_service(client, "render_temperature", str(tmp_path))

    assert result["n"] == 256
    assert abs(result["min"] - 295.0) < 0.01, result["min"]
    assert abs(result["max"] - 295.0) < 0.01, result["max"]


def test_atlas_shader_group_falls_back_when_no_atlas_image(executable):
    """No ``HeatSim_Temperature_Atlas`` image registered (render_domain=VERTEX, the atlas is
    never built) -> the Image Texture node has no image, but the graph must still build (no
    exceptions) and every node this test can reach must exist -- the byte-identical VERTEX
    guarantee is enforced by the mix factor being multiplied by the OBJECT-level gate, which
    is never stamped (and so defaults to 0) whenever ``write_frame_attributes`` is called
    without an ``atlas_plan`` -- this test only guards the structural half (no crash, no
    dangling/unlinked sockets) since the zero-default-attribute behavior itself is core
    Blender semantics, not something under test here.
    """
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        result = call_service(client, "shader_graphs", register_atlas=False)

    gray = result["gray"]
    assert gray["tex_image_name"] is None, "an atlas image was unexpectedly registered"
    assert gray["mix_factor_linked"] and gray["mix_ab_linked"]
    assert result["aov"]["value_linked"], "AOV Value input is not linked"
    assert result["aov"]["mix_reaches_aov"], "Mix result does not reach the OutputAOV"


def test_enter_restore_round_trip(executable):
    """Thermal passes preserve shared meshes, face assignments and object overrides."""
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        result = call_service(client, "enter_restore")

    assert result["setup_preserved"], "thermal setup changed source material assignments"
    assert result["override_ok"], "view layers did not receive the gray-body material override"
    assert result["restored"], "thermal pass did not restore scene state"


def test_meshes_without_materials_get_a_temperature_carrying_surface(executable):
    """A mesh with no material (or an empty slot) has no shader to write the value
    AOV, so it renders 0 K however well it was simulated. It must be given one."""
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        result = call_service(client, "default_surface")

    # Fixture preconditions: the setup must actually exercise the bare and empty-slot cases.
    assert result["bare_slots_before"] == 0, "bare fixture did not start slotless"
    assert result["half_empty_before"], "half fixture did not start with an empty slot"

    for name in ("bare", "authored", "half"):
        assert result["slots"][name] > 0, f"{name}: no material slot"
        assert result["aov_counts"][name] > 0, f"{name}: no temperature AOV on its material (or an empty slot)"

    # The stand-in must not disturb the authored material.
    assert result["authored_kept"], "authored material was replaced"

    # It is shared, not one material per object.
    assert result["bare_default"], "bare mesh did not get the shared default surface"
    assert result["default_count"] == 1, f"expected one shared default surface, got {result['default_count']}"

    # Idempotent: a second pass must not add more slots.
    assert result["idempotent"], "a second setup pass added material slots"


def test_radiance_matches_gray_body_for_per_vertex_emissivity(executable):
    """Rendered radiance must equal eps*sigma*T^4 + (1-eps)*sigma*T_amb^4, per surface.

    Two defects made this false. The gray-body shader baked ONE emissivity constant into
    the mix factor and assigned that single material to every mesh, so the per-vertex
    ``emissivity`` attribute the solve writes was never read and every surface radiated as
    if eps=0.9. And the thermal world emitted the ambient temperature directly where the
    reflected ``(1 - eps)`` term needs a radiance, ``sigma*T_amb^4``.

    The second was a ~4% error while eps was pinned at 0.9 and invisible; it dominates at
    low emissivity, where a surface is mostly a mirror. Both matter for LWIR: the preset
    library spans eps 0.05 (polished aluminium) to 0.98 (skin), and a low-emissivity
    surface reading close to ambient IS the physical behaviour that makes polished metal
    hard to measure with a thermal camera.
    """
    with BlenderClient.spawn(executable=executable, timeout=120, service=SERVICE_PATH) as client:
        result = call_service(client, "gray_body_radiance")
    sigma = result["sigma"]
    t_amb = result["t_amb"]
    t_hot = result["t_hot"]
    rendered = result["rendered"]

    for key, got in rendered.items():
        eps = float(key)
        want = eps * sigma * t_hot**4 + (1.0 - eps) * sigma * t_amb**4
        rel = abs(got - want) / want
        assert rel < 0.02, f"eps={eps}: rendered {got:.2f}, gray body predicts {want:.2f} ({rel:.1%} off)"

    # And the emissivity must actually be read per surface, not collapsed to one constant.
    assert rendered["0.98"] - rendered["0.05"] > 300.0, (
        f"emissivity barely changed radiance: {rendered['0.05']:.1f} vs {rendered['0.98']:.1f}"
    )


def test_rgb_unchanged_across_thermal_renders(executable, tmp_path):
    """The RGB/thermal render loop preserves RGB pixels and cleans up failed passes."""
    with BlenderClient.spawn(executable=executable, timeout=120, service=SERVICE_PATH) as client:
        result = call_service(client, "rgb_thermal_loop", str(tmp_path))

    assert result["n_frames"] == 3, result["n_frames"]
    assert result["identical"], "RGB pixels changed across thermal renders"
    assert result["n_radiance"] == 2, result["n_radiance"]
    assert result["radiance_finite"], "thermal radiance contains non-finite values"
    assert result["radiance_max"] > 100, "thermal override did not emit radiance"
    assert result["cleanup_ok"], "failed thermal pass did not clean up scene state"
    assert result["propagation_ok"], "injected render failure was not propagated"
    # Each injected failure must surface after exactly the expected render calls -- the
    # loop must not swallow the failure and retry silently.
    assert result["call_counts"] == [1, 2], result["call_counts"]
