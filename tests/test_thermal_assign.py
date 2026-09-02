from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import thermal_assign

from visionsim.simulate.heatsim.materials import MAX_DIRICHLET_K, MIN_DIRICHLET_K, preset_keys

# --- dump -------------------------------------------------------------------


class _Poly:
    def __init__(self, material_index, area):
        self.material_index = material_index
        self.area = area


class _Mesh:
    def __init__(self, polygons, n_verts):
        self.polygons = list(polygons)
        self.vertices = [object()] * n_verts


class _Node:
    def __init__(self, type_, inputs=None):
        self.type = type_
        self.inputs = inputs or {}


class _Socket:
    def __init__(self, value):
        self.default_value = value


class _Material:
    def __init__(self, name, use_nodes=True, nodes=None):
        self.name = name
        self.use_nodes = use_nodes
        self.node_tree = type("T", (), {"nodes": list(nodes or [])})()


class _Obj:
    def __init__(self, name, mesh, mats, hide_render=False):
        self.name = name
        self.type = "MESH"
        self.data = mesh
        self.material_slots = [type("S", (), {"material": m})() for m in mats]
        self.hide_render = hide_render

    def visible_get(self):
        return True


def _scene(objects):
    return type("S", (), {"objects": list(objects)})()


def _data(mats):
    return type("D", (), {"materials": list(mats)})()


def test_face_area_share_ranks_by_coverage_and_sums_to_one():
    wall, knob = _Material("MUROS"), _Material("bronce")
    obj = _Obj("o", _Mesh([_Poly(0, 90.0), _Poly(1, 10.0)], 8), [wall, knob])
    dump = thermal_assign.collect_scene_materials(_scene([obj]), _data([wall, knob]))

    shares = {m["name"]: m["face_area_share"] for m in dump["materials"]}
    assert shares["MUROS"] == pytest.approx(0.9)
    assert shares["bronce"] == pytest.approx(0.1)
    assert [m["name"] for m in dump["materials"]] == ["MUROS", "bronce"], "sorted by share, descending"
    assert dump["n_mesh_objects"] == 1 and dump["n_vertices"] == 8


def test_emission_and_bsdf_are_captured():
    lamp = _Material("ilum", nodes=[_Node("EMISSION", {"Strength": _Socket(4.0)})])
    wood = _Material("MADERA", nodes=[_Node("BSDF_PRINCIPLED", {
        "Base Color": _Socket([0.40, 0.32, 0.23, 1.0]), "Metallic": _Socket(0.51),
    })])
    obj = _Obj("o", _Mesh([_Poly(0, 1.0), _Poly(1, 1.0)], 6), [lamp, wood])
    dump = thermal_assign.collect_scene_materials(_scene([obj]), _data([lamp, wood]))
    by_name = {m["name"]: m for m in dump["materials"]}

    assert by_name["ilum"]["emission"] == {"is_emissive": True, "strength": pytest.approx(4.0)}
    assert by_name["MADERA"]["bsdf"]["metallic"] == pytest.approx(0.51)
    assert by_name["MADERA"]["emission"]["is_emissive"] is False


def test_principled_emission_strength_with_black_color_is_not_emissive():
    """Blender 4.x defaults Principled BSDF to Emission Strength=1.0 with BLACK emission
    color, i.e. zero actual emission - strength alone must not flag it."""
    mat = _Material("wall", nodes=[_Node("BSDF_PRINCIPLED", {
        "Emission Strength": _Socket(1.0), "Emission Color": _Socket([0.0, 0.0, 0.0, 1.0]),
    })])
    obj = _Obj("o", _Mesh([_Poly(0, 1.0)], 3), [mat])
    dump = thermal_assign.collect_scene_materials(_scene([obj]), _data([mat]))
    assert dump["materials"][0]["emission"]["is_emissive"] is False


def test_principled_emission_strength_with_non_black_color_is_emissive():
    mat = _Material("lamp", nodes=[_Node("BSDF_PRINCIPLED", {
        "Emission Strength": _Socket(1.0), "Emission Color": _Socket([1.0, 0.9, 0.7, 1.0]),
    })])
    obj = _Obj("o", _Mesh([_Poly(0, 1.0)], 3), [mat])
    dump = thermal_assign.collect_scene_materials(_scene([obj]), _data([mat]))
    assert dump["materials"][0]["emission"] == {"is_emissive": True, "strength": pytest.approx(1.0)}


def test_principled_emission_strength_without_color_socket_is_emissive():
    """Older Blender with no Emission Color socket: strength alone is the right signal."""
    mat = _Material("lamp_old", nodes=[_Node("BSDF_PRINCIPLED", {"Emission Strength": _Socket(2.0)})])
    obj = _Obj("o", _Mesh([_Poly(0, 1.0)], 3), [mat])
    dump = thermal_assign.collect_scene_materials(_scene([obj]), _data([mat]))
    assert dump["materials"][0]["emission"] == {"is_emissive": True, "strength": pytest.approx(2.0)}


def test_unused_materials_hidden_objects_and_node_free_materials_are_handled():
    used, unused = _Material("USED"), _Material("UNUSED", use_nodes=False)
    hidden = _Obj("h", _Mesh([_Poly(0, 100.0)], 3), [used], hide_render=True)
    shown = _Obj("s", _Mesh([_Poly(0, 1.0)], 3), [used])
    dump = thermal_assign.collect_scene_materials(_scene([hidden, shown]), _data([used, unused]))
    by_name = {m["name"]: m for m in dump["materials"]}

    assert by_name["UNUSED"]["face_area_share"] == pytest.approx(0.0)
    assert by_name["UNUSED"]["bsdf"] == {}
    assert by_name["USED"]["objects"] == ["s"], "hidden objects must not contribute"


# --- assign -----------------------------------------------------------------


def _dump(materials):
    return {"schema_version": 1, "n_mesh_objects": 1, "n_vertices": 10, "materials": materials}


def _material(name, emissive=False, share=0.5):
    return {"name": name, "textures": [], "bsdf": {}, "node_types": [], "objects": ["o"],
            "emission": {"is_emissive": emissive, "strength": 4.0 if emissive else 0.0},
            "face_area_share": share}


def test_prompt_lists_the_materials_and_the_closed_preset_menu():
    prompt = thermal_assign.build_prompt(_dump([_material("MADERA ESTANTES"), _material("ilum", emissive=True)]))
    assert "MADERA ESTANTES" in prompt and "ilum" in prompt
    assert "aluminium_polished" in prompt and "metal_painted" in prompt
    assert "emissive" in prompt.lower()
    for key in preset_keys():
        assert key in prompt


def test_emissive_material_is_forced_to_a_source_role():
    """Deterministic guard: the EMISSION node is ground truth and outranks the model."""
    raw = [{"material": "ilum", "preset": "glass", "role": "FEM_PARTICIPANT",
            "dirichlet_K": None, "confidence": 0.6, "reason": "lamp glass"}]
    with pytest.warns(UserWarning, match="overrid"):
        out = thermal_assign.apply_guards(_dump([_material("ilum", emissive=True)]), raw)
    assert out["ilum"]["role"] == "DIRICHLET_SOURCE"
    assert out["ilum"]["dirichlet_K"] is not None


def test_emissive_material_already_marked_source_does_not_warn_about_override():
    """No warning-spam: the model already got it right, so the force is a no-op."""
    raw = [{"material": "ilum", "preset": "glass", "role": "DIRICHLET_SOURCE",
            "dirichlet_K": 350.0, "confidence": 0.9, "reason": "lamp element"}]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = thermal_assign.apply_guards(_dump([_material("ilum", emissive=True)]), raw)
    assert not any("overrid" in str(w.message) for w in caught)
    assert out["ilum"]["role"] == "DIRICHLET_SOURCE"
    assert out["ilum"]["dirichlet_K"] == pytest.approx(350.0)


def test_emissive_material_skipped_by_model_is_forced_to_source_and_warns_accurately():
    """A skipped emissive material is forced to DIRICHLET_SOURCE, not left 'unassigned'."""
    with pytest.warns(UserWarning, match="EMISSION node"):
        out = thermal_assign.apply_guards(_dump([_material("ilum", emissive=True)]), [])
    assert out["ilum"]["role"] == "DIRICHLET_SOURCE"
    assert out["ilum"]["dirichlet_K"] == pytest.approx(thermal_assign.DEFAULT_LAMP_K)
    assert MIN_DIRICHLET_K <= out["ilum"]["dirichlet_K"] <= MAX_DIRICHLET_K


def test_unknown_role_becomes_fem_participant_and_warns():
    raw = [{"material": "MUROS", "preset": "plaster", "role": "MOLTEN",
            "dirichlet_K": None, "confidence": 0.4, "reason": "??"}]
    with pytest.warns(UserWarning, match="role"):
        out = thermal_assign.apply_guards(_dump([_material("MUROS")]), raw)
    assert out["MUROS"]["role"] == "FEM_PARTICIPANT"
    assert out["MUROS"]["dirichlet_K"] is None


def test_non_emissive_material_keeps_its_model_role():
    raw = [{"material": "MUROS", "preset": "plaster", "role": "FEM_PARTICIPANT",
            "dirichlet_K": None, "confidence": 0.9, "reason": "walls"}]
    out = thermal_assign.apply_guards(_dump([_material("MUROS")]), raw)
    assert out["MUROS"]["role"] == "FEM_PARTICIPANT" and out["MUROS"]["dirichlet_K"] is None


def test_out_of_enum_preset_becomes_unassigned_and_warns():
    raw = [{"material": "_87", "preset": "unobtainium", "role": "FEM_PARTICIPANT",
            "dirichlet_K": None, "confidence": 0.2, "reason": "no idea"}]
    with pytest.warns(UserWarning, match="unobtainium"):
        out = thermal_assign.apply_guards(_dump([_material("_87")]), raw)
    assert out["_87"]["preset"] is None


def test_out_of_band_dirichlet_temperature_is_dropped_and_warns():
    """A non-emissive DIRICHLET_SOURCE with an out-of-band K must not stay pinned at
    ambient (which would silently act as a heat sink): the role is degraded too."""
    raw = [{"material": "hob", "preset": "steel", "role": "DIRICHLET_SOURCE",
            "dirichlet_K": 5000.0, "confidence": 0.7, "reason": "stove"}]
    with pytest.warns(UserWarning, match="dirichlet_K"):
        out = thermal_assign.apply_guards(_dump([_material("hob")]), raw)
    assert out["hob"]["dirichlet_K"] is None
    assert out["hob"]["role"] == "FEM_PARTICIPANT"


def test_out_of_band_dirichlet_temperature_on_emissive_material_still_forced_to_source():
    """Ground truth (the EMISSION node) still wins even after the out-of-band-K degrade:
    the role guard drops to FEM_PARTICIPANT first, then the emission check forces it back
    to DIRICHLET_SOURCE with DEFAULT_LAMP_K instead of the rejected out-of-band value."""
    raw = [{"material": "ilum", "preset": "glass", "role": "DIRICHLET_SOURCE",
            "dirichlet_K": 5000.0, "confidence": 0.7, "reason": "lamp"}]
    with pytest.warns(UserWarning, match="dirichlet_K"):
        out = thermal_assign.apply_guards(_dump([_material("ilum", emissive=True)]), raw)
    assert out["ilum"]["role"] == "DIRICHLET_SOURCE"
    assert out["ilum"]["dirichlet_K"] == pytest.approx(thermal_assign.DEFAULT_LAMP_K)


def test_skipped_material_is_added_back_and_hallucinated_one_is_dropped():
    raw = [
        {"material": "A", "preset": "wood", "role": "FEM_PARTICIPANT", "dirichlet_K": None,
         "confidence": 0.9, "reason": "wood"},
        {"material": "GHOST", "preset": "steel", "role": "FEM_PARTICIPANT", "dirichlet_K": None,
         "confidence": 0.9, "reason": "invented"},
    ]
    with pytest.warns(UserWarning):
        out = thermal_assign.apply_guards(_dump([_material("A"), _material("B")]), raw)
    assert set(out) == {"A", "B"}, "B added back, GHOST dropped"
    assert out["B"]["preset"] is None


def test_sidecar_round_trips_through_the_loader(tmp_path):
    import json

    from visionsim.simulate.heatsim.materials import load_assignments

    dump = _dump([_material("MADERA"), _material("ilum", emissive=True)])
    raw = [{"material": "MADERA", "preset": "wood", "role": "FEM_PARTICIPANT",
            "dirichlet_K": None, "confidence": 0.95, "reason": "madera = wood"}]
    with pytest.warns(UserWarning):
        guarded = thermal_assign.apply_guards(dump, raw)
    sidecar = thermal_assign.to_sidecar(dump, guarded, "kitchen1.blend")

    path = tmp_path / "kitchen1.thermal.json"
    path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    parsed = load_assignments(path)
    assert parsed.scene == "kitchen1.blend"
    entry = parsed.entry_for("MADERA")
    assert entry is not None and entry.preset is not None and entry.preset.key == "wood"
    lamp = parsed.entry_for("ilum")
    assert lamp is not None and lamp.role == "DIRICHLET_SOURCE"


# --- response parsing -------------------------------------------------------


def test_parse_assignments_accepts_a_bare_json_object():
    out = thermal_assign.parse_assignments_content('{"assignments": [{"material": "A", "preset": "wood"}]}')
    assert out == [{"material": "A", "preset": "wood"}]


def test_parse_assignments_strips_a_markdown_fence():
    """Reasoning models (GLM etc.) fence their JSON even under response_format=json_object."""
    fenced = '```json\n{"assignments": [{"material": "A", "preset": "wood"}]}\n```'
    out = thermal_assign.parse_assignments_content(fenced)
    assert out == [{"material": "A", "preset": "wood"}]


def test_parse_assignments_strips_a_bare_triple_backtick_fence():
    fenced = '```\n{"assignments": []}\n```'
    assert thermal_assign.parse_assignments_content(fenced) == []


def test_parse_assignments_rejects_non_json():
    with pytest.raises(ValueError, match="did not return JSON"):
        thermal_assign.parse_assignments_content("I cannot help with that.")


def test_parse_assignments_rejects_json_without_the_assignments_key():
    with pytest.raises(ValueError, match="assignments"):
        thermal_assign.parse_assignments_content('{"result": "ok"}')


# --- report -----------------------------------------------------------------


def _report_inputs():
    dump = _dump([_material("MUROS", share=0.7), _material("_87", share=0.2),
                  _material("ilum", emissive=True, share=0.1)])
    sidecar = {
        "schema_version": 1, "scene": "kitchen1.blend", "defaults": {"preset": "plaster"},
        "materials": {
            "MUROS": {"preset": "drywall", "role": "FEM_PARTICIPANT", "dirichlet_K": None,
                      "confidence": 0.95, "reason": "muros = walls"},
            "_87": {"preset": None, "role": "FEM_PARTICIPANT", "dirichlet_K": None,
                    "confidence": 0.0, "reason": "opaque name"},
            "ilum": {"preset": "glass", "role": "DIRICHLET_SOURCE", "dirichlet_K": 345.0,
                     "confidence": 1.0, "reason": "EMISSION node"},
        },
    }
    return dump, sidecar


def test_report_is_self_contained_area_ordered_and_flags_the_rows_needing_attention():
    html = thermal_assign.build_report(*_report_inputs())
    assert html.lstrip().startswith("<!DOCTYPE html>") and "<style>" in html
    assert "http://" not in html and "https://" not in html, "must not reference external assets"
    assert html.index("MUROS") < html.index("_87") < html.index("ilum")
    assert "UNASSIGNED" in html and "SOURCE" in html and "345" in html
    assert "drywall" in html and "0.31" in html  # resolved alpha is shown


def test_report_escapes_material_names():
    dump, sidecar = _report_inputs()
    dump["materials"][0]["name"] = "<script>x</script>"
    sidecar["materials"]["<script>x</script>"] = sidecar["materials"].pop("MUROS")
    html = thermal_assign.build_report(dump, sidecar)
    assert "<script>x</script>" not in html and "&lt;script&gt;" in html
