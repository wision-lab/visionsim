from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np
import pytest

from visionsim.simulate.heatsim import constants, materials

# ---------------------------------------------------------------------------
# 1. Preset library
# ---------------------------------------------------------------------------

# Seeded from the vendored constants.py dicts; alpha/rho/c must agree exactly so
# the new library and the addon-parity constants cannot silently drift apart.
_SEEDED = ["aluminium", "pvc", "glass", "copper", "polystyrene", "wood", "steel",
           "brick", "concrete", "plaster", "asphalt", "iron", "li_ion"]


def test_library_is_well_formed():
    assert len(materials.PRESETS) == 29
    for key, preset in materials.PRESETS.items():
        assert preset.key == key
        assert preset.alpha_mm2_s > 0.0, key
        assert preset.density_kg_m3 > 0.0, key
        assert preset.specific_heat_J_kgK > 0.0, key
        assert 0.0 <= preset.emissivity_ir <= 1.0, key
        assert preset.notes, key


def test_preset_keys_are_sorted_and_complete():
    keys = materials.preset_keys()
    assert keys == sorted(keys)
    assert set(keys) == set(materials.PRESETS)


def test_seeded_presets_agree_with_vendored_constants():
    for key in _SEEDED:
        preset = materials.PRESETS[key]
        assert preset.alpha_mm2_s == pytest.approx(constants.TDiff[key]), key
        # constants.Density is kg/mm^3 (kg/m^3 divided by 1000**3).
        assert preset.density_kg_m3 == pytest.approx(constants.Density[key] * 1.0e9), key
        assert preset.specific_heat_J_kgK == pytest.approx(constants.SpecificHeat[key]), key


def test_polished_and_painted_metal_are_distinct():
    """The single largest visual lever in LWIR - see spec section 4b."""
    assert materials.PRESETS["aluminium_polished"].emissivity_ir < 0.10
    assert materials.PRESETS["metal_painted"].emissivity_ir > 0.85


def test_alpha_spans_three_orders_of_magnitude():
    """Sanity: the library must actually discriminate insulators from conductors."""
    alphas = np.array([p.alpha_mm2_s for p in materials.PRESETS.values()])
    assert alphas.max() / alphas.min() > 1000.0


def test_presets_are_immutable():
    # ThermalPreset is frozen: resolve_material/resolve_vertex_materials hand the SAME
    # instance to every caller, so a mutable preset would let one caller poison the
    # global library. Assert the specific error rather than bare Exception.
    with pytest.raises(dataclasses.FrozenInstanceError):
        materials.PRESETS["wood"].alpha_mm2_s = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. Sidecar parsing
# ---------------------------------------------------------------------------


def _write(tmp_path, block, default_preset="plaster", name="scene.thermal.json"):
    path = tmp_path / name
    path.write_text(json.dumps({
        "schema_version": 1,
        "scene": "kitchen1.blend",
        "defaults": {"preset": default_preset},
        "materials": block,
    }), encoding="utf-8")
    return path


def test_loads_a_well_formed_sidecar(tmp_path):
    path = _write(tmp_path, {
        "MADERA ESTANTES": {"preset": "wood", "role": "FEM_PARTICIPANT", "confidence": 0.95, "reason": "madera"},
    })
    sa = materials.load_assignments(path)
    assert sa.scene == "kitchen1.blend"
    assert sa.default_preset is not None and sa.default_preset.key == "plaster"
    entry = sa.entry_for("MADERA ESTANTES")
    assert entry is not None and entry.preset is not None
    assert entry.preset.key == "wood"
    assert entry.role == "FEM_PARTICIPANT" and entry.dirichlet_K is None


def test_material_absent_from_the_sidecar_returns_none(tmp_path):
    sa = materials.load_assignments(_write(tmp_path, {"a": {"preset": "wood"}}))
    assert sa.entry_for("NOT PRESENT") is None


def test_null_reason_does_not_become_the_literal_string_none(tmp_path):
    """``spec.get("reason", "")`` would turn a JSON ``null`` into the string "None"
    (``str(None)``); the ``or``-fallback used elsewhere in this file must be used here too."""
    path = _write(tmp_path, {"x": {"preset": "wood", "reason": None}})
    entry = materials.load_assignments(path).entry_for("x")
    assert entry is not None
    assert entry.reason == ""


def test_null_scene_does_not_become_the_literal_string_none(tmp_path):
    """Same ``.get(key, default)`` vs ``or``-fallback hazard for the top-level ``scene``
    field, which reaches the solve cache key - a stray "None" there would be a silently
    wrong (but stable-looking) cache key component."""
    path = tmp_path / "scene.thermal.json"
    path.write_text(json.dumps({
        "schema_version": 1, "scene": None, "defaults": {}, "materials": {},
    }), encoding="utf-8")
    sa = materials.load_assignments(path)
    assert sa.scene == path.name


def test_unknown_preset_becomes_unassigned_and_warns(tmp_path):
    """An out-of-enum preset must never be silently guessed at - spec section 5."""
    path = _write(tmp_path, {"weird": {"preset": "unobtainium"}})
    with pytest.warns(UserWarning, match="unobtainium"):
        sa = materials.load_assignments(path)
    entry = sa.entry_for("weird")
    assert entry is not None and entry.preset is None


def test_dirichlet_source_keeps_an_in_band_temperature(tmp_path):
    path = _write(tmp_path, {"ilum": {"preset": "glass", "role": "DIRICHLET_SOURCE", "dirichlet_K": 345.0}})
    entry = materials.load_assignments(path).entry_for("ilum")
    assert entry is not None
    assert entry.role == "DIRICHLET_SOURCE" and entry.dirichlet_K == 345.0


@pytest.mark.parametrize("bad", [12.0, 5000.0])
def test_out_of_band_dirichlet_temperature_is_dropped_and_warns(tmp_path, bad):
    """An out-of-band dirichlet_K must not leave the slot pinned at ambient with role
    still DIRICHLET_SOURCE (alpha=0, no incident flux) - that would silently turn an
    intended heat source into a heat sink. The role degrades to FEM_PARTICIPANT too."""
    path = _write(tmp_path, {"ilum": {"preset": "glass", "role": "DIRICHLET_SOURCE", "dirichlet_K": bad}})
    with pytest.warns(UserWarning, match="dirichlet_K"):
        sa = materials.load_assignments(path)
    entry = sa.entry_for("ilum")
    assert entry is not None and entry.dirichlet_K is None
    assert entry.role == "FEM_PARTICIPANT"


def test_unknown_role_falls_back_to_fem_and_warns(tmp_path):
    path = _write(tmp_path, {"x": {"preset": "wood", "role": "TELEPORTER"}})
    with pytest.warns(UserWarning, match="role"):
        sa = materials.load_assignments(path)
    entry = sa.entry_for("x")
    assert entry is not None and entry.role == "FEM_PARTICIPANT"


def test_role_defaults_to_fem_participant(tmp_path):
    entry = materials.load_assignments(_write(tmp_path, {"x": {"preset": "wood"}})).entry_for("x")
    assert entry is not None and entry.role == "FEM_PARTICIPANT"


def test_unknown_default_preset_warns_and_disables_the_default(tmp_path):
    path = _write(tmp_path, {"x": {"preset": "wood"}}, default_preset="nonsense")
    with pytest.warns(UserWarning, match="nonsense"):
        sa = materials.load_assignments(path)
    assert sa.default_preset is None


def test_bad_schema_version_raises(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"schema_version": 99, "scene": "s", "defaults": {}, "materials": {}}), "utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        materials.load_assignments(path)


def test_non_numeric_dirichlet_k_raises_naming_the_sidecar(tmp_path):
    path = _write(tmp_path, {"ilum": {"preset": "glass", "role": "DIRICHLET_SOURCE", "dirichlet_K": "hot"}})
    with pytest.raises(ValueError, match="scene.thermal.json"):
        materials.load_assignments(path)


def test_null_defaults_block_raises_naming_the_sidecar(tmp_path):
    path = tmp_path / "scene.thermal.json"
    path.write_text(json.dumps({
        "schema_version": 1, "scene": "kitchen1.blend", "defaults": None, "materials": {},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="scene.thermal.json"):
        materials.load_assignments(path)


def test_null_materials_block_raises_naming_the_sidecar(tmp_path):
    path = tmp_path / "scene.thermal.json"
    path.write_text(json.dumps({
        "schema_version": 1, "scene": "kitchen1.blend", "defaults": {}, "materials": None,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="scene.thermal.json"):
        materials.load_assignments(path)


def test_non_dict_material_spec_raises_naming_the_sidecar_and_material(tmp_path):
    path = _write(tmp_path, {"ilum": ["not", "a", "dict"]})
    with pytest.raises(ValueError, match=r"scene\.thermal\.json.*'ilum'"):
        materials.load_assignments(path)


def test_digest_tracks_file_bytes(tmp_path):
    path = _write(tmp_path, {"x": {"preset": "wood"}})
    sa = materials.load_assignments(path)
    assert sa.digest == hashlib.sha256(path.read_bytes()).hexdigest()

    # Editing the sidecar must change the digest, so the solve cache invalidates.
    path.write_text(path.read_text(encoding="utf-8").replace("wood", "steel"), encoding="utf-8")
    assert materials.load_assignments(path).digest != sa.digest


# ---------------------------------------------------------------------------
# 3. Slot -> per-vertex resolution
# ---------------------------------------------------------------------------

# Distinctive globals: no value here can be confused with a preset or a
# PropertyGroup default (mirrors tests/test_heatsim_adapter.py).
_FALLBACK = {
    "initial_temperature_K": 300.0,
    "thermal_diffusivity_mm2_s": 0.42,
    "density_kg_m3": 1234.0,
    "specific_heat_J_kgK": 777.0,
    "emissivity": 0.5,
    "thermal_role": "FEM_PARTICIPANT",
    "dirichlet_temperature_K": 0.0,
}


class _Poly:
    def __init__(self, vertices, material_index, area):
        self.vertices = list(vertices)
        self.material_index = material_index
        self.area = area


class _Mesh:
    def __init__(self, n_verts, polygons):
        self.vertices = [object()] * n_verts
        self.polygons = list(polygons)


class _Obj:
    def __init__(self, name, mesh, slot_names):
        self.name = name
        self.data = mesh
        self.material_slots = [type("S", (), {"material": type("M", (), {"name": n})()})() for n in slot_names]


def _stool():
    """Verts 0,1 pure wood; verts 4,5 pure steel; verts 2,3 on the seam.

       0---1---(2---3)---4---5    wood area 6 on the left, steel area 2 on the right
    """
    polys = [_Poly([0, 1, 2, 3], 0, area=6.0), _Poly([2, 3, 4, 5], 1, area=2.0)]
    return _Obj("stool", _Mesh(6, polys), ["MADERA BANQUETAS", "METALBANQUETAS"])


def _sidecar(tmp_path, block, default_preset=None):
    return materials.load_assignments(_write(tmp_path, block, default_preset=default_preset))


def test_interior_vertices_get_their_slot_preset_exactly(tmp_path):
    sa = _sidecar(tmp_path, {"MADERA BANQUETAS": {"preset": "wood"}, "METALBANQUETAS": {"preset": "steel"}})
    out = materials.resolve_vertex_materials(_stool(), sa, _FALLBACK)
    assert out is not None
    assert out["alpha"][1] == pytest.approx(materials.PRESETS["wood"].alpha_mm2_s)
    assert out["eps"][1] == pytest.approx(materials.PRESETS["wood"].emissivity_ir)
    assert out["alpha"][4] == pytest.approx(materials.PRESETS["steel"].alpha_mm2_s)
    assert out["rho"][4] == pytest.approx(materials.PRESETS["steel"].density_kg_m3)


def test_seam_vertices_get_the_area_weighted_mean(tmp_path):
    wood, steel = materials.PRESETS["wood"], materials.PRESETS["steel"]
    sa = _sidecar(tmp_path, {"MADERA BANQUETAS": {"preset": "wood"}, "METALBANQUETAS": {"preset": "steel"}})
    out = materials.resolve_vertex_materials(_stool(), sa, _FALLBACK)
    assert out is not None

    expected_alpha = (wood.alpha_mm2_s * 6.0 + steel.alpha_mm2_s * 2.0) / 8.0
    expected_c = (wood.specific_heat_J_kgK * 6.0 + steel.specific_heat_J_kgK * 2.0) / 8.0
    for v in (2, 3):
        assert out["alpha"][v] == pytest.approx(expected_alpha)
        assert out["c"][v] == pytest.approx(expected_c)
    assert wood.alpha_mm2_s < out["alpha"][2] < steel.alpha_mm2_s


def test_categorical_role_is_dominant_not_blended(tmp_path):
    """A vertex is pinned or it is not - a 75/25 split must not make it 'partly pinned'."""
    sa = _sidecar(tmp_path, {
        "MADERA BANQUETAS": {"preset": "wood"},
        "METALBANQUETAS": {"preset": "steel", "role": "DIRICHLET_SOURCE", "dirichlet_K": 345.0},
    })
    out = materials.resolve_vertex_materials(_stool(), sa, _FALLBACK)
    assert out is not None
    assert not out["dirichlet_mask"][2] and not out["dirichlet_mask"][3]  # wood dominates the seam
    assert out["dirichlet_mask"][4] and out["dirichlet_mask"][5]
    assert out["t0"][4] == pytest.approx(345.0)


def test_dominant_flips_when_the_area_split_flips(tmp_path):
    sa = _sidecar(tmp_path, {
        "A": {"preset": "wood"},
        "B": {"preset": "steel", "role": "DIRICHLET_SOURCE", "dirichlet_K": 350.0},
    })
    obj = _Obj("o", _Mesh(6, [_Poly([0, 1, 2, 3], 0, 1.0), _Poly([2, 3, 4, 5], 1, 9.0)]), ["A", "B"])
    out = materials.resolve_vertex_materials(obj, sa, _FALLBACK)
    assert out is not None
    assert out["dirichlet_mask"][2] and out["dirichlet_mask"][3]


def test_unassigned_material_falls_back_to_the_object_defaults(tmp_path):
    sa = _sidecar(tmp_path, {"MADERA BANQUETAS": {"preset": "wood"}})  # slot 1 absent
    out = materials.resolve_vertex_materials(_stool(), sa, _FALLBACK)
    assert out is not None
    assert out["alpha"][4] == pytest.approx(_FALLBACK["thermal_diffusivity_mm2_s"])
    assert out["eps"][4] == pytest.approx(_FALLBACK["emissivity"])


def test_sidecar_default_preset_covers_unlisted_materials(tmp_path):
    sa = _sidecar(tmp_path, {"MADERA BANQUETAS": {"preset": "wood"}}, default_preset="plaster")
    out = materials.resolve_vertex_materials(_stool(), sa, _FALLBACK)
    assert out is not None
    assert out["alpha"][4] == pytest.approx(materials.PRESETS["plaster"].alpha_mm2_s)


def test_loose_vertices_take_the_object_defaults(tmp_path):
    """kitchen1 has objects with unreferenced verts; they must not produce NaN."""
    sa = _sidecar(tmp_path, {"A": {"preset": "wood"}})
    obj = _Obj("o", _Mesh(4, [_Poly([0, 1], 0, 3.0)]), ["A"])
    out = materials.resolve_vertex_materials(obj, sa, _FALLBACK)
    assert out is not None
    assert out["alpha"][2] == pytest.approx(_FALLBACK["thermal_diffusivity_mm2_s"])
    assert np.all(np.isfinite(out["alpha"])) and not out["dirichlet_mask"][2]


def test_zero_area_faces_still_vote(tmp_path):
    sa = _sidecar(tmp_path, {"A": {"preset": "wood"}})
    obj = _Obj("o", _Mesh(3, [_Poly([0, 1, 2], 0, 0.0)]), ["A"])
    out = materials.resolve_vertex_materials(obj, sa, _FALLBACK)
    assert out is not None
    assert np.all(np.isfinite(out["alpha"]))
    assert out["alpha"][0] == pytest.approx(materials.PRESETS["wood"].alpha_mm2_s)


def test_out_of_range_material_index_is_clamped(tmp_path):
    """Blender allows a polygon material_index past the end of material_slots."""
    sa = _sidecar(tmp_path, {"A": {"preset": "wood"}})
    obj = _Obj("o", _Mesh(3, [_Poly([0, 1, 2], 7, 1.0)]), ["A"])
    out = materials.resolve_vertex_materials(obj, sa, _FALLBACK)
    assert out is not None
    assert out["alpha"][0] == pytest.approx(materials.PRESETS["wood"].alpha_mm2_s)


def test_dirichlet_without_temperature_uses_the_object_initial_temperature(tmp_path):
    sa = _sidecar(tmp_path, {"A": {"preset": "steel", "role": "DIRICHLET_SOURCE"}})
    obj = _Obj("o", _Mesh(3, [_Poly([0, 1, 2], 0, 1.0)]), ["A"])
    out = materials.resolve_vertex_materials(obj, sa, _FALLBACK)
    assert out is not None
    assert out["dirichlet_mask"][0]
    assert out["t0"][0] == pytest.approx(_FALLBACK["initial_temperature_K"])


@pytest.mark.parametrize("slot_names,n_verts", [([], 3), (["A"], 0)])
def test_returns_none_when_slots_cannot_drive_resolution(tmp_path, slot_names, n_verts):
    sa = _sidecar(tmp_path, {"A": {"preset": "wood"}})
    obj = _Obj("o", _Mesh(n_verts, []), slot_names)
    assert materials.resolve_vertex_materials(obj, sa, _FALLBACK) is None


def test_shapes_dtypes_and_emissivity_range(tmp_path):
    sa = _sidecar(tmp_path, {
        "MADERA BANQUETAS": {"preset": "aluminium_polished"},
        "METALBANQUETAS": {"preset": "skin"},
    })
    out = materials.resolve_vertex_materials(_stool(), sa, _FALLBACK)
    assert out is not None
    for key in ("t0", "alpha", "rho", "c", "eps"):
        assert out[key].shape == (6,) and out[key].dtype == np.float64
    assert out["dirichlet_mask"].shape == (6,) and out["dirichlet_mask"].dtype == np.bool_
    assert np.all(out["eps"] >= 0.0) and np.all(out["eps"] <= 1.0)
