from __future__ import annotations

import inspect
import json
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pytest

from visionsim.simulate.blender import BlenderService
from visionsim.simulate.config import ThermalConfig
from visionsim.simulate.heatsim import adapter, materials

_DEFAULTS = {
    "initial_temperature_K": 300.0,
    "thermal_diffusivity_mm2_s": 0.42,
    "density_kg_m3": 1234.0,
    "specific_heat_J_kgK": 777.0,
    "emissivity": 0.5,
    "irradiance_scale": 1.0,
}
_SOLVER_CFG = {"domain": "POINTS"}


class _Poly:
    def __init__(self, vertices, material_index, area):
        self.vertices = list(vertices)
        self.material_index = material_index
        self.area = area


class _AttrData(list):
    """Mirrors the bit of bpy's attribute-data collection ``_write_point_float_attr`` uses."""

    def foreach_set(self, prop, values):
        for elem, value in zip(self, values):
            setattr(elem, prop, float(value))


class _Attr:
    def __init__(self, n):
        self.data = _AttrData(type("D", (), {"value": 0.0})() for _ in range(n))


class _Attrs(dict):
    def __init__(self, n):
        super().__init__()
        self._n = n

    def new(self, name, type, domain):
        self[name] = _Attr(self._n)
        return self[name]


class _Mesh:
    def __init__(self, verts_xyz, polygons):
        self._xyz = np.asarray(verts_xyz, dtype=np.float64)
        self.vertices = [object()] * len(self._xyz)
        self.polygons = list(polygons)
        self.attributes = _Attrs(len(self._xyz))

    def update(self):
        pass


class _Obj(dict):
    # dict-backed so ``obj["key"] = value`` mirrors bpy custom properties, but real
    # bpy.types.Object instances are hashable by identity - restore that here,
    # since plain ``dict`` (unlike ``object``) sets __hash__ = None.
    __hash__ = object.__hash__
    __eq__ = object.__eq__

    def __init__(self, name, mesh, slot_names):
        super().__init__()
        self.name = name
        self.type = "MESH"
        self.data = mesh
        self.material_slots = [type("S", (), {"material": type("M", (), {"name": n})()})() for n in slot_names]
        self.heat_sim_material = None


def _square():
    """4 verts, 2 coplanar tris; tri 0 uses slot 0, tri 1 uses slot 1."""
    xyz = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    return _Obj("square", _Mesh(xyz, [_Poly([0, 1, 2], 0, 0.5), _Poly([0, 2, 3], 1, 0.5)]), ["WOODY", "STEELY"])


def _sidecar(tmp_path, block):
    path = tmp_path / "s.thermal.json"
    path.write_text(json.dumps({
        "schema_version": 1, "scene": "s.blend", "defaults": {"preset": None}, "materials": block,
    }), encoding="utf-8")
    return materials.load_assignments(path)


@pytest.fixture(autouse=True)
def _stub_geometry(monkeypatch):
    """_extract_geometry needs bpy/mathutils; feed it straight from the fake mesh."""
    def fake(obj):
        verts = np.asarray(obj.data._xyz, dtype=np.float64) * 1000.0  # m -> mm
        faces = np.asarray([list(p.vertices) for p in obj.data.polygons], dtype=np.int32)
        return verts, faces, len(verts)

    monkeypatch.setattr(adapter, "_extract_geometry", fake)


# --- _combine ---------------------------------------------------------------


def test_without_assignment_arrays_stay_constant_per_object():
    """Backward-compat guard: assignment=None must reproduce today's behaviour exactly."""
    combined = adapter._combine([_square()], {}, _DEFAULTS, _SOLVER_CFG)
    assert combined is not None
    assert np.allclose(combined.alpha, _DEFAULTS["thermal_diffusivity_mm2_s"])
    assert np.allclose(combined.eps, _DEFAULTS["emissivity"])
    assert np.allclose(combined.t0, _DEFAULTS["initial_temperature_K"])
    assert np.all(combined.boundary_mask)


def test_with_assignment_alpha_varies_within_one_object(tmp_path):
    sa = _sidecar(tmp_path, {"WOODY": {"preset": "wood"}, "STEELY": {"preset": "steel"}})
    combined = adapter._combine([_square()], {}, _DEFAULTS, _SOLVER_CFG, assignment=sa)
    assert combined is not None
    # Vert 1 is only in the wood tri; vert 3 only in the steel tri.
    assert combined.alpha[1] == pytest.approx(materials.PRESETS["wood"].alpha_mm2_s)
    assert combined.alpha[3] == pytest.approx(materials.PRESETS["steel"].alpha_mm2_s)
    assert combined.alpha.min() < combined.alpha.max()
    assert combined.eps[1] == pytest.approx(materials.PRESETS["wood"].emissivity_ir)


def test_density_is_converted_to_kg_per_mm3(tmp_path):
    sa = _sidecar(tmp_path, {"WOODY": {"preset": "wood"}, "STEELY": {"preset": "wood"}})
    combined = adapter._combine([_square()], {}, _DEFAULTS, _SOLVER_CFG, assignment=sa)
    assert combined is not None
    assert combined.density[0] == pytest.approx(materials.PRESETS["wood"].density_kg_m3 / 1.0e9)


def test_per_vertex_dirichlet_pins_only_the_source_vertices(tmp_path):
    sa = _sidecar(tmp_path, {
        "WOODY": {"preset": "wood"},
        "STEELY": {"preset": "steel", "role": "DIRICHLET_SOURCE", "dirichlet_K": 350.0},
    })
    obj = _square()
    combined = adapter._combine([obj], {obj: np.full(4, 10.0)}, _DEFAULTS, _SOLVER_CFG, assignment=sa)
    assert combined is not None

    pinned = ~combined.boundary_mask
    assert pinned.any() and not pinned.all(), "expected a partially pinned object"
    assert np.allclose(combined.alpha[pinned], 0.0)
    assert np.allclose(combined.irradiance[pinned], 0.0)
    assert np.all(combined.irradiance[~pinned] > 0.0)
    assert np.allclose(combined.t0[pinned], 350.0)
    assert np.allclose(combined.t0[~pinned], _DEFAULTS["initial_temperature_K"])


def test_object_with_no_slots_keeps_the_object_level_path(tmp_path):
    sa = _sidecar(tmp_path, {"WOODY": {"preset": "wood"}})
    obj = _square()
    obj.material_slots = []
    combined = adapter._combine([obj], {}, _DEFAULTS, _SOLVER_CFG, assignment=sa)
    assert combined is not None
    assert np.allclose(combined.alpha, _DEFAULTS["thermal_diffusivity_mm2_s"])


def test_topology_changing_modifier_falls_back_to_object_level(tmp_path, monkeypatch, caplog):
    """A modifier makes the evaluated geometry a different vertex count than the base mesh.

    resolve_vertex_materials sizes its arrays to the base mesh (4 verts here); _combine
    operates on the evaluated geometry. When they disagree the per-slot arrays cannot be
    combined with the (evaluated-sized) flux/mask, so the object must degrade to the
    object-level path with a warning rather than raise a broadcast error (the real-scene
    crash on kitchen1's modifier-carrying 'Circle' lamp: dmask (6018,) vs irr (6870,)).
    """
    # Evaluated geometry has 6 verts / 2 tris while the base _square() mesh has 4.
    ev_xyz = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0),
                       (0.0, 1.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.0)], dtype=np.float64)
    ev_faces = np.array([[0, 1, 4], [2, 3, 5]], dtype=np.int32)
    monkeypatch.setattr(adapter, "_extract_geometry", lambda obj: (ev_xyz * 1000.0, ev_faces, 6))

    sa = _sidecar(tmp_path, {"WOODY": {"preset": "wood"}, "STEELY": {"preset": "steel"}})
    obj = _square()
    with caplog.at_level("WARNING"):
        combined = adapter._combine([obj], {}, _DEFAULTS, _SOLVER_CFG, assignment=sa)

    assert combined is not None
    assert combined.alpha.shape == (6,), "arrays must match the evaluated geometry, not the base mesh"
    # Fell back to the object-level constant rather than per-slot values.
    assert np.allclose(combined.alpha, _DEFAULTS["thermal_diffusivity_mm2_s"])
    assert np.allclose(combined.eps, _DEFAULTS["emissivity"])
    assert "topology-changing modifier" in caplog.text


# --- write_frame_attributes -------------------------------------------------


def _emissivity(mesh):
    return np.array([d.value for d in mesh.attributes["emissivity"].data])


def test_without_assignment_emissivity_is_constant():
    obj = _square()
    scene = type("S", (), {"objects": [obj]})()
    adapter.write_frame_attributes(scene, {"square": np.full((2, 4), 305.0)}, -1, _DEFAULTS)
    assert np.allclose(_emissivity(obj.data), _DEFAULTS["emissivity"])


def test_with_assignment_emissivity_varies_per_slot(tmp_path):
    """The single largest lever on how a thermal frame looks - spec section 4b."""
    sa = _sidecar(tmp_path, {
        "WOODY": {"preset": "aluminium_polished"},
        "STEELY": {"preset": "metal_painted"},
    })
    obj = _square()
    scene = type("S", (), {"objects": [obj]})()
    adapter.write_frame_attributes(scene, {"square": np.full((2, 4), 305.0)}, -1, _DEFAULTS, assignment=sa)

    eps = _emissivity(obj.data)
    assert eps[1] == pytest.approx(materials.PRESETS["aluminium_polished"].emissivity_ir, abs=1e-6)
    assert eps[3] == pytest.approx(materials.PRESETS["metal_painted"].emissivity_ir, abs=1e-6)
    assert eps.max() - eps.min() > 0.5


def test_temperature_attribute_is_still_written(tmp_path):
    sa = _sidecar(tmp_path, {"WOODY": {"preset": "wood"}, "STEELY": {"preset": "wood"}})
    obj = _square()
    scene = type("S", (), {"objects": [obj]})()
    history = {"square": np.stack([np.full(4, 300.0), np.full(4, 307.0)])}
    adapter.write_frame_attributes(scene, history, -1, _DEFAULTS, assignment=sa)
    assert np.allclose([d.value for d in obj.data.attributes["sim_temperature"].data], 307.0)


def test_unsimulated_mesh_still_gets_the_fallback_property(tmp_path):
    sa = _sidecar(tmp_path, {"WOODY": {"preset": "wood"}, "STEELY": {"preset": "wood"}})
    obj = _square()
    scene = type("S", (), {"objects": [obj]})()
    adapter.write_frame_attributes(scene, {}, -1, _DEFAULTS, assignment=sa)
    assert obj["heatsim_default_temperature"] == pytest.approx(_DEFAULTS["initial_temperature_K"])


def test_thermal_config_has_an_assignments_field():
    assert "assignments" in {f.name for f in fields(ThermalConfig)}
    assert ThermalConfig().assignments is None
    assert ThermalConfig(assignments=Path("a.json")).assignments == Path("a.json")


def test_asdict_dispatch_matches_both_service_signatures():
    """job.py and cli/blender.py both call **asdict(config.thermal); every key must land."""
    keys = set(asdict(ThermalConfig()))
    for method in (BlenderService.exposed_prepare_thermal, BlenderService.exposed_heatsim_solve):
        missing = keys - (set(inspect.signature(method).parameters) - {"self"})
        assert not missing, f"{method.__name__} is missing {sorted(missing)}"
