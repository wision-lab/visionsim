from __future__ import annotations

import json
import logging
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

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


# ---------------------------------------------------------------------------
# Fake-bpy fixtures (mirrors tests/test_heatsim_assignments_integration.py)
# ---------------------------------------------------------------------------


class _Poly:
    def __init__(self, vertices, material_index, area):
        self.vertices = list(vertices)
        self.material_index = material_index
        self.area = area


class _Mesh:
    def __init__(self, verts_xyz, polygons):
        self._xyz = np.asarray(verts_xyz, dtype=np.float64)
        self.vertices = [object()] * len(self._xyz)
        self.polygons = list(polygons)


class _Obj:
    __hash__ = object.__hash__
    __eq__ = object.__eq__

    def __init__(self, name, mesh, slot_names):
        self.name = name
        self.type = "MESH"
        self.data = mesh
        self.material_slots = [type("S", (), {"material": type("M", (), {"name": n})()})() for n in slot_names]
        self.heat_sim_material = None


class _NS:
    """Like ``SimpleNamespace`` but hashable by identity (mirrors real ``bpy`` objects,
    which ``_combine``'s ``flux_by_obj`` dict keys on)."""

    __hash__ = object.__hash__
    __eq__ = object.__eq__

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _square(name="square"):
    """4 verts, 2 coplanar tris; tri 0 uses slot 0 (WOODY), tri 1 uses slot 1 (STEELY)."""
    xyz = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    return _Obj(name, _Mesh(xyz, [_Poly([0, 1, 2], 0, 0.5), _Poly([0, 2, 3], 1, 0.5)]), ["WOODY", "STEELY"])


def _sidecar(tmp_path, block, name="s.thermal.json"):
    path = tmp_path / name
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


# ---------------------------------------------------------------------------
# materials.resolve_face_materials
# ---------------------------------------------------------------------------


def test_resolve_face_materials_exact_per_face(tmp_path):
    sa = _sidecar(tmp_path, {
        "WOODY": {"preset": "wood"},
        "STEELY": {"preset": "steel", "role": "DIRICHLET_SOURCE", "dirichlet_K": 350.0},
    })
    obj = _square()
    face_slots = np.array([0, 1], dtype=np.int32)  # face 0 -> WOODY, face 1 -> STEELY

    out = materials.resolve_face_materials(obj, sa, _DEFAULTS, face_slots)

    assert out is not None
    for key in ("t0", "alpha", "rho", "c", "eps"):
        assert out[key].shape == (2,)
    assert out["alpha"][0] == pytest.approx(materials.PRESETS["wood"].alpha_mm2_s)
    assert out["alpha"][1] == pytest.approx(materials.PRESETS["steel"].alpha_mm2_s)
    # No averaging: the two faces' values are exactly the two distinct presets, not a blend.
    assert out["alpha"][0] != pytest.approx(out["alpha"][1])
    assert not (materials.PRESETS["wood"].alpha_mm2_s < out["alpha"][0] < materials.PRESETS["steel"].alpha_mm2_s)

    # Dirichlet slot faces are flagged; the FEM slot's face is not.
    assert not out["dirichlet_mask"][0]
    assert out["dirichlet_mask"][1]
    assert out["t0"][1] == pytest.approx(350.0)


def test_resolve_face_materials_returns_none_without_slots(tmp_path):
    sa = _sidecar(tmp_path, {"WOODY": {"preset": "wood"}})
    obj = _square()
    obj.material_slots = []
    assert materials.resolve_face_materials(obj, sa, _DEFAULTS, np.array([0], dtype=np.int32)) is None


def test_resolve_face_materials_out_of_range_index_is_clamped(tmp_path):
    sa = _sidecar(tmp_path, {"WOODY": {"preset": "wood"}, "STEELY": {"preset": "steel"}})
    obj = _square()
    out = materials.resolve_face_materials(obj, sa, _DEFAULTS, np.array([7], dtype=np.int32))
    assert out is not None
    assert out["alpha"][0] == pytest.approx(materials.PRESETS["steel"].alpha_mm2_s)


# ---------------------------------------------------------------------------
# adapter._combine TEXEL mode
# ---------------------------------------------------------------------------


def _texel_table(k, face, face_material_index):
    rng = np.random.default_rng(0)
    return {
        "position_mm": rng.uniform(0.0, 100.0, size=(k, 3)),
        "normal": np.tile(np.array([0.0, 0.0, 1.0]), (k, 1)),
        "uv": np.zeros((k, 2), dtype=np.float64),
        "face": np.asarray(face, dtype=np.int64),
        "face_material_index": np.asarray(face_material_index, dtype=np.int32),
    }


def test_combine_texel_mode_mixes_texel_and_vertex_objects():
    k = 5
    sparse_obj = _NS(name="sparse", heat_sim_material=None)
    dense_obj = _square("dense")

    atlas_plan = SimpleNamespace(texels={
        "sparse": _texel_table(k, face=np.zeros(k, dtype=np.int64), face_material_index=[0]),
    })

    combined = adapter._combine([sparse_obj, dense_obj], {}, _DEFAULTS, _SOLVER_CFG, atlas_plan=atlas_plan)

    assert combined is not None
    n_dense = 4
    assert combined.verts.shape[0] == k + n_dense
    assert combined.alpha.shape[0] == k + n_dense
    assert combined.layout == [
        ("sparse", 0, k, "TEXEL"),
        ("dense", k, n_dense, "VERTEX"),
    ]
    # TEXEL object contributes no faces (POINTS-only); VERTEX object's faces are offset by k.
    assert combined.faces.shape[0] == 2  # only "dense"'s two triangles
    assert combined.faces.min() >= k


def test_combine_texel_mode_object_level_materials_when_no_assignment():
    """No sidecar: a TEXEL object still gets the object-level constant, broadcast per-texel."""
    k = 3
    obj = _NS(name="sparse", heat_sim_material=None)
    atlas_plan = SimpleNamespace(texels={
        "sparse": _texel_table(k, face=np.zeros(k, dtype=np.int64), face_material_index=[0]),
    })

    combined = adapter._combine([obj], {}, _DEFAULTS, _SOLVER_CFG, atlas_plan=atlas_plan)

    assert combined is not None
    assert np.allclose(combined.alpha, _DEFAULTS["thermal_diffusivity_mm2_s"])
    assert np.allclose(combined.eps, _DEFAULTS["emissivity"])
    assert np.allclose(combined.t0, _DEFAULTS["initial_temperature_K"])
    assert np.all(combined.boundary_mask)


def test_combine_texel_dirichlet_pinning_per_texel(tmp_path):
    sa = _sidecar(tmp_path, {
        "WOODY": {"preset": "wood"},
        "STEELY": {"preset": "steel", "role": "DIRICHLET_SOURCE", "dirichlet_K": 350.0},
    })
    obj = _square()  # name "square", slots WOODY (face 0), STEELY (face 1)
    tex = _texel_table(4, face=[0, 0, 1, 1], face_material_index=[0, 1])
    atlas_plan = SimpleNamespace(texels={"square": tex})
    flux_by_obj = {obj: np.array([10.0, 10.0, 10.0, 10.0])}

    combined = adapter._combine([obj], flux_by_obj, _DEFAULTS, _SOLVER_CFG, assignment=sa, atlas_plan=atlas_plan)

    assert combined is not None
    pinned = ~combined.boundary_mask
    assert pinned.tolist() == [False, False, True, True]
    assert np.allclose(combined.alpha[pinned], 0.0)
    assert np.allclose(combined.irradiance[pinned], 0.0)
    assert np.all(combined.irradiance[~pinned] > 0.0)
    assert np.allclose(combined.t0[pinned], 350.0)
    assert np.allclose(combined.t0[~pinned], _DEFAULTS["initial_temperature_K"])


def test_combine_vertex_mode_unchanged_when_no_plan():
    """Backward-compat gate: atlas_plan=None (or omitted) reproduces today's behaviour."""
    obj = _square()
    combined_default = adapter._combine([obj], {}, _DEFAULTS, _SOLVER_CFG)
    combined_explicit = adapter._combine([obj], {}, _DEFAULTS, _SOLVER_CFG, atlas_plan=None)

    for combined in (combined_default, combined_explicit):
        assert combined is not None
        assert np.allclose(combined.alpha, _DEFAULTS["thermal_diffusivity_mm2_s"])
        assert np.allclose(combined.eps, _DEFAULTS["emissivity"])
        assert np.allclose(combined.t0, _DEFAULTS["initial_temperature_K"])
        assert np.all(combined.boundary_mask)
        assert combined.layout == [("square", 0, 4, "VERTEX")]

    assert np.array_equal(combined_default.alpha, combined_explicit.alpha)
    assert np.array_equal(combined_default.verts, combined_explicit.verts)
    assert np.array_equal(combined_default.faces, combined_explicit.faces)


# ---------------------------------------------------------------------------
# adapter.build_atlas_plan
# ---------------------------------------------------------------------------


class _AtlasMesh:
    def __init__(self, n_verts):
        self.vertices = [object()] * n_verts


class _AtlasObj:
    def __init__(self, name, n_verts):
        self.name = name
        self.data = _AtlasMesh(n_verts)
        self.heat_sim_material = None


def _big_plane_geom(side_mm=10_000.0):
    """4 verts spanning a 10m x 10m plane => 100 m^2 for ~0.04 verts/m^2 (well under any
    reasonable atlas_texel_density, so select_for_atlas admits it)."""
    verts = np.array([
        [0.0, 0.0, 0.0], [side_mm, 0.0, 0.0], [side_mm, side_mm, 0.0], [0.0, side_mm, 0.0],
    ])
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return verts, faces, 4


_ATLAS_CFG = {"atlas_texel_density": 50.0, "atlas_tile_min": 16, "atlas_tile_max": 512, "atlas_texel_soft_max": 500_000}


def test_uv_failure_demotes_to_vertex_path_with_warning(monkeypatch, caplog):
    good = _AtlasObj("good", 4)
    bad = _AtlasObj("bad", 4)
    geoms = {"good": _big_plane_geom(), "bad": _big_plane_geom()}

    monkeypatch.setattr(adapter, "_extract_geometry", lambda obj: geoms[obj.name])
    monkeypatch.setattr(adapter, "_prepare_bake_uv", lambda obj: None)

    captured = {}

    def spy_write(o, tile, atlas_size, src_layer_name):
        captured[o.name] = (tile, atlas_size)

    monkeypatch.setattr(adapter, "_write_atlas_uv_layer", spy_write)

    raw_uv = np.array([
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    ])

    def fake_uv(obj, layer_name):
        if obj.name == "bad":
            return None  # simulates an unwrap that never produced a usable UV layer
        # Mirror the real _write_atlas_uv_layer -> _extract_evaluated_face_uv_and_slots
        # round trip (as test_build_atlas_plan_vertex_count_mismatch_no_longer_demotes
        # does): apply the forward tile-local -> atlas-global remap here, so
        # build_atlas_plan's inverse remap gets back exactly `raw_uv` instead of
        # double-remapping already tile-local coordinates out of [0, 1].
        tile, atlas_size = captured[obj.name]
        aw, ah = atlas_size
        tw, th = tile.size
        tx, ty = tile.offset
        atlas_uv = np.empty_like(raw_uv)
        atlas_uv[..., 0] = (tx + raw_uv[..., 0] * tw) / aw
        atlas_uv[..., 1] = (ty + raw_uv[..., 1] * th) / ah
        face_material_index = np.array([0, 0], dtype=np.int32)
        return atlas_uv, face_material_index

    monkeypatch.setattr(adapter, "_extract_evaluated_face_uv_and_slots", fake_uv)

    with caplog.at_level(logging.WARNING):
        plan = adapter.build_atlas_plan(scene=None, sim_objects=[good, bad], cfg=_ATLAS_CFG)

    assert "good" in plan.texels
    assert "bad" not in plan.texels
    assert plan.texels["good"]["position_mm"].shape[0] > 0

    messages = [rec.getMessage() for rec in caplog.records]
    assert any("bad" in m and "demoted" in m for m in messages)


def test_build_atlas_plan_vertex_count_mismatch_no_longer_demotes(monkeypatch):
    """Regression for the evaluated-mesh rasterization fix: a base/evaluated vertex-count
    mismatch (the hallmark of a topology-changing modifier, e.g. Bevel or Geometry Nodes)
    must NOT demote the object anymore. A temperature atlas is UV-addressed, not
    vertex-addressed: :func:`build_atlas_plan` reads both geometry and UVs from the
    evaluated mesh, so it no longer cares that the base mesh disagrees on vertex count.

    Simulates the write-then-read-back round trip _write_atlas_uv_layer/
    _extract_evaluated_face_uv_and_slots perform: a spy captures the (tile, atlas_size)
    the real write call would have used, and the fake evaluated-UV reader applies the
    exact same forward remap _write_atlas_uv_layer does, so build_atlas_plan's inverse
    remap round-trips back to the original tile-local UV.
    """
    obj = _AtlasObj("mod_obj", n_verts=6)  # base mesh reports 6 verts
    evaluated = _big_plane_geom()  # evaluated geometry has 4 verts

    monkeypatch.setattr(adapter, "_extract_geometry", lambda o: evaluated)
    prep_calls = []
    monkeypatch.setattr(adapter, "_prepare_bake_uv", lambda o: prep_calls.append(o.name))

    captured = {}

    def spy_write(o, tile, atlas_size, src_layer_name):
        captured["tile"] = tile
        captured["atlas_size"] = atlas_size

    monkeypatch.setattr(adapter, "_write_atlas_uv_layer", spy_write)

    raw_uv = np.array([
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    ])

    def fake_evaluated_uv(o, layer_name):
        tile = captured["tile"]
        aw, ah = captured["atlas_size"]
        tw, th = tile.size
        tx, ty = tile.offset
        atlas_uv = np.empty_like(raw_uv)
        atlas_uv[..., 0] = (tx + raw_uv[..., 0] * tw) / aw
        atlas_uv[..., 1] = (ty + raw_uv[..., 1] * th) / ah
        face_material_index = np.array([0, 0], dtype=np.int32)
        return atlas_uv, face_material_index

    monkeypatch.setattr(adapter, "_extract_evaluated_face_uv_and_slots", fake_evaluated_uv)

    plan = adapter.build_atlas_plan(scene=None, sim_objects=[obj], cfg=_ATLAS_CFG)

    assert "mod_obj" in plan.texels
    assert prep_calls == ["mod_obj"]  # bake UV prep still runs before the atlas UV write
    assert plan.texels["mod_obj"]["position_mm"].shape[0] > 0


def test_build_atlas_plan_excludes_dense_objects(monkeypatch):
    """An object whose native vertex density already exceeds the target keeps the vertex path."""
    dense = _AtlasObj("dense", n_verts=50_000)
    dense_geom = (np.random.default_rng(0).uniform(0, 10, size=(50_000, 3)),
                  np.array([[0, 1, 2]], dtype=np.int32), 50_000)

    monkeypatch.setattr(adapter, "_extract_geometry", lambda o: dense_geom)

    plan = adapter.build_atlas_plan(scene=None, sim_objects=[dense], cfg=_ATLAS_CFG)
    assert plan.texels == {}


def _build_two_object_plan(monkeypatch, *, drop_second: bool):
    """Shared setup for the digest-participation tests below: two same-sized objects on
    the same allocation (so ``layout.tiles`` is identical either way); ``drop_second``
    controls whether "obj_b" makes it through rasterization (mirrors
    ``test_uv_failure_demotes_to_vertex_path_with_warning``'s UV-extraction-failure
    setup) or fully participates."""
    obj_a = _AtlasObj("obj_a", 4)
    obj_b = _AtlasObj("obj_b", 4)
    geoms = {"obj_a": _big_plane_geom(), "obj_b": _big_plane_geom()}

    monkeypatch.setattr(adapter, "_extract_geometry", lambda obj: geoms[obj.name])
    monkeypatch.setattr(adapter, "_prepare_bake_uv", lambda obj: None)

    captured = {}

    def spy_write(o, tile, atlas_size, src_layer_name):
        captured[o.name] = (tile, atlas_size)

    monkeypatch.setattr(adapter, "_write_atlas_uv_layer", spy_write)

    raw_uv = np.array([
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    ])

    def fake_uv(obj, layer_name):
        if drop_second and obj.name == "obj_b":
            return None  # simulates an unwrap that never produced a usable UV layer
        tile, atlas_size = captured[obj.name]
        aw, ah = atlas_size
        tw, th = tile.size
        tx, ty = tile.offset
        atlas_uv = np.empty_like(raw_uv)
        atlas_uv[..., 0] = (tx + raw_uv[..., 0] * tw) / aw
        atlas_uv[..., 1] = (ty + raw_uv[..., 1] * th) / ah
        face_material_index = np.array([0, 0], dtype=np.int32)
        return atlas_uv, face_material_index

    monkeypatch.setattr(adapter, "_extract_evaluated_face_uv_and_slots", fake_uv)

    return adapter.build_atlas_plan(scene=None, sim_objects=[obj_a, obj_b], cfg=_ATLAS_CFG)


def test_atlas_digest_reflects_realized_texel_participation(monkeypatch):
    """Finding 1 regression: `atlas.allocate` assigns tiles for both "obj_a" and "obj_b"
    before rasterization runs, so the tile layout alone (name/size/offset) is identical
    whether "obj_b" ends up contributing texels or gets dropped by a UV failure. The
    digest must still differ, because a materially different simulation (one fewer
    object's temperature actually solved into the atlas) must not silently reuse a
    cached solve keyed on the allocation alone."""
    plan_both = _build_two_object_plan(monkeypatch, drop_second=False)
    plan_dropped = _build_two_object_plan(monkeypatch, drop_second=True)

    assert "obj_a" in plan_both.texels and "obj_b" in plan_both.texels
    assert "obj_a" in plan_dropped.texels and "obj_b" not in plan_dropped.texels
    # Same objects, same density/config => same tile allocation either way.
    assert plan_both.layout.tiles.keys() == plan_dropped.layout.tiles.keys()

    assert plan_both.digest != plan_dropped.digest


def test_atlas_digest_stable_across_identical_builds(monkeypatch):
    """The digest must be deterministic: two builds over the same inputs produce the
    same digest, regardless of dict/set iteration order."""
    plan1 = _build_two_object_plan(monkeypatch, drop_second=False)
    plan2 = _build_two_object_plan(monkeypatch, drop_second=False)

    assert plan1.digest == plan2.digest


# ---------------------------------------------------------------------------
# solve_scene cache key
# ---------------------------------------------------------------------------


class _FakeScene:
    def __init__(self):
        self.objects = []


def test_cache_key_gains_atlas_digest(tmp_path, monkeypatch):
    """Finding 1 regression: with ``atlas_plan=None`` the ``"atlas"`` key must be
    entirely ABSENT from key_cfg -- not present with value ``None`` -- so the JSON
    (and therefore the SHA1 cache key) is byte-identical to before the atlas feature
    existed, per solve_scene's docstring guarantee ("reproduces today's behaviour
    exactly, including the cache key"). Only once an AtlasPlan is actually supplied
    does the key gain the layout-sensitive ``"atlas"`` entry."""
    captured = []
    from visionsim.simulate.heatsim import cache as cache_mod

    real_cache_key = cache_mod.cache_key

    def spy(blend_path, key_cfg):
        captured.append(key_cfg)
        return real_cache_key(blend_path, key_cfg)

    monkeypatch.setattr(adapter, "gather_meshes", lambda scene: [])
    monkeypatch.setattr(adapter.cache, "cache_key", spy)

    scene = _FakeScene()
    adapter.solve_scene(scene, defaults=_DEFAULTS, solver_cfg=_SOLVER_CFG, cache_root=tmp_path)

    fake_plan = SimpleNamespace(
        texels={}, density=50.0, tile_min=16, tile_max=512, soft_max=500_000, digest="deadbeef",
    )
    adapter.solve_scene(
        scene, defaults=_DEFAULTS, solver_cfg=_SOLVER_CFG, cache_root=tmp_path, atlas_plan=fake_plan,
    )

    assert len(captured) == 2
    assert "atlas" not in captured[0]
    assert captured[1]["atlas"] == {
        "density": 50.0, "tile_min": 16, "tile_max": 512, "soft_max": 500_000, "layout_digest": "deadbeef",
    }


def test_cache_key_is_byte_identical_to_pre_atlas_baseline(tmp_path, monkeypatch):
    """Stronger form of the above: reconstruct the pre-atlas key_cfg by hand (no
    "atlas" field at all) and assert solve_scene's atlas_plan=None key_cfg matches it
    exactly, so an existing .heatsim cache from before the atlas feature is not busted."""
    captured = []
    monkeypatch.setattr(adapter, "gather_meshes", lambda scene: [])
    monkeypatch.setattr(adapter.cache, "cache_key", lambda blend_path, key_cfg: captured.append(key_cfg) or "k")

    scene = _FakeScene()
    adapter.solve_scene(scene, defaults=_DEFAULTS, solver_cfg=_SOLVER_CFG, cache_root=tmp_path)

    expected_pre_atlas_key_cfg = {
        "solver": dict(_SOLVER_CFG),
        "defaults": dict(_DEFAULTS),
        "objects": [],
        "assignments": None,
    }
    assert captured[0] == expected_pre_atlas_key_cfg


# ---------------------------------------------------------------------------
# solve_scene: atlas objects skip the per-vertex irradiance pass (Finding 2)
# ---------------------------------------------------------------------------


def test_solve_scene_only_computes_vertex_irradiance_for_non_atlas_objects(tmp_path, monkeypatch):
    """_compute_irradiance must not do the expensive per-vertex albedo-bake + shadow-ray
    work for objects the atlas plan is going to solve at texel resolution and whose
    per-vertex result would be discarded by _combine's TEXEL branch."""
    atlas_obj = _NS(name="atlas_obj", heat_sim_material=None)
    vertex_obj = _NS(name="vertex_obj", heat_sim_material=None)

    monkeypatch.setattr(adapter, "gather_meshes", lambda scene: [atlas_obj, vertex_obj])

    vertex_call_objects: list = []

    def fake_compute_irradiance(scene, sim_objects, solver_cfg, defaults):
        vertex_call_objects.extend(sim_objects)
        return {}

    texel_calls: list = []

    def fake_compute_texel_irradiance(scene, sim_objects, atlas_plan, solver_cfg, defaults):
        texel_calls.append(list(sim_objects))
        return {}

    monkeypatch.setattr(adapter, "_compute_irradiance", fake_compute_irradiance)
    monkeypatch.setattr(adapter, "_compute_texel_irradiance", fake_compute_texel_irradiance)
    monkeypatch.setattr(adapter, "_combine", lambda *a, **kw: None)

    atlas_plan = SimpleNamespace(
        texels={"atlas_obj": _texel_table(3, face=np.zeros(3, dtype=np.int64), face_material_index=[0])},
        density=50.0, tile_min=16, tile_max=512, soft_max=500_000, digest="abc123",
    )

    scene = _FakeScene()
    adapter.solve_scene(
        scene, defaults=_DEFAULTS, solver_cfg=_SOLVER_CFG, cache_root=tmp_path, atlas_plan=atlas_plan,
    )

    # Only the object NOT covered by the atlas plan reaches the per-vertex kernel...
    assert vertex_call_objects == [vertex_obj]
    # ...while the texel path still sees every sim object (it filters by atlas_plan.texels
    # itself, and still needs to trigger the albedo bake for its own participants).
    assert texel_calls == [[atlas_obj, vertex_obj]]


def test_solve_scene_runs_vertex_irradiance_for_all_objects_without_an_atlas_plan(tmp_path, monkeypatch):
    """Backward-compat gate: atlas_plan=None must not filter anything out."""
    obj_a = _NS(name="a", heat_sim_material=None)
    obj_b = _NS(name="b", heat_sim_material=None)
    monkeypatch.setattr(adapter, "gather_meshes", lambda scene: [obj_a, obj_b])

    vertex_call_objects: list = []
    monkeypatch.setattr(
        adapter, "_compute_irradiance",
        lambda scene, sim_objects, solver_cfg, defaults: (vertex_call_objects.extend(sim_objects) or {}),
    )
    monkeypatch.setattr(adapter, "_combine", lambda *a, **kw: None)

    scene = _FakeScene()
    adapter.solve_scene(scene, defaults=_DEFAULTS, solver_cfg=_SOLVER_CFG, cache_root=tmp_path)

    assert vertex_call_objects == [obj_a, obj_b]


# ---------------------------------------------------------------------------
# _compute_texel_irradiance: shared BVH build + shadow-ray count (Findings 3 & 4)
# ---------------------------------------------------------------------------


def test_compute_texel_irradiance_builds_bvh_once_and_threads_shadow_ray_count(executable):
    """_compute_texel_irradiance must build the scene BVH backend exactly once per call
    (not once per atlas-participating object -- irradiance_kernel.compute_irradiance_at_points
    builds a full scene BVH internally whenever no backend is supplied) and must thread
    solver_cfg['direct_kernel_soft_shadow_rays'] through to the kernel instead of relying on
    its hardcoded default of 8. Needs real bpy (irradiance_kernel imports it unconditionally),
    so this spies via direct module-attribute patching inside a Blender subprocess -- the same
    pattern test_heatsim_irradiance.py's under-bpy tests use.
    """
    code = r"""
import bpy, numpy as np
from visionsim.simulate.heatsim import register, adapter, bvh_backend, irradiance_kernel

register()
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()


def make_plane(name, x):
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(x, 0.0, 0.0))
    obj = bpy.context.active_object
    obj.name = name
    obj.heat_simulation_enabled = True
    mat = bpy.data.materials.new(name + '_mat')
    mat.use_nodes = True
    obj.data.materials.append(mat)
    return obj


make_plane('A', 0.0)
make_plane('B', 5.0)

bpy.ops.object.light_add(type='SUN')
bpy.context.active_object.data.energy = 10.0
world = bpy.context.scene.world
world.use_nodes = True
bg = world.node_tree.nodes.get('Background')
bg.inputs['Strength'].default_value = 1.0

atlas_cfg = dict(atlas_texel_density=64.0, atlas_tile_min=16, atlas_tile_max=64, atlas_texel_soft_max=500_000)
sim_objects = adapter.gather_meshes(bpy.context.scene)
plan = adapter.build_atlas_plan(bpy.context.scene, sim_objects, atlas_cfg)
assert set(plan.texels) == {'A', 'B'}, plan.texels.keys()

build_calls = []
best_available_calls = []
orig_best_available = bvh_backend.best_available


def counting_best_available():
    best_available_calls.append(1)
    backend = orig_best_available()
    orig_build = backend.build_for_meshes

    def counted(meshes):
        build_calls.append(1)
        return orig_build(meshes)
    backend.build_for_meshes = counted
    return backend


bvh_backend.best_available = counting_best_available

captured_n_samples = []
orig_compute_at_points = irradiance_kernel.compute_irradiance_at_points


def spy_compute_at_points(scene, positions, normals, albedo, **kwargs):
    captured_n_samples.append(kwargs.get('n_samples_for_area'))
    return orig_compute_at_points(scene, positions, normals, albedo, **kwargs)


irradiance_kernel.compute_irradiance_at_points = spy_compute_at_points

defaults = dict(initial_temperature_K=295.0, thermal_diffusivity_mm2_s=0.17,
                density_kg_m3=1330.0, specific_heat_J_kgK=880.0, emissivity=0.9,
                irradiance_scale=100.0)

# 1. A custom shadow-ray count in solver_cfg must reach the kernel, and the BVH must be
#    built exactly once even though there are 2 atlas-participating objects.
solver_cfg_custom = dict(sim_time_s=0.1, timestep_s=0.05, domain='POINTS',
                          laplacian_backend='ROBUST', device='cpu',
                          direct_kernel_soft_shadow_rays=3)
flux = adapter._compute_texel_irradiance(bpy.context.scene, sim_objects, plan, solver_cfg_custom, defaults)
assert set(o.name for o in flux) == {'A', 'B'}, flux.keys()
assert len(best_available_calls) == 1, best_available_calls
assert len(build_calls) == 1, build_calls
assert len(captured_n_samples) == 2, captured_n_samples
assert all(n == 3 for n in captured_n_samples), captured_n_samples

# 2. Omitting the knob from solver_cfg must fall back to the kernel's original hardcoded
#    default (8), not silently pass 0/None through.
build_calls.clear(); best_available_calls.clear(); captured_n_samples.clear()
solver_cfg_default = dict(sim_time_s=0.1, timestep_s=0.05, domain='POINTS',
                           laplacian_backend='ROBUST', device='cpu')
adapter._compute_texel_irradiance(bpy.context.scene, sim_objects, plan, solver_cfg_default, defaults)
assert len(best_available_calls) == 1, best_available_calls
assert len(build_calls) == 1, build_calls
assert all(n == 8 for n in captured_n_samples), captured_n_samples

print('TEXEL_BVH_ONCE_AND_SHADOW_RAYS_OK')
"""
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True, check=False)
    assert "TEXEL_BVH_ONCE_AND_SHADOW_RAYS_OK" in out.stdout, out.stdout + "\n" + out.stderr


# ---------------------------------------------------------------------------
# End-to-end smoke test (real bpy): build_atlas_plan -> solve_scene(atlas_plan=...)
# -> write_frame_attributes, on a genuine low-vertex-density plane. Not one of the
# brief's named fake-bpy tests, but the fake fixtures above cannot catch a real bpy
# API mismatch (foreach_get signatures, UV layer creation, ...), so this closes that
# gap the same way test_heatsim_irradiance.py's executable tests do for the bake path.
# ---------------------------------------------------------------------------


def test_texel_pipeline_solves_end_to_end_under_bpy(executable, tmp_path):
    code = f"""
import bpy, numpy as np
from pathlib import Path
from visionsim.simulate.heatsim import register, adapter
from visionsim.simulate.heatsim.constants import ATLAS_UV_LAYER_NAME

register()
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# A coarse (4-vertex) plane spanning 4 m^2: at any reasonable density this is
# well under the atlas_texel_density target, so it must join the atlas.
bpy.ops.mesh.primitive_plane_add(size=2.0)
plane = bpy.context.active_object
plane.name = 'CoarsePlane'
plane.heat_simulation_enabled = True

mat = bpy.data.materials.new('plane_mat')
mat.use_nodes = True
plane.data.materials.append(mat)

bpy.ops.object.light_add(type='SUN')
bpy.context.active_object.data.energy = 10.0
world = bpy.context.scene.world
world.use_nodes = True
bg = world.node_tree.nodes.get('Background')
bg.inputs['Strength'].default_value = 1.0

defaults = dict(initial_temperature_K=295.0, thermal_diffusivity_mm2_s=0.17,
                density_kg_m3=1330.0, specific_heat_J_kgK=880.0, emissivity=0.9,
                irradiance_scale=100.0)
solver_cfg = dict(sim_time_s=0.1, timestep_s=0.05, domain='POINTS',
                  laplacian_backend='ROBUST', device='cpu')
atlas_cfg = dict(atlas_texel_density=64.0, atlas_tile_min=16, atlas_tile_max=64,
                  atlas_texel_soft_max=500_000)

sim_objects = adapter.gather_meshes(bpy.context.scene)
plan = adapter.build_atlas_plan(bpy.context.scene, sim_objects, atlas_cfg)
assert 'CoarsePlane' in plan.texels, 'coarse plane should have joined the atlas'
n_texels = plan.texels['CoarsePlane']['position_mm'].shape[0]
assert n_texels > len(plane.data.vertices), (n_texels, len(plane.data.vertices))

# HeatSim_Atlas_UV must have been written for the render-time shader lookup.
assert ATLAS_UV_LAYER_NAME in plane.data.uv_layers

hist = adapter.solve_scene(bpy.context.scene, defaults=defaults, solver_cfg=solver_cfg,
                           cache_root=Path(r'{tmp_path}'), atlas_plan=plan)
assert 'CoarsePlane' in hist, list(hist.keys())
T_hist = np.asarray(hist['CoarsePlane'])
assert T_hist.ndim == 2 and T_hist.shape[1] == n_texels, T_hist.shape
assert np.isfinite(T_hist).all()
assert T_hist.min() > 200 and T_hist.max() < 2000, (float(T_hist.min()), float(T_hist.max()))

print('TEXEL_PIPELINE_OK', n_texels, len(plane.data.vertices))
"""
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True, check=False)
    assert "TEXEL_PIPELINE_OK" in out.stdout, out.stdout + "\n" + out.stderr


# ---------------------------------------------------------------------------
# Real-bpy regression: topology-changing modifiers no longer demote an object
# from the atlas (evaluated-mesh rasterization fix).
# ---------------------------------------------------------------------------


def test_build_atlas_plan_promotes_object_with_topology_changing_modifier(executable):
    """Before the fix, an object whose evaluated (post-modifier) vertex count
    disagreed with its base mesh's was unconditionally demoted from the atlas -
    and, since the per-vertex write-back path also can't handle a shape mismatch,
    its solved temperatures were silently discarded (flat ambient at render time).

    A Subdivision Surface modifier reliably changes a plane's vertex count
    headless (4 base verts -> many more evaluated verts), so it stands in for the
    Bevel/Geometry-Nodes/EdgeSplit modifiers the design doc calls out. Asserts the
    object IS an atlas participant with a nonzero texel count, and that every
    rasterized texel position lies within the object's evaluated world-space
    bounding box - a coordinate-convention regression (e.g. mixing up tile-local
    and atlas-global UV space) would place texels far outside this box.
    """
    code = r"""
import bpy, numpy as np
from visionsim.simulate.heatsim import register, adapter
from visionsim.simulate.heatsim.constants import ATLAS_UV_LAYER_NAME

register()
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

bpy.ops.mesh.primitive_plane_add(size=2.0)
plane = bpy.context.active_object
plane.name = 'ModPlane'
plane.heat_simulation_enabled = True

mat = bpy.data.materials.new('plane_mat')
mat.use_nodes = True
plane.data.materials.append(mat)

subsurf = plane.modifiers.new(name='Subsurf', type='SUBSURF')
subsurf.levels = 2
subsurf.render_levels = 2

bpy.ops.object.light_add(type='SUN')
bpy.context.active_object.data.energy = 10.0
world = bpy.context.scene.world
world.use_nodes = True
bg = world.node_tree.nodes.get('Background')
bg.inputs['Strength'].default_value = 1.0

base_n_verts = len(plane.data.vertices)

depsgraph = bpy.context.evaluated_depsgraph_get()
eval_mesh = plane.evaluated_get(depsgraph).data
eval_n_verts = len(eval_mesh.vertices)
# Sanity check on the test setup itself: the modifier must actually change the
# vertex count, or this test would not exercise the bug at all.
assert eval_n_verts != base_n_verts, (base_n_verts, eval_n_verts)

verts_local = np.array([tuple(v.co) for v in eval_mesh.vertices], dtype=np.float64)
mw = np.array(plane.matrix_world, dtype=np.float64)
verts_world_mm = ((verts_local @ mw[:3, :3].T) + mw[:3, 3]) * 1000.0
bbox_lo = verts_world_mm.min(axis=0) - 1.0e-3
bbox_hi = verts_world_mm.max(axis=0) + 1.0e-3

atlas_cfg = dict(atlas_texel_density=64.0, atlas_tile_min=16, atlas_tile_max=64,
                  atlas_texel_soft_max=500_000)
sim_objects = adapter.gather_meshes(bpy.context.scene)
plan = adapter.build_atlas_plan(bpy.context.scene, sim_objects, atlas_cfg)

assert 'ModPlane' in plan.texels, 'topology-changing-modifier object was demoted from the atlas'
pos_mm = plan.texels['ModPlane']['position_mm']
n_texels = pos_mm.shape[0]
assert n_texels > 0, n_texels

assert np.all(pos_mm >= bbox_lo) and np.all(pos_mm <= bbox_hi), (
    pos_mm.min(axis=0), pos_mm.max(axis=0), bbox_lo, bbox_hi,
)

# The atlas UV layer must have landed on the BASE mesh (so both the modifier
# stack and the render-time shader can see it), not just the evaluated copy.
assert ATLAS_UV_LAYER_NAME in plane.data.uv_layers

print('TOPOLOGY_MODIFIER_ATLAS_PROMOTED_OK', n_texels, base_n_verts, eval_n_verts)
"""
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True, check=False)
    assert "TOPOLOGY_MODIFIER_ATLAS_PROMOTED_OK" in out.stdout, out.stdout + "\n" + out.stderr
