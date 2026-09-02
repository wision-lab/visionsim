"""Tests for Task 3: the atlas EXR writer (``adapter.write_atlas``) and the render-time
plumbing that loads/packs it. Both node-graph shader tests and the
``render_domain="TEXEL"`` config-dispatch parity tests live in sibling files
(``tests/test_heatsim_shader.py``, ``tests/test_heatsim_config.py``); this file covers
the writer itself and the ``write_frame_attributes``/``global_temperature_range`` TEXEL
behavior specified in the Task 3 brief.
"""

from __future__ import annotations

import subprocess

import Imath
import numpy as np
import OpenEXR
import pytest

from visionsim.simulate.heatsim import adapter, atlas


def _tile_layout(atlas_size, tiles):
    return atlas.AtlasLayout(atlas_size=atlas_size, tiles=tiles, effective_density=500.0, rescaled=False)


def _plan(atlas_size, tiles, texels, digest="testdigest"):
    return adapter.AtlasPlan(layout=_tile_layout(atlas_size, tiles), texels=texels, digest=digest)


# ---------------------------------------------------------------------------
# write_atlas: needs bpy (image creation/save), run inside Blender.
# ---------------------------------------------------------------------------


def test_write_atlas_scatters_dilates_and_marks_alpha(executable, tmp_path):
    code = f"""
import numpy as np
from pathlib import Path
from visionsim.simulate.heatsim import adapter, atlas

# One object, one 6x6 tile at atlas offset (0,0). Two solved texels at (1,1) and (4,4)
# (far apart so their dilation margins don't overlap and mask each other's zeros).
# Size the tile from the dilation count so an unwritten region provably survives: the
# margin grows by one texel per pass, so a corner further than that from every solved
# texel must stay alpha=0. Hardcoding a 6x6 tile silently stopped testing anything when
# _ATLAS_DILATE_ITERATIONS was raised from 1 to 8 -- the margin then covered the whole tile.
_D = adapter._ATLAS_DILATE_ITERATIONS
_N = 2 * _D + 6          # room for two solved texels and an untouched corner
tile = atlas.TileSpec(obj_name='obj', size=(_N, _N), offset=(0, 0))
layout = atlas.AtlasLayout(atlas_size=(_N, _N), tiles={{'obj': tile}}, effective_density=500.0, rescaled=False)
texels = {{'obj': {{'xy': np.array([[1, 1], [_N - 2, _N - 2]], dtype=np.int64)}}}}
plan = adapter.AtlasPlan(layout=layout, texels=texels, digest='rt')

history = {{'obj': np.array([[300.0, 300.0], [310.0, 320.0]])}}  # (T=2, K=2); final row used

out_path = adapter.write_atlas(history, plan, Path(r'{tmp_path}'))
assert out_path.exists(), out_path
assert out_path.name == 'atlas_temperature.exr'
assert 'rt' in str(out_path.parent.name)

import bpy
img = bpy.data.images.load(str(out_path))
w, h = img.size
assert (w, h) == (_N, _N), (w, h)
px = np.array(img.pixels[:], dtype=np.float64).reshape(h, w, 4)

# Scattered texels: match (within EXR write/load round-trip precision -- Blender's
# Image.save() has no exposed lossless-compression knob for a plain generated
# image, so a few mK of drift is expected and harmless for a Kelvin-scale field)
# at (x=1,y=1) -> 310.0 and (x=4,y=4) -> 320.0, alpha=1.
assert abs(px[1, 1, 0] - 310.0) < 0.05, px[1, 1]
assert abs(px[_N - 2, _N - 2, 0] - 320.0) < 0.05, px[_N - 2, _N - 2]
assert px[1, 1, 3] == 1.0
assert px[_N - 2, _N - 2, 3] == 1.0

# A direct 8-neighbor of a solved texel is inside the dilation margin: alpha=1 (valid
# coverage per the dilated mask) and a nonzero temperature pulled from that neighbor
# (never the raw zero an un-dilated scatter would leave).
assert px[1, 2, 3] == 1.0, px[1, 2]
assert px[1, 2, 0] > 0.0, px[1, 2]

# Far corner, well outside the (small, capped) dilation margin from either solved texel:
# still unwritten -> alpha=0, temperature left at the initialized zero.
unwritten = np.argwhere(px[:, :, 3] == 0.0)
assert unwritten.size > 0, 'dilation covered the whole tile; margin is not bounded'
_r, _c = unwritten[0]
assert px[_r, _c, 3] == 0.0, px[_r, _c]
assert px[_r, _c, 0] == 0.0, ('unwritten texel carries a temperature', px[_r, _c])

print('WRITE_ATLAS_OK')
"""
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True, check=False)
    assert "WRITE_ATLAS_OK" in out.stdout, out.stdout + "\n" + out.stderr

    # Absolute-value check: read the EXR with the standalone OpenEXR package (never bpy --
    # bpy's load applies the same colorspace-tagged decode as the write's encode, so a
    # write->bpy-load round trip only proves symmetry, not that the file holds the true
    # Kelvin values). This is the assertion that catches a Non-Color tag regression: an
    # untagged write would land here as ~11.2 (sRGB-OETF-encoded 310.0), not 310.0.
    exr_path = tmp_path / "atlas_rt" / "atlas_temperature.exr"
    assert exr_path.exists(), exr_path
    exr = OpenEXR.InputFile(str(exr_path))
    dw = exr.header()["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    float_t = Imath.PixelType(Imath.PixelType.FLOAT)
    r_raw = np.frombuffer(exr.channel("R", float_t), dtype=np.float32).reshape(h, w)
    # OpenEXR rows are top-down while Blender's Image.pixels buffer (and the (x, y) texel
    # coordinates used above) are bottom-up, so the on-disk row is (h - 1 - y).
    # Same geometry the Blender-side half used: tile side derived from the dilation count,
    # with the second solved texel at (_N - 2, _N - 2).
    _n = 2 * adapter._ATLAS_DILATE_ITERATIONS + 6
    assert (w, h) == (_n, _n), (w, h)
    assert abs(float(r_raw[h - 1 - 1, 1]) - 310.0) < 1e-2, r_raw[h - 1 - 1, 1]
    assert abs(float(r_raw[h - 1 - (_n - 2), _n - 2]) - 320.0) < 1e-2, r_raw[h - 1 - (_n - 2), _n - 2]


def test_write_atlas_no_texels_writes_empty_placeholder(executable, tmp_path):
    """No atlas objects this solve (render_domain=TEXEL requested but nothing qualified,
    or atlas_plan built from an empty scene) -> write_atlas must still return a stable,
    loadable, all-invalid path rather than raising."""
    code = f"""
from pathlib import Path
from visionsim.simulate.heatsim import adapter, atlas

layout = atlas.AtlasLayout(atlas_size=(0, 0), tiles={{}}, effective_density=500.0, rescaled=False)
plan = adapter.AtlasPlan(layout=layout, texels={{}}, digest='empty')

out_path = adapter.write_atlas({{}}, plan, Path(r'{tmp_path}'))
assert out_path.exists(), out_path

import bpy
img = bpy.data.images.load(str(out_path))
import numpy as np
px = np.array(img.pixels[:], dtype=np.float64)
assert np.all(px[3::4] == 0.0)  # every alpha channel is 0 (nothing valid)

print('WRITE_ATLAS_EMPTY_OK')
"""
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True, check=False)
    assert "WRITE_ATLAS_EMPTY_OK" in out.stdout, out.stdout + "\n" + out.stderr


# ---------------------------------------------------------------------------
# F3: dilation must not bridge the inter-tile packing padding (pure numpy, no bpy).
# ---------------------------------------------------------------------------


def test_scatter_atlas_arrays_dilation_does_not_bridge_inter_tile_padding():
    """Two adjacent tiles, solved at the edges nearest each other with different
    temperatures. The dilation margin from each tile must not reach far enough to pull
    its neighbour's temperature into the other tile's region (or the padding gap right
    next to it) -- this is only true while `_ATLAS_DILATE_ITERATIONS` (grows the valid
    region by 1 texel/pass) stays strictly less than the packing `_ATLAS_PACKING_PADDING`
    gap between tiles."""
    pad = adapter._ATLAS_PACKING_PADDING
    # Tiles must be wider than the dilation margin, or the margin swallows the tile and
    # the test stops distinguishing "did not bridge" from "filled everything".
    side = 2 * adapter._ATLAS_DILATE_ITERATIONS + 2
    tile_a = atlas.TileSpec("tile_a", (side, side), (0, 0))
    tile_b = atlas.TileSpec("tile_b", (side, side), (side + pad, 0))
    atlas_size = (side + pad + side, side)
    layout = atlas.AtlasLayout(atlas_size=atlas_size, tiles={"tile_a": tile_a, "tile_b": tile_b},
                                effective_density=500.0, rescaled=False)
    # Solved texel at each tile's edge closest to the other tile, so any bleed shows up
    # as fast as possible.
    texels = {
        "tile_a": {"xy": np.array([[side - 1, 0]], dtype=np.int64)},
        "tile_b": {"xy": np.array([[0, 0]], dtype=np.int64)},
    }
    plan = adapter.AtlasPlan(layout=layout, texels=texels, digest="bleedtest")
    history = {"tile_a": np.array([[400.0]]), "tile_b": np.array([[300.0]])}

    temp, _alpha = adapter._scatter_atlas_arrays(history, plan)

    b_start = side + pad
    tile_b_region = temp[:, b_start : b_start + side]
    tile_a_region = temp[:, 0:side]
    # The gap's near-A column may legitimately carry tile A's dilated value (and the
    # near-B column may legitimately carry tile B's) -- that's the single-texel push-out
    # margin working as intended. What must never happen is A's temperature reaching
    # tile B's region or the gap column immediately adjacent to B (and symmetrically for
    # B's temperature reaching tile A's side).
    gap_near_b = temp[:, b_start - 1 : b_start]
    gap_near_a = temp[:, side : side + 1]
    assert not np.any(np.isclose(tile_b_region, 400.0)), tile_b_region
    assert not np.any(np.isclose(gap_near_b, 400.0)), gap_near_b
    assert not np.any(np.isclose(tile_a_region, 300.0)), tile_a_region
    assert not np.any(np.isclose(gap_near_a, 300.0)), gap_near_a
    # The middle gap column must stay untouched (0.0). With 2*iterations <= padding the
    # two tiles' push-outs can never meet; if a future bump violated that invariant, the
    # meeting texels would hold the MEAN of both tiles (e.g. 350.0 here) - catch it.
    gap_middle = temp[:, 5:6]
    assert np.all(gap_middle == 0.0), gap_middle


# ---------------------------------------------------------------------------
# write_frame_attributes TEXEL behavior (no bpy needed via a minimal fake scene/mesh).
# ---------------------------------------------------------------------------


class _FakeAttrData(list):
    def foreach_set(self, prop, values):
        for elem, value in zip(self, values):
            setattr(elem, prop, float(value))


class _FakeAttr:
    def __init__(self, n):
        self.data = _FakeAttrData(type("D", (), {"value": 0.0})() for _ in range(n))


class _FakeAttrs(dict):
    def __init__(self, n):
        super().__init__()
        self._n = n

    def new(self, name, type, domain):
        self[name] = _FakeAttr(self._n)
        return self[name]

    def remove(self, attr):
        for key, value in list(self.items()):
            if value is attr:
                del self[key]
                return


class _FakeMesh:
    def __init__(self, n_verts):
        self.vertices = [object()] * n_verts
        self.attributes = _FakeAttrs(n_verts)

    def update(self):
        pass


class _FakeObj:
    def __init__(self, name, n_verts):
        self.name = name
        self.type = "MESH"
        self.data = _FakeMesh(n_verts)
        self.heat_sim_material = None
        self._props = {}

    def __setitem__(self, key, value):
        self._props[key] = value

    def __getitem__(self, key):
        return self._props[key]

    def __contains__(self, key):
        return key in self._props

    def __delitem__(self, key):
        del self._props[key]


class _FakeScene:
    def __init__(self, objects):
        self.objects = objects


_DEFAULTS = {
    "initial_temperature_K": 295.0,
    "thermal_diffusivity_mm2_s": 0.17,
    "density_kg_m3": 1330.0,
    "specific_heat_J_kgK": 880.0,
    "emissivity": 0.9,
}


def test_write_frame_attributes_texel_objects_get_fallback_only():
    vertex_obj = _FakeObj("vertex_mesh", n_verts=3)
    atlas_obj = _FakeObj("atlas_mesh", n_verts=3)  # same vertex count as its texel count on purpose

    # atlas_mesh's "history" has K=3 texels -- deliberately the SAME as its vertex count, so a
    # shape-coincidence could otherwise slip through the old (implicit) shape-mismatch fallback.
    history = {
        "vertex_mesh": np.array([[295.0, 295.0, 295.0], [300.0, 301.0, 302.0]]),
        "atlas_mesh": np.array([[295.0, 295.0, 295.0], [350.0, 360.0, 370.0]]),
    }
    plan = _plan((8, 8), {"atlas_mesh": atlas.TileSpec("atlas_mesh", (8, 8), (0, 0))}, {"atlas_mesh": {"xy": np.zeros((3, 2), dtype=np.int64)}})

    scene = _FakeScene([vertex_obj, atlas_obj])
    adapter.write_frame_attributes(scene, history, -1, _DEFAULTS, atlas_plan=plan)

    # Vertex-path object: unchanged, per-vertex sim_temperature written from the final row.
    vals = [d.value for d in vertex_obj.data.attributes["sim_temperature"].data]
    assert vals == pytest.approx([300.0, 301.0, 302.0])
    assert "heatsim_default_temperature" not in vertex_obj._props

    # Atlas object: NO per-vertex sim_temperature attribute written (even though the shapes
    # would have "matched" under the old implicit-mismatch fallback) -- only the OBJECT-level
    # fallback, at the ambient default (FEM participant).
    assert "sim_temperature" not in atlas_obj.data.attributes
    assert "emissivity" not in atlas_obj.data.attributes
    assert atlas_obj["heatsim_default_temperature"] == pytest.approx(295.0)

    # Coverage gate: 1.0 for the atlas participant, 0.0 for the vertex-path object.
    assert atlas_obj["heatsim_atlas_coverage"] == pytest.approx(1.0)
    assert vertex_obj["heatsim_atlas_coverage"] == pytest.approx(0.0)


def test_write_frame_attributes_vertex_mode_unaffected_by_atlas_plan_none():
    """atlas_plan=None (or omitted) reproduces exactly today's VERTEX-only behavior --
    no coverage-gate property is stamped anywhere."""
    obj = _FakeObj("mesh", n_verts=2)
    history = {"mesh": np.array([[295.0, 295.0], [305.0, 306.0]])}

    scene = _FakeScene([obj])
    adapter.write_frame_attributes(scene, history, -1, _DEFAULTS)

    vals = [d.value for d in obj.data.attributes["sim_temperature"].data]
    assert vals == pytest.approx([305.0, 306.0])
    assert "heatsim_atlas_coverage" not in obj._props


def test_write_frame_attributes_atlas_participant_clears_stale_vertex_attrs():
    """F1: a VERTEX->TEXEL mode switch on a long-lived service must not let a
    pre-existing (stale) per-vertex sim_temperature/emissivity attribute bleed through
    atlas holes -- the shader's vertex-path fallback treats sim_temperature > 1.0 as
    valid, so a leftover attribute would win over the fresh OBJECT-level fallback."""
    atlas_obj = _FakeObj("atlas_mesh", n_verts=3)
    # Simulate a prior VERTEX-mode run: stale per-vertex attributes already present.
    atlas_obj.data.attributes.new(name="sim_temperature", type="FLOAT", domain="POINT")
    atlas_obj.data.attributes.new(name="emissivity", type="FLOAT", domain="POINT")
    for d in atlas_obj.data.attributes["sim_temperature"].data:
        d.value = 999.0

    plan = _plan(
        (8, 8),
        {"atlas_mesh": atlas.TileSpec("atlas_mesh", (8, 8), (0, 0))},
        {"atlas_mesh": {"xy": np.zeros((3, 2), dtype=np.int64)}},
    )
    scene = _FakeScene([atlas_obj])
    adapter.write_frame_attributes(scene, {}, -1, _DEFAULTS, atlas_plan=plan)

    assert "sim_temperature" not in atlas_obj.data.attributes
    assert "emissivity" not in atlas_obj.data.attributes
    assert atlas_obj["heatsim_default_temperature"] == pytest.approx(295.0)


def test_write_frame_attributes_vertex_mode_clears_stale_atlas_coverage_gate():
    """F4: a TEXEL->VERTEX mode switch must clear a stale heatsim_atlas_coverage=1.0
    object property left by a prior TEXEL run -- otherwise the shader's atlas mix gate
    (stale alpha * stale coverage) can stay open and select stale atlas texels over the
    fresh per-vertex temperatures written this call."""
    obj = _FakeObj("mesh", n_verts=2)
    obj["heatsim_atlas_coverage"] = 1.0  # left over from a prior TEXEL run
    history = {"mesh": np.array([[295.0, 295.0], [305.0, 306.0]])}

    scene = _FakeScene([obj])
    adapter.write_frame_attributes(scene, history, -1, _DEFAULTS, atlas_plan=None)

    assert "heatsim_atlas_coverage" not in obj._props


# ---------------------------------------------------------------------------
# write_frame_attributes constant-fill on impossible write-back (Fix 2: the 0-K
# regression guard). Same no-bpy fake scene/mesh as above.
# ---------------------------------------------------------------------------


class _FakeMat:
    """Stand-in for ``obj.heat_sim_material`` (mirrors test_heatsim_adapter.py's)."""

    def __init__(self, *, always_set: bool, **values):
        self._always_set = always_set
        for k, v in values.items():
            setattr(self, k, v)

    def is_property_set(self, attr):
        return self._always_set


def test_write_frame_attributes_shape_mismatch_writes_constant_fill(caplog):
    """Fix 2 / 0-K regression guard: a modifier changed the vertex count between the
    evaluated mesh the FEM solve ran on (history's vertex axis) and the base mesh
    (n_verts). The old behaviour dropped sim_temperature entirely -- absent, so the
    shader's `sim_temperature > 1.0` validity gate falls through to
    heatsim_default_temperature for the RADIANCE pass but the temperature AOV (which has
    no such gate) emits 0 K. The fix must still fill a constant sim_temperature equal to
    the mean of the solved field's final timestep, must not leave emissivity absent
    either, and must warn (naming the object)."""
    obj = _FakeObj("mismatched_mesh", n_verts=4)  # base mesh: 4 verts
    final_row = [300.0, 302.0, 304.0, 306.0, 308.0, 310.0]  # solve-time (evaluated): 6
    history = {"mismatched_mesh": np.array([[295.0] * 6, final_row])}

    scene = _FakeScene([obj])
    with caplog.at_level("WARNING"):
        adapter.write_frame_attributes(scene, history, -1, _DEFAULTS)

    # NOT absent -- this is the actual 0-K regression guard.
    assert "sim_temperature" in obj.data.attributes
    vals = [d.value for d in obj.data.attributes["sim_temperature"].data]
    expected_mean = float(np.mean(final_row))
    assert vals == pytest.approx([expected_mean] * 4)
    assert expected_mean > _DEFAULTS["initial_temperature_K"]  # real heating preserved

    # emissivity must not be left absent either (same reasoning).
    assert "emissivity" in obj.data.attributes
    eps = [d.value for d in obj.data.attributes["emissivity"].data]
    assert eps == pytest.approx([_DEFAULTS["emissivity"]] * 4)

    assert any("mismatched_mesh" in rec.message for rec in caplog.records)


def test_write_frame_attributes_missing_history_writes_fallback_fill():
    """Fix 2: an object entirely absent from `history` (e.g. a DIRICHLET_SOURCE fluid
    whose topology changes every frame, so no per-vertex field survives) must render at
    its RESERVOIR temperature -- not ambient -- and must not be left with an absent
    sim_temperature attribute."""
    obj = _FakeObj("dirichlet_mesh", n_verts=3)
    obj.heat_sim_material = _FakeMat(
        always_set=True,
        initial_temperature_K=295.0,
        thermal_diffusivity_mm2_s=0.17,
        density_kg_m3=1330.0,
        specific_heat_J_kgK=880.0,
        emissivity=0.9,
        thermal_role="DIRICHLET_SOURCE",
        dirichlet_temperature_K=350.0,
    )

    scene = _FakeScene([obj])
    adapter.write_frame_attributes(scene, {}, -1, _DEFAULTS)

    assert "sim_temperature" in obj.data.attributes
    vals = [d.value for d in obj.data.attributes["sim_temperature"].data]
    assert vals == pytest.approx([350.0, 350.0, 350.0])  # reservoir, not ambient (295 K)
    assert obj["heatsim_default_temperature"] == pytest.approx(350.0)
    assert "emissivity" in obj.data.attributes


def test_atlas_participants_still_have_no_vertex_attribute():
    """Fix 2 must not touch atlas participants: their per-pixel signal comes from the
    atlas image, not a per-vertex mesh attribute, and an earlier fix already strips any
    stale one left by a prior VERTEX-mode run. Confirms that invariant survives Fix 2's
    new constant-fill branches (an atlas participant is also absent from `history`, which
    would otherwise hit the same code path a non-participant does)."""
    atlas_obj = _FakeObj("atlas_mesh", n_verts=5)
    plan = _plan(
        (8, 8),
        {"atlas_mesh": atlas.TileSpec("atlas_mesh", (8, 8), (0, 0))},
        {"atlas_mesh": {"xy": np.zeros((5, 2), dtype=np.int64)}},
    )

    scene = _FakeScene([atlas_obj])
    # No history entry for the atlas object -- its signal lives in the atlas image.
    adapter.write_frame_attributes(scene, {}, -1, _DEFAULTS, atlas_plan=plan)

    assert "sim_temperature" not in atlas_obj.data.attributes
    assert "emissivity" not in atlas_obj.data.attributes
    assert atlas_obj["heatsim_default_temperature"] == pytest.approx(295.0)


# ---------------------------------------------------------------------------
# global_temperature_range already pools whatever is in `history`, TEXEL entries included
# (they arrive there via the same _split_history/solve_scene path as VERTEX entries) --
# this locks that behavior explicitly rather than relying on it being incidental.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# End-to-end integration: drive the real BlenderService.exposed_prepare_thermal /
# exposed_include_thermal with render_domain="TEXEL". Not one of the brief's named
# fake-bpy tests, but the unit tests above (write_atlas, write_frame_attributes, the
# shader node graph) each exercise one piece in isolation -- this closes the gap the
# same way test_heatsim_animated.py's BlenderService-driven test does for the animated
# path, catching a real orchestration mismatch between _thermal_solve/write_atlas/the
# atlas-image load+pack/setup_temperature_aov that no single-function test would see.
# ---------------------------------------------------------------------------


def test_prepare_and_include_thermal_texel_mode_end_to_end(executable, tmp_path):
    code = f"""
import bpy
from visionsim.simulate.heatsim import register
register()

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# A coarse (4-vertex) plane spanning 4 m^2: well under any reasonable
# atlas_texel_density, so it must join the atlas (mirrors test_heatsim_texel_mode.py's
# end-to-end atlas smoke test).
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

blend_path = r'{tmp_path}/texel_test.blend'
root_path = r'{tmp_path}'
bpy.ops.wm.save_as_mainfile(filepath=blend_path)

from visionsim.simulate.blender import BlenderService
from visionsim.simulate.heatsim.constants import ATLAS_COVERAGE_PROP, ATLAS_IMAGE_NAME

service = BlenderService()
service.exposed_initialize(blend_path, root_path)

service.exposed_prepare_thermal(
    render_domain='TEXEL',
    atlas_texel_density=64.0,
    atlas_tile_min=16,
    atlas_tile_max=64,
    atlas_texel_soft_max=500_000,
    domain='POINTS',
    laplacian_backend='ROBUST',
    device='cpu',
    sim_time_s=0.1,
    timestep_s=0.05,
    initial_temperature_K=295.0,
    thermal_diffusivity_mm2_s=0.17,
    density_kg_m3=1330.0,
    specific_heat_J_kgK=880.0,
    emissivity=0.9,
    irradiance_scale=100.0,
)

assert service._thermal_atlas_plan is not None
assert 'CoarsePlane' in service._thermal_atlas_plan.texels

plane_obj = bpy.data.objects['CoarsePlane']
assert 'sim_temperature' not in plane_obj.data.attributes, 'atlas object must not get a per-vertex write'
assert plane_obj[ATLAS_COVERAGE_PROP] == 1.0

atlas_img = bpy.data.images.get(ATLAS_IMAGE_NAME)
assert atlas_img is not None, 'atlas image was not loaded/packed'
assert atlas_img.packed_file is not None, 'atlas image was not packed'

# Re-fetch the material via the object rather than the captured Python `mat` handle --
# the albedo bake path (triggered by texel irradiance for the atlas) may swap/copy
# material slots, invalidating the original `bpy.types.Material` reference. That same
# albedo bake also leaves its OWN ShaderNodeTexImage node (the bake target) on the
# material, so look for the atlas-specific one by its image rather than assuming
# there is exactly one Image Texture node.
live_mat = plane_obj.material_slots[0].material
assert live_mat is not None
mat_nodes = live_mat.node_tree.nodes
aov_nodes = [n for n in mat_nodes if n.type == 'OUTPUT_AOV']
assert len(aov_nodes) == 1
atlas_tex_nodes = [n for n in mat_nodes if n.bl_idname == 'ShaderNodeTexImage' and n.image is atlas_img]
assert len(atlas_tex_nodes) == 1, [n.name for n in mat_nodes if n.bl_idname == 'ShaderNodeTexImage']

service.exposed_include_thermal(render_domain='TEXEL')
assert 'temperature' in service.render_layers.outputs

print('TEXEL_E2E_OK')
"""
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True, check=False)
    assert "TEXEL_E2E_OK" in out.stdout, out.stdout + "\n" + out.stderr


def test_prepare_thermal_texel_static_branch_keeps_dirichlet_reservoir_fallback(executable, tmp_path):
    """F2: exposed_prepare_thermal's static (non-animated) branch must stamp
    heatsim_default_temperature (ambient) BEFORE write_frame_attributes runs, not after --
    otherwise the stamp clobbers the Dirichlet-reservoir fallback write_frame_attributes
    just set for a DIRICHLET_SOURCE atlas object back down to ambient."""
    code = f"""
import bpy
from visionsim.simulate.heatsim import register
register()

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

bpy.ops.mesh.primitive_plane_add(size=2.0)
plane = bpy.context.active_object
plane.name = 'HotReservoir'
plane.heat_simulation_enabled = True
plane.heat_sim_material.thermal_role = 'DIRICHLET_SOURCE'
plane.heat_sim_material.dirichlet_temperature_K = 350.0

mat = bpy.data.materials.new('plane_mat')
mat.use_nodes = True
plane.data.materials.append(mat)

bpy.ops.object.light_add(type='SUN')
bpy.context.active_object.data.energy = 10.0
world = bpy.context.scene.world
world.use_nodes = True
bg = world.node_tree.nodes.get('Background')
bg.inputs['Strength'].default_value = 1.0

blend_path = r'{tmp_path}/texel_dirichlet_test.blend'
root_path = r'{tmp_path}'
bpy.ops.wm.save_as_mainfile(filepath=blend_path)

from visionsim.simulate.blender import BlenderService
from visionsim.simulate.heatsim.constants import ATLAS_COVERAGE_PROP

service = BlenderService()
service.exposed_initialize(blend_path, root_path)

service.exposed_prepare_thermal(
    render_domain='TEXEL',
    atlas_texel_density=64.0,
    atlas_tile_min=16,
    atlas_tile_max=64,
    atlas_texel_soft_max=500_000,
    domain='POINTS',
    laplacian_backend='ROBUST',
    device='cpu',
    sim_time_s=0.1,
    timestep_s=0.05,
    initial_temperature_K=295.0,
    thermal_diffusivity_mm2_s=0.17,
    density_kg_m3=1330.0,
    specific_heat_J_kgK=880.0,
    emissivity=0.9,
    irradiance_scale=100.0,
)

plane_obj = bpy.data.objects['HotReservoir']
assert plane_obj[ATLAS_COVERAGE_PROP] == 1.0, 'expected this DIRICHLET_SOURCE object to be an atlas participant'
# Must be the reservoir temperature (350K), NOT ambient (295K) -- a wrong stamp/write
# order would have clobbered it back down to ambient.
assert abs(plane_obj['heatsim_default_temperature'] - 350.0) < 1e-6, plane_obj['heatsim_default_temperature']

print('DIRICHLET_FALLBACK_OK')
"""
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True, check=False)
    assert "DIRICHLET_FALLBACK_OK" in out.stdout, out.stdout + "\n" + out.stderr


def test_global_temperature_range_includes_texels():
    # A "vertex" object near ambient plus an "atlas" (texel) object running much hotter --
    # both keyed into `history` exactly the same way (solve_scene/​_split_history don't
    # distinguish TEXEL from VERTEX entries), so the pooled range must span both.
    history = {
        "vertex_mesh": np.stack([np.full(50, 295.0), np.full(50, 295.5)]),
        "atlas_mesh": np.stack([np.full(400, 295.0), np.full(400, 340.0)]),
    }

    tmin, tmax = adapter.global_temperature_range(history, default_K=295.0)

    # The texel object's 340 K dominates the pool (400 texels vs 50 vertices), so P99 must
    # land near it, not be capped near the near-ambient vertex object's ~295.5 K.
    assert tmax > 330.0, tmax
    assert tmin <= 295.5, tmin
