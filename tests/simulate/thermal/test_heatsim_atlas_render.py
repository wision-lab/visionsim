"""Atlas storage and render integration tests."""

from __future__ import annotations

import Imath
import numpy as np
import OpenEXR
import pytest

from tests.simulate.thermal.heatsim_test_service import SERVICE_PATH, call_service
from visionsim.simulate.blender import BlenderClient
from visionsim.simulate.heatsim import adapter, atlas


def _tile_layout(atlas_size, tiles):
    return atlas.AtlasLayout(atlas_size=atlas_size, tiles=tiles, effective_density=500.0, rescaled=False)


def _plan(atlas_size, tiles, texels, digest="testdigest"):
    return adapter.AtlasPlan(layout=_tile_layout(atlas_size, tiles), texels=texels, digest=digest)


# write_atlas: needs bpy (image creation/save), run inside Blender.


def test_write_atlas_scatters_dilates_and_marks_alpha(executable, tmp_path):
    # One object, one tile sized from the dilation count so an unwritten region provably
    # survives: the margin grows by one texel per pass, so a corner further than that from
    # every solved texel must stay alpha=0. Hardcoding a 6x6 tile silently stopped testing
    # anything when _ATLAS_DILATE_ITERATIONS was raised from 1 to 8 (the margin then covered
    # the whole tile). Two solved texels are placed far apart so their margins don't mask
    # each other's zeros.
    n = 2 * adapter._ATLAS_DILATE_ITERATIONS + 6
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        written = call_service(
            client,
            "write_atlas",
            str(tmp_path),
            {"obj": {"size": [n, n], "offset": [0, 0]}},
            {"obj": [[1, 1], [n - 2, n - 2]]},
            {"obj": [310.0, 320.0]},
            [n, n],
            digest="rt",
        )
        sampled = call_service(
            client,
            "load_atlas",
            written["atlas"],
            {"texel1": [1, 1], "texel2": [n - 2, n - 2], "neighbor": [1, 2]},
        )

    assert written["exists"], written["atlas"]
    assert written["name"] == "atlas_temperature.exr"
    assert written["parent_prefix"] == "atlas"
    assert written["size"] == [n, n], written["size"]

    # Scattered texels: match (within EXR write/load round-trip precision -- Blender's
    # Image.save() has no exposed lossless-compression knob for a plain generated image, so
    # a few mK of drift is expected and harmless for a Kelvin-scale field).
    assert abs(sampled["samples"]["texel1"][0] - 310.0) < 0.05, sampled["samples"]["texel1"]
    assert abs(sampled["samples"]["texel2"][0] - 320.0) < 0.05, sampled["samples"]["texel2"]
    assert sampled["samples"]["texel1"][1] == 1.0 and sampled["samples"]["texel2"][1] == 1.0

    # A direct 8-neighbor of a solved texel is inside the dilation margin: alpha=1 and a
    # nonzero temperature pulled from that neighbor (never the raw zero an un-dilated
    # scatter would leave).
    assert sampled["samples"]["neighbor"][1] == 1.0, sampled["samples"]["neighbor"]
    assert sampled["samples"]["neighbor"][0] > 0.0, sampled["samples"]["neighbor"]

    # Far corner, well outside the (small, capped) dilation margin: alpha=0, temperature 0.
    assert written["unwritten_count"] > 0, "dilation covered the whole tile; margin is not bounded"
    assert written["unwritten_value"] == [0.0, 0.0], written["unwritten_value"]

    # Absolute-value check: read the EXR with the standalone OpenEXR package (never bpy --
    # bpy's load applies the same colorspace-tagged decode as the write's encode, so a
    # write->bpy-load round trip only proves symmetry, not that the file holds the true
    # Kelvin values). This is the assertion that catches a Non-Color tag regression: an
    # untagged write would land here as ~11.2 (sRGB-OETF-encoded 310.0), not 310.0.
    exr_path = next(tmp_path.glob("atlas_*/atlas_temperature.exr"))
    assert exr_path.exists(), exr_path
    exr = OpenEXR.InputFile(str(exr_path))
    assert str(exr.header()["compression"]) == "NO_COMPRESSION"
    dw = exr.header()["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    float_t = Imath.PixelType(Imath.PixelType.FLOAT)
    r_raw = np.frombuffer(exr.channel("R", float_t), dtype=np.float32).reshape(h, w)
    # OpenEXR rows are top-down while Blender's Image.pixels buffer (and the (x, y) texel
    # coordinates used above) are bottom-up, so the on-disk row is (h - 1 - y).
    assert (w, h) == (n, n), (w, h)
    assert abs(float(r_raw[h - 1 - 1, 1]) - 310.0) < 1e-2, r_raw[h - 1 - 1, 1]
    assert abs(float(r_raw[h - 1 - (n - 2), n - 2]) - 320.0) < 1e-2, r_raw[h - 1 - (n - 2), n - 2]


def test_write_atlas_no_texels_writes_empty_placeholder(executable, tmp_path):
    """No atlas objects this solve (render_domain=TEXEL requested but nothing qualified,
    or atlas_plan built from an empty scene) -> write_atlas must still return a stable,
    loadable, all-invalid path rather than raising."""
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        result = call_service(client, "write_atlas", str(tmp_path), {}, {}, {}, [0, 0], digest="empty")

    assert result["exists"], result["atlas"]
    assert result["all_alpha_zero"], "placeholder atlas has a valid (non-zero alpha) texel"


def test_atlas_path_changes_with_solved_values(executable, tmp_path):
    tiles = {"wall": {"size": [4, 4], "offset": [0, 0]}}
    texels = {"wall": [[1, 1]]}
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        first = call_service(
            client, "write_atlas", str(tmp_path), tiles, texels, {"wall": [300.0]}, [4, 4], digest="same"
        )
        second = call_service(
            client, "write_atlas", str(tmp_path), tiles, texels, {"wall": [320.0]}, [4, 4], digest="same"
        )

    assert first["atlas"] != second["atlas"], "identical path for different values"
    assert first["exists"] and second["exists"], (first["exists"], second["exists"])


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
    layout = atlas.AtlasLayout(
        atlas_size=atlas_size, tiles={"tile_a": tile_a, "tile_b": tile_b}, effective_density=500.0, rescaled=False
    )
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


# write_frame_attributes TEXEL behavior (no bpy needed via a minimal fake scene/mesh).


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
    plan = _plan(
        (8, 8),
        {"atlas_mesh": atlas.TileSpec("atlas_mesh", (8, 8), (0, 0))},
        {"atlas_mesh": {"xy": np.zeros((3, 2), dtype=np.int64)}},
    )

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
    """Switching to TEXEL clears vertex attributes left by a prior solve."""
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
    """Switching to VERTEX clears coverage left by a prior atlas solve."""
    obj = _FakeObj("mesh", n_verts=2)
    obj["heatsim_atlas_coverage"] = 1.0  # left over from a prior TEXEL run
    history = {"mesh": np.array([[295.0, 295.0], [305.0, 306.0]])}

    scene = _FakeScene([obj])
    adapter.write_frame_attributes(scene, history, -1, _DEFAULTS, atlas_plan=None)

    assert "heatsim_atlas_coverage" not in obj._props


# write_frame_attributes constant-fill on impossible write-back (Fix 2: the 0-K
# regression guard). Same no-bpy fake scene/mesh as above.


class _FakeMat:
    """Stand-in for ``obj.heat_sim_material`` (mirrors test_heatsim_adapter.py's)."""

    def __init__(self, *, always_set: bool, **values):
        self._always_set = always_set
        for k, v in values.items():
            setattr(self, k, v)

    def is_property_set(self, attr):
        return self._always_set


def test_write_frame_attributes_rejects_unsafe_vertex_mapping():
    obj = _FakeObj("mismatched_mesh", n_verts=4)
    history = {"mismatched_mesh": np.array([[295.0] * 6, [300.0, 302.0, 304.0, 306.0, 308.0, 310.0]])}
    with pytest.raises(RuntimeError, match="mismatched_mesh.*cannot be written"):
        adapter.write_frame_attributes(_FakeScene([obj]), history, -1, _DEFAULTS)


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


def test_prepare_and_include_thermal_texel_mode_end_to_end(executable, tmp_path):
    blend = str(tmp_path / "texel_test.blend")
    with BlenderClient.spawn(executable=executable, timeout=120, service=SERVICE_PATH) as client:
        # A coarse (4-vertex) plane spanning 4 m^2: well under any reasonable
        # atlas_texel_density, so it must join the atlas.
        call_service(client, "build_scene", "plane", "CoarsePlane", material="plain")
        configured = call_service(client, "configure_thermal", str(tmp_path), blend)
        frozen = call_service(client, "freeze", blend, str(tmp_path / "thermal_frozen.blend"))

    assert configured["in_plan"] and configured["obj_name"] == "CoarsePlane", configured["texels"]
    assert configured["has_vertex_attr"] is False, "atlas object must not get a per-vertex write"
    assert configured["coverage"] == 1.0, configured["coverage"]

    assert configured["atlas_loaded"], "atlas image was not loaded/packed"
    assert configured["atlas_packed"], "atlas image was not packed"

    # Re-fetch the material via the object rather than a captured handle -- the albedo
    # bake path may swap/copy material slots. The albedo bake also leaves its OWN
    # ShaderNodeTexImage node on the material, so the atlas one is matched by image.
    assert configured["aov_count"] == 1, configured["aov_count"]
    assert configured["atlas_tex_count"] == 1, configured["atlas_tex_count"]
    assert configured["has_temperature_pass"], "temperature pass missing from the render layers"

    assert frozen["persistent_restored"], "persistent-data flag was not restored on reopen"
    assert frozen["reopened_packed"], "atlas image was not packed into the frozen blend"
    assert frozen["reopened_wired"], "atlas image not wired into the reopened material"


def test_prepare_thermal_texel_static_branch_keeps_dirichlet_reservoir_fallback(executable, tmp_path):
    """Atlas Dirichlet fallback survives default-temperature stamping."""
    blend = str(tmp_path / "texel_dirichlet_test.blend")
    with BlenderClient.spawn(executable=executable, timeout=120, service=SERVICE_PATH) as client:
        call_service(client, "build_scene", "plane", "HotReservoir", material="plain", dirichlet_K=350.0)
        configured = call_service(client, "configure_thermal", str(tmp_path), blend)

    assert configured["in_plan"], "expected this DIRICHLET_SOURCE object to be an atlas participant"
    assert configured["coverage"] == 1.0, configured["coverage"]
    # Must be the reservoir temperature (350K), NOT ambient (295K) -- a wrong stamp/write
    # order would have clobbered it back down to ambient.
    assert abs(configured["default_temperature"] - 350.0) < 1e-6, configured["default_temperature"]


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
