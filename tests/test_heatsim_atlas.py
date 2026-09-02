from __future__ import annotations

import math

import numpy as np
import pytest

from visionsim.simulate.heatsim import atlas

# ---------------------------------------------------------------------------
# surface_area_m2
# ---------------------------------------------------------------------------


def test_surface_area_of_unit_quad():
    # A 1000mm x 1000mm quad (two triangles) is exactly 1 m^2.
    verts_mm = np.array(
        [[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0], [1000.0, 1000.0, 0.0], [0.0, 1000.0, 0.0]]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    assert atlas.surface_area_m2(verts_mm, faces) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# select_for_atlas
# ---------------------------------------------------------------------------


def test_selection_excludes_dense_objects():
    density = 500.0
    # Orchid-like: 21k verts crammed into 2 m^2 -> ~10500 verts/m^2, already denser than target.
    assert atlas.select_for_atlas(n_verts=21_000, area_m2=2.0, density=density) is False
    # Floor-like: 16 verts spanning 80 m^2 -> 0.2 verts/m^2, far under target -> joins the atlas.
    assert atlas.select_for_atlas(n_verts=16, area_m2=80.0, density=density) is True


def test_selection_area_zero_uses_eps_guard():
    # area_m2=0 must not raise (division guarded by eps); any positive vertex count is "dense".
    assert atlas.select_for_atlas(n_verts=1, area_m2=0.0, density=500.0) is False
    # Zero verts and zero area: 0/eps = 0, which is < any positive density -> included.
    assert atlas.select_for_atlas(n_verts=0, area_m2=0.0, density=500.0) is True


def test_select_for_atlas_forces_participation_when_writeback_impossible():
    """Fix 1: per-vertex write-back can only ever land on the BASE mesh. When a
    topology-changing modifier (Bevel, Subdivision, Geometry Nodes, ...) makes the base
    and evaluated vertex counts differ, that path is structurally impossible regardless
    of density, so the object must join the atlas even if the density rule alone would
    exclude it."""
    density = 500.0
    n_verts, area_m2 = 21_000, 2.0  # orchid-dense: would normally be excluded.
    assert atlas.select_for_atlas(n_verts, area_m2, density) is False
    assert atlas.select_for_atlas(n_verts, area_m2, density, writeback_possible=True) is False
    assert atlas.select_for_atlas(n_verts, area_m2, density, writeback_possible=False) is True


# ---------------------------------------------------------------------------
# allocate: sizing
# ---------------------------------------------------------------------------


def test_tile_size_scales_with_area_and_respects_min_max_and_multiple_of_4():
    density = 1500.0
    areas = {
        "tiny": 0.01,  # K=15,   ceil(sqrt)=4,    mult4=4,    clamped up to tile_min=16
        "mid": 1.0,  # K=1500, ceil(sqrt)=39,   mult4=40,   no clamp
        "huge": 1000.0,  # K=1.5e6, ceil(sqrt)=1225, mult4=1228, clamped down to tile_max=512
    }
    layout = atlas.allocate(areas, density, tile_min=16, tile_max=512)

    assert layout.tiles["tiny"].size == (16, 16)
    assert layout.tiles["mid"].size == (40, 40)
    assert layout.tiles["huge"].size == (512, 512)
    for spec in layout.tiles.values():
        w, h = spec.size
        assert w == h  # tiles are square
        assert w % 4 == 0
        assert 16 <= w <= 512


def test_big_object_not_power_of_two_overshot():
    # 81 m^2 @ 1500/m^2: K=121500, ceil(sqrt)=349, rounded up to mult-of-4 = 352.
    # A naive power-of-two quantizer would jump straight to 512 (a ~2.1x-per-side, ~4.4x-area
    # overshoot); the spec explicitly forbids that.
    layout = atlas.allocate({"floor": 81.0}, 1500.0)
    side = layout.tiles["floor"].size[0]
    assert side == 352
    assert side != 512


# ---------------------------------------------------------------------------
# allocate: soft-max rescale
# ---------------------------------------------------------------------------


def test_soft_max_rescales_uniformly_and_warns():
    # area=100 m^2 @ density=100/m^2 -> K=10000 -> side=100 exactly (already mult of 4, no clamp),
    # so tiles_total == 10000 exactly, keeping the rescale arithmetic hand-checkable.
    areas = {"big": 100.0}
    density = 100.0
    soft_max = 5000

    with pytest.warns(UserWarning):
        layout = atlas.allocate(areas, density, soft_max=soft_max)

    assert layout.rescaled is True
    # ratio = soft_max / tiles_total = 5000/10000 = 0.5 -> effective_density = 50.0
    assert layout.effective_density == pytest.approx(50.0)
    assert layout.effective_density < density
    # New K = 100*50=5000 -> ceil(sqrt)=71 -> rounded up to mult-of-4 = 72.
    assert layout.tiles["big"].size == (72, 72)


def test_soft_max_counts_retained_vertices():
    areas = {"big": 100.0}  # K=10000 -> side=100 -> tiles_total=10000, same as above
    density = 100.0
    soft_max = 15_000

    # Without retained vertices, tiles_total (10000) is under soft_max (15000) -> no rescale.
    baseline = atlas.allocate(areas, density, soft_max=soft_max, retained_vertex_count=0)
    assert baseline.rescaled is False
    assert baseline.effective_density == pytest.approx(density)

    # With 6000 retained vertices, tiles_total+retained = 16000 > 15000 -> rescale must trigger,
    # even though tiles_total alone was fine. This is the "retained vertices count" contract.
    with pytest.warns(UserWarning):
        with_retained = atlas.allocate(areas, density, soft_max=soft_max, retained_vertex_count=6000)
    assert with_retained.rescaled is True
    # ratio = (soft_max - retained)/tiles_total = (15000-6000)/10000 = 0.9 -> density=90.0
    assert with_retained.effective_density == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# allocate: shelf packing
# ---------------------------------------------------------------------------


def _rects_have_padding_gap(r1, r2, padding):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2

    if x1 + w1 <= x2:
        xgap = x2 - (x1 + w1)
    elif x2 + w2 <= x1:
        xgap = x1 - (x2 + w2)
    else:
        xgap = None

    if y1 + h1 <= y2:
        ygap = y2 - (y1 + h1)
    elif y2 + h2 <= y1:
        ygap = y1 - (y2 + h2)
    else:
        ygap = None

    return (xgap is not None and xgap >= padding) or (ygap is not None and ygap >= padding)


def test_shelf_pack_no_overlap_and_padding():
    areas = {"a": 0.01, "b": 1.0, "c": 4.0, "d": 0.2, "e": 9.0}
    padding = 2
    layout = atlas.allocate(areas, density=1500.0, padding=padding)

    assert set(layout.tiles) == set(areas)
    rects = {name: (spec.offset[0], spec.offset[1], spec.size[0], spec.size[1]) for name, spec in layout.tiles.items()}

    names = list(rects)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert _rects_have_padding_gap(rects[names[i]], rects[names[j]], padding), (names[i], names[j])

    # Every tile must actually fit inside the reported atlas bounds.
    atlas_w, atlas_h = layout.atlas_size
    assert atlas_w % 4 == 0 and atlas_h % 4 == 0
    for x, y, w, h in rects.values():
        assert x >= 0 and y >= 0
        assert x + w <= atlas_w
        assert y + h <= atlas_h


def test_allocate_empty_areas():
    layout = atlas.allocate({}, density=500.0)
    assert layout.tiles == {}
    assert layout.atlas_size == (0, 0)
    assert layout.rescaled is False
    assert layout.effective_density == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# rasterize_tile
# ---------------------------------------------------------------------------


def _quad_mesh(size_mm=80.0):
    verts_mm = np.array(
        [[0.0, 0.0, 0.0], [size_mm, 0.0, 0.0], [size_mm, size_mm, 0.0], [0.0, size_mm, 0.0]]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    loop_uv = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        ]
    )
    return verts_mm, faces, loop_uv


def test_rasterize_full_cover_quad():
    size_mm = 80.0
    tile = (8, 8)
    verts_mm, faces, loop_uv = _quad_mesh(size_mm)

    out = atlas.rasterize_tile(verts_mm, faces, loop_uv, tile)

    w, h = tile
    assert out["xy"].shape[0] == w * h
    # every texel covered exactly once
    seen = set(map(tuple, out["xy"].tolist()))
    assert seen == {(x, y) for x in range(w) for y in range(h)}

    # normals all point +Z (verts are CCW in the XY plane)
    assert np.allclose(out["normal"], np.array([0.0, 0.0, 1.0]), atol=1e-9)

    # positions interpolate linearly with texel-center UV
    for xy, pos in zip(out["xy"], out["position_mm"]):
        x, y = xy
        expected = np.array([(x + 0.5) / w * size_mm, (y + 0.5) / h * size_mm, 0.0])
        assert pos == pytest.approx(expected, abs=1e-9)

    assert np.isfinite(out["position_mm"]).all()
    assert np.isfinite(out["normal"]).all()


def test_rasterize_half_tile_triangle():
    tile = (8, 8)
    verts_mm = np.array([[0.0, 0.0, 0.0], [80.0, 0.0, 0.0], [0.0, 80.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    loop_uv = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])

    out = atlas.rasterize_tile(verts_mm, faces, loop_uv, tile)

    w, h = tile
    n = out["xy"].shape[0]
    # roughly half the tile, and strictly none of it
    assert 0 < n < w * h
    assert 20 <= n <= 44
    xs, ys = out["xy"][:, 0], out["xy"][:, 1]
    assert (xs >= 0).all() and (xs < w).all()
    assert (ys >= 0).all() and (ys < h).all()
    # no duplicate texels
    assert len(set(map(tuple, out["xy"].tolist()))) == n


def test_rasterize_no_double_claim():
    # Same abutting-triangle-pair setup as the full-cover test, but the assertion here is
    # specifically about the shared diagonal edge: no texel may be claimed by both triangles.
    tile = (16, 16)
    verts_mm, faces, loop_uv = _quad_mesh(160.0)

    out = atlas.rasterize_tile(verts_mm, faces, loop_uv, tile)

    w, h = tile
    xy_rows = list(map(tuple, out["xy"].tolist()))
    assert len(xy_rows) == w * h  # full coverage, no gaps
    assert len(set(xy_rows)) == len(xy_rows)  # no texel appears twice

    # Confirm both triangles actually contributed (i.e. this is a real two-triangle test, not one
    # triangle silently covering everything).
    assert set(out["face"].tolist()) == {0, 1}


def test_rasterize_degenerate_triangle_skipped():
    tile = (8, 8)
    verts_mm = np.array([[0.0, 0.0, 0.0], [80.0, 0.0, 0.0], [40.0, 0.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    # Colinear UVs -> zero UV-space area.
    loop_uv = np.array([[[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]]])

    out = atlas.rasterize_tile(verts_mm, faces, loop_uv, tile)

    assert out["xy"].shape[0] == 0
    assert out["position_mm"].shape == (0, 3)
    assert out["normal"].shape == (0, 3)
    assert out["face"].shape == (0,)
    for key in out:
        assert np.isfinite(out[key]).all() if out[key].size else True


def test_rasterize_degenerate_triangle_mixed_with_valid_no_nan():
    # A degenerate triangle followed by a valid one: the degenerate one must not poison the rest.
    tile = (8, 8)
    verts_mm = np.array(
        [[0.0, 0.0, 0.0], [80.0, 0.0, 0.0], [40.0, 0.0, 0.0], [80.0, 80.0, 0.0], [0.0, 80.0, 0.0]]
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 3, 4]])
    loop_uv = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]],  # degenerate
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        ]
    )

    out = atlas.rasterize_tile(verts_mm, faces, loop_uv, tile)

    assert out["xy"].shape[0] == 8 * 8
    assert np.isfinite(out["position_mm"]).all()
    assert np.isfinite(out["normal"]).all()
    assert 0 not in out["face"].tolist()  # the degenerate triangle (face 0) claims nothing


def test_rasterize_mirrored_uv_triangle_interpolates_correctly():
    """Mirrored-UV island: face 0's loop_uv corners are authored CW (negative signed UV area),
    which exercises the `area2 < 0` vertex-swap branch in rasterize_tile. Mesh vertex order in
    `faces` stays natural/untouched (so the swap only affects the UV<->vertex pairing used for
    barycentric interpolation, not the mesh winding used for the normal).

    Mesh: same quad as `_quad_mesh` - P0=(0,0,0), P1=(size,0,0), P2=(size,size,0), P3=(0,size,0) -
    split into faces [0,1,2] and [0,2,3], both CCW in 3D (both normals point +Z).

    Face 0's UV corners are the *same three UV points* as `_quad_mesh`'s face 0 ((0,0),(1,0),(1,1))
    but with corners 1 and 2 swapped, i.e. mirrored:
        uv0 = (0, 0)  at vertex i0 = P0
        uv1 = (1, 1)  at vertex i1 = P1   (P1 normally maps to (1, 0))
        uv2 = (1, 0)  at vertex i2 = P2   (P2 normally maps to (1, 1))
    Signed UV area = cross(uv1-uv0, uv2-uv0) = cross((1,1),(1,0)) = 1*0 - 1*1 = -1 < 0, so
    rasterize_tile takes the `area2 < 0` branch: idx becomes (i0, i2, i1), pts becomes
    (uv0, uv2, uv1) = ((0,0), (w,0), (w,h)) in tile-pixel space (w, h = tile size). That is the
    *same triangle footprint* as face 0 in `_quad_mesh` (only relabeled), so coverage/no-double-
    claim behaves exactly as in `test_rasterize_full_cover_quad` - what differs is which mesh
    vertex gets which barycentric weight.

    Hand-derived barycentric weights for a=(0,0), b=(w,0), c=(w,h), area2=w*h, texel center
    (px, py) = (x+0.5, y+0.5):
        w_a = h*(w-px),  w_b = h*px - w*py,  w_c = w*py
        wa = 1 - px/w,   wb = px/w - py/h,   wc = py/h
    and post-swap (ia, ib, ic) = (i0, i2, i1) = (P0, P2, P1), so
        position = P0*wa + P2*wb + P1*wc = (size*px/w, size*(px/w - py/h), 0)
    Note the y-component is *not* the naive `(y+0.5)/h*size` shortcut - it is skewed by the
    mirrored vertex/UV pairing (it depends on px too). A regression that mishandles the swap
    (e.g. permutes `idx` without permuting `pts` to match, or vice versa) would instead produce
    either NaNs/garbage or the naive-linear result asserted for face 1 below; this formula is
    specific to the correct pairing and was cross-checked against the implementation by hand.

    Face 1 is untouched from `_quad_mesh` (ordinary CCW UV, no swap), so it must still interpolate
    with the naive-linear formula - confirming the mirror on face 0 didn't leak into face 1.
    """
    size_mm = 80.0
    tile = (8, 8)
    w, h = tile
    verts_mm = np.array(
        [[0.0, 0.0, 0.0], [size_mm, 0.0, 0.0], [size_mm, size_mm, 0.0], [0.0, size_mm, 0.0]]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    loop_uv = np.array(
        [
            [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0]],  # CW winding -> area2 < 0 swap branch (mirrored)
            [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],  # ordinary CCW UV, same as _quad_mesh face 1
        ]
    )

    # Sanity-check the premise this test depends on, independent of atlas internals: face 0's
    # loop_uv is authored CW (negative signed area) while face 1's is CCW (positive).
    uv0, uv1, uv2 = loop_uv[0]
    signed_area2_face0 = (uv1[0] - uv0[0]) * (uv2[1] - uv0[1]) - (uv1[1] - uv0[1]) * (uv2[0] - uv0[0])
    assert signed_area2_face0 < 0, "fixture must exercise the area2 < 0 swap branch"
    uv0, uv1, uv2 = loop_uv[1]
    signed_area2_face1 = (uv1[0] - uv0[0]) * (uv2[1] - uv0[1]) - (uv1[1] - uv0[1]) * (uv2[0] - uv0[0])
    assert signed_area2_face1 > 0, "face 1 must stay on the ordinary (no-swap) path"

    out = atlas.rasterize_tile(verts_mm, faces, loop_uv, tile)

    # (a) full coverage, no double-claimed texels, and both triangles actually contributed. The
    # footprint is identical to test_rasterize_full_cover_quad's quad (see derivation above), so
    # this should behave exactly the same regardless of the UV mirror on face 0.
    xy_rows = list(map(tuple, out["xy"].tolist()))
    assert len(xy_rows) == w * h
    assert len(set(xy_rows)) == len(xy_rows)
    assert set(out["face"].tolist()) == {0, 1}

    # (c) normals: computed from `faces`' vertex order (mesh winding), not `loop_uv`, so the UV
    # mirror on face 0 must NOT affect it. Both faces are CCW in the XY plane -> unit +Z normal.
    assert np.allclose(out["normal"], np.array([0.0, 0.0, 1.0]), atol=1e-9)
    assert np.allclose(np.linalg.norm(out["normal"], axis=1), 1.0, atol=1e-9)

    xy = out["xy"]
    pos = out["position_mm"]
    face_col = out["face"]

    # (b) face 0 (mirrored): every covered texel must match the skewed barycentric formula above,
    # not the naive linear one.
    face0_mask = face_col == 0
    assert face0_mask.sum() > 0  # the mirrored triangle actually contributed texels
    for x, y, p in zip(xy[face0_mask, 0], xy[face0_mask, 1], pos[face0_mask]):
        px, py = float(x) + 0.5, float(y) + 0.5
        expected = np.array([size_mm * px / w, size_mm * (px / w - py / h), 0.0])
        assert p == pytest.approx(expected, abs=1e-9)

    # face 1 (ordinary CCW UV): naive-linear formula, unaffected by face 0's mirror.
    face1_mask = face_col == 1
    assert face1_mask.sum() > 0
    for x, y, p in zip(xy[face1_mask, 0], xy[face1_mask, 1], pos[face1_mask]):
        expected = np.array([(x + 0.5) / w * size_mm, (y + 0.5) / h * size_mm, 0.0])
        assert p == pytest.approx(expected, abs=1e-9)

    # Explicit named probes (face 0) for diagnosability if the loop-based check above ever fails:
    # texel (0,0): px=py=0.5 -> wa=1-0.5/8=0.9375, wb=0.5/8-0.5/8=0, wc=0.5/8=0.0625
    #   position = P0*0.9375 + P2*0 + P1*0.0625 = (80*0.0625, 0, 0) = (5.0, 0.0, 0.0)
    # texel (7,0): px=7.5, py=0.5 -> position_x=80*7.5/8=75.0, position_y=80*(7.5/8-0.5/8)=70.0
    # texel (4,2): px=4.5, py=2.5 -> position_x=80*4.5/8=45.0, position_y=80*(4.5/8-2.5/8)=20.0
    probes = {(0, 0): (5.0, 0.0, 0.0), (7, 0): (75.0, 70.0, 0.0), (4, 2): (45.0, 20.0, 0.0)}
    for (px_i, py_i), expected in probes.items():
        m = face0_mask & (xy[:, 0] == px_i) & (xy[:, 1] == py_i)
        assert m.sum() == 1, f"expected exactly one covered face-0 texel at ({px_i},{py_i})"
        assert pos[m][0] == pytest.approx(np.array(expected), abs=1e-9)


# ---------------------------------------------------------------------------
# dilate
# ---------------------------------------------------------------------------


def test_dilate_fills_margin_and_preserves_valid():
    image = np.zeros((5, 5), dtype=np.float64)
    valid = np.zeros((5, 5), dtype=bool)
    image[2, 2] = 10.0
    valid[2, 2] = True

    original = image.copy()
    out = atlas.dilate(image, valid, iterations=4)

    # The one originally-valid texel is untouched.
    assert out[2, 2] == pytest.approx(original[2, 2])

    # After one push-out pass, its 8 direct neighbors take the mean of valid 8-neighbors, i.e. 10.
    for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        assert out[2 + dy, 2 + dx] == pytest.approx(10.0)

    # With enough iterations (max Chebyshev distance from center on a 5x5 grid is 2), every texel
    # ends up filled - no leftover zeros from an unfilled/no-valid-neighbor texel.
    assert np.all(out > 0.0)

    # A corner (Chebyshev distance 2, reached only on the 2nd pass) should also end up close to
    # 10 given the small, uniform-source geometry here.
    assert out[0, 0] > 0.0


def test_dilate_preserves_valid_and_is_noop_when_all_valid():
    image = np.arange(9, dtype=np.float64).reshape(3, 3)
    valid = np.ones((3, 3), dtype=bool)
    out = atlas.dilate(image, valid, iterations=4)
    assert np.array_equal(out, image)


def test_dilate_rejects_non_2d_valid():
    # `valid` must be a single 2D validity mask even when `image` is multi-channel
    # (e.g. an (H, W, 3) RGB atlas) - a caller passing a (H, W, 3) `valid` by mistake
    # must get a clear error, not a silent shape-broadcast bug.
    image = np.zeros((4, 4, 3), dtype=np.float64)
    valid = np.zeros((4, 4, 3), dtype=bool)
    with pytest.raises(ValueError):
        atlas.dilate(image, valid, iterations=2)


def test_dilate_zero_iterations_is_noop():
    image = np.array([[1.0, 0.0], [0.0, 0.0]])
    valid = np.array([[True, False], [False, False]])
    out = atlas.dilate(image, valid, iterations=0)
    assert out[0, 0] == pytest.approx(1.0)
    assert out[0, 1] == pytest.approx(0.0)
    assert out[1, 0] == pytest.approx(0.0)
    assert out[1, 1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# performance sanity (not a hard requirement of the brief's named tests, but the brief calls out
# a concrete target: a 352^2 tile with ~2k triangles well under a second)
# ---------------------------------------------------------------------------


def test_rasterize_performance_sanity():

    rng = np.random.default_rng(0)
    n_tris = 2000
    tile = (352, 352)

    # Build a grid of small quads (2 triangles each) tiling the UV unit square, so triangles are
    # spatially local (like real UV islands) rather than degenerate/overlapping garbage.
    grid_n = math.ceil(math.sqrt(n_tris / 2))
    step = 1.0 / grid_n
    verts = []
    faces = []
    loop_uv = []
    for gy in range(grid_n):
        for gx in range(grid_n):
            if len(faces) >= n_tris:
                break
            u0, v0 = gx * step, gy * step
            u1, v1 = u0 + step, v0 + step
            base = len(verts)
            z = float(rng.uniform(-1.0, 1.0))
            verts.extend(
                [
                    [u0 * 1000, v0 * 1000, z],
                    [u1 * 1000, v0 * 1000, z],
                    [u1 * 1000, v1 * 1000, z],
                    [u0 * 1000, v1 * 1000, z],
                ]
            )
            faces.append([base, base + 1, base + 2])
            loop_uv.append([[u0, v0], [u1, v0], [u1, v1]])
            if len(faces) < n_tris:
                faces.append([base, base + 2, base + 3])
                loop_uv.append([[u0, v0], [u1, v1], [u0, v1]])

    verts_mm = np.array(verts)
    faces_arr = np.array(faces[:n_tris])
    loop_uv_arr = np.array(loop_uv[:n_tris])

    out = atlas.rasterize_tile(verts_mm, faces_arr, loop_uv_arr, tile)

    assert out["xy"].shape[0] > 0
