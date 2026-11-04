"""Validate thermal UV layouts and preserve authored Blender UV state."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import bpy  # type: ignore[import-not-found]
import numpy as np

UVState = tuple[str | None, str | None]  # (active_uv_name, active_render_uv_name)
UVSnapshot = list[tuple["bpy.types.Object", UVState]]


def _overlapping_uv_faces(uv: np.ndarray, polygons: np.ndarray, candidates: set[int]) -> set[int]:
    """Refine Blender's overlap candidates using the renderer's triangles.

    Shared edges can become tiny intersections after float32 UV packing. A
    separating-axis test rejects these coordinate-rounding contacts while
    retaining genuine overlaps, including ones smaller than a bake pixel.
    """
    selected = np.isin(polygons, list(candidates))
    triangles, faces = uv[selected], polygons[selected]
    if len(triangles) > 10_000:
        return candidates  # Keep large invalid layouts conservative without quadratic work.
    low, high = triangles.min(axis=1), triangles.max(axis=1)
    precision = 2 * np.finfo(np.float32).eps * max(float(np.abs(triangles).max()), float(np.finfo(np.float32).tiny))
    bad: set[int] = set()
    for i, triangle in enumerate(triangles):
        others = np.flatnonzero(
            (np.arange(len(triangles)) > i)
            & (np.minimum(high, high[i]) - np.maximum(low, low[i]) > precision).all(axis=1)
        )
        if not len(others):
            continue
        other = triangles[others]
        edges = np.concatenate(
            [np.broadcast_to(np.roll(triangle, -1, axis=0) - triangle, other.shape), np.roll(other, -1, axis=1) - other],
            axis=1,
        )
        axes = np.stack([-edges[..., 1], edges[..., 0]], axis=-1)
        a = np.einsum("vd,nad->nva", triangle, axes)
        b = np.einsum("nvd,nad->nva", other, axes)
        width = np.minimum(a.max(axis=1), b.max(axis=1)) - np.maximum(a.min(axis=1), b.min(axis=1))
        hits = others[(width > precision * np.linalg.norm(axes, axis=2)).all(axis=1)]
        if len(hits):
            bad.add(int(faces[i]))
            bad.update(faces[hits].tolist())
    return bad


def _invalid_bake_uv_faces(mesh, source_mesh=None) -> set[int]:
    """Check actual UV triangles, including overlaps smaller than one bake pixel.

    Called on the disposable, physically scaled projection mesh. Returns in
    Edit mode, with mesh faces selected and overlapping UVs selected.
    """
    import bmesh  # type: ignore[import-not-found]

    bpy.ops.object.mode_set(mode="OBJECT")
    source = source_mesh if source_mesh is not None else mesh
    source.calc_loop_triangles()
    count = len(source.loop_triangles)
    vertices = np.empty((len(mesh.vertices), 3))
    triangles: np.ndarray = np.empty((count, 3), dtype=np.int32)
    loops: np.ndarray = np.empty((count, 3), dtype=np.int32)
    polygons: np.ndarray = np.empty(count, dtype=np.int32)
    uv = np.empty((len(mesh.loops), 2))
    mesh.vertices.foreach_get("co", vertices.reshape(-1))
    source.loop_triangles.foreach_get("vertices", triangles.reshape(-1))
    source.loop_triangles.foreach_get("loops", loops.reshape(-1))
    source.loop_triangles.foreach_get("polygon_index", polygons)
    mesh.uv_layers.active.data.foreach_get("uv", uv.reshape(-1))
    if not np.isfinite(vertices).all():
        raise ValueError("Thermal geometry contains non-finite coordinates")

    p, t = vertices[triangles], uv[loops]
    a, b = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    u, v = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    physical_area = np.linalg.norm(np.cross(a, b), axis=1)
    uv_area = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
    # For a locally isotropic map this ratio is 2; a 4:1 stretch gives 4.25.
    # Work with the unnormalised Jacobian so collapsed UVs require no division.
    energy = np.sum((a * v[:, 1, None] - b * u[:, 1, None]) ** 2, axis=1)
    energy += np.sum((b * u[:, 0, None] - a * v[:, 0, None]) ** 2, axis=1)
    # Blender stores positions as float32. Faces thinner than the object's
    # coordinate precision (including collapsed bevels) have no reliable area.
    position_noise = 8 * np.finfo(np.float32).eps * np.linalg.norm(np.ptp(vertices, axis=0))
    longest_edge = np.maximum.reduce([np.linalg.norm(edge, axis=1) for edge in (a, b, a - b)])
    longest_uv_edge = np.maximum.reduce([np.linalg.norm(edge, axis=1) for edge in (u, v, u - v)])
    # Expanding an already thin physical face improves bake coverage. Reject
    # anisotropy when it makes the UV triangle thinner than the physical one.
    worsens_shape = longest_uv_edge**2 * physical_area > longest_edge**2 * uv_area
    real = physical_area > position_noise * longest_edge
    invalid = (
        ~np.isfinite(t).all(axis=(1, 2))
        | (t < -1e-6).any(axis=(1, 2))
        | (t > 1 + 1e-6).any(axis=(1, 2))
        | (uv_area == 0)
        | ((energy > 4.25 * physical_area * uv_area) & worsens_shape)
    )
    bad = set(polygons[real & invalid].tolist())
    winding = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    # The renderer's triangles within a polygon must not fold over each other.
    bad.update(np.intersect1d(polygons[real & (winding > 0)], polygons[real & (winding < 0)]).tolist())
    active: np.ndarray = np.zeros(len(mesh.polygons), dtype=bool)
    active[polygons[real & (uv_area > 0)]] = True

    bpy.ops.object.mode_set(mode="EDIT")
    if np.isfinite(uv).all() and active.any():
        settings = bpy.context.scene.tool_settings
        selection = settings.uv_select_mode, settings.uv_sticky_select_mode
        settings.uv_select_mode, settings.uv_sticky_select_mode = "FACE", "DISABLED"
        bpy.ops.uv.select_all(action="DESELECT")
        bm = bmesh.from_edit_mesh(mesh)
        # Collapsed bevels have no reliable physical area but may occupy UV
        # space. Keep the overlap query on the same real surfaces checked above.
        for face in bm.faces:
            face.select = bool(active[face.index])
            face.hide = not active[face.index]
        bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
        try:
            bpy.ops.uv.select_overlap(extend=False)
            bm = bmesh.from_edit_mesh(mesh)
            layer = bm.loops.layers.uv.active
            modern_selection = bpy.app.version >= (5, 2, 0)
            candidates = {
                face.index
                for face in bm.faces
                if active[face.index]
                and all(loop.uv_select_vert if modern_selection else loop[layer].select for loop in face.loops)
            }
            if candidates:
                bad.update(_overlapping_uv_faces(t[real & (uv_area > 0)], polygons[real & (uv_area > 0)], candidates))
        finally:
            settings.uv_select_mode, settings.uv_sticky_select_mode = selection
            for face in bm.faces:
                face.hide = False
                face.select = True
            bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
    return bad


def _unfold_bake_uv_faces(mesh, bad: set[int], source_mesh=None) -> None:
    """Flatten the renderer's exact triangles without changing polygon topology.

    Each remaining bad face gets its own island. In particular, a warped quad
    must unfold across Blender's render diagonal, rather than a different split
    chosen internally by the unwrap solver.
    """
    bpy.ops.object.mode_set(mode="OBJECT")
    source = source_mesh if source_mesh is not None else mesh
    source.calc_loop_triangles()
    triangle_loops: np.ndarray = np.empty((len(source.loop_triangles), 3), dtype=np.int32)
    polygons: np.ndarray = np.empty(len(source.loop_triangles), dtype=np.int32)
    source.loop_triangles.foreach_get("loops", triangle_loops.reshape(-1))
    source.loop_triangles.foreach_get("polygon_index", polygons)
    vertices = np.empty((len(mesh.vertices), 3))
    loop_vertices: np.ndarray = np.empty(len(source.loops), dtype=np.int32)
    mesh.vertices.foreach_get("co", vertices.reshape(-1))
    source.loops.foreach_get("vertex_index", loop_vertices)
    positions = vertices[loop_vertices]
    uv = np.empty((len(mesh.loops), 2))
    mesh.uv_layers.active.data.foreach_get("uv", uv.reshape(-1))
    p, t = positions[triangle_loops], uv[triangle_loops]
    world_area = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1).sum()
    a, b = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    uv_area = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum()
    scale = np.sqrt(uv_area / world_area) if np.isfinite(uv_area) and uv_area > 0 else 1.0
    side = int(np.ceil(np.sqrt(len(bad))))
    unfolded = []
    for index in sorted(bad):
        start = int(np.searchsorted(polygons, index))
        end = int(np.searchsorted(polygons, index + 1))
        pending = triangle_loops[start:end].tolist()
        # Start with the largest triangle so a degenerate corner cannot set the basis.
        pending.sort(
            key=lambda t: np.linalg.norm(np.cross(positions[t[1]] - positions[t[0]], positions[t[2]] - positions[t[0]]))
        )
        a, b, c = pending.pop()
        edge = positions[b] - positions[a]
        length = np.linalg.norm(edge)
        x = np.dot(positions[c] - positions[a], edge) / length
        height = np.linalg.norm(np.cross(edge, positions[c] - positions[a])) / length
        mapped = {a: np.zeros(2), b: np.array([length, 0]), c: np.array([x, height])}
        while pending:
            for triangle in pending:
                missing = [i for i, loop in enumerate(triangle) if loop not in mapped]
                if len(missing) > 1:
                    continue
                if missing:
                    i = missing[0]
                    a, b, c = triangle[(i + 1) % 3], triangle[(i + 2) % 3], triangle[i]
                    edge = positions[b] - positions[a]
                    length = np.linalg.norm(edge)
                    if length == 0:
                        mapped[c] = mapped[a].copy()
                    else:
                        direction = (mapped[b] - mapped[a]) / length
                        x = np.dot(positions[c] - positions[a], edge) / length
                        height = np.linalg.norm(np.cross(edge, positions[c] - positions[a])) / length
                        mapped[c] = mapped[a] + x * direction + height * np.array([-direction[1], direction[0]])
                pending.remove(triangle)
                break
            else:
                raise ValueError("Disconnected triangulation in thermal UV repair")
        corners = list(source.polygons[index].loop_indices)
        flat = np.array([mapped[loop] for loop in corners])
        flat -= flat.min(axis=0)
        unfolded.append((corners, flat * scale))
    spacing = 2 * max(np.ptp(flat, axis=0).max() for _, flat in unfolded)
    for island, (corners, flat) in enumerate(unfolded):
        # Match physical texel density and separate adjacent repaired faces.
        uv[corners] = flat + (2 + spacing * (island % side), spacing * (island // side))
    mesh.uv_layers.active.data.foreach_set("uv", uv.reshape(-1))
    bpy.ops.object.mode_set(mode="EDIT")


def repair_bake_uv(mesh, object_name: str, source_mesh=None) -> None:
    """Validate every generated layout; rechart only when it fails.

    Relax connected charts first so thin faces can share bake pixels with their
    neighbors. If that makes the layout worse, retain the original charts and
    unfold only failing faces. Never accept a partial/overlapping map.
    The caller owns the Edit-mode context and restores scene selection settings.
    """
    bad = _invalid_bake_uv_faces(mesh, source_mesh)
    if not bad:
        return
    bpy.ops.object.mode_set(mode="OBJECT")
    original: np.ndarray = np.empty((len(mesh.loops), 2), dtype=np.float32)
    mesh.uv_layers.active.data.foreach_get("uv", original.reshape(-1))
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.uv.select_all(action="SELECT")
    bpy.ops.uv.pin(clear=True)
    bpy.ops.mesh.mark_seam(clear=True)
    bpy.ops.uv.seams_from_islands(mark_seams=True, mark_sharp=False)
    bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.002, correct_aspect=False)
    bpy.ops.uv.average_islands_scale()
    bpy.ops.uv.pack_islands(
        rotate=True, margin_method="FRACTION", margin=0.002, shape_method="CONCAVE", merge_overlap=False
    )
    relaxed = _invalid_bake_uv_faces(mesh, source_mesh)
    if len(relaxed) <= len(bad):
        bad = relaxed
    else:
        bpy.ops.object.mode_set(mode="OBJECT")
        mesh.uv_layers.active.data.foreach_set("uv", original.reshape(-1))
        bpy.ops.object.mode_set(mode="EDIT")
    failed: set[int] = set()
    for attempt in range(4):
        if not bad:
            return
        failed.update(bad)
        if attempt == 3:
            raise ValueError(f"{object_name!r}: thermal UV repair left {len(bad)} invalid faces")
        logging.getLogger("rich").info(
            "thermal: repairing UVs for %s (%d faces, attempt %d)", object_name, len(bad), attempt + 1
        )
        _unfold_bake_uv_faces(mesh, failed, source_mesh)
        bpy.ops.uv.select_all(action="SELECT")
        bpy.ops.uv.pack_islands(
            rotate=True, margin_method="FRACTION", margin=0.002, shape_method="CONCAVE", merge_overlap=False
        )
        bad = _invalid_bake_uv_faces(mesh, source_mesh)


def get_uv_state(obj: bpy.types.Object) -> UVState:
    """Return (active_uv_name, active_render_uv_name) for a mesh object."""
    if obj is None or obj.type != "MESH":
        return (None, None)
    mesh = getattr(obj, "data", None)
    if mesh is None or not getattr(mesh, "uv_layers", None):
        return (None, None)

    active = mesh.uv_layers.active.name if mesh.uv_layers.active else None
    render = None
    for uv in mesh.uv_layers:
        if getattr(uv, "active_render", False):
            render = uv.name
            break
    if render is None:
        render = active
    return (active, render)


def _set_active_uv(mesh: bpy.types.Mesh, uv_name: str | None) -> None:
    if not mesh or not getattr(mesh, "uv_layers", None) or not uv_name:
        return
    if uv_name not in mesh.uv_layers:
        return
    try:
        mesh.uv_layers[uv_name].active = True
    except Exception:  # noqa: BLE001
        # Some Blender versions are finicky; fall back to active_index.
        try:
            mesh.uv_layers.active_index = list(mesh.uv_layers).index(mesh.uv_layers[uv_name])
        except Exception:
            logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)


def _set_render_uv(mesh: bpy.types.Mesh, uv_name: str | None) -> None:
    if not mesh or not getattr(mesh, "uv_layers", None) or not uv_name:
        return
    if uv_name not in mesh.uv_layers:
        return
    try:
        for uv in mesh.uv_layers:
            if getattr(uv, "active_render", None) is not None:
                uv.active_render = False
        mesh.uv_layers[uv_name].active_render = True
    except Exception:
        logging.getLogger(__name__).debug("Cannot set render UV state", exc_info=True)


def set_uv_state(obj: bpy.types.Object, state: UVState) -> None:
    """Set active/render UV for a mesh object, if the UV names exist."""
    if obj is None or obj.type != "MESH":
        return
    mesh = getattr(obj, "data", None)
    if mesh is None or not getattr(mesh, "uv_layers", None):
        return
    active_name, render_name = state
    if active_name:
        _set_active_uv(mesh, active_name)
    if render_name:
        _set_render_uv(mesh, render_name)


def snapshot_uv_states(objects: Iterable[bpy.types.Object]) -> UVSnapshot:
    """Capture UV state for objects (only MESH objects with UV layers)."""
    snap: UVSnapshot = []
    for obj in objects:
        if obj is None or obj.type != "MESH":
            continue
        mesh = getattr(obj, "data", None)
        if mesh is None or not getattr(mesh, "uv_layers", None) or len(mesh.uv_layers) == 0:
            continue
        snap.append((obj, get_uv_state(obj)))
    return snap


def restore_uv_states(snapshot: UVSnapshot) -> None:
    """Restore a previously-captured UV snapshot."""
    for obj, state in snapshot:
        try:
            set_uv_state(obj, state)
        except Exception:
            logging.getLogger(__name__).debug("Cannot restore UV state", exc_info=True)
