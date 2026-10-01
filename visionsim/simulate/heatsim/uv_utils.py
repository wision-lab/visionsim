"""Snapshot and restore Blender UV state around thermal bakes."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Optional

import bpy  # type: ignore[import-not-found]

UVState = tuple[Optional[str], Optional[str]]  # (active_uv_name, active_render_uv_name)
UVSnapshot = list[tuple["bpy.types.Object", UVState]]


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
