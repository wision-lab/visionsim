"""Temporary object material overrides that preserve authored slot assignments."""

from __future__ import annotations

from dataclasses import dataclass

import bpy  # type: ignore[import-not-found]


@dataclass
class MaterialSlotState:
    """Keep effective, mesh and hidden object assignments alive during a bake."""

    materials: list[bpy.types.Material | None]
    data_materials: list[bpy.types.Material | None]
    object_materials: list[bpy.types.Material | None]
    links: list[str]
    active_index: int
    bare_material_indices: list[int] | None


def snapshot_material_slots(obj: bpy.types.Object) -> MaterialSlotState:
    """Capture object overrides even when the slot currently uses its DATA link."""
    materials = [slot.material for slot in obj.material_slots]
    links = [slot.link for slot in obj.material_slots]
    object_materials = []
    for slot, link in zip(obj.material_slots, links):
        try:
            slot.link = "OBJECT"
            object_materials.append(slot.material)
        finally:
            slot.link = link
    return MaterialSlotState(
        materials=materials,
        data_materials=list(obj.data.materials),
        object_materials=object_materials,
        links=links,
        active_index=obj.active_material_index,
        # Even a slot-less mesh can carry authored, out-of-range face indices.
        bare_material_indices=[face.material_index for face in obj.data.polygons] if not obj.data.materials else None,
    )


def set_material_override(obj: bpy.types.Object, index: int, material: bpy.types.Material) -> None:
    """Use OBJECT links so a shared mesh's DATA materials remain untouched."""
    while len(obj.material_slots) <= index:
        obj.data.materials.append(None)
    slot = obj.material_slots[index]
    slot.link = "OBJECT"
    slot.material = material


def restore_material_slots(obj: bpy.types.Object, state: MaterialSlotState) -> None:
    """Restore in place; rebuilding existing slots resets polygon material indices."""
    if not state.data_materials:
        # Only an originally bare mesh may have its entire slot array removed.
        obj.data.materials.clear()
        if state.bare_material_indices is not None:
            obj.data.polygons.foreach_set("material_index", state.bare_material_indices)
        obj.active_material_index = state.active_index
        return
    while len(obj.data.materials) < len(state.data_materials):
        obj.data.materials.append(None)
    for index, material in enumerate(state.data_materials):
        obj.data.materials[index] = material
    while len(obj.data.materials) > len(state.data_materials):
        index = len(obj.data.materials) - 1
        obj.material_slots[index].link = "DATA"
        obj.data.materials.pop(index=index)
    for slot, link, material in zip(obj.material_slots, state.links, state.object_materials):
        slot.link = "OBJECT"
        slot.material = material
        slot.link = link
    obj.active_material_index = state.active_index
