"""Inventory Blender materials and review a thermal assignment sidecar.

Run ``dump`` inside Blender; run ``template`` and ``check`` with Python. The
render path reads the resulting JSON directly and does not use this script.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return data


def _inventory_names(inventory: dict[str, Any]) -> list[str]:
    materials = inventory.get("materials")
    if not isinstance(materials, list):
        raise TypeError("inventory needs a materials list")
    names = []
    for item in materials:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise TypeError("each inventory material needs a name")
        names.append(item["name"])
    if len(names) != len(set(names)):
        raise ValueError("inventory contains duplicate material names")
    return names


def collect_materials(scene: Any, bpy_data: Any) -> dict[str, Any]:
    """Inventory renderable meshes, their material slots and useful visual clues."""
    area: dict[str, float] = {}
    objects: dict[str, set[str]] = {}
    materials: dict[str, Any] = {}

    for obj in scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        slots = [slot.material for slot in obj.material_slots]
        for material in slots:
            if material is None:
                continue
            materials[material.name] = material
            objects.setdefault(material.name, set()).add(obj.name)
        for face in obj.data.polygons:
            index = face.material_index
            if 0 <= index < len(slots) and slots[index] is not None:
                name = slots[index].name
                area[name] = area.get(name, 0.0) + face.area

    total_area = sum(area.values())
    entries = []
    for name, material in materials.items():
        tree = getattr(material, "node_tree", None)
        nodes = tree.nodes if tree else []
        textures = sorted({node.image.name for node in nodes if node.type == "TEX_IMAGE" and node.image})
        emission_nodes = [node.type for node in nodes if node.type == "EMISSION"]
        entries.append(
            {
                "name": name,
                "objects": sorted(objects[name]),
                "textures": textures,
                "diffuse_color": [round(float(x), 4) for x in material.diffuse_color[:3]],
                "emission_nodes": emission_nodes,
                "face_area_share": area.get(name, 0.0) / total_area if total_area else 0.0,
            }
        )
    entries.sort(key=lambda entry: (-entry["face_area_share"], entry["name"]))
    return {"schema_version": 1, "materials": entries}


def make_template(inventory: dict[str, Any]) -> dict[str, Any]:
    """Make a valid sidecar scaffold without guessing physical materials."""
    return {
        "schema_version": 1,
        "scene": str(inventory.get("scene", "")),
        "materials": {name: {"preset": None} for name in _inventory_names(inventory)},
    }


def check_assignments(inventory: dict[str, Any], sidecar: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Check that a sidecar names real materials and uses the renderer's vocabulary."""
    from visionsim.simulate.heatsim.materials import MAX_DIRICHLET_K, MIN_DIRICHLET_K, PRESETS

    names = set(_inventory_names(inventory))
    errors: list[str] = []
    warnings: list[str] = []
    if sidecar.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    unknown_top = set(sidecar) - {"schema_version", "scene", "defaults", "materials"}
    if unknown_top:
        errors.append(f"unknown top-level fields: {', '.join(sorted(unknown_top))}")

    defaults = sidecar.get("defaults", {})
    if not isinstance(defaults, dict):
        errors.append("defaults must be an object")
    elif set(defaults) - {"preset"}:
        errors.append("defaults only accepts preset")
    elif defaults.get("preset") is not None and (
        not isinstance(defaults["preset"], str) or defaults["preset"] not in PRESETS
    ):
        errors.append(f"unknown default preset: {defaults['preset']!r}")

    entries = sidecar.get("materials")
    if not isinstance(entries, dict):
        return errors + ["materials must be an object"], warnings
    for name in sorted(names - entries.keys()):
        errors.append(f"missing material: {name!r}")
    for name in sorted(entries.keys() - names):
        errors.append(f"material not in inventory: {name!r}")

    for name, entry in entries.items():
        if not isinstance(entry, dict):
            errors.append(f"{name!r}: entry must be an object")
            continue
        extra = set(entry) - {"preset", "role", "dirichlet_K", "confidence", "reason"}
        if extra:
            errors.append(f"{name!r}: unknown fields: {', '.join(sorted(extra))}")
        preset = entry.get("preset")
        if preset is None:
            warnings.append(f"{name!r}: no preset; renderer will use the sidecar or object defaults")
        elif not isinstance(preset, str) or preset not in PRESETS:
            errors.append(f"{name!r}: unknown preset {preset!r}")
        role = entry.get("role", "FEM_PARTICIPANT")
        if not isinstance(role, str) or role not in {"FEM_PARTICIPANT", "DIRICHLET_SOURCE"}:
            errors.append(f"{name!r}: unknown role {role!r}")
        source_K = entry.get("dirichlet_K")
        valid_temperature = (
            isinstance(source_K, (int, float)) and not isinstance(source_K, bool) and math.isfinite(source_K)
        )
        if source_K is not None and not valid_temperature:
            errors.append(f"{name!r}: dirichlet_K must be a finite number")
        if role == "DIRICHLET_SOURCE":
            if source_K is None:
                errors.append(f"{name!r}: source needs a numeric dirichlet_K")
            elif valid_temperature and not MIN_DIRICHLET_K <= source_K <= MAX_DIRICHLET_K:
                errors.append(f"{name!r}: dirichlet_K must be {MIN_DIRICHLET_K:g}–{MAX_DIRICHLET_K:g} K")
        elif valid_temperature:
            warnings.append(f"{name!r}: dirichlet_K is ignored for a FEM_PARTICIPANT")
        if "confidence" in entry:
            confidence = entry["confidence"]
            if (
                not isinstance(confidence, (int, float))
                or isinstance(confidence, bool)
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1
            ):
                errors.append(f"{name!r}: confidence must be a number from 0 to 1")
        if "reason" in entry and not isinstance(entry["reason"], str):
            errors.append(f"{name!r}: reason must be text")
    return errors, warnings


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    dump = commands.add_parser("dump", help="inventory the open Blender scene")
    dump.add_argument("--output", type=Path, required=True)
    template = commands.add_parser("template", help="make an unassigned sidecar scaffold")
    template.add_argument("inventory", type=Path)
    template.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("check", help="review material names and sidecar values")
    check.add_argument("inventory", type=Path)
    check.add_argument("sidecar", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.command == "dump":
            import bpy  # type: ignore

            result = collect_materials(bpy.context.scene, bpy.data)
            result["scene"] = bpy.path.basename(bpy.data.filepath)
            output = result
        elif args.command == "template":
            output = make_template(_load_json(args.inventory))
        else:
            errors, warnings = check_assignments(_load_json(args.inventory), _load_json(args.sidecar))
            for warning in warnings:
                print(f"warning: {warning}")
            for error in errors:
                print(f"error: {error}", file=sys.stderr)
            print(f"Checked assignments: {len(errors)} errors, {len(warnings)} warnings")
            return 1 if errors else 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
        return 0
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
