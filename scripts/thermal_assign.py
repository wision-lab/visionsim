"""Offline thermal material assignment for VisionSim scenes: dump -> assign -> report.

**Deliberately outside the ``visionsim`` package.** It needs an API key and makes a
network call, neither of which belongs in a simulator library. It imports from
``visionsim``; nothing in ``visionsim`` imports it. Run once per scene; the
committed sidecar is the artifact, and the render path reads only that JSON.

Subcommands::

    # 1. inventory a scene's materials (runs inside Blender)
    blender -b scene.blend --python scripts/thermal_assign.py -- dump --output scene.materials.json

    # 2. draft a sidecar with an LLM (host python; the only network call)
    python scripts/thermal_assign.py assign scene.materials.json --output scene.thermal.json

    # 3. render a review contact sheet, then correct the sidecar by hand
    python scripts/thermal_assign.py report scene.materials.json scene.thermal.json --output review.html

The LLM is provider-neutral: a plain OpenAI-compatible ``/chat/completions`` POST
over stdlib ``urllib``, so there is no SDK dependency and any compatible backend
works (OpenAI, a LiteLLM proxy, OpenRouter, vLLM, a local model). Configure with:

* ``LLM_API_KEY``  - required (``OPENAI_API_KEY`` also accepted)
* ``LLM_BASE_URL`` - default ``https://api.openai.com/v1``
* ``LLM_MODEL``    - default ``gpt-4o-mini``

Whatever the model returns passes through :func:`apply_guards` before it is
written, so a hallucinated preset, a skipped material, or a nonsense reservoir
temperature degrades to "unassigned, use globals" with a warning - never to a
silent wrong guess. The guards are the contract, not the server's schema support.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import urllib.request
import warnings
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visionsim.simulate.heatsim.materials import (
    MAX_DIRICHLET_K,
    MIN_DIRICHLET_K,
    PRESETS,
    preset_keys,
)

DEFAULT_LAMP_K = 345.0
"""Reservoir temperature for an emissive material with no model-supplied value."""

_ROLES = ("FEM_PARTICIPANT", "DIRICHLET_SOURCE")
_LOW_CONFIDENCE = 0.5


# ---------------------------------------------------------------------------
# dump
# ---------------------------------------------------------------------------


def _socket(node: Any, name: str) -> Any:
    inputs = getattr(node, "inputs", None)
    if inputs is None:
        return None
    try:
        value = getattr(inputs[name], "default_value", None)
    except (KeyError, TypeError, IndexError):
        return None
    if value is None:
        return None
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        return [float(component) for component in list(value)[:3]]
    return float(value)


def _describe(material: Any) -> dict[str, Any]:
    """Name, textures, a Principled-BSDF summary, and emission state.

    The BSDF numbers are context for the model, never a decision rule: these scenes
    are artist-authored for looks, so wood shows up with metallic=0.51 and glass
    with metallic=1.0. Emission is the one reliable signal.
    """
    bsdf: dict[str, Any] = {}
    emission = {"is_emissive": False, "strength": 0.0}
    textures: list[str] = []
    node_types: list[str] = []

    if getattr(material, "use_nodes", False):
        for node in getattr(getattr(material, "node_tree", None), "nodes", []):
            node_type = str(getattr(node, "type", ""))
            node_types.append(node_type)
            if node_type == "TEX_IMAGE":
                image = getattr(node, "image", None)
                name = getattr(image, "name", None) if image is not None else None
                if name:
                    textures.append(str(name))
            elif node_type == "BSDF_PRINCIPLED" and not bsdf:
                bsdf = {
                    "base_color": _socket(node, "Base Color"),
                    "metallic": _socket(node, "Metallic"),
                    "roughness": _socket(node, "Roughness"),
                    "transmission": _socket(node, "Transmission Weight"),
                }
                strength = _socket(node, "Emission Strength")
                color = _socket(node, "Emission Color")
                # Blender 4.x defaults Principled BSDF to Emission Strength=1.0 with a
                # BLACK emission color, i.e. zero actual emission. Strength alone would
                # spuriously flag most materials as heat sources on a modern scene. An
                # absent color socket means an older Blender where strength alone is
                # the right signal (these are older archviz assets).
                if strength and (color is None or max(color) > 1e-6):
                    emission = {"is_emissive": True, "strength": float(strength)}
            elif node_type == "EMISSION":
                emission = {"is_emissive": True, "strength": float(_socket(node, "Strength") or 0.0)}

    return {"name": str(material.name), "textures": sorted(set(textures)), "bsdf": bsdf,
            "emission": emission, "node_types": sorted(set(node_types))}


def collect_scene_materials(scene: Any, bpy_data: Any) -> dict[str, Any]:
    """Build the material inventory for *scene*.

    ``face_area_share`` is what makes review tractable: it ranks materials by how
    much surface they actually cover, so attention goes to the walls and floor
    before the 44-vertex logo.
    """
    objects = []
    for obj in getattr(scene, "objects", []):
        if getattr(obj, "type", None) != "MESH" or getattr(obj, "hide_render", False):
            continue
        visible = getattr(obj, "visible_get", None)
        if visible is not None and not visible():
            continue
        objects.append(obj)

    area_by_material: dict[str, float] = {}
    objects_by_material: dict[str, list[str]] = {}
    n_vertices = 0

    for obj in objects:
        mesh = getattr(obj, "data", None)
        if mesh is None:
            continue
        n_vertices += len(getattr(mesh, "vertices", []))
        slot_names: list[str | None] = []
        for slot in getattr(obj, "material_slots", []):
            material = getattr(slot, "material", None)
            slot_names.append(None if material is None else str(material.name))
        if not slot_names:
            continue

        for poly in getattr(mesh, "polygons", []):
            index = min(max(int(getattr(poly, "material_index", 0)), 0), len(slot_names) - 1)
            name = slot_names[index]
            if name is None:
                continue
            area_by_material[name] = area_by_material.get(name, 0.0) + float(getattr(poly, "area", 0.0))
            users = objects_by_material.setdefault(name, [])
            if obj.name not in users:
                users.append(obj.name)

    total = sum(area_by_material.values())
    entries = []
    for material in getattr(bpy_data, "materials", []):
        entry = _describe(material)
        entry["face_area_share"] = (area_by_material.get(entry["name"], 0.0) / total) if total > 0.0 else 0.0
        entry["objects"] = sorted(objects_by_material.get(entry["name"], []))
        entries.append(entry)
    entries.sort(key=lambda m: (-m["face_area_share"], m["name"]))

    return {"schema_version": 1, "n_mesh_objects": len(objects), "n_vertices": n_vertices, "materials": entries}


# ---------------------------------------------------------------------------
# assign
# ---------------------------------------------------------------------------


def build_prompt(dump: dict[str, Any]) -> str:
    """Render the per-scene prompt: the closed preset menu, then the material inventory."""
    menu = "\n".join(
        f"- {key}: alpha={p.alpha_mm2_s} mm2/s, rho={p.density_kg_m3} kg/m3, "
        f"c={p.specific_heat_J_kgK} J/kgK, emissivity={p.emissivity_ir} - {p.notes}"
        for key, p in sorted(PRESETS.items())
    )

    lines = []
    for material in dump["materials"]:
        parts = [f"name={material['name']!r}", f"area_share={material['face_area_share']:.4f}"]
        if material.get("objects"):
            parts.append(f"used_by={material['objects'][:6]}")
        if material.get("textures"):
            parts.append(f"textures={material['textures'][:6]}")
        if material.get("emission", {}).get("is_emissive"):
            parts.append(f"EMISSIVE(strength={material['emission']['strength']})")
        if material.get("bsdf"):
            parts.append(f"bsdf={material['bsdf']}")
        lines.append("- " + ", ".join(parts))

    return f"""You are assigning thermal material properties to the materials of a Blender interior scene, \
so a finite-element heat-transfer simulation can produce physically plausible thermal (LWIR) imagery.

Assign exactly one preset from this closed library to every material listed below:

{menu}

Guidance:
- The material NAME is your strongest signal. These scenes come from an international asset community, \
so names may be Spanish, Portuguese, Polish, German or English (e.g. "MADERA" = wood, "CRISTAL" = glass, \
"MUROS" = walls, "AGUA" = water, "hojas" = leaves, "CESPED" = grass).
- Texture filenames are the next best signal, especially for opaque names like "_87" or "archmodels66_033_1".
- The Principled-BSDF numbers are CONTEXT ONLY, never a rule. These scenes were authored for appearance, \
not physics: wood appears with metallic=0.51 and glass with metallic=1.0. Do not classify from them.
- Emissivity matters more than diffusivity for how a thermal frame looks. Choose "aluminium_polished" \
(emissivity 0.05) only for genuinely bare, shiny metal; choose "metal_painted" (emissivity 0.92) for \
anything coated, painted or powder-coated. When a metal object is ambiguous, prefer "metal_painted".
- Set role="DIRICHLET_SOURCE" with a plausible dirichlet_K (between {MIN_DIRICHLET_K:.0f} and \
{MAX_DIRICHLET_K:.0f} K) ONLY for things genuinely warmer than the room: emissive lamp elements \
(~340-360 K), an active stove or oven surface (~400-450 K), a radiator (~330-340 K), a running TV, \
monitor or laptop (~305-315 K), a fridge's warm compressor grille (~305 K), human skin (~307 K). \
Everything else is role="FEM_PARTICIPANT" with dirichlet_K=null.
- Do NOT mark something a source just because it belongs to a lamp or appliance: a lamp's metal stem and \
glass shade are FEM participants; only the emissive element is a source.
- Report a low confidence rather than inventing a specific answer. Confidence is 0.0 to 1.0.
- Give a one-line reason naming the evidence you used.

Materials in this scene (sorted by the fraction of surface area they cover):
{chr(10).join(lines)}

Respond with JSON only, in exactly this shape, with one entry for EVERY material above, \
using the exact name string given:

{{"assignments": [{{"material": "<exact name>", "preset": "<one of the library keys>", \
"role": "FEM_PARTICIPANT" or "DIRICHLET_SOURCE", "dirichlet_K": <number or null>, \
"confidence": <0.0-1.0>, "reason": "<one line>"}}]}}
"""


def parse_assignments_content(content: str) -> list[dict[str, Any]]:
    """Extract the ``assignments`` list from a chat-completion message body.

    Reasoning-capable models (GLM, several others) wrap their answer in a
    ```json ... ``` markdown fence even when asked for ``response_format:
    json_object``, so a bare ``json.loads`` fails on the leading fence. Strip an
    optional fence, then parse. Raises ``ValueError`` with the offending text if
    the payload still isn't the expected shape, so a bad endpoint fails loudly
    rather than silently yielding an empty draft.
    """
    text = content.strip()
    if text.startswith("```"):
        # Drop the opening fence line (``` or ```json) and the closing fence.
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model did not return JSON: {content[:200]!r}") from exc
    if not isinstance(parsed, dict) or "assignments" not in parsed:
        raise ValueError(f"model JSON lacks an 'assignments' array: {content[:200]!r}")
    return list(parsed["assignments"])


def request_assignments(dump: dict[str, Any]) -> list[dict[str, Any]]:  # pragma: no cover - network
    """POST the prompt to an OpenAI-compatible ``/chat/completions`` endpoint."""
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set LLM_API_KEY (any OpenAI-compatible endpoint).\n"
            "  Optional: LLM_BASE_URL (default https://api.openai.com/v1), LLM_MODEL (default gpt-4o-mini)"
        )
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    body = json.dumps({
        "model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        "messages": [{"role": "user", "content": build_prompt(dump)}],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return parse_assignments_content(payload["choices"][0]["message"]["content"])


def apply_guards(dump: dict[str, Any], raw_assignments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Deterministically correct the model's output. Warns on every correction.

    1. An assignment for a material not in the scene is dropped.
    2. A material the model skipped is added back as unassigned.
    3. An out-of-enum preset becomes ``None`` rather than a silent guess.
    4. An unknown role becomes ``FEM_PARTICIPANT``.
    5. A material with an EMISSION node is **forced** to ``DIRICHLET_SOURCE`` -
       the node is ground truth and outranks the model's judgement.
    6. A ``dirichlet_K`` outside the sane band is dropped; if the role was
       ``DIRICHLET_SOURCE`` (and the material is not itself emissive, which would force
       it back), the role is also degraded to ``FEM_PARTICIPANT`` -- otherwise the slot
       would stay pinned at the ambient temperature and silently act as a heat sink.
    """
    known = set(preset_keys())
    scene_materials = {m["name"]: m for m in dump["materials"]}
    out: dict[str, dict[str, Any]] = {}

    for item in raw_assignments:
        name = str(item.get("material", ""))
        if name not in scene_materials:
            warnings.warn(f"thermal assign: {name!r} is not a material in this scene; dropping it",
                          UserWarning, stacklevel=2)
            continue

        preset = str(item["preset"]) if item.get("preset") is not None else None
        if preset is not None and preset not in known:
            warnings.warn(f"thermal assign: {name!r} got unknown preset {preset!r}; marking unassigned",
                          UserWarning, stacklevel=2)
            preset = None

        role = str(item.get("role") or "FEM_PARTICIPANT").upper()
        if role not in _ROLES:
            warnings.warn(f"thermal assign: {name!r} got unknown role {item.get('role')!r}; "
                          "using FEM_PARTICIPANT", UserWarning, stacklevel=2)
            role = "FEM_PARTICIPANT"

        dirichlet_K = item.get("dirichlet_K")
        if dirichlet_K is not None:
            value = float(dirichlet_K)
            if MIN_DIRICHLET_K <= value <= MAX_DIRICHLET_K:
                dirichlet_K = value
            elif role == "DIRICHLET_SOURCE":
                # Dropping the temperature but leaving role=DIRICHLET_SOURCE would pin this
                # slot at the ambient/initial temperature (alpha=0, no incident flux) -- an
                # intended-but-nonsense-valued heat source silently becomes an ambient heat
                # SINK. Degrade to FEM_PARTICIPANT instead (mirrors materials.load_assignments'
                # identical guard). The EMISSION-node check below still overrides this back to
                # DIRICHLET_SOURCE (with DEFAULT_LAMP_K) when the node is ground truth.
                warnings.warn(f"thermal assign: {name!r} got dirichlet_K={value} outside "
                              f"[{MIN_DIRICHLET_K}, {MAX_DIRICHLET_K}] K; dropping it and degrading role "
                              "DIRICHLET_SOURCE -> FEM_PARTICIPANT (an out-of-band source pinned at "
                              "ambient would otherwise silently act as a heat sink)",
                              UserWarning, stacklevel=2)
                role = "FEM_PARTICIPANT"
                dirichlet_K = None
            else:
                warnings.warn(f"thermal assign: {name!r} got dirichlet_K={value} outside "
                              f"[{MIN_DIRICHLET_K}, {MAX_DIRICHLET_K}] K; dropping it", UserWarning, stacklevel=2)
                dirichlet_K = None

        # Ground truth beats judgement: an EMISSION node IS a heat source.
        if scene_materials[name].get("emission", {}).get("is_emissive"):
            if role != "DIRICHLET_SOURCE":
                warnings.warn(f"thermal assign: {name!r} has an EMISSION node; overriding model role "
                              f"{role!r} to DIRICHLET_SOURCE", UserWarning, stacklevel=2)
            role = "DIRICHLET_SOURCE"
            if dirichlet_K is None:
                dirichlet_K = DEFAULT_LAMP_K
        if role != "DIRICHLET_SOURCE":
            dirichlet_K = None

        out[name] = {"preset": preset, "role": role, "dirichlet_K": dirichlet_K,
                     "confidence": float(item.get("confidence", 0.0)), "reason": str(item.get("reason", ""))}

    for name, entry in scene_materials.items():
        if name not in out:
            # Ground truth beats absence: an EMISSION node IS a heat source even if
            # the model skipped the material entirely. That is a different, more
            # consequential correction than "unassigned", so it gets its own warning
            # rather than the generic "marking unassigned" one.
            if entry.get("emission", {}).get("is_emissive"):
                warnings.warn(f"thermal assign: no assignment returned for {name!r}, but it has an "
                              "EMISSION node; forcing DIRICHLET_SOURCE", UserWarning, stacklevel=2)
                role = "DIRICHLET_SOURCE"
                dirichlet_K = DEFAULT_LAMP_K
            else:
                warnings.warn(f"thermal assign: no assignment returned for {name!r}; marking unassigned",
                              UserWarning, stacklevel=2)
                role = "FEM_PARTICIPANT"
                dirichlet_K = None
            out[name] = {"preset": None, "role": role, "dirichlet_K": dirichlet_K,
                         "confidence": 0.0, "reason": "no assignment returned by the model"}
    return out


def to_sidecar(dump: dict[str, Any], guarded: dict[str, dict[str, Any]], scene_name: str) -> dict[str, Any]:
    """Wrap guarded assignments in the envelope ``materials.load_assignments`` reads."""
    ordered = [m["name"] for m in dump["materials"]]
    return {
        "schema_version": 1,
        "scene": scene_name,
        "generated_by": "scripts/thermal_assign.py - REVIEW AND CORRECT BY HAND BEFORE PUBLISHING",
        "defaults": {"preset": "plaster"},
        "materials": {name: guarded[name] for name in ordered if name in guarded},
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

_CSS = """
body { font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; background: #fff; }
h1 { font-size: 1.4rem; margin-bottom: 0.2rem; }
.sub { color: #666; margin-bottom: 1.5rem; font-size: 0.9rem; }
table { border-collapse: collapse; width: 100%; font-size: 0.85rem; }
th, td { border-bottom: 1px solid #e4e4e4; padding: 0.4rem 0.6rem; text-align: left; vertical-align: top; }
th { background: #fafafa; position: sticky; top: 0; font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.swatch { display: inline-block; width: 1rem; height: 1rem; border: 1px solid #bbb; vertical-align: -2px; }
.bar { display: inline-block; height: 0.55rem; background: #4a7fb5; vertical-align: 1px; }
.tag { font-size: 0.7rem; font-weight: 700; padding: 0.1rem 0.35rem; border-radius: 3px; }
.unassigned { background: #ffe0e0; color: #a00; }
.source { background: #ffeccc; color: #a35200; }
.lowconf { background: #fff6cc; color: #8a6d00; }
tr.flagged { background: #fffaf5; }
"""


def _swatch(base_color: Any) -> str:
    if not base_color or len(base_color) < 3:
        return '<span class="swatch" style="background:#fff"></span>'
    r, g, b = (max(0, min(255, round(float(x) ** (1 / 2.2) * 255))) for x in base_color[:3])
    return f'<span class="swatch" style="background:rgb({r},{g},{b})"></span>'


def build_report(dump: dict[str, Any], sidecar: dict[str, Any]) -> str:
    """Render the review contact sheet, ordered by the surface area each material covers."""
    entries = dict(sidecar.get("materials", {}))
    rows: list[str] = []
    n_unassigned = n_sources = n_low = 0

    for material in dump["materials"]:
        name = str(material["name"])
        entry = entries.get(name, {})
        preset_key = entry.get("preset")
        preset = PRESETS.get(preset_key) if preset_key else None
        confidence = float(entry.get("confidence", 0.0))
        share = float(material.get("face_area_share", 0.0))

        tags = []
        if preset is None:
            tags.append('<span class="tag unassigned">UNASSIGNED</span>')
            n_unassigned += 1
        if str(entry.get("role", "")) == "DIRICHLET_SOURCE":
            temp = entry.get("dirichlet_K")
            label = f"SOURCE {temp:.0f} K" if temp is not None else "SOURCE"
            tags.append(f'<span class="tag source">{html.escape(label)}</span>')
            n_sources += 1
        if preset is not None and confidence < _LOW_CONFIDENCE:
            tags.append('<span class="tag lowconf">LOW CONF</span>')
            n_low += 1

        # Precomputed because f-string expressions cannot contain backslashes or
        # reuse the outer quote character before Python 3.12.
        row_class = ' class="flagged"' if tags else ""
        swatch = _swatch((material.get("bsdf") or {}).get("base_color"))
        preset_cell = html.escape(preset_key) if preset_key else "&mdash;"
        alpha_cell = f"{preset.alpha_mm2_s:g}" if preset else "&mdash;"
        eps_cell = f"{preset.emissivity_ir:.2f}" if preset else "&mdash;"
        bar_px = max(1, int(share * 120))
        reason = html.escape(str(entry.get("reason", "")))
        objects = html.escape(", ".join(material.get("objects", [])[:4]))

        rows.append(
            f"<tr{row_class}>"
            f"<td>{swatch} <code>{html.escape(name)}</code></td>"
            f'<td class="num">{share:.1%}<br><span class="bar" style="width:{bar_px}px"></span></td>'
            f"<td>{preset_cell}</td>"
            f'<td class="num">{alpha_cell}</td>'
            f'<td class="num">{eps_cell}</td>'
            f'<td class="num">{confidence:.2f}</td>'
            f'<td>{" ".join(tags)}</td>'
            f"<td>{reason}</td>"
            f"<td>{objects}</td></tr>"
        )

    scene = html.escape(str(sidecar.get("scene", "?")))
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Thermal assignments &mdash; {scene}</title>
<style>{_CSS}</style></head><body>
<h1>Thermal material assignments &mdash; {scene}</h1>
<div class="sub">{len(dump['materials'])} materials &middot; {dump.get('n_mesh_objects', '?')} mesh objects
&middot; {dump.get('n_vertices', '?')} vertices &middot; <b>{n_unassigned}</b> unassigned &middot;
<b>{n_sources}</b> heat sources &middot; <b>{n_low}</b> low confidence &middot;
sorted by surface area covered</div>
<table><thead><tr>
<th>Material</th><th>Area</th><th>Preset</th><th>&alpha; mm&sup2;/s</th><th>&epsilon;</th>
<th>Conf.</th><th>Flags</th><th>Reason</th><th>Objects</th>
</tr></thead><tbody>
{chr(10).join(rows)}
</tbody></table></body></html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover - CLI
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(prog="thermal_assign", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_dump = sub.add_parser("dump", help="inventory a scene's materials (run inside Blender)")
    p_dump.add_argument("--output", type=Path, required=True)

    p_assign = sub.add_parser("assign", help="draft a sidecar with an LLM")
    p_assign.add_argument("dump", type=Path)
    p_assign.add_argument("--output", type=Path, required=True)

    p_report = sub.add_parser("report", help="render a review contact sheet")
    p_report.add_argument("dump", type=Path)
    p_report.add_argument("sidecar", type=Path)
    p_report.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.command == "dump":
        import bpy  # type: ignore

        result = collect_scene_materials(bpy.context.scene, bpy.data)
        result["scene"] = str(bpy.path.basename(bpy.data.filepath) or "")
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"[thermal_assign] wrote {args.output}: {len(result['materials'])} materials")

    elif args.command == "assign":
        dump = json.loads(args.dump.read_text(encoding="utf-8"))
        scene = str(dump.get("scene") or args.dump.name.replace(".materials.json", ".blend"))
        guarded = apply_guards(dump, request_assignments(dump))
        args.output.write_text(
            json.dumps(to_sidecar(dump, guarded, scene), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        unassigned = [n for n, e in guarded.items() if e["preset"] is None]
        sources = [n for n, e in guarded.items() if e["role"] == "DIRICHLET_SOURCE"]
        low = [n for n, e in guarded.items() if e["preset"] is not None and e["confidence"] < _LOW_CONFIDENCE]
        print(f"[thermal_assign] wrote {args.output}: {len(guarded)} materials")
        print(f"[thermal_assign]   unassigned: {len(unassigned)} {unassigned[:8]}")
        print(f"[thermal_assign]   heat sources: {len(sources)} {sources[:8]}")
        print(f"[thermal_assign]   low confidence: {len(low)} {low[:8]}")

    else:
        dump = json.loads(args.dump.read_text(encoding="utf-8"))
        sidecar = json.loads(args.sidecar.read_text(encoding="utf-8"))
        args.output.write_text(build_report(dump, sidecar), encoding="utf-8")
        print(f"[thermal_assign] wrote {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
