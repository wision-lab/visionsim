"""Thermal material properties: preset library, per-scene assignment, per-vertex resolution.

Three layers, one job: turn a Blender object's **material slots** into the
per-vertex ``(N,)`` arrays the FEM solver already accepts.

1. :data:`PRESETS` - a fixed library of named thermal materials.
2. :func:`load_assignments` - parse a scene's committed sidecar, which maps
   Blender material names onto those presets.
3. :func:`resolve_vertex_materials` - walk ``mesh.polygons[].material_index``
   and produce the per-vertex arrays.

Units are SI-with-mm-diffusivity, matching :mod:`visionsim.simulate.heatsim.adapter`:
``alpha_mm2_s`` mm^2/s, ``density_kg_m3`` kg/m^3, ``specific_heat_J_kgK`` J/(kg.K),
``emissivity_ir`` dimensionless in [0, 1], temperatures in K.

There is deliberately **no** solar-absorptivity column: absorbed flux is already
computed post-``(1 - albedo)`` from a Cycles bake of the scene's real textures
(``irradiance_kernel``), so a per-preset absorptivity would double-count.

Nothing here calls out to a network or an LLM. Sidecars are authored offline by
``scripts/thermal_assign.py`` and committed; this module only reads them.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MIN_DIRICHLET_K = 280.0
"""Below this a 'heat source' is really a cold sink; almost always an authoring slip."""

MAX_DIRICHLET_K = 500.0
"""Above this we are out of interior-scene territory (an oven element, not a room)."""

_SCHEMA_VERSION = 1
_ROLES = ("FEM_PARTICIPANT", "DIRICHLET_SOURCE")

# Degenerate (zero-area) faces still have to vote, otherwise their vertices end up
# with no incident area and fall back to the object defaults - which reads as
# "unassigned" when the material is in fact known.
_MIN_FACE_AREA = 1.0e-12


# ---------------------------------------------------------------------------
# 1. Preset library
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThermalPreset:
    """One named thermal material. Frozen so the shared library cannot be mutated."""

    key: str
    alpha_mm2_s: float
    density_kg_m3: float
    specific_heat_J_kgK: float
    emissivity_ir: float
    notes: str


# key, alpha (mm^2/s), density (kg/m^3), specific heat (J/kg.K), IR emissivity, notes.
#
# The first 13 are seeded from the vendored ``constants.py`` dicts; their
# alpha/rho/c must stay in lockstep with them (guarded by a test) so the two
# cannot silently drift. Emissivities are literature LWIR values - constants.py
# has no emissivity column (its ``Absorptivity`` dict is solar, and dead code).
_PRESET_TABLE = (
    ("aluminium", 97.0, 2700.0, 978.0, 0.20, "Mill-finish / lightly oxidised aluminium."),
    ("pvc", 0.17, 1330.0, 880.0, 0.93, "Generic rigid plastic; also the global default."),
    ("glass", 0.34, 2500.0, 840.0, 0.92, "Soda-lime glass. Opaque in LWIR despite visible transparency."),
    ("copper", 111.0, 8960.0, 385.0, 0.15, "Oxidised copper."),
    ("polystyrene", 0.5, 1050.0, 1300.0, 0.90, "Expanded / rigid polystyrene."),
    ("wood", 0.082, 897.0, 2380.0, 0.90, "Generic hardwood."),
    ("steel", 4.2, 7930.0, 280.0, 0.28, "Bare rolled steel."),
    ("brick", 0.52, 2200.0, 800.0, 0.93, "Common fired brick."),
    ("concrete", 0.5, 2300.0, 880.0, 0.92, "Structural concrete."),
    ("plaster", 0.4, 1200.0, 1090.0, 0.91, "Plastered / rendered wall."),
    ("asphalt", 0.3, 2100.0, 920.0, 0.95, "Asphalt / bitumen."),
    ("iron", 18.0, 7200.0, 450.0, 0.31, "Cast iron."),
    ("li_ion", 0.2, 2500.0, 1100.0, 0.88, "Li-ion cell pack; rare in interior scenes."),
    ("aluminium_polished", 97.0, 2700.0, 978.0, 0.05, "Mirror-polished aluminium; the low emissivity is the point."),
    ("stainless_steel", 4.0, 7900.0, 500.0, 0.16, "Brushed stainless: appliance bodies, sinks, cookware."),
    ("metal_painted", 4.2, 7930.0, 280.0, 0.92, "Steel bulk, painted/powder-coated surface. Use for any coated metal."),
    ("ceramic", 0.6, 2300.0, 850.0, 0.93, "Glazed ceramic tile, sanitaryware, pottery."),
    ("porcelain", 0.7, 2400.0, 840.0, 0.92, "Vitrified porcelain: tableware, basins."),
    ("marble", 1.2, 2700.0, 880.0, 0.94, "Marble / granite / engineered stone worktops."),
    ("drywall", 0.31, 800.0, 1090.0, 0.90, "Gypsum plasterboard; the commonest interior wall/ceiling."),
    ("fabric", 0.09, 300.0, 1300.0, 0.95, "Upholstery, curtains, bedding, clothing."),
    ("carpet", 0.06, 200.0, 1300.0, 0.90, "Carpet and rugs. Lowest diffusivity in the library."),
    ("leather", 0.11, 900.0, 1500.0, 0.95, "Leather and leatherette upholstery."),
    ("rubber", 0.10, 1200.0, 1400.0, 0.94, "Rubber, silicone, gaskets, cable sheathing."),
    ("paper", 0.13, 700.0, 1340.0, 0.93, "Paper, card, books, posters, lampshade paper."),
    ("water", 0.143, 997.0, 4182.0, 0.96, "Liquid water. Surface FEM ignores convection, so approximate."),
    ("foliage", 0.15, 700.0, 3000.0, 0.96, "Live and artificial plants, leaves, grass."),
    ("food", 0.14, 1000.0, 3200.0, 0.95, "Food items; high water content dominates the bulk properties."),
    ("skin", 0.11, 1050.0, 3470.0, 0.98, "Human skin. Pair with role=DIRICHLET_SOURCE at ~307 K."),
)


def _build_library() -> dict[str, ThermalPreset]:
    library: dict[str, ThermalPreset] = {}
    for key, alpha, rho, c, eps, notes in _PRESET_TABLE:
        if alpha <= 0.0 or rho <= 0.0 or c <= 0.0:
            raise ValueError(f"preset {key!r}: alpha, density and specific heat must all be > 0")
        if not 0.0 <= eps <= 1.0:
            raise ValueError(f"preset {key!r}: emissivity_ir must lie in [0, 1]")
        if key in library:
            raise ValueError(f"duplicate preset key {key!r}")
        library[key] = ThermalPreset(key, alpha, rho, c, eps, notes)
    return library


PRESETS: dict[str, ThermalPreset] = _build_library()
"""The thermal preset library, keyed by preset name."""


def preset_keys() -> list[str]:
    """Sorted preset keys. This is the closed enum the offline assignment tool chooses from."""
    return sorted(PRESETS)


# ---------------------------------------------------------------------------
# 2. Per-scene assignment sidecar
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaterialEntry:
    """One material's assignment. ``preset is None`` means 'fall back to the globals'."""

    preset: ThermalPreset | None
    role: str
    dirichlet_K: float | None
    confidence: float
    reason: str


@dataclass(frozen=True)
class SceneAssignment:
    """A scene's sidecar, keyed by Blender material name."""

    scene: str
    digest: str
    default_preset: ThermalPreset | None
    materials: dict[str, MaterialEntry]

    def entry_for(self, material_name: str) -> MaterialEntry | None:
        """Return the entry for *material_name*, or ``None`` if the sidecar omits it."""
        return self.materials.get(material_name)


def _lookup_preset(key: Any, where: str) -> ThermalPreset | None:
    if key is None:
        return None
    preset = PRESETS.get(str(key))
    if preset is None:
        warnings.warn(
            f"thermal assignment {where}: unknown preset {str(key)!r}; falling back to the global defaults",
            UserWarning,
            stacklevel=3,
        )
    return preset


def load_assignments(path: Path) -> SceneAssignment:
    """Parse and validate the sidecar at *path*.

    Every recoverable inconsistency degrades to the global defaults **with a
    warning** rather than a silent guess: an unknown preset key, an unknown role,
    or an out-of-band Dirichlet temperature. Structural problems raise.
    """
    path = Path(path)
    raw_bytes = path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    raw = json.loads(raw_bytes.decode("utf-8"))

    version = int(raw.get("schema_version", 0))
    if version != _SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version {version} != expected {_SCHEMA_VERSION}")

    defaults_block = raw.get("defaults", {})
    if not isinstance(defaults_block, dict):
        raise ValueError(f"{path}: 'defaults' must be an object, got {type(defaults_block).__name__}")
    default_preset = _lookup_preset(defaults_block.get("preset"), f"{path.name}:defaults")

    materials_block = raw.get("materials", {})
    if not isinstance(materials_block, dict):
        raise ValueError(f"{path}: 'materials' must be an object, got {type(materials_block).__name__}")

    entries: dict[str, MaterialEntry] = {}
    for name, spec in materials_block.items():
        where = f"{path.name}:{name}"
        if not isinstance(spec, dict):
            raise ValueError(f"{path}: material {name!r} spec must be an object, got {type(spec).__name__}")
        preset = _lookup_preset(spec.get("preset"), where)

        role = str(spec.get("role") or "FEM_PARTICIPANT").upper()
        if role not in _ROLES:
            warnings.warn(f"thermal assignment {where}: unknown role {spec.get('role')!r}; "
                          "treating as FEM_PARTICIPANT", UserWarning, stacklevel=2)
            role = "FEM_PARTICIPANT"

        dirichlet_K: float | None = None
        if spec.get("dirichlet_K") is not None:
            try:
                value = float(spec["dirichlet_K"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}: material {name!r} has non-numeric dirichlet_K {spec['dirichlet_K']!r}"
                ) from exc
            if MIN_DIRICHLET_K <= value <= MAX_DIRICHLET_K:
                dirichlet_K = value
            elif role == "DIRICHLET_SOURCE":
                # An out-of-band K on a DIRICHLET_SOURCE must not silently keep the role: with
                # dirichlet_K dropped but role left as DIRICHLET_SOURCE, _slot_tables would still
                # mark the slot dirichlet and pin it at the *ambient* fallback temperature (alpha=0,
                # no incident flux, excluded from the radiation/convection boundary) -- an intended
                # 5000 K source silently becomes an ambient-temperature heat SINK. Degrading the role
                # to FEM_PARTICIPANT instead lets the slot behave as an ordinary participant, which is
                # the closer-to-harmless failure mode for "the authored temperature was nonsense".
                warnings.warn(
                    f"thermal assignment {where}: dirichlet_K={value} outside "
                    f"[{MIN_DIRICHLET_K}, {MAX_DIRICHLET_K}] K; dropping it and degrading role "
                    "DIRICHLET_SOURCE -> FEM_PARTICIPANT (an out-of-band source pinned at ambient "
                    "would otherwise silently act as a heat sink)", UserWarning, stacklevel=2,
                )
                role = "FEM_PARTICIPANT"
            else:
                warnings.warn(f"thermal assignment {where}: dirichlet_K={value} outside "
                              f"[{MIN_DIRICHLET_K}, {MAX_DIRICHLET_K}] K; ignoring it", UserWarning, stacklevel=2)

        entries[str(name)] = MaterialEntry(
            preset=preset,
            role=role,
            dirichlet_K=dirichlet_K,
            confidence=float(spec.get("confidence", 0.0)),
            reason=str(spec.get("reason") or ""),
        )

    return SceneAssignment(
        scene=str(raw.get("scene") or path.name),
        digest=digest,
        default_preset=default_preset,
        materials=entries,
    )


# ---------------------------------------------------------------------------
# 3. Slot -> per-vertex resolution
# ---------------------------------------------------------------------------


def _slot_tables(obj: Any, assignment: SceneAssignment, fallback: dict[str, Any]) -> dict[str, np.ndarray]:
    """Per-slot scalar tables. Unassigned slots inherit the object-level *fallback*."""
    names = []
    for slot in obj.material_slots:
        material = getattr(slot, "material", None)
        names.append(None if material is None else str(getattr(material, "name", "")))

    n = len(names)
    table: dict[str, np.ndarray] = {key: np.empty(n, dtype=np.float64) for key in ("t0", "alpha", "rho", "c", "eps")}
    table["is_dirichlet"] = np.zeros(n, dtype=bool)
    fallback_T = float(fallback["initial_temperature_K"])

    for i, name in enumerate(names):
        entry = assignment.entry_for(name) if name is not None else None
        preset = entry.preset if entry is not None else None
        if preset is None:
            preset = assignment.default_preset  # sidecar-wide fallback, then the object-level one

        if preset is None:
            table["alpha"][i] = float(fallback["thermal_diffusivity_mm2_s"])
            table["rho"][i] = float(fallback["density_kg_m3"])
            table["c"][i] = float(fallback["specific_heat_J_kgK"])
            table["eps"][i] = float(fallback["emissivity"])
        else:
            table["alpha"][i] = preset.alpha_mm2_s
            table["rho"][i] = preset.density_kg_m3
            table["c"][i] = preset.specific_heat_J_kgK
            table["eps"][i] = preset.emissivity_ir

        if entry is not None and entry.role == "DIRICHLET_SOURCE":
            table["is_dirichlet"][i] = True
            table["t0"][i] = float(entry.dirichlet_K) if entry.dirichlet_K is not None else fallback_T
        else:
            table["t0"][i] = fallback_T

    return table


def resolve_vertex_materials(
    obj: Any,
    assignment: SceneAssignment,
    fallback: dict[str, Any],
) -> dict[str, np.ndarray] | None:
    """Return per-vertex thermal arrays for *obj*, or ``None`` if slots cannot drive it.

    Blender stores materials on **faces**; the solver wants one value per
    **vertex**. Almost every vertex is interior to a single material region, so
    the conversion is a lookup. At a **seam** - a vertex touching faces of two
    different materials - the rule differs by the kind of quantity:

    * **Continuous** (alpha, rho, c, eps, T0) -> **area-weighted mean**. A
      wood-to-metal joint really is a material gradient, so a blend is closer to
      reality than an arbitrary hard jump.
    * **Categorical** (pinned or not, and at what temperature) -> **dominant**
      incident material by area. A vertex cannot be "60% pinned".

    Both rules share one accumulation pass building face area per
    ``(vertex, slot)``; only the final reduction differs (weighted mean vs
    ``argmax``). The cheaper alternative - "pinned if *any* touching face is a
    source" - is rejected because it leaks a lamp filament's temperature outward
    into the surrounding glass shade.

    Args:
        obj: A Blender ``MESH`` object (or duck-typed stand-in) exposing
            ``data.vertices``, ``data.polygons`` and ``material_slots``.
        assignment: The scene's parsed sidecar.
        fallback: The object-level resolution from ``adapter.resolve_material``,
            used for unassigned slots and untouched vertices - so an explicitly
            authored ``heat_sim_material`` still wins where the sidecar is silent.

    Returns:
        ``{"t0", "alpha", "rho", "c", "eps", "dirichlet_mask"}``, each ``(N,)``,
        or ``None`` when the object has no slots or no vertices (the caller then
        keeps its existing object-level path).

        ``t0`` is only meaningful where ``dirichlet_mask`` is ``True`` -- there it
        holds the vertex's exact reservoir temperature. Where ``dirichlet_mask`` is
        ``False`` it instead holds the same area-weighted mean as every other
        continuous field, which at a seam onto a Dirichlet slot blends in a slice of
        that reservoir's temperature even though the vertex itself is not pinned.
        Callers must supply the ambient initial temperature themselves for unpinned
        vertices rather than using this array's value wholesale (see
        ``adapter._combine``, which does exactly that).
    """
    mesh = getattr(obj, "data", None)
    if mesh is None:
        return None
    n_verts = len(getattr(mesh, "vertices", []))
    n_slots = len(getattr(obj, "material_slots", []))
    if n_verts == 0 or n_slots == 0:
        return None

    area: np.ndarray = np.zeros((n_verts, n_slots), dtype=np.float64)
    for poly in getattr(mesh, "polygons", []):
        slot = min(max(int(getattr(poly, "material_index", 0)), 0), n_slots - 1)
        face_area = max(float(getattr(poly, "area", 0.0)), _MIN_FACE_AREA)
        for vert in poly.vertices:
            index = int(vert)
            if 0 <= index < n_verts:
                area[index, slot] += face_area

    table = _slot_tables(obj, assignment, fallback)
    total = area.sum(axis=1)
    touched = total > 0.0

    fallback_scalar = {
        "t0": float(fallback["initial_temperature_K"]),
        "alpha": float(fallback["thermal_diffusivity_mm2_s"]),
        "rho": float(fallback["density_kg_m3"]),
        "c": float(fallback["specific_heat_J_kgK"]),
        "eps": float(fallback["emissivity"]),
    }
    out: dict[str, np.ndarray] = {}
    for key, default in fallback_scalar.items():
        values: np.ndarray = np.full(n_verts, default, dtype=np.float64)
        if np.any(touched):
            values[touched] = (area[touched] @ table[key]) / total[touched]
        out[key] = values

    dominant = area.argmax(axis=1)
    out["dirichlet_mask"] = table["is_dirichlet"][dominant] & touched

    # A pinned vertex holds its reservoir temperature exactly - an area-weighted
    # blend with a neighbouring participant would quietly cool the source.
    if np.any(out["dirichlet_mask"]):
        out["t0"][out["dirichlet_mask"]] = table["t0"][dominant][out["dirichlet_mask"]]

    out["eps"] = np.clip(out["eps"], 0.0, 1.0)
    return out


def resolve_face_materials(
    obj: Any,
    assignment: SceneAssignment,
    fallback: dict[str, Any],
    face_slots: np.ndarray,
) -> dict[str, np.ndarray] | None:
    """Return per-FACE thermal arrays for *obj*, or ``None`` if slots cannot drive it.

    The texel-sim analogue of :func:`resolve_vertex_materials`: a texel belongs to exactly
    one face, and a face belongs to exactly one material slot, so there is no seam to blend
    across - unlike the per-vertex path this is an exact lookup, never an area-weighted mean.

    Args:
        obj: A Blender ``MESH`` object (or duck-typed stand-in) exposing ``material_slots``.
        assignment: The scene's parsed sidecar.
        fallback: The object-level resolution from ``adapter.resolve_material``, used for
            unassigned slots - so an explicitly authored ``heat_sim_material`` still wins
            where the sidecar is silent.
        face_slots: ``(M,)`` int array of ``material_index`` per face. The caller is
            responsible for keeping face order consistent with whatever ``face`` indices it
            later uses to index into the result (e.g. ``atlas.rasterize_tile``'s ``face``
            output, which indexes into the same triangulated face array this was built
            from). An out-of-range index is clamped to the last slot, mirroring
            :func:`resolve_vertex_materials`'s ``material_index`` handling.

    Returns:
        ``{"t0", "alpha", "rho", "c", "eps", "dirichlet_mask"}``, each ``(M,)`` where
        ``M = len(face_slots)``, or ``None`` when the object has no material slots.

        ``t0`` here is always the face's exact assigned value (the reservoir temperature
        where ``dirichlet_mask`` is True, else the slot/object ambient) since there is no
        seam to blend across a whole face. A caller that wants the vertex path's "an
        unpinned face starts at ambient regardless of a neighboring Dirichlet slot"
        convention gets that for free here already - there is no neighbor contribution to
        exclude. Callers combining this with an object-level Dirichlet override should
        still apply that themselves (see ``adapter._combine``).
    """
    n_slots = len(getattr(obj, "material_slots", []))
    if n_slots == 0:
        return None

    table = _slot_tables(obj, assignment, fallback)
    slots = np.clip(np.asarray(face_slots, dtype=np.int64), 0, n_slots - 1)

    out: dict[str, np.ndarray] = {}
    for key in ("t0", "alpha", "rho", "c", "eps"):
        out[key] = table[key][slots]
    out["dirichlet_mask"] = table["is_dirichlet"][slots]
    out["eps"] = np.clip(out["eps"], 0.0, 1.0)
    return out
