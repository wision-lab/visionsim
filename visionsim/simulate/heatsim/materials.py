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
computed post-``(1 - albedo)`` from Cycles bakes of the scene's real textures,
so a per-preset absorptivity would double-count.

Sidecars are authored offline as JSON and committed; this module reads them.
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
# Bulk properties are approximate constants; alpha = 1e6 * k / (rho * c).
# Emissivity (eps) is one gray-body scalar shared by the heat solve and thermal render.
# Source finishes, temperatures and bands are approximations for that model, not camera calibration.
# T below means total spectrum; LW means 8-14 um. Uncited emissivities remain surface estimates.
# [1] NIST TN 1681, section 5.2.1, equations 5.6a and 5.7a evaluated at 20 C.
#     https://nvlpubs.nist.gov/nistpubs/technicalnotes/nist.tn.1681.pdf
# [2] NETZSCH, PS: Polystyrene, Properties table.
#     https://analyzing-testing.netzsch.com/en-AU/polymers-netzsch-com/commodity-thermoplastics/ps-polystyrene/
# [3] NIST, physical properties of selected metals at 295 K.
#     https://www.nist.gov/ncnr/neutron-instruments/sample-environment/sample-mounting/reference-tables
# [4] NIST Chemistry WebBook, aluminium solid heat capacity near 298 K.
#     https://webbook.nist.gov/cgi/cbook.cgi?ID=C7429905&Mask=2&Table=on&Type=JANAFS
# [5] ASHRAE Fundamentals, chapter 26, table 1 (bulk properties at 24 C), table 9 (rocks).
#     https://handbook.ashrae.org/Handbooks/F25/SI/F25_Ch26/F25_Ch26_si.aspx
# [6] NIST Chemistry WebBook, copper solid heat capacity near 298 K.
#     https://webbook.nist.gov/cgi/cbook.cgi?ID=C7440508&Mask=2&Table=on&Type=JANAFS
# [7] Outokumpu Core range datasheet, table 7, Core 304/4301 at 20 C.
#     https://www.outokumpu.com/en/products/product-ranges/-/media/files/products/core/outokumpu-core-range-datasheet.pdf?modified=20251117111909&revision=025e9931-a1d5-4c8f-8ff5-f881d38916da
# [8] NIST GCR 15-917-36, section 5.4.6: estimated cast-iron pan specific heat.
#     https://nvlpubs.nist.gov/nistpubs/gcr/2015/NIST.GCR.15-917-36.pdf
# [9] Pilkington ATS-129, page 2: soda-lime silica float glass at 75 F (~24 C).
#     https://www.pilkington.com/-/media/pilkington/site-content/usa/window-manufacturers/technical-bulletins/ats-129---properties-of-soda-lime-silica-float-glass.pdf
# [10] NETZSCH, PVC-U: Polyvinyl Chloride (Unplasticized), Properties table.
#     https://analyzing-testing.netzsch.com/en/polymers-netzsch-com/commodity-thermoplastics/pvc-u-polyvinyl-chloride-unplasticized/
# [11] NISTIR 4973, Appendix E, pp. 89-90: cotton and cellulose specific heat.
#     https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=917009
# [12] NETZSCH, Q: Silicone rubber, Properties table (page URL uses an HNBR slug).
#     https://analyzing-testing.netzsch.com/en-US/polymers-netzsch-com/elastomers/hnbr-hydrogenated-acrylonitrile-butadiene-rubber-1
# [13] IAPWS SR6-08(2011), table 8: liquid water at 298.15 K and 0.1 MPa.
#     https://iapws.org/technical-guidance/release/LiquidWater.download
# [14] ASHRAE Refrigeration, chapter 19: composition-dependent food properties.
#     https://handbook.ashrae.org/Handbooks/R18/SI/r18_ch19/r18_ch19_si.aspx
# [15] Buyel et al. (2016), leaf thermal properties, Journal of Biotechnology 217, 100-108.
#     https://pubmed.ncbi.nlm.nih.gov/26608794/
# [16] Lidbeck and Syed (2017), experimental Li-ion cell characterization, abstract.
#     https://odr.chalmers.se/items/87a21437-25d3-4026-9e7c-f71fa964af37
# [17] IT'IS Tissue Properties Database, Skin averages (2024-06-04 data file).
#     https://itis.swiss/virtual-population/tissue-properties/database
# [18] FLIR, Emissivity tables, table 22.1 (surface, temperature and spectrum columns).
#     https://support.flir.com/docdownload/assets/web/27eh/en-us/T505002.xml.html
# [19] Optotherm, Emissivity Values (approximate surface-dependent ranges).
#     https://www.optotherm.com/slides/slide/emissivity-values-30
_PRESET_TABLE = (
    # [3, 4] Pure aluminium: k = 235 W/(m.K), c rounded to 900; alpha rounded to 97.
    # [19] eps: rough aluminium, estimate within 0.10-0.30.
    ("aluminium", 97.0, 2700.0, 900.0, 0.20, "Uncoated aluminium with a rough surface."),
    # [10] Choose rho = 1400, c = 880, k = 0.21 within the listed ranges; alpha ~ 0.17.
    # [18] eps: textured dull PVC flooring, LW at 70 C; surface proxy.
    ("pvc", 0.17, 1400.0, 880.0, 0.93, "Representative unplasticized PVC."),
    # [9] k = 0.937 W/(m.K); derived alpha ~ 0.43.
    # [9] eps: hemispherical value at 24 C, used for the gray-body approximation.
    ("glass", 0.43, 2500.0, 880.0, 0.84, "Uncoated soda-lime float glass. Opaque in LWIR."),
    # [3, 6] Pure copper: k = 400 W/(m.K), c rounded to 385; derived alpha ~ 116.
    # [18] eps: midpoint of oxidized copper's 0.6-0.7 range, T at 50 C.
    ("copper", 116.0, 8960.0, 385.0, 0.65, "Oxidised copper."),
    # [2] Use k = 0.16 W/(m.K), the midpoint of 0.14-0.18; derived alpha rounded to 0.12.
    ("polystyrene", 0.12, 1050.0, 1300.0, 0.90, "Solid polystyrene; excludes expanded foam."),
    # [5] Oak at 12% moisture: choose rho = 700 (660-750), k = 0.17 (0.16-0.18).
    # Use hardwood c = 1630; derived alpha ~ 0.15 approximates conduction across the grain.
    # [18] eps: planed oak, T at 20 C.
    ("wood", 0.15, 700.0, 1630.0, 0.90, "Representative oak at 12% moisture; scalar approximation."),
    # [1] Structural steel: c ~ 440 J/(kg.K), k ~ 53.3 W/(m.K); derived alpha ~ 15.4.
    # [18] eps: freshly rolled steel, T at 20 C.
    ("steel", 15.4, 7850.0, 440.0, 0.24, "Freshly rolled bare structural steel."),
    # [5] Fired clay, rho = 1920: choose k = 0.90 within 0.81-0.98; derived alpha ~ 0.59.
    # [18] eps: common red brick, T at 20 C.
    ("brick", 0.59, 1920.0, 800.0, 0.93, "Representative fired-clay brick."),
    # [5] Aggregate concrete, rho = 2240: midpoints k = 1.95 (1.3-2.6), c = 900 (800-1000).
    # [18] eps: concrete, T at 20 C.
    ("concrete", 0.97, 2240.0, 900.0, 0.92, "Representative sand/gravel or stone aggregate concrete."),
    # [5] Gypsum plaster: rho = 1120, k = 0.38; c = 1090 remains an estimate; alpha ~ 0.31.
    # [18] eps: rough plaster, T at 20 C.
    ("plaster", 0.31, 1120.0, 1090.0, 0.91, "Gypsum plaster; not cement render."),
    # [5] Filled bitumen: rho = 1900, k = 0.58; c = 920 remains an estimate; alpha ~ 0.33.
    # [19] eps: estimate within the asphalt range, 0.90-1.00.
    ("asphalt", 0.33, 1900.0, 920.0, 0.95, "Asphalt / bitumen with inert fill."),
    # Bulk estimates; c = 450 is close to the 460 J/(kg.K) estimate in [8].
    # Density and alpha are retained assumptions, not values from that reference.
    ("iron", 18.0, 7200.0, 450.0, 0.31, "Approximate cast-iron properties; grade unspecified."),
    # [16] c = 1100 measured for pouch/prismatic cells; density and alpha remain estimates.
    # eps estimates a coated casing; a bare metal casing needs a metal surface value.
    ("li_ion", 0.2, 2500.0, 1100.0, 0.88, "Approximate cell bulk; scalar model ignores directional conduction."),
    # [3, 4] Same pure-aluminium bulk properties as aluminium above.
    # [18] eps: polished aluminium, T at 100 C.
    ("aluminium_polished", 97.0, 2700.0, 900.0, 0.05, "Mirror-polished aluminium."),
    # [7] Use grade 304 bulk properties: k = 15 W/(m.K); derived alpha ~ 3.8.
    # [18] eps: buffed 18-8 stainless, T at 20 C.
    ("stainless_steel", 3.8, 7900.0, 500.0, 0.16, "Grade 304 stainless with a buffed finish."),
    # [1] Same structural-steel bulk properties as steel above.
    # [18] eps: paint proxy, LW range 0.92-0.94 at 70 C.
    ("metal_painted", 15.4, 7850.0, 440.0, 0.92, "Steel with nonmetallic paint; coating thermal mass neglected."),
    # Ceramic and porcelain bulk values are estimates without a verified source for these tuples.
    # [19] Ceramic eps: estimate within 0.90-0.95.
    ("ceramic", 0.6, 2300.0, 850.0, 0.93, "Approximate glazed ceramic: tile, sanitaryware, pottery."),
    # [18] Porcelain eps: glazed surface, T at 20 C.
    ("porcelain", 0.7, 2400.0, 840.0, 0.92, "Approximate vitrified porcelain: tableware, basins."),
    # Bulk estimates imply k ~ 2.85, within [5] table 9's marble range (1.2-4.3 W/(m.K)).
    # That range does not source the individual density, specific heat, or alpha values.
    ("marble", 1.2, 2700.0, 880.0, 0.94, "Approximate natural marble; other worktop materials may differ."),
    # [5] Gypsum/plaster board: k = 0.16 W/(m.K); derived alpha ~ 0.22.
    # [19] eps: estimate within the gypsum range, 0.85-0.95.
    ("drywall", 0.22, 640.0, 1150.0, 0.90, "Representative gypsum plasterboard."),
    # [11] Use solid-cotton c ~ 1300 as a proxy; fabric density and alpha remain estimates.
    # [19] eps: upper end of the close-weave textile range, 0.70-0.95.
    ("fabric", 0.09, 300.0, 1300.0, 0.95, "Approximate textile properties; weave and fiber dependent."),
    # [5] Estimate k ~ 0.045 W/(m.K), guided by a 19 mm carpet/pad with R = 0.42 m^2.K/W.
    # Density and specific heat remain estimates, not values from that assembly; alpha ~ 0.17.
    # [19] eps: estimate within the carpet range, 0.85-1.00.
    ("carpet", 0.17, 200.0, 1300.0, 0.90, "Approximate effective carpet properties."),
    # Leather bulk values remain estimates without a verified source for this tuple.
    # eps uses [19]'s 0.95-1.00 estimate; [18] reports 0.75-0.80 for tanned leather.
    ("leather", 0.11, 900.0, 1500.0, 0.95, "Approximate natural leather; synthetic leather may differ."),
    # [12] c = 1400 falls within the silicone range (1300-1500); use as an elastomer proxy.
    # Density and alpha remain generic estimates, not the silicone values in [12].
    # [18] eps: approximate the hard-rubber value of 0.95, T at 20 C.
    ("rubber", 0.10, 1200.0, 1400.0, 0.94, "Approximate solid elastomer; formulation dependent."),
    # [11] Use cellulose c = 1340 as a paper proxy; density and alpha remain estimates.
    # [18] eps: white bond paper, T at 20 C.
    ("paper", 0.13, 700.0, 1340.0, 0.93, "Approximate paper/card; coatings and porosity may differ."),
    # [13] Round rho to 997, c to 4180; k ~ 0.6065 W/(m.K) gives alpha ~ 0.146 at 25 C.
    # [18] eps: distilled water, T at 20 C.
    ("water", 0.146, 997.0, 4180.0, 0.96, "Liquid water at 25 C; surface FEM ignores convection."),
    # [15] c = 3000 is an estimate between species means of 2253 and 3661 J/(kg.K).
    # Density and alpha remain estimates; transpiration is omitted.
    ("foliage", 0.15, 700.0, 3000.0, 0.96, "Approximate live foliage; artificial plants need their actual material."),
    # Bulk estimates for unfrozen food; [14] gives composition models, not this specific tuple.
    # [19] eps: estimate within the food range, 0.85-1.00.
    ("food", 0.14, 1000.0, 3200.0, 0.95, "Approximate unfrozen food; composition dependent."),
    # [17] Mean rho = 1109, c ~ 3391, k ~ 0.372 W/(m.K); alpha ~ 0.099. Perfusion omitted.
    # [18] eps: human skin, T at 32 C.
    ("skin", 0.099, 1109.0, 3391.0, 0.98, "Human skin. Pair with role=DIRICHLET_SOURCE at ~307 K."),
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
            f"thermal assignment {where}: unknown preset {str(key)!r}; using fallback material properties",
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
            warnings.warn(
                f"thermal assignment {where}: unknown role {spec.get('role')!r}; treating as FEM_PARTICIPANT",
                UserWarning,
                stacklevel=2,
            )
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
                    "would otherwise silently act as a heat sink)",
                    UserWarning,
                    stacklevel=2,
                )
                role = "FEM_PARTICIPANT"
            else:
                warnings.warn(
                    f"thermal assignment {where}: dirichlet_K={value} outside "
                    f"[{MIN_DIRICHLET_K}, {MAX_DIRICHLET_K}] K; ignoring it",
                    UserWarning,
                    stacklevel=2,
                )

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
    """Per-slot scalar tables. Unassigned slots use the scene preset, then the object-level *fallback*."""
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
