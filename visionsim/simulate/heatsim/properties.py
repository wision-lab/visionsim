"""Per-object thermal material PropertyGroup for visionsim heat simulation.

Existing Blender object property names are retained so authored scenes load their
material overrides.
"""

from __future__ import annotations

import logging

from visionsim.simulate.config import ThermalConfig

try:
    import bpy  # type: ignore
    from bpy.props import BoolProperty, EnumProperty, FloatProperty, PointerProperty  # type: ignore

    _BPY_AVAILABLE = True
except ImportError:
    bpy = None  # type: ignore
    _BPY_AVAILABLE = False

# Only define the PropertyGroup when bpy is available (i.e. inside Blender).
if _BPY_AVAILABLE:

    class HeatSimObjectMaterialProperties(bpy.types.PropertyGroup):  # type: ignore[name-defined]
        """Per-object material + initial condition configuration for heat simulation.

        Values are stored in SI units:
            thermal_diffusivity_mm2_s: mm²/s
            density_kg_m3:             kg/m³
            specific_heat_J_kgK:       J/(kg·K)
            initial_temperature_K:     K
        """

        initial_temperature_K: FloatProperty(  # type: ignore[valid-type, assignment]
            name="Initial Temp (K)",
            description="Initial temperature for this object (used if no per-vertex attribute override is provided)",
            default=ThermalConfig.initial_temperature_K,
            min=0.0,
        )

        thermal_diffusivity_mm2_s: FloatProperty(  # type: ignore[valid-type, assignment]
            name="Thermal Diffusivity α (mm²/s)",
            description=(
                "Thermal diffusivity for this object in mm²/s (used if no per-vertex attribute override is provided). "
                "Note: your solver uses mm-units internally, so mm²/s is the natural unit here."
            ),
            default=ThermalConfig.thermal_diffusivity_mm2_s,
            min=0.0,
        )

        density_kg_m3: FloatProperty(  # type: ignore[valid-type, assignment]
            name="Density ρ (kg/m³)",
            description="Material density in kg/m^3 (used if no per-vertex attribute override is provided)",
            default=ThermalConfig.density_kg_m3,
            min=0.0,
        )

        specific_heat_J_kgK: FloatProperty(  # type: ignore[valid-type, assignment]
            name="Specific Heat c (J/kgK)",
            description="Specific heat capacity in J/(kg*K) (used if no per-vertex attribute override is provided)",
            default=ThermalConfig.specific_heat_J_kgK,
            min=0.0,
        )

        emissivity: FloatProperty(  # type: ignore[valid-type, assignment]
            name="Emissivity ε",
            description="Surface emissivity for thermal rendering and radiation calculations (0-1)",
            default=ThermalConfig.emissivity,
            min=0.0,
            max=1.0,
        )

        thermal_role: EnumProperty(  # type: ignore[valid-type, assignment]
            name="Thermal Role",
            description=(
                "How this object participates in the FEM heat sim. "
                "FEM Participant (default): full transient FEM; topology must be stable. "
                "Dirichlet Source: vertex temperatures are pinned to a constant value every step."
            ),
            items=[
                ("FEM_PARTICIPANT", "FEM Participant", "Full transient simulation; stable topology required"),
                ("DIRICHLET_SOURCE", "Dirichlet Source (constant T)", "Vertex temperatures pinned; topology may change per frame"),
            ],
            default="FEM_PARTICIPANT",
        )

        dirichlet_temperature_K: FloatProperty(  # type: ignore[valid-type, assignment]
            name="Dirichlet Temperature (K)",
            description=(
                "Constant temperature for this object's vertices when Thermal Role = "
                "Dirichlet Source. When set to 0 (or below), falls back to this "
                "object's Initial Temp."
            ),
            default=0.0,
            min=0.0,
            soft_min=0.0,
            soft_max=2000.0,
        )


def register() -> None:
    """Register the per-object thermal material PropertyGroup on ``bpy.types.Object``."""
    if not _BPY_AVAILABLE or bpy is None:
        return
    bpy.utils.register_class(HeatSimObjectMaterialProperties)  # type: ignore[name-defined]
    bpy.types.Object.heat_sim_material = PointerProperty(type=HeatSimObjectMaterialProperties)  # type: ignore[name-defined,assignment]
    bpy.types.Object.heat_simulation_enabled = BoolProperty(  # type: ignore[name-defined,assignment]
        name="Heat Simulation Enabled",
        description=(
            "If disabled, this object is excluded from FEM/Zombie solving, but it will still render using "
            "its initial/default temperature."
        ),
        default=True,
    )


def unregister() -> None:
    """Unregister the thermal material PropertyGroup."""
    if not _BPY_AVAILABLE or bpy is None:
        return
    try:
        del bpy.types.Object.heat_simulation_enabled  # type: ignore[name-defined]
    except Exception:
        logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)
    try:
        del bpy.types.Object.heat_sim_material  # type: ignore[name-defined]
    except Exception:
        logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)
    try:
        bpy.utils.unregister_class(HeatSimObjectMaterialProperties)  # type: ignore[name-defined]
    except Exception:
        logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)
