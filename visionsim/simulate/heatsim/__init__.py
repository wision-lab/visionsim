from __future__ import annotations

from visionsim.simulate.heatsim import constants, laplacian, properties, solver

__all__ = ["constants", "laplacian", "properties", "solver"]


def register() -> None:
    """Register the per-object thermal material PropertyGroup on ``bpy.types.Object``."""
    properties.register()


def unregister() -> None:
    """Unregister the thermal material PropertyGroup."""
    properties.unregister()
