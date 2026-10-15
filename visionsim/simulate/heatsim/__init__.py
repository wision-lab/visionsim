from __future__ import annotations

import importlib
from typing import Any

__all__ = ["laplacian", "properties", "solver"]


def __getattr__(name: str) -> Any:
    """Import a public submodule on first access.

    These stay lazy so that importing this package is cheap: ``BlenderService.initialize``
    imports it only to register the property group below, and an eager ``solver`` import
    would pull torch and scipy (about 1.1 s) into every Blender session, thermal or not.
    """
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def register() -> None:
    """Register the per-object thermal material PropertyGroup on ``bpy.types.Object``."""
    from visionsim.simulate.heatsim import properties

    properties.register()


def unregister() -> None:
    """Unregister the thermal material PropertyGroup."""
    from visionsim.simulate.heatsim import properties

    properties.unregister()
