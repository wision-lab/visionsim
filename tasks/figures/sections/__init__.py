"""Figure collections for the ``sections`` documentation pages."""

from __future__ import annotations

from invoke import Collection

from . import interpolation, sensors

ns = Collection()
ns.add_task(interpolation.build)
ns.add_collection(Collection.from_module(sensors))
