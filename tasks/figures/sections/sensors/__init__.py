"""Figure collections for the ``sections/sensors`` documentation pages."""

from __future__ import annotations

from invoke import Collection

from . import itof

ns = Collection()
ns.add_task(itof.build)
