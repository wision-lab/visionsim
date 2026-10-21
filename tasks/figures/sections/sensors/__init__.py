"""Figure collections for the ``sections/sensors`` documentation pages."""

from __future__ import annotations

from invoke import Collection

from . import imu, itof, rgb

ns = Collection()
ns.add_task(imu.build)
ns.add_task(itof.build)
ns.add_task(rgb.build)
