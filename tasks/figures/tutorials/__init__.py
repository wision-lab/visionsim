"""Figure collections for the ``tutorials`` documentation pages."""

from __future__ import annotations

from invoke import Collection

from . import playblast, stereo

ns = Collection()
ns.add_task(playblast.build)
ns.add_task(stereo.build)
