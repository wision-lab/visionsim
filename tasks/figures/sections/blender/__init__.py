"""Figure collections for the ``sections/blender`` documentation pages."""

from __future__ import annotations

from invoke import Collection

from . import groundtruths, keyframing

ns = Collection()
ns.add_task(groundtruths.build)
ns.add_task(keyframing.build)
