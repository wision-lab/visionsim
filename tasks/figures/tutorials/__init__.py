"""Figure collections for the ``tutorials`` documentation pages."""

from __future__ import annotations

from invoke import Collection

from . import dvs, hdr, light_passes, playblast, stereo, varying_focal_lengths

ns = Collection()
ns.add_task(dvs.build)
ns.add_task(hdr.build)
ns.add_task(light_passes.build)
ns.add_task(playblast.build)
ns.add_task(stereo.build)
ns.add_task(varying_focal_lengths.build)
