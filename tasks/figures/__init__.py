"""Per-page documentation figure tasks.

Each page module exposes a single task as ``build``, already named after the
page, so ``inv --list`` shows the page itself and not a ``build`` child. The
pages' nodes are also merged into ``figures.all``, which builds every page at
once so shared intermediates are provisioned only once.

``figures.arrange-node-trees`` is not a page's: it re-arranges the node tree exports the figures draw, in place.
"""

from __future__ import annotations

from invoke import Collection

from . import quick_start, sections, tutorials
from ._page import all_task, arrange_node_trees
from .sections import interpolation
from .sections.blender import groundtruths
from .sections.sensors import imu, itof, rgb
from .tutorials import dvs, hdr, light_passes, playblast, stereo, varying_focal_lengths
from .tutorials import itof as tutorial_itof

ns = Collection()
ns.add_task(
    all_task(
        (
            quick_start,
            interpolation,
            groundtruths,
            imu,
            itof,
            rgb,
            dvs,
            hdr,
            light_passes,
            playblast,
            stereo,
            varying_focal_lengths,
            tutorial_itof,
        )
    )
)
ns.add_task(quick_start.build)
ns.add_task(arrange_node_trees)
ns.add_collection(Collection.from_module(sections))
ns.add_collection(Collection.from_module(tutorials))
