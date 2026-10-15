"""Per-page documentation figure tasks.

Each page module exposes a single task as ``build``, already named after the
page, so ``inv --list`` shows the page itself and not a ``build`` child. The
pages' nodes are also merged into ``figures.all``, which builds every page at
once so shared intermediates are provisioned only once.
"""

from __future__ import annotations

from invoke import Collection

from . import quick_start, sections, tutorials
from ._page import all_task
from .sections import interpolation
from .sections.sensors import itof
from .tutorials import light_passes, playblast, stereo

ns = Collection()
ns.add_task(all_task((quick_start, interpolation, itof, light_passes, playblast, stereo)))
ns.add_task(quick_start.build)
ns.add_collection(Collection.from_module(sections))
ns.add_collection(Collection.from_module(tutorials))
