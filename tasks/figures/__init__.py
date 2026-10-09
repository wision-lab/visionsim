"""Per-page documentation figure tasks.

Each page module exposes a single task as ``build``, already named after the
page, so ``inv --list`` shows the page itself and not a ``build`` child.
"""

from __future__ import annotations

from invoke import Collection

from . import quick_start, sections, tutorials

ns = Collection()
ns.add_task(quick_start.build)
ns.add_collection(Collection.from_module(sections))
ns.add_collection(Collection.from_module(tutorials))
