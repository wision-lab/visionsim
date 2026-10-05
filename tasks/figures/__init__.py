"""Per-page documentation figure tasks.

The tree mirrors ``docs/source``, so a page's figures live at the same dotted
address as the page itself, e.g. ``docs/source/sections/sensors/itof.rst`` is
regenerated with ``inv figures.sections.sensors.itof``.  Pages without generated
figures have no module here.

Each page module exposes a single task as ``build``, already named after the
page, so ``inv --list`` shows the page itself and not a ``build`` child.
"""

from __future__ import annotations

from invoke import Collection

from . import quick_start, sections

ns = Collection()
ns.add_task(quick_start.build)
ns.add_collection(Collection.from_module(sections))
