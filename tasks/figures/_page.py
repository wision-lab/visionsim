"""Shared helpers for per-page documentation figure tasks.

Every documentation page with generated figures gets a module under this package
whose path mirrors the page's path below ``docs/source``. Each page exposes its
nodes as ``NODES`` so :func:`all_task` can merge every page into one graph.

Figures and datasets are rebuilt only when the files a node declares are missing.
A page task walks up from its leaf figures to see what has to be re-provisioned
and then runs those nodes back down the graph.  ``--force`` rebuilds every figure
and whatever it requires even when their files are present.  ``--dry-run`` only
reports what would happen.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from types import ModuleType

from invoke import Exit, task
from rich.console import Console

console = Console()

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
STATIC = ROOT / "docs" / "source" / "_static"


@dataclass(frozen=True)
class Node:
    """One node of a page's figure graph.

    A node either produces a deliverable figure or an intermediate artifact that
    other nodes require; the only difference is whether ``is_figure`` is set.
    """

    name: str
    """Unique within a page. Other nodes refer to this node by it through ``requires``, and it is how the node appears in log output."""

    recipe: Callable[..., None]
    """Performs the work, and is the only thing that ever runs. Takes the blender executable for the current run, or ``None`` to use whichever blender is on ``$PATH``, and whether the run is forced."""

    files: tuple[Path, ...] = ()
    """The paths the recipe writes. The node is stale when any of them is missing or empty, and each is checked after the run. Order matters only for log output, which reports the first."""

    requires: tuple[str, ...] = ()
    """Names of the nodes that must be satisfied first."""

    is_figure: bool = False
    """Marks a deliverable figure rather than an intermediate, which only selects log phrasing: figures report ``would build``/``built``, and intermediates report ``would run``/``provisioning``."""

    check: Callable[[], bool] | None = None
    """Optional extra staleness test, run in addition to the presence test on ``files``.

    Presence alone cannot tell a run that finished from one that died partway with its
    outputs already created: a render that was interrupted leaves a populated frame folder
    and a short metadata database behind, and neither the missing-``files`` test nor the
    non-empty test notices. A node may supply a ``check`` for such outputs, which is given
    no arguments and returns whether the output is genuinely complete, e.g. that a
    ``transforms.db`` holds one row per rendered file.

    Both tests must pass for a node to count as satisfied: any absent ``files`` entry
    makes it stale on its own, and a present-but-failing ``check`` does too. A node
    without a ``check`` behaves exactly as before, so only outputs whose completeness is
    not implied by existence need one."""


def _command_recipe(name: str, *commands: str) -> Callable[[str | None], None]:
    """Build a recipe that runs shell ``commands`` in order from ``cache/``.

    Steps are separate strings rather than one ``&&``-joined command, so a failing
    step is reported with its position and command, and the remaining steps are
    skipped. The provisioning lines are printed from inside the recipe rather than
    by :func:`run_node`, so every recipe stays a uniform callable and ``Node``
    needs no extra field.

    Args:
        name: Node name, used in the provisioning lines.
        *commands: Shell commands to run in order. Each may contain the
            ``{executable}`` placeholder, which is filled with
            ``--config.executable=<path>`` when the caller supplies a blender
            executable and with an empty string otherwise, and the ``{force}``
            placeholder, which is filled with ``--config.no-allow-skips`` when
            the node is being forced and with an empty string otherwise.

    Returns:
        A recipe taking the blender executable path and whether to force, or
        ``None`` for the latter.
    """

    def recipe(executable: str | None, force: bool = False) -> None:
        rendered = [
            c.format(
                executable=f" --config.executable={executable}" if executable else "",
                force=" --config.no-allow-skips" if force else "",
            )
            for c in commands
        ]
        for i, cmd in enumerate(rendered, start=1):
            step = f" ({i}/{len(rendered)})" if len(rendered) > 1 else ""
            console.print(f"[yellow]provisioning {name}{step}:[/yellow] {cmd}")
            subprocess.run(cmd, shell=True, check=True, cwd=CACHE)

    return recipe


# Intermediate nodes shared across pages: name -> node.  Each recipe lists the
# steps of ``examples/quickstart.sh`` or of the interpolation render/interpolate
# pair, one command per step so a failure names the exact step; ``requires`` is
# read off each command's ``--input-dir=``/``--depth-dir=``.  ``lego-interp`` has
# no owning figure and is reachable only through :attr:`Node.requires`, which is
# why it lives here.
INTERMEDIATES: dict[str, Node] = {
    # Used in quick-start
    "lego-gt": Node(
        name="lego-gt",
        files=(CACHE / "quickstart" / "lego-gt",),
        recipe=_command_recipe(
            "lego-gt",
            "visionsim blender.render-animation lego.blend quickstart/lego-gt/"
            " --config.keyframe-multiplier=5.0 --config.include-depths{executable}",
        ),
    ),
    "lego-interp": Node(
        name="lego-interp",
        files=(CACHE / "quickstart" / "lego-interp",),
        requires=("lego-gt",),
        recipe=_command_recipe(
            "lego-interp",
            "visionsim interpolate.dataset --input-dir=quickstart/lego-gt/frames"
            " --output-dir=quickstart/lego-interp/ --n=32",
        ),
    ),
    "lego-rgb25fps": Node(
        name="lego-rgb25fps",
        files=(CACHE / "quickstart" / "lego-rgb25fps",),
        requires=("lego-interp",),
        recipe=_command_recipe(
            "lego-rgb25fps",
            "visionsim emulate.rgb --input-dir=quickstart/lego-interp/"
            " --output-dir=quickstart/lego-rgb25fps/ --chunk-size=160 --readout-std=0",
        ),
    ),
    "lego-spc4kHz": Node(
        name="lego-spc4kHz",
        files=(CACHE / "quickstart" / "lego-spc4kHz" / "preview",),
        requires=("lego-interp",),
        recipe=_command_recipe(
            "lego-spc4kHz",
            "visionsim emulate.spad --input-dir=quickstart/lego-interp/ --output-dir=quickstart/lego-spc4kHz/ --force",
            "visionsim ffmpeg.animate --input-dir=quickstart/lego-spc4kHz/ --outfile=quickstart/lego-spc4kHz/preview.mp4 --fps=25 --force",
            "visionsim ffmpeg.extract --input-file=quickstart/lego-spc4kHz/preview.mp4 --output-dir=quickstart/lego-spc4kHz/preview/",
        ),
    ),
    "lego-dvs125fps": Node(
        name="lego-dvs125fps",
        files=(CACHE / "quickstart" / "lego-dvs125fps" / "preview",),
        requires=("lego-gt",),
        recipe=_command_recipe(
            "lego-dvs125fps",
            "visionsim emulate.events --input-dir=quickstart/lego-gt/frames"
            " --output-dir=quickstart/lego-dvs125fps/ --fps=125 --preview-step=1 --force",
        ),
    ),
    "lego-itof": Node(
        name="lego-itof",
        # ``preview/`` alone is created even when the run produces nothing, so the
        # node is only done once a tap directory exists inside it.
        files=(CACHE / "quickstart" / "lego-itof" / "preview" / "tap_0",),
        requires=("lego-gt",),
        recipe=_command_recipe(
            "lego-itof",
            "visionsim emulate.itof --input-dir=quickstart/lego-gt/frames"
            " --depth-dir=quickstart/lego-gt/depths --output-dir=quickstart/lego-itof/"
            " --scheme=convSin --n-captures=4 --freq=120e6 --preview --force",
        ),
    ),
    # Used for the playblast tutorial
    "playblast-full": Node(
        name="playblast-full",
        files=(CACHE / "playblast" / "full" / "frames",),
        recipe=_command_recipe(
            "playblast-full",
            "visionsim blender.render-animation loft.blend playblast/full/{executable}{force}",
        ),
    ),
    "playblast": Node(
        name="playblast",
        files=(CACHE / "playblast" / "playblast",),
        recipe=_command_recipe(
            "playblast",
            "visionsim blender.render-playblast loft.blend playblast/ --no-video{executable}{force}",
        ),
    ),
    # Used in the interpolation docs
    "lego-0025": Node(
        name="lego-0025",
        files=(CACHE / "interpolation" / "lego0025-interp",),
        recipe=_command_recipe(
            "lego-0025",
            "visionsim blender.render-animation lego.blend interpolation/lego-0025/"
            " --config.keyframe-multiplier=0.25 --config.width=320 --config.height=320{executable}",
            "visionsim interpolate.dataset --input-dir=interpolation/lego-0025/frames"
            " --output-dir=interpolation/lego0025-interp/ --n=64",
        ),
    ),
    "lego-0050": Node(
        name="lego-0050",
        files=(CACHE / "interpolation" / "lego0050-interp",),
        recipe=_command_recipe(
            "lego-0050",
            "visionsim blender.render-animation lego.blend interpolation/lego-0050/"
            " --config.keyframe-multiplier=0.5 --config.width=320 --config.height=320{executable}",
            "visionsim interpolate.dataset --input-dir=interpolation/lego-0050/frames"
            " --output-dir=interpolation/lego0050-interp/ --n=32",
        ),
    ),
    "lego-0100": Node(
        name="lego-0100",
        files=(CACHE / "interpolation" / "lego0100-interp",),
        recipe=_command_recipe(
            "lego-0100",
            "visionsim blender.render-animation lego.blend interpolation/lego-0100/"
            " --config.keyframe-multiplier=1.0 --config.width=320 --config.height=320{executable}",
            "visionsim interpolate.dataset --input-dir=interpolation/lego-0100/frames"
            " --output-dir=interpolation/lego0100-interp/ --n=16",
        ),
    ),
    "lego-0200": Node(
        name="lego-0200",
        files=(CACHE / "interpolation" / "lego0200-interp",),
        recipe=_command_recipe(
            "lego-0200",
            "visionsim blender.render-animation lego.blend interpolation/lego-0200/"
            " --config.keyframe-multiplier=2.0 --config.width=320 --config.height=320{executable}",
            "visionsim interpolate.dataset --input-dir=interpolation/lego-0200/frames"
            " --output-dir=interpolation/lego0200-interp/ --n=8",
        ),
    ),
}


def resolve_nodes(nodes: tuple[Node, ...]) -> dict[str, Node]:
    """Expand a page's nodes to the closed set the graph needs, and validate it.

    The graph is validated here, at import time, so a typo in ``requires`` or a
    cycle fails ``inv --list`` rather than a run half-way through. Unknown names
    are rejected before the ``TopologicalSorter`` is built, because
    ``TopologicalSorter.add`` auto-adds unknown predecessors without error and
    would silently accept a typo as an already-satisfied dependency.

    Args:
        nodes: The page's own nodes, typically its figures.

    Returns:
        Every name reachable from ``nodes``, mapped to its node.

    Raises:
        ValueError: If a name is defined twice, or if the requirements form a
            cycle.
        KeyError: If a name is in neither ``nodes`` nor ``INTERMEDIATES``.
    """
    resolved: dict[str, Node] = {}
    queue = list(nodes)
    while queue:
        node = queue.pop()
        if node.name in resolved:
            raise ValueError(f"duplicate node name {node.name!r}")
        resolved[node.name] = node
        for req in node.requires:
            if req in resolved:
                continue
            if req in INTERMEDIATES:
                queue.append(INTERMEDIATES[req])
            elif not any(pending.name == req for pending in queue):
                raise KeyError(f"{node.name} requires unknown node {req!r}")

    try:
        TopologicalSorter({n: set(nd.requires) for n, nd in resolved.items()}).prepare()
    except CycleError as exc:
        raise ValueError(f"node dependency cycle: {' -> '.join(map(str, exc.args[1]))}") from exc
    return resolved


def _present(path: Path) -> bool:
    """Return whether ``path`` exists and counts as produced.

    A directory only counts if it holds something; an empty directory is what a
    previous run leaves behind when it got as far as creating the output but
    produced nothing in it, so treating it as done would skip the work and let
    every dependent fail on a missing input instead.

    Args:
        path: Path an intermediate promised to produce.

    Returns:
        ``True`` if the path is a non-empty directory or any other existing path.
    """
    if not path.exists():
        return False
    return not path.is_dir() or any(path.iterdir())


def _satisfied(node: Node) -> bool:
    """Return whether every output a node declares is present and complete.

    This is the presence test on :attr:`Node.files` plus the node's own
    :attr:`Node.check`, both of which must pass. Short-circuiting on absence keeps a
    node that was never run from paying for a check that would only re-discover that
    its inputs are missing, and a check is only meaningful when the file it inspects
    is there to begin with.

    Args:
        node: The node whose outputs to test.

    Returns:
        ``True`` if all of the node's files are present and its check, if it has one,
        reports the output as complete.
    """
    if not all(_present(p) for p in node.files):
        return False
    return node.check is None or node.check()


def select_nodes(
    nodes: tuple[Node, ...],
    by_name: dict[str, Node],
    force: bool,
) -> set[str]:
    """Return the nodes a run has to attempt.

    A node is stale when any of its files is missing, or its ``check`` reports the
    output incomplete, or any node it requires is itself stale -- staleness
    propagates downstream, so a figure whose dataset was re-rendered is rebuilt even
    though its own file is present and complete. Requirements are evaluated first,
    which is what lets the walk stop at a satisfied leaf: nothing above it changes,
    so nothing above it is attempted.

    ``--force`` seeds the figures as stale, so it re-provisions their requirements
    too, as far up as the graph reaches.

    Args:
        nodes: The page's own nodes, typically its figures.
        by_name: The page's closed node set, used to look requirements up.
        force: Rebuild every figure and whatever it requires, even when the files
            they declare exist.

    Returns:
        The names of the nodes to attempt, in no particular order; callers run them
        in topological order.
    """
    stale: set[str] = set()
    seen: set[str] = set()

    def walk(node: Node) -> bool:
        """Return whether ``node`` is stale, recording it and its requirements."""
        if node.name in seen:
            return node.name in stale
        seen.add(node.name)

        # Requirements first: a node whose output exists but whose input is stale has
        # to be rebuilt, and a check that reports an output complete says nothing
        # about what that output was built from.
        upstream = any(walk(by_name[req]) for req in node.requires)
        if upstream or (node.is_figure and force) or not node.files or not _satisfied(node):
            stale.add(node.name)
        return node.name in stale

    for node in nodes:
        walk(node)
    return stale


def run_node(
    node: Node,
    ledger: dict[str, str],
    executable: str | None,
    selected: bool,
    force: bool,
) -> str | None:
    """Ensure ``node`` is satisfied, caching the outcome in ``ledger``.

    Every outcome is recorded in ``ledger`` so a dependent visited later can tell
    whether this node succeeded without re-running it. Requirements were already
    processed, because :func:`run_nodes` walks the graph in topological order, so
    this never recurses.

    Args:
        node: The node to satisfy.
        ledger: Maps node name to ``"ok"`` or a failure reason. Shared across the
            whole invocation, and read for each requirement.
        executable: Path to the blender binary, or ``None`` to use whichever one is
            on ``$PATH``.
        selected: Whether :func:`select_nodes` picked this node to be attempted.
        force: Whether the run is forced, which recipes that render pass on so
            Blender re-renders frames it would otherwise keep.

    Returns:
        ``None`` when the node is satisfied, otherwise the reason it was skipped.
    """
    if not selected:
        if node.is_figure:
            console.print(f"{node.name}: up to date")
        ledger[node.name] = "ok"
        return None

    if node.name in ledger:
        return None if ledger[node.name] == "ok" else ledger[node.name]

    for req in node.requires:
        if ledger.get(req) != "ok":
            ledger[node.name] = f"upstream {req} failed"
            return ledger[node.name]

    for p in node.files:
        p.parent.mkdir(parents=True, exist_ok=True)
    try:
        node.recipe(executable, force)
    except subprocess.CalledProcessError as exc:  # other nodes may still be fine
        ledger[node.name] = str(exc)
        return ledger[node.name]

    for p in node.files:
        if not _present(p):
            ledger[node.name] = f"missing output {p}"
            return ledger[node.name]
    if node.check is not None and not node.check():
        ledger[node.name] = f"incomplete output {node.files[0]}"
        return ledger[node.name]
    if node.is_figure:
        console.print(f"[green]built {node.name} -> {node.files[0]}[/green]")
    ledger[node.name] = "ok"
    return None


def run_nodes(nodes: tuple[Node, ...], force: bool, dry_run: bool, executable: str | None = None) -> int:
    """Provision the intermediates and build the figures of a page's graph.

    Nodes are attempted in topological order, so a requirement is always built
    before the node that needs it, and each node is attempted at most once. Which
    ones are attempted is decided the other way around, by walking up from the
    leaf figures (see :func:`select_nodes`), so only the lowest nodes necessary
    run. A node whose requirement failed is skipped with one line naming that
    requirement.

    Args:
        nodes: The page's own nodes, typically its figures.
        force: Rebuild figures even when they are not stale.
        dry_run: Report what would happen without running any recipe.
        executable: Path to the blender binary, or ``None`` to use whichever one
            is on ``$PATH``.

    Returns:
        ``1`` if any node was skipped or failed, otherwise ``0``.
    """
    by_name = resolve_nodes(nodes)
    selected = select_nodes(nodes, by_name, force)
    ledger: dict[str, str] = {}
    failed = False
    for name in TopologicalSorter({n: set(nd.requires) for n, nd in by_name.items()}).static_order():
        node = by_name[name]
        if dry_run:
            if node.is_figure:
                console.print(f"would build {node.name} -> {node.files[0] if node.files else node.name}")
            else:
                console.print(f"would run {node.name}")
            ledger[node.name] = "ok"
            continue
        reason = run_node(node, ledger, executable, name in selected, force)
        if reason is None:
            continue
        failed = True
        if reason.startswith("upstream"):
            console.print(f"{node.name}: skipping, {reason}")
        else:
            console.print(f"[red]{node.name}: skipping, {reason}[/red]")
    return 1 if failed else 0


def gifski(pattern: str, step: int, out: Path, width: int = 320, height: int = 320, fps: int = 25) -> None:
    """Animate frames matching ``pattern`` (relative to ``cache/``) into ``out``.

    Rendered frames are sharded into numbered subfolders, hence the ``$(ls ... |
    sed ...)`` form: ``step`` keeps every n-th frame so a 400-frame render does not
    become a 400-frame gif.

    Args:
        pattern: Glob, relative to ``cache/``, matching the source frames.
        step: Keep every ``step``-th frame.
        out: Destination path for the gif.
        width: Output width in pixels.
        height: Output height in pixels.
        fps: Playback rate of the gif.
    """
    cmd = f"gifski $(ls -1a {pattern} | sed -n '1~{step}p') --fps {fps} -o {out} --width={width} --height={height}"
    subprocess.run(cmd, shell=True, check=True, cwd=CACHE)


def page_task(nodes: tuple[Node, ...], name: str, doc: str):
    """Build a page's single task, under the page's own name.

    The task carries the page's name so it shows up in ``inv --list`` as
    ``figures.<page>`` rather than as a ``build`` child of a collection named after
    the module. Callers add it to their package's collection with
    :meth:`invoke.Collection.add_task`.

    Args:
        nodes: The page's own nodes, typically its figures.
        name: Task name, which should match the page's module name.
        doc: One-line summary shown by ``inv --list`` and ``inv --help``.

    Returns:
        The collection's default task.
    """
    resolve_nodes(nodes)  # reject unknown requirements and cycles at import time

    @task(name=name)
    def build(c, force=False, dry_run=False, executable=None):
        # A task's return value is only collected into the executor's results
        # dict, never turned into an exit status, so signal failure explicitly.
        if run_nodes(nodes, force, dry_run, executable):
            raise Exit(1)

    build.__doc__ = doc
    return build


def all_task(pages: tuple[ModuleType, ...]):
    """Build the single ``figures.all`` task over every page's figures.

    Each page module exposes its nodes as ``NODES``. They are merged into one
    graph so shared intermediates are provisioned once instead of once per page.

    Args:
        pages: The page modules whose ``NODES`` make up the combined graph.

    Returns:
        The collection's default task.
    """
    nodes = tuple(node for page in pages for node in page.NODES)
    resolve_nodes(nodes)  # reject unknown requirements and cycles at import time

    @task(name="all")
    def build(c, force=False, dry_run=False, executable=None):
        """Regenerate the figures of every documentation page."""
        if run_nodes(nodes, force, dry_run, executable):
            raise Exit(1)

    return build


def gif_recipe(pattern: str, step: int, name: str) -> Callable[[str | None], None]:
    """Build a gifski recipe writing ``STATIC/<name>.gif``.

    Args:
        pattern: Glob, relative to ``cache/``, matching the source frames.
        step: Keep every ``step``-th frame.
        name: Output file path relative to ``docs/source/_static``, without the
            ``.gif`` suffix.

    Returns:
        A recipe that ignores the blender executable it is handed.
    """

    def recipe(executable: str | None, force: bool = False) -> None:
        out = STATIC / f"{name}.gif"
        out.parent.mkdir(parents=True, exist_ok=True)
        gifski(pattern, step, out)

    return recipe


def cached(*parts: str) -> Path:
    """Build a path below ``cache/``.

    Args:
        *parts: Path components to append to ``cache/``.

    Returns:
        The joined path.
    """
    return CACHE.joinpath(*parts)


def dataset_check(root: Path, expected: int | None = None) -> Callable[[], bool]:
    """Build a :attr:`Node.check` for a rendered dataset directory.

    A render writes its frames and its ``transforms.db`` together, one metadata row
    per frame, so an interrupted run leaves the two consistent with each other while
    both fall short of the frame range that was asked for. Comparing the two against
    each other catches the case where they diverge; passing ``expected`` additionally
    catches the case where they agree on a truncated range.

    The rows are counted from the database rather than assumed, and the files are
    counted on disk under ``root``, which is where Blender shards them (``0000/``,
    ``0001/``, ...). Counts must match exactly: a dataset is only complete if every
    row has its file and every file has its row.

    Args:
        root: Dataset directory holding ``transforms.db`` and the sharded frames.
        expected: Number of frames the render was asked for. When given, the
            database must hold exactly this many rows, which is what distinguishes a
            finished render from one that stopped early and left a consistent but
            short dataset behind.

    Returns:
        A predicate returning whether the dataset at ``root`` is complete.
    """

    def check() -> bool:
        db = root / "transforms.db"
        if not db.exists():
            return False

        # Imported here rather than at module scope: this package is imported by
        # ``inv --list``, which should not have to pay for the dataset stack.
        import peewee
        from pydantic import ValidationError

        from visionsim.dataset import Metadata

        try:
            frames = len(Metadata.load(db).frames)
        # A truncated file, a table the schema did not get to create or a malformed
        # row are all what this check exists to catch; anything else is a real bug
        # and should surface rather than be reported as stale.
        except (peewee.DatabaseError, ValidationError):
            return False

        if expected is not None and frames != expected:
            return False
        return frames == sum(1 for p in root.rglob("*") if p.is_file() and p.suffix != ".db")

    return check


def static(name: str) -> Path:
    """Build a path to a file in ``docs/source/_static``.

    Args:
        name: File name within the static directory.

    Returns:
        The joined path.
    """
    return STATIC / name
