"""Shared helpers for per-page documentation figure tasks.

Every documentation page with generated figures gets a module under this package
whose path mirrors the page's path below ``docs/source``. Each page exposes its
nodes as ``NODES`` so :func:`all_task` can merge every page into one graph.

Figures and datasets are rebuilt only when the files a node declares are missing.
A page task walks up from its leaf figures to see what has to be re-provisioned
and then runs those nodes back down the graph.  ``--force`` marks every figure
stale, so its requirements are re-evaluated and re-run only when their own
outputs are missing or their check fails; intermediates already present are not
rebuilt.  ``--dry-run`` only reports what would happen.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from invoke import Exit, task
from rich.console import Console

if TYPE_CHECKING:
    from visionsim.simulate.blender import BlenderClient

console = Console()

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
STATIC = ROOT / "docs" / "source" / "_static"

NODE_TREE_SERVICE = ROOT / "tasks" / "figures" / "node_trees.py"
"""BlenderService subclass that exports node trees and lays those exports out, spawned by
:func:`export_node_trees` and :func:`arrange_node_trees`."""

NODE_TREES = (
    *(STATIC / "blender" / "nodes").glob("*.json"),
    ROOT / "examples" / "blender" / "mesh_camera.json",
)
"""The Tree Clipper exports the documentation draws its node graphs from: the compositor
trees the ground-truth figures show, and the geometry nodes group the IMU page shows."""

BLENDER_SERVICE = ROOT / "visionsim" / "simulate" / "blender.py"
"""The library's own BlenderService, for figures that drive it through the client API."""


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


def _command_recipe(name: str, *commands: str) -> Callable[[str | None, bool], None]:
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
# read off each command's ``--input-dir=``/``--depth-dir=``.
INTERMEDIATES: dict[str, Node] = {
    # Used in quick-start
    "ultramodern-gt": Node(
        name="ultramodern-gt",
        files=(CACHE / "quickstart" / "ultramodern-gt",),
        recipe=_command_recipe(
            "ultramodern-gt",
            "visionsim blender.render-animation ultramodern.blend quickstart/ultramodern-gt/"
            " --frame-start=150 --frame-end=350"
            " --config.keyframe-multiplier=1.0 --config.include-depths{executable}",
        ),
    ),
    "ultramodern-interp": Node(
        name="ultramodern-interp",
        files=(CACHE / "quickstart" / "ultramodern-interp",),
        requires=("ultramodern-gt",),
        recipe=_command_recipe(
            "ultramodern-interp",
            "visionsim interpolate.dataset --input-dir=quickstart/ultramodern-gt/frames"
            " --output-dir=quickstart/ultramodern-interp/ --n=32",
        ),
    ),
    "ultramodern-rgb25fps": Node(
        name="ultramodern-rgb25fps",
        files=(CACHE / "quickstart" / "ultramodern-rgb25fps",),
        requires=("ultramodern-interp",),
        recipe=_command_recipe(
            "ultramodern-rgb25fps",
            "visionsim emulate.rgb --input-dir=quickstart/ultramodern-interp/"
            " --output-dir=quickstart/ultramodern-rgb25fps/ --chunk-size=64 --readout-std=0{force}",
        ),
    ),
    "ultramodern-spc1600fps": Node(
        name="ultramodern-spc1600fps",
        files=(CACHE / "quickstart" / "ultramodern-spc1600fps" / "preview",),
        requires=("ultramodern-interp",),
        recipe=_command_recipe(
            "ultramodern-spc1600fps",
            "visionsim emulate.spad --input-dir=quickstart/ultramodern-interp/"
            " --output-dir=quickstart/ultramodern-spc1600fps/ --force",
            "visionsim ffmpeg.animate --input-dir=quickstart/ultramodern-spc1600fps/"
            " --outfile=quickstart/ultramodern-spc1600fps/preview.mp4 --fps=25 --force",
            "visionsim ffmpeg.extract --input-file=quickstart/ultramodern-spc1600fps/preview.mp4"
            " --output-dir=quickstart/ultramodern-spc1600fps/preview/",
        ),
    ),
    "ultramodern-dvs1600fps": Node(
        name="ultramodern-dvs1600fps",
        files=(CACHE / "quickstart" / "ultramodern-dvs1600fps" / "preview",),
        requires=("ultramodern-interp",),
        recipe=_command_recipe(
            "ultramodern-dvs1600fps",
            "visionsim emulate.events --input-dir=quickstart/ultramodern-interp/"
            " --output-dir=quickstart/ultramodern-dvs1600fps/ --fps=1600 --preview-step=1 --force",
        ),
    ),
    "ultramodern-itof": Node(
        name="ultramodern-itof",
        # ``preview/`` alone is created even when the run produces nothing, so the
        # node is only done once a tap directory exists inside it.
        files=(CACHE / "quickstart" / "ultramodern-itof" / "preview" / "tap_0",),
        requires=("ultramodern-gt",),
        recipe=_command_recipe(
            "ultramodern-itof",
            "visionsim emulate.itof --input-dir=quickstart/ultramodern-gt/frames"
            " --depth-dir=quickstart/ultramodern-gt/depths --output-dir=quickstart/ultramodern-itof/"
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

    Figures are the leaves of the graph and requirements point up the chain to them.
    A figure whose own output is present and complete is final: nothing is done for
    it and its chain is not walked, because its inputs were already consumed into it.
    Only when a figure is missing or the run is forced does the chain get climbed,
    and the climb stops at the first requirement that is itself satisfied -- from
    there down to the figure, every unsatisfied node is re-provisioned.

    ``--force`` treats every figure as stale, which is what re-provisions a chain
    whose outputs are all present.

    Args:
        nodes: The page's own nodes, typically its figures.
        by_name: The page's closed node set, used to look requirements up.
        force: Treat every figure as stale, even when the file it declares exists.

    Returns:
        The names of the nodes to attempt, in no particular order; callers run them
        in topological order.
    """
    stale: set[str] = set()
    seen: set[str] = set()

    def visit(node: Node) -> bool:
        """Return whether ``node`` has to be rebuilt, recording it and its stale requirements."""
        if node.name in seen:
            return node.name in stale
        seen.add(node.name)

        # A satisfied figure is the deliverable, so it and everything above it are left
        # alone; a missing requirement does not make it stale. --force is the way to
        # re-provision a chain whose outputs are all present.
        if node.is_figure and not force and _satisfied(node):
            return False

        # A node is rebuilt when it is a forced figure, or when its own output is
        # missing or incomplete. A satisfied requirement stops the climb: it is what
        # the stale node below it will be built from, so its own chain is not visited.
        if (node.is_figure and force) or not node.files or not _satisfied(node):
            stale.add(node.name)
            for req in node.requires:
                visit(by_name[req])
            return True
        return False

    for node in nodes:
        if node.is_figure:
            visit(node)
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
            if name not in selected:
                if node.is_figure:
                    console.print(f"{node.name}: up to date")
                continue
            if node.is_figure:
                console.print(f"would build {node.name} -> {node.files[0] if node.files else node.name}")
            else:
                console.print(f"would run {node.name}")
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


def encode_video(pattern: str, step: int, out: Path, size: int = 320, fps: int = 25, crf: int = 23) -> None:
    """Encode frames matching ``pattern`` (relative to ``cache/``) into ``out``.

    Rendered frames are sharded into numbered subfolders, hence the glob, and ``step``
    keeps every n-th frame so a 6400-frame render does not become a 6400-frame clip.
    The kept frames are re-timestamped to play at ``fps``, so a decimated sequence
    still runs at the speed the scene was authored at.

    H.264 rather than a gif: the same clip at the same resolution is an order of
    magnitude smaller, which matters most on the pages that play six of them together.

    Args:
        pattern: Glob, relative to ``cache/``, matching the source frames.
        step: Keep every ``step``-th frame.
        out: Destination path for the video.
        size: Output edge length in pixels.
        fps: Playback rate of the video.
        crf: Quality, lower being better. The default is transparent at preview
            resolutions; previews whose pixels are noise, like a photon or event
            camera's, cost several megabytes a clip at that setting and are the ones
            worth spending quality on.
    """
    filters = f"select='not(mod(n\\,{step}))',setpts=N/({fps}*TB),scale={size}:{size}:flags=lanczos"
    cmd = (
        f"ffmpeg -y -loglevel error -pattern_type glob -i '{pattern}'"
        f' -vf "{filters}" -r {fps} -fps_mode cfr'
        f" -an -c:v libx264 -crf {crf} -tune animation -pix_fmt yuv420p -movflags +faststart {out}"
    )
    subprocess.run(cmd, shell=True, check=True, cwd=CACHE)


@contextmanager
def blender_service(
    service: Path,
    executable: str | None = None,
    timeout: float = 300.0,
) -> Iterator[BlenderClient]:
    """Run a BlenderService subclass in its own blender and connect to it.

    The service is started by hand, which is the documented way of running a service
    subclass, and this checkout is prepended to ``PYTHONPATH`` so blender imports it
    rather than whichever copy its installation points at. ``--factory-startup`` keeps
    a run independent of the machine's blender preferences.

    Args:
        service: Script holding the ``BlenderService`` subclass to run.
        executable: Blender to run, ``None`` for ``$VSIM_BLENDER``, or ``blender`` on ``$PATH``.
        timeout: Seconds to wait for the spawned blender to accept connections.

    Yields:
        The connected :class:`BlenderClient <visionsim.simulate.blender.BlenderClient>`.
    """
    from ephemeral_port_reserve import reserve as port_reserve  # type: ignore

    from visionsim.simulate.blender import BlenderClient

    blender = executable or os.environ.get("VSIM_BLENDER") or "blender"
    pythonpath = os.pathsep.join(part for part in (str(ROOT), os.environ.get("PYTHONPATH")) if part)
    port = port_reserve("localhost")
    log = CACHE / "logs" / f"{service.stem}.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    # `--python-use-system-env` is what lets the PYTHONPATH below reach blender.
    command = [
        blender,
        "--background",
        "--factory-startup",
        "--enable-autoexec",
        "--python-use-system-env",
        "--python",
        str(service),
        "--",
        "--port",
        str(port),
    ]
    console.print(f"Starting {blender} on port {port}, logging to {log.relative_to(ROOT)}")
    with log.open("w") as handle:
        process = subprocess.Popen(
            command,
            env=os.environ | {"PYTHONPATH": pythonpath},
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        try:
            with BlenderClient(("localhost", port), timeout=timeout) as client:
                yield client
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def encode_webp(source: str | Path, out: Path, quality: int = 90) -> None:
    """Encode a rendered still into ``out`` as WebP.

    Stills that are meant to be looked at rather than measured do not have to be
    lossless: the same picture as a PNG is several times the bytes, and a page carries
    half a dozen of them. Alpha survives the encode, which a JPEG would flatten to
    black, and WebP needs nothing from a reader that these docs do not already require.

    Args:
        source: Image to encode.
        out: Destination path for the WebP file.
        quality: libwebp quality, higher being better.
    """
    cmd = f"ffmpeg -y -loglevel error -i {source} -c:v libwebp -quality {quality} {out}"
    subprocess.run(cmd, shell=True, check=True, cwd=CACHE)


def export_node_trees(
    executable: str | None,
    blend: Path,
    output_dir: Path,
    dataset: str,
    passes: tuple[tuple[str, str | None, dict[str, Any]], ...],
    timeout: float = 300.0,
) -> None:
    """Run :data:`NODE_TREE_SERVICE` in its own blender and export one tree per pass.

    Each pass is a ``(figure, method, kwargs)`` triple: the service is initialized on
    ``blend``, ``method`` (a call on the connected :class:`BlenderClient
    <visionsim.simulate.blender.BlenderClient>`) sets a ground-truth pass up on it, and
    the tree that leaves behind is written to ``output_dir`` as ``figure.json``. Nothing
    is rendered, so ``dataset`` only names the directory those passes would write to,
    which is what keeps their output paths out of the figures.

    The layout in the files this overwrites is not reproduced: what is written here is the
    tree as the service created it, which :func:`arrange_node_trees` lays out.

    Args:
        executable: Blender to run, ``None`` for ``$VSIM_BLENDER``, or ``blender`` on ``$PATH``.
        blend: Blend file whose compositor trees are exported.
        output_dir: Directory the JSON files are written to.
        dataset: Name of the directory the exported passes would render into.
        passes: ``(figure, client method, its arguments)`` per tree to write.
        timeout: Seconds to wait for the spawned blender to accept connections.
    """
    root = Path(tempfile.gettempdir()) / "visionsim-node-trees" / dataset
    with blender_service(NODE_TREE_SERVICE, executable, timeout) as client:
        for figure, method, kwargs in passes:
            client.initialize(blend, root)
            if method is not None:
                getattr(client, method)(**kwargs)
            print(client.export_tree(figure, str(output_dir)))


@task(name="arrange-node-trees")
def arrange_node_trees(c, executable=None):
    """Re-arrange the node tree exports the documentation draws, in place.

    One blender serves every file: :data:`NODE_TREE_SERVICE` is started once and each export
    is handed to it in turn. The blender it starts needs 5.0 or newer and has to carry Tree
    Clipper and the Node Arrange extension.
    """
    with blender_service(NODE_TREE_SERVICE, executable) as client:
        for path in sorted(NODE_TREES):
            print(client.arrange_tree(str(path)))


def page_task(nodes: tuple[Node, ...], name: str, doc: str):
    """Build a task over a node graph, under the given name.

    The task carries the graph's name so it shows up in ``inv --list`` as
    ``figures.<name>`` rather than as a ``build`` child of a collection named after
    the module. Callers add it to their package's collection with
    :meth:`invoke.Collection.add_task`.

    Args:
        nodes: The graph's top-level nodes, typically a page's figures.
        name: Task name, which should match the page's module name, or ``all`` for
            the graph merged from every page.
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
    return page_task(
        tuple(node for page in pages for node in page.NODES),
        "all",
        "Regenerate the figures of every documentation page.",
    )


def video_recipe(
    pattern: str,
    step: int,
    name: str,
    size: int = 320,
    fps: int = 25,
    crf: int = 23,
) -> Callable[[str | None, bool], None]:
    """Build a preview-video recipe writing ``STATIC/<name>.mp4``.

    Args:
        pattern: Glob, relative to ``cache/``, matching the source frames.
        step: Keep every ``step``-th frame.
        name: Output file path relative to ``docs/source/_static``, without the
            ``.mp4`` suffix.
        size: Output edge length in pixels; pages whose render is not the default
            320px square pass their own.
        fps: Playback rate of the video, for scenes rendered at a rate other than 25.
        crf: Quality, for previews whose pixels are noise rather than a picture.

    Returns:
        A recipe that ignores the blender executable it is handed.
    """

    def recipe(executable: str | None, force: bool = False) -> None:
        out = STATIC / f"{name}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        encode_video(pattern, step, out, size=size, fps=fps, crf=crf)

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
