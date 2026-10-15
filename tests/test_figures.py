"""Tests for the documentation figure task graph, `tasks/figures/_page.py`.

Only the staleness walk is covered here; the recipes shell out to Blender and are
exercised by the docs figure tasks themselves. The interesting behaviour is which
nodes a run decides to attempt, because getting it wrong either skips a figure that
has to be rebuilt or re-renders a whole dataset for a figure that is already there.
"""

from __future__ import annotations

from pathlib import Path

from tasks.figures._page import Node, resolve_nodes, select_nodes


def _noop(executable: str | None, force: bool = False) -> None:
    """A recipe that does nothing; selection is decided before any recipe runs."""


def _written(tmp_path: Path, name: str) -> Path:
    """Create a non-empty file that counts as a produced output."""
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    return path


def _absent(tmp_path: Path, name: str) -> Path:
    """A path that is never created, so it counts as a missing output."""
    return tmp_path / name


def _intermediate(name: str, requires: tuple[str, ...], *files: Path) -> Node:
    return Node(name=name, requires=requires, files=tuple(files), recipe=_noop)


def _figure(name: str, requires: tuple[str, ...], *files: Path, check=None) -> Node:
    return Node(name=name, is_figure=True, requires=requires, files=tuple(files), recipe=_noop, check=check)


def _selected(nodes: tuple[Node, ...], force: bool = False) -> set[str]:
    return select_nodes(nodes, resolve_nodes(nodes), force)


def test_present_figure_is_left_alone(tmp_path):
    """A figure who exists is not rebuilt, and its chain is not walked, even when the
    dataset it was built from is gone."""
    nodes = (
        _figure("fig", ("mid",), _written(tmp_path, "fig.gif")),
        _intermediate("mid", ("root",), _absent(tmp_path, "mid")),
        _intermediate("root", (), _absent(tmp_path, "root")),
    )
    assert _selected(nodes) == set()


def test_missing_figure_rebuilds_only_itself(tmp_path):
    """A missing figure is rebuilt from the inputs that are still there."""
    nodes = (
        _figure("fig", ("mid",), _absent(tmp_path, "fig.gif")),
        _intermediate("mid", ("root",), _written(tmp_path, "mid")),
        _intermediate("root", (), _written(tmp_path, "root")),
    )
    assert _selected(nodes) == {"fig"}


def test_missing_figure_climbs_to_first_satisfied_requirement(tmp_path):
    """The climb stops at a requirement that is present, so the missing input above it is
    not re-provisioned."""
    nodes = (
        _figure("fig", ("mid",), _absent(tmp_path, "fig.gif")),
        _intermediate("mid", ("root",), _written(tmp_path, "mid")),
        _intermediate("root", (), _absent(tmp_path, "root")),
    )
    assert _selected(nodes) == {"fig"}


def test_missing_chain_is_rebuilt_all_the_way_up(tmp_path):
    """Every unsatisfied node between the figure and the first present input is rebuilt."""
    nodes = (
        _figure("fig", ("mid",), _absent(tmp_path, "fig.gif")),
        _intermediate("mid", ("root",), _absent(tmp_path, "mid")),
        _intermediate("root", (), _written(tmp_path, "root")),
    )
    assert _selected(nodes) == {"fig", "mid"}


def test_fully_missing_chain_selects_every_node(tmp_path):
    """With nothing present, the whole chain from the figure up is attempted."""
    nodes = (
        _figure("fig", ("mid",), _absent(tmp_path, "fig.gif")),
        _intermediate("mid", ("root",), _absent(tmp_path, "mid")),
        _intermediate("root", (), _absent(tmp_path, "root")),
    )
    assert _selected(nodes) == {"fig", "mid", "root"}


def test_force_reprovisions_present_figure_and_missing_inputs(tmp_path):
    """``--force`` rebuilds every figure, pulling in only the requirements that are absent."""
    present = tmp_path / "present"
    nodes = (
        _figure("fig", ("mid",), _written(present, "fig.gif")),
        _intermediate("mid", ("root",), _written(present, "mid")),
        _intermediate("root", (), _written(present, "root")),
    )
    assert _selected(nodes, force=True) == {"fig"}

    absent = tmp_path / "absent"
    nodes = (
        _figure("fig", ("mid",), _absent(absent, "fig.gif")),
        _intermediate("mid", ("root",), _absent(absent, "mid")),
        _intermediate("root", (), _written(absent, "root")),
    )
    assert _selected(nodes, force=True) == {"fig", "mid"}


def test_incomplete_output_is_stale(tmp_path):
    """A node whose ``check`` reports its output incomplete is re-run even though the file
    is there."""
    nodes = (
        _figure("fig", ("mid",), _written(tmp_path, "fig.gif"), check=lambda: False),
        _intermediate("mid", ("root",), _written(tmp_path, "mid")),
        _intermediate("root", (), _written(tmp_path, "root")),
    )
    assert _selected(nodes) == {"fig"}


def test_shared_requirement_is_not_rebuilt_for_a_present_figure(tmp_path):
    """A figure sharing an input with a stale figure stays as-is: only the stale figure and
    the inputs it needs run."""
    nodes = (
        _figure("stale", ("shared",), _absent(tmp_path, "stale.gif")),
        _figure("fresh", ("shared",), _written(tmp_path, "fresh.gif")),
        _intermediate("shared", ("root",), _written(tmp_path, "shared")),
        _intermediate("root", (), _absent(tmp_path, "root")),
    )
    assert _selected(nodes) == {"stale"}
