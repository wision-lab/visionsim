"""Thermal cache maintenance commands."""

from __future__ import annotations

from pathlib import Path


def prune_cache(cache_root: Path, keep: int = 0) -> None:
    """Remove completed transient thermal trajectories from a cache root.

    Args:
        cache_root: Directory containing ``transient-*`` trajectory directories.
        keep: Number of newest completed trajectories to retain.
    """
    from visionsim.cli import _log  # avoid circular import
    from visionsim.simulate.heatsim.cache import prune_transient_trajectories

    removed = prune_transient_trajectories(cache_root, keep=keep)
    noun = "directory" if len(removed) == 1 else "directories"
    _log.info("Removed %d transient %s from %s", len(removed), noun, cache_root)
