from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def cache_key(blend_path: Path, solver_cfg: dict) -> str:
    """Stable cache key from the blend identity and solver-relevant config.

    Args:
        blend_path: Path to the source blend file.
        solver_cfg: Solver-relevant config values that affect the result.

    Returns:
        A short hex digest used as the cache subdirectory name.
    """
    blend_path = Path(blend_path)
    try:
        mtime = blend_path.stat().st_mtime_ns
    except OSError:
        mtime = 0
    payload = json.dumps({"p": str(blend_path), "m": mtime, "c": solver_cfg}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def write_temperatures(cache_root: Path, key: str, per_object: dict[str, np.ndarray], meta: dict) -> Path:
    """Write per-object temperature histories to ``<cache_root>/<key>/temperatures.npz``.

    Args:
        cache_root: Root directory for thermal caches.
        key: Cache key from :func:`cache_key`.
        per_object: Mapping of object name to a ``(timesteps, vertices)`` array.
        meta: JSON-serializable metadata stored alongside the arrays.

    Returns:
        The path to the written ``.npz`` archive.
    """
    out_dir = Path(cache_root) / key
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "temperatures.npz"
    save_data: dict[str, Any] = {"__meta__": np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)}
    save_data.update(per_object)
    np.savez_compressed(out, **save_data)
    return out


def read_temperatures(cache_root: Path, key: str) -> dict[str, np.ndarray] | None:
    """Read per-object temperature histories, or return ``None`` on a cache miss.

    Args:
        cache_root: Root directory for thermal caches.
        key: Cache key from :func:`cache_key`.

    Returns:
        Mapping of object name to its history array, or ``None`` if absent.
    """
    path = Path(cache_root) / key / "temperatures.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return {k: data[k] for k in data.files if k != "__meta__"}


def write_animated(cache_dir: Path, history: dict[str, np.ndarray], frames: list[int], meta: dict) -> Path:
    """Write per-frame animated temperature histories to ``<cache_dir>/frames.npz``.

    Args:
        cache_dir: Directory for this specific cache key (already resolved via :func:`cache_key`).
        history: Mapping of object name to a ``(n_frames, n_vertices)`` array.
        frames: Solved Blender frame numbers, aligned with each array's leading axis.
        meta: JSON-serializable metadata stored alongside the arrays.

    Returns:
        The path to the written ``.npz`` archive.
    """
    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "frames.npz"
    save_data: dict[str, Any] = {
        "__meta__": np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
        "__frames__": np.asarray(frames, dtype=np.int64),
    }
    save_data.update(history)
    np.savez_compressed(out, **save_data)
    return out


def read_animated(cache_dir: Path) -> tuple[dict[str, np.ndarray], list[int]] | None:
    """Read per-frame animated temperature histories, or return ``None`` on a cache miss.

    Args:
        cache_dir: Directory for this specific cache key (already resolved via :func:`cache_key`).

    Returns:
        A tuple of (mapping of object name to its ``(n_frames, n_vertices)`` history array,
        list of solved frame numbers), or ``None`` if absent.
    """
    path = Path(cache_dir) / "frames.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        frames = data["__frames__"].tolist()
        history = {k: data[k] for k in data.files if k not in ("__meta__", "__frames__")}
    return history, frames
