from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

CACHE_SCHEMA_VERSION = 2
_log = logging.getLogger("rich")


def cache_key(blend_path: Path, solver_cfg: dict, source_digest: str = "") -> str:
    """Stable cache key from the blend identity and solver-relevant config.

    Args:
        blend_path: Path to the source blend file.
        solver_cfg: Solver-relevant config values that affect the result.

    Returns:
        A short hex digest used as the cache subdirectory name.
    """
    blend_path = Path(blend_path)
    payload = json.dumps(
        {"schema": CACHE_SCHEMA_VERSION, "p": str(blend_path), "source": source_digest, "c": solver_cfg},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(blend_data: Any) -> str | None:
    """Identify an unmodified saved blend and its external image/library inputs.

    Unsaved edits and unavailable inputs cannot be identified safely, so callers
    must bake and solve again in those cases.
    """
    blend_path = Path(str(getattr(blend_data, "filepath", "")))
    if not blend_path.is_file() or bool(getattr(blend_data, "is_dirty", True)):
        return None
    digest = hashlib.sha256(file_digest(blend_path).encode())
    for collection in (getattr(blend_data, "images", ()), getattr(blend_data, "libraries", ())):
        for item in collection:
            if bool(getattr(item, "is_dirty", False)):
                return None
            if getattr(item, "packed_file", None) is not None:
                continue
            filepath = str(getattr(item, "filepath", ""))
            if not filepath:
                continue
            if getattr(item, "source", "FILE") not in {"FILE", "GENERATED"}:
                return None
            library = getattr(item, "library", None)
            library_path = str(getattr(library, "filepath", "")) if library is not None else ""
            if library_path.startswith("//"):
                library_path = str(blend_path.parent / library_path[2:])
            base_dir = Path(library_path).parent if library_path else blend_path.parent
            resolved = (base_dir / filepath[2:]) if filepath.startswith("//") else Path(filepath)
            if not resolved.is_file():
                return None
            digest.update(str(resolved.resolve()).encode())
            digest.update(file_digest(resolved).encode())
    return digest.hexdigest()


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
    meta = {**meta, "schema": CACHE_SCHEMA_VERSION, "objects": sorted(per_object),
            "timesteps": next(iter(per_object.values())).shape[0] if per_object else 0}
    save_data: dict[str, Any] = {"__meta__": np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)}
    save_data.update(per_object)
    with tempfile.NamedTemporaryFile(dir=out_dir, suffix=".npz", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        np.savez_compressed(temp_path, **save_data)
        os.replace(temp_path, out)
    finally:
        temp_path.unlink(missing_ok=True)
    return out


def read_temperatures(
    cache_root: Path, key: str, expected_counts: dict[str, int] | None = None,
) -> dict[str, np.ndarray] | None:
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

    def reject(reason: str) -> None:
        _log.warning("thermal cache %s is invalid (%s); recomputing", path, reason)

    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(data["__meta__"].tobytes())
            if meta["schema"] != CACHE_SCHEMA_VERSION:
                reject("schema mismatch")
                return None
            result = {k: data[k] for k in data.files if k != "__meta__"}
            if set(result) != set(meta["objects"]):
                reject("object membership mismatch")
                return None
            if expected_counts is not None and set(result) != set(expected_counts):
                reject("scene membership changed")
                return None
            for name, values in result.items():
                if values.ndim != 2 or values.shape[0] != meta["timesteps"]:
                    reject(f"invalid history shape for {name}")
                    return None
                if expected_counts is not None and values.shape[1] != expected_counts[name]:
                    reject(f"sample count changed for {name}")
                    return None
                if not np.isfinite(values).all():
                    reject(f"non-finite history for {name}")
                    return None
            return result
    except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
        reject(str(exc))
        return None
