# Vendored from heat-sim-blender:addon/lib/temperature_io.py @ e5b4afe
"""External `.npz` storage for per-object FEM temperature histories.

The .blend keeps only the live-display `sim_temperature` vertex attribute
plus small metadata custom properties (heatsim_num_timesteps, heatsim_temp_min,
heatsim_temp_max, heatsim_run_mode, heatsim_data_uri, heatsim_data_abspath,
heatsim_data_key).

The full per-vertex history goes to
`<blend_dir>/<basename>.heatsim/latest/temperatures.npz`, keyed by object name,
with a sidecar `manifest.json`. The reader resolution order is:

    1. cached open NpzFile for the resolved archive path
    2. `obj["heatsim_data_uri"]` (Blender //-relative path) -> archive
    3. `obj["heatsim_data_abspath"]` -> archive
    4. legacy `obj["heatsim_temperature_data"]` (pre-migration .blend files)
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any, Optional

import bpy
import numpy as np

_log = logging.getLogger("rich")

ARCHIVE_FILENAME = "temperatures.npz"
MANIFEST_FILENAME = "manifest.json"
IRRADIANCE_CACHE_FILENAME = "irradiance_cache.npz"
BAKES_DIRNAME = "bakes"
BAKES_INDEX_FILENAME = "index.json"
KERNEL_BAKES_DIRNAME = "kernel_bakes"
KERNEL_ALBEDO_FILENAME = "albedo_cache.npz"
SKY_VISIBILITY_FILENAME = "sky_visibility_cache.npz"


# Cache of opened NpzFile handles, keyed by absolute archive path.
# Each entry: {"npz": np.lib.npyio.NpzFile, "path": str}
_CACHE: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _archive_dir_for_blend() -> Optional[str]:
    """Absolute path to `<blend>.heatsim/latest/`, or None if .blend is unsaved."""
    blend = bpy.data.filepath
    if not blend:
        return None
    blend_dir = os.path.dirname(blend)
    base = os.path.splitext(os.path.basename(blend))[0]
    return os.path.join(blend_dir, f"{base}.heatsim", "latest")


def _archive_dir_fallback(scene_name: str) -> str:
    """Fallback under bpy.app.tempdir when the .blend is unsaved."""
    return os.path.join(bpy.app.tempdir, f"heatsim_{scene_name}")


def _now_utc_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _stamp_object_props(obj: bpy.types.Object, archive_abspath: str, key: str) -> None:
    """Set the lookup custom properties used by the reader."""
    blend = bpy.data.filepath
    rel_uri = ""
    if blend:
        try:
            rel_uri = bpy.path.relpath(archive_abspath)
        except Exception:
            rel_uri = ""
    obj["heatsim_data_uri"] = rel_uri
    obj["heatsim_data_abspath"] = archive_abspath
    obj["heatsim_data_key"] = key
    # Drop the legacy custom property if it's still around.
    if "heatsim_temperature_data" in obj:
        try:
            del obj["heatsim_temperature_data"]
        except Exception:
            pass


def _resolve_archive_path(obj: bpy.types.Object) -> Optional[str]:
    """Try the relative URI first (handles 'blend was moved with .heatsim/'),
    then the stored absolute path. Returns None if neither resolves to a file
    that exists."""
    uri = obj.get("heatsim_data_uri", "")
    if uri:
        try:
            abspath = bpy.path.abspath(uri)
            if abspath and os.path.isfile(abspath):
                return abspath
        except Exception:
            pass
    abspath = obj.get("heatsim_data_abspath", "")
    if abspath and os.path.isfile(abspath):
        return abspath
    return None


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _open_archive(path: str):
    """Return a memmap-backed NpzFile, caching by absolute path."""
    cached = _CACHE.get(path)
    if cached is not None and cached.get("npz") is not None:
        return cached["npz"]
    npz = np.load(path, mmap_mode="r", allow_pickle=False)
    _CACHE[path] = {"npz": npz, "path": path}
    return npz


def _close_archive(path: str) -> None:
    entry = _CACHE.pop(path, None)
    if entry is None:
        return
    try:
        entry["npz"].close()
    except Exception:
        pass


def invalidate_cache() -> None:
    """Close all cached handles. Called after writes (so the next read sees
    the new file) and on `load_post` (so a stale handle doesn't survive a
    .blend reload)."""
    for path in list(_CACHE.keys()):
        _close_archive(path)


def delete_archive(scene: bpy.types.Scene) -> bool:
    """Drop the on-disk temperature archive for this .blend.

    Used by the Reset Simulation operator and by tab-switch auto-reset to
    abandon accumulated incremental state. Closes any cached mmap handle
    first so Windows can replace the file. Best-effort — missing archive
    isn't an error. Returns True iff a file was actually deleted.
    """
    archive_dir = _archive_dir_for_blend()
    if archive_dir is None:
        archive_dir = _archive_dir_fallback(getattr(scene, "name", "default"))
    archive_path = os.path.join(archive_dir, ARCHIVE_FILENAME)
    manifest_path = os.path.join(archive_dir, MANIFEST_FILENAME)
    _close_archive(archive_path)
    deleted = False
    for p in (archive_path, manifest_path):
        try:
            if os.path.isfile(p):
                os.remove(p)
                deleted = True
        except Exception as e:
            _log.debug(f"[HeatSim] WARNING: failed to delete {p}: {e}")
    return deleted


# ---------------------------------------------------------------------------
# Irradiance cache (sidecar) — used by static + incremental so click 2+
# can reuse the per-vertex flux from click 1's bake instead of paying the
# Cycles bake cost every click. The cache stores per-object (N_obj,) flux
# arrays keyed by obj.name. It is invalidated by Reset Simulation and by
# tab-switch auto-reset.
# ---------------------------------------------------------------------------


def _irradiance_cache_path(scene: bpy.types.Scene) -> str:
    """Absolute path to the sidecar irradiance cache for this .blend."""
    archive_dir = _archive_dir_for_blend()
    if archive_dir is None:
        archive_dir = _archive_dir_fallback(getattr(scene, "name", "default"))
    return os.path.join(archive_dir, IRRADIANCE_CACHE_FILENAME)


def write_irradiance_cache(
    scene: bpy.types.Scene,
    per_object_irradiance: dict[str, np.ndarray],
) -> Optional[str]:
    """Write per-object surface flux arrays (W/mm² per vertex) to the
    sidecar cache. Called once per fresh static-incremental run so
    subsequent Run clicks can skip the Cycles bake.

    `per_object_irradiance`: maps `obj.name` -> 1D float32 array of length
    `num_surface_vertices` for that object. Empty dict is a no-op.
    """
    if not per_object_irradiance:
        return None
    archive_dir = _archive_dir_for_blend()
    if archive_dir is None:
        archive_dir = _archive_dir_fallback(getattr(scene, "name", "default"))
    try:
        os.makedirs(archive_dir, exist_ok=True)
    except Exception as e:
        _log.debug(f"[HeatSim] WARNING: cannot create archive dir for irradiance cache: {e}")
        return None
    path = os.path.join(archive_dir, IRRADIANCE_CACHE_FILENAME)
    arrays = {
        name: np.asarray(arr, dtype=np.float32).reshape(-1)
        for name, arr in per_object_irradiance.items()
    }
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp_path, path)
        return path
    except Exception as e:
        _log.debug(f"[HeatSim] WARNING: failed to write irradiance cache {path}: {e}")
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return None


def read_irradiance_cache(scene: bpy.types.Scene) -> Optional[dict[str, np.ndarray]]:
    """Load the per-object flux arrays cached by ``write_irradiance_cache``.
    Returns ``None`` if no cache exists. Returns a fresh dict (not mmap-backed
    — these arrays are tiny, ~24 KB / 6 k verts / object)."""
    archive_dir = _archive_dir_for_blend()
    if archive_dir is None:
        archive_dir = _archive_dir_fallback(getattr(scene, "name", "default"))
    path = os.path.join(archive_dir, IRRADIANCE_CACHE_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as npz:
            return {k: np.array(npz[k], dtype=np.float32) for k in npz.files}
    except Exception as e:
        _log.debug(f"[HeatSim] WARNING: failed to read irradiance cache {path}: {e}")
        return None


def delete_irradiance_cache(scene: bpy.types.Scene) -> bool:
    """Drop the on-disk irradiance cache. Best-effort."""
    archive_dir = _archive_dir_for_blend()
    if archive_dir is None:
        archive_dir = _archive_dir_fallback(getattr(scene, "name", "default"))
    path = os.path.join(archive_dir, IRRADIANCE_CACHE_FILENAME)
    if not os.path.isfile(path):
        return False
    try:
        os.remove(path)
        return True
    except Exception as e:
        _log.debug(f"[HeatSim] WARNING: failed to delete irradiance cache {path}: {e}")
        return False


# ---------------------------------------------------------------------------
# Per-frame bake cache (Phase 2 disk persistence)
# ---------------------------------------------------------------------------
#
# Animate Scene full-run / animate-incremental bakes are persisted to
# `<blend>.heatsim/latest/bakes/` so they survive .blend save/reload and
# can be read back by:
#   - the next Run with `irradiance_bake_source == USE_CACHED` (skip Cycles)
#   - the View Flux follow-timeline handler (display the nearest-earlier
#     cached bake on every frame change)
#
# Layout:
#   bakes/index.json    — sorted list of cached frame numbers + per-bake
#                         metadata (bake_mode, per-object texture sizes)
#   bakes/<frame>.npz   — keys: "<obj_name>__irradiance",
#                         "<obj_name>__albedo" (float32 ndarray (H, W, 3))
#
# A per-frame npz is the unit of write/read, atomic via tmp+rename. A
# partial bake (ADAPTIVE re-bakes only some objects) writes only those
# objects' keys; reads return whatever keys are present.


def _bakes_dir(scene: bpy.types.Scene) -> str:
    """Absolute path to the bakes/ directory for this .blend (created on demand)."""
    archive_dir = _archive_dir_for_blend()
    if archive_dir is None:
        archive_dir = _archive_dir_fallback(getattr(scene, "name", "default"))
    return os.path.join(archive_dir, BAKES_DIRNAME)


def _bakes_index_path(scene: bpy.types.Scene) -> str:
    return os.path.join(_bakes_dir(scene), BAKES_INDEX_FILENAME)


def _bakes_frame_path(scene: bpy.types.Scene, frame: int) -> str:
    return os.path.join(_bakes_dir(scene), f"{int(frame)}.npz")


def read_bake_index(scene: bpy.types.Scene) -> Optional[dict]:
    """Load `bakes/index.json`. Returns None if the cache directory doesn't
    exist or the index is unreadable."""
    path = _bakes_index_path(scene)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        _log.debug(f"[HeatSim] WARNING: failed to read bake index {path}: {e}")
        return None


def _write_bake_index(scene: bpy.types.Scene, idx: dict) -> bool:
    """Atomic-write bakes/index.json."""
    bdir = _bakes_dir(scene)
    try:
        os.makedirs(bdir, exist_ok=True)
    except Exception as e:
        _log.debug(f"[HeatSim] WARNING: cannot create bakes dir {bdir}: {e}")
        return False
    path = _bakes_index_path(scene)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(idx, f, indent=2)
        os.replace(tmp_path, path)
        return True
    except Exception as e:
        _log.debug(f"[HeatSim] WARNING: failed to write bake index {path}: {e}")
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return False


def write_frame_bake(
    scene: bpy.types.Scene,
    frame: int,
    per_object_textures: dict[str, dict[str, Optional[np.ndarray]]],
    *,
    bake_mode: str,
    texture_sizes: dict[str, int],
) -> Optional[str]:
    """Write one frame's per-object bake to disk and update the index.

    `per_object_textures`: ``{obj.name: {"irradiance": (H,W,3) float32,
    "albedo": (H,W,3) float32 or None}, ...}``. Albedo None is allowed
    and the key is simply skipped.

    `bake_mode`: PER_OBJECT or SHARED (recorded in the index for
    read-time mismatch detection).

    `texture_sizes`: per-object pixel size of the bake (recorded in the
    index for read-time size-mismatch detection).

    Returns the absolute path of the written npz, or None on failure
    (disk-full, unsaved .blend with fallback unwritable, etc.).
    """
    if not per_object_textures:
        return None

    bdir = _bakes_dir(scene)
    try:
        os.makedirs(bdir, exist_ok=True)
    except Exception as e:
        _log.debug(f"[HeatSim] WARNING: cannot create bakes dir {bdir}: {e}")
        return None

    arrays: dict[str, np.ndarray] = {}
    for obj_name, payload in per_object_textures.items():
        irr = payload.get("irradiance")
        if irr is not None:
            arrays[f"{obj_name}__irradiance"] = np.asarray(irr, dtype=np.float32)
        alb = payload.get("albedo")
        if alb is not None:
            arrays[f"{obj_name}__albedo"] = np.asarray(alb, dtype=np.float32)
    if not arrays:
        return None

    frame_path = _bakes_frame_path(scene, frame)
    tmp_path = frame_path + ".tmp"
    try:
        with open(tmp_path, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp_path, frame_path)
    except Exception as e:
        _log.debug(f"[HeatSim] WARNING: failed to write frame bake {frame_path}: {e}")
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return None

    # Update / merge index.
    idx = read_bake_index(scene) or {}
    frames = sorted(set(int(f) for f in idx.get("frames", [])) | {int(frame)})
    objects = sorted(set(idx.get("objects", [])) | set(per_object_textures.keys()))
    sizes = dict(idx.get("texture_sizes", {}))
    sizes.update({k: int(v) for k, v in texture_sizes.items()})
    idx_new = {
        "frames": frames,
        "objects": objects,
        "bake_mode": str(bake_mode).upper(),
        "texture_sizes": sizes,
        "written_at_utc": _now_utc_iso(),
    }
    _write_bake_index(scene, idx_new)
    return frame_path


def read_frame_bake(
    scene: bpy.types.Scene,
    frame: int,
) -> Optional[dict[str, dict[str, np.ndarray]]]:
    """Load one frame's bake from disk. Returns
    ``{obj_name: {"irradiance": ndarray, "albedo": ndarray or None}, ...}``
    or None if the file doesn't exist."""
    path = _bakes_frame_path(scene, frame)
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as npz:
            files = list(npz.files)
            out: dict[str, dict[str, np.ndarray]] = {}
            for k in files:
                if "__" not in k:
                    continue
                obj_name, _, kind = k.partition("__")
                if kind not in ("irradiance", "albedo"):
                    continue
                arr = np.array(npz[k], dtype=np.float32)
                out.setdefault(obj_name, {"irradiance": None, "albedo": None})[kind] = arr
            return out or None
    except Exception as e:
        _log.debug(f"[HeatSim] WARNING: failed to read frame bake {path}: {e}")
        return None


def find_nearest_earlier_baked_frame(
    scene: bpy.types.Scene,
    frame: int,
) -> Optional[int]:
    """Return the largest cached frame ≤ `frame`, or the smallest cached
    frame if all cached frames are > `frame` (so the viewer never sees
    blank when scrubbing before the earliest bake). Returns None if the
    cache is empty or the index is unreadable."""
    idx = read_bake_index(scene)
    if not idx:
        return None
    frames = sorted(int(f) for f in idx.get("frames", []))
    if not frames:
        return None
    # Filter out frames whose npz is actually missing on disk (e.g. user
    # manually deleted some) — trust the file system as authoritative.
    extant = [f for f in frames if os.path.isfile(_bakes_frame_path(scene, f))]
    if not extant:
        return None
    candidates = [f for f in extant if f <= int(frame)]
    if candidates:
        return candidates[-1]
    return extant[0]


def delete_bakes(scene: bpy.types.Scene) -> int:
    """Remove the entire bakes/ directory. Returns the number of files
    deleted (incl. index.json). Best-effort — missing directory is fine."""
    bdir = _bakes_dir(scene)
    if not os.path.isdir(bdir):
        return 0
    deleted = 0
    try:
        for entry in os.listdir(bdir):
            p = os.path.join(bdir, entry)
            try:
                if os.path.isfile(p):
                    os.remove(p)
                    deleted += 1
            except Exception as e:
                _log.debug(f"[HeatSim] WARNING: failed to delete {p}: {e}")
        try:
            os.rmdir(bdir)
        except OSError:
            # Directory not empty (some unexpected file) — leave it.
            pass
    except Exception as e:
        _log.debug(f"[HeatSim] WARNING: failed to clear bakes dir {bdir}: {e}")
    return deleted


# ---------------------------------------------------------------------------
# Direct-Kernel disk caches
# ---------------------------------------------------------------------------
#
# The Direct-Kernel irradiance source bypasses the Cycles bake but still
# produces two reusable artifacts per object:
#
#   - Per-vertex absorbed flux (W/m²) per simulated frame, stored under
#     `<blend>.heatsim/latest/kernel_bakes/<frame>.npz`. Used by the
#     View Flux follow-timeline handler so the user can scrub the
#     timeline and see the baked direct-kernel flux at each frame
#     without re-running the kernel.
#   - Per-vertex grayscale albedo (one-shot Cycles bake reduced to a
#     scalar), stored under `<blend>.heatsim/latest/albedo_cache.npz`.
#     Reused on every kernel run; invalidated only by Reset Simulation
#     or source switch.
#
# Both caches are atomically written via tmp+rename and best-effort —
# disk-full / unsaved-blend failures log and swallow.


def _kernel_bakes_dir(scene: bpy.types.Scene) -> str:
    """Absolute path to the kernel_bakes/ directory for this .blend."""
    archive_dir = _archive_dir_for_blend()
    if archive_dir is None:
        archive_dir = _archive_dir_fallback(getattr(scene, "name", "default"))
    return os.path.join(archive_dir, KERNEL_BAKES_DIRNAME)


def _kernel_bake_frame_path(scene: bpy.types.Scene, frame: int) -> str:
    return os.path.join(_kernel_bakes_dir(scene), f"{int(frame)}.npz")


def _kernel_albedo_path(scene: bpy.types.Scene) -> str:
    archive_dir = _archive_dir_for_blend()
    if archive_dir is None:
        archive_dir = _archive_dir_fallback(getattr(scene, "name", "default"))
    return os.path.join(archive_dir, KERNEL_ALBEDO_FILENAME)


def write_kernel_frame_bake(
    scene: bpy.types.Scene,
    frame: int,
    per_object_flux: dict[str, np.ndarray],
) -> Optional[str]:
    """Persist one frame's direct-kernel absorbed flux (W/m² per vertex)
    to disk. Returns the file path, or None on failure."""
    if not per_object_flux:
        return None
    bdir = _kernel_bakes_dir(scene)
    try:
        os.makedirs(bdir, exist_ok=True)
    except Exception as e:
        _log.debug(f"[HeatSim] WARNING: cannot create kernel_bakes dir {bdir}: {e}")
        return None
    arrays = {
        name: np.asarray(arr, dtype=np.float32).reshape(-1)
        for name, arr in per_object_flux.items()
    }
    if not arrays:
        return None
    path = _kernel_bake_frame_path(scene, frame)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp_path, path)
        return path
    except Exception as e:
        _log.debug(f"[HeatSim] WARNING: failed to write kernel frame bake {path}: {e}")
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return None


def read_kernel_frame_bake(
    scene: bpy.types.Scene,
    frame: int,
) -> Optional[dict[str, np.ndarray]]:
    """Load one frame's direct-kernel flux from disk, or None if absent."""
    path = _kernel_bake_frame_path(scene, frame)
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as npz:
            return {k: np.array(npz[k], dtype=np.float32) for k in npz.files}
    except Exception as e:
        _log.debug(f"[HeatSim] WARNING: failed to read kernel frame bake {path}: {e}")
        return None


def find_nearest_earlier_kernel_frame(
    scene: bpy.types.Scene,
    frame: int,
) -> Optional[int]:
    """Return the largest cached kernel-bake frame ≤ ``frame``, falling
    back to the smallest cached frame if none precede ``frame``. Returns
    None if the kernel cache is empty."""
    bdir = _kernel_bakes_dir(scene)
    if not os.path.isdir(bdir):
        return None
    extant: list[int] = []
    try:
        for entry in os.listdir(bdir):
            if not entry.endswith(".npz"):
                continue
            stem = entry[:-4]
            try:
                extant.append(int(stem))
            except ValueError:
                continue
    except Exception:
        return None
    if not extant:
        return None
    extant.sort()
    candidates = [f for f in extant if f <= int(frame)]
    return candidates[-1] if candidates else extant[0]


def delete_kernel_bakes(scene: bpy.types.Scene) -> int:
    """Remove the kernel_bakes/ directory. Returns # files deleted."""
    bdir = _kernel_bakes_dir(scene)
    if not os.path.isdir(bdir):
        return 0
    deleted = 0
    try:
        for entry in os.listdir(bdir):
            p = os.path.join(bdir, entry)
            try:
                if os.path.isfile(p):
                    os.remove(p)
                    deleted += 1
            except Exception as e:
                _log.debug(f"[HeatSim] WARNING: failed to delete {p}: {e}")
        try:
            os.rmdir(bdir)
        except OSError:
            pass
    except Exception as e:
        _log.debug(f"[HeatSim] WARNING: failed to clear kernel_bakes dir {bdir}: {e}")
    return deleted


def write_kernel_albedo_cache(
    scene: bpy.types.Scene,
    per_object_albedo: dict[str, np.ndarray],
) -> Optional[str]:
    """Write per-vertex grayscale albedo (one-shot Cycles bake) to disk."""
    if not per_object_albedo:
        return None
    archive_dir = _archive_dir_for_blend()
    if archive_dir is None:
        archive_dir = _archive_dir_fallback(getattr(scene, "name", "default"))
    try:
        os.makedirs(archive_dir, exist_ok=True)
    except Exception as e:
        _log.debug(f"[HeatSim] WARNING: cannot create archive dir for albedo cache: {e}")
        return None
    path = _kernel_albedo_path(scene)
    arrays = {
        name: np.asarray(arr, dtype=np.float32).reshape(-1)
        for name, arr in per_object_albedo.items()
    }
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp_path, path)
        return path
    except Exception as e:
        _log.debug(f"[HeatSim] WARNING: failed to write kernel albedo cache {path}: {e}")
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return None


def read_kernel_albedo_cache(scene: bpy.types.Scene) -> Optional[dict[str, np.ndarray]]:
    """Read the per-vertex albedo cache, or None if absent."""
    path = _kernel_albedo_path(scene)
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as npz:
            return {k: np.array(npz[k], dtype=np.float32) for k in npz.files}
    except Exception as e:
        _log.debug(f"[HeatSim] WARNING: failed to read kernel albedo cache {path}: {e}")
        return None


def delete_kernel_albedo_cache(scene: bpy.types.Scene) -> bool:
    """Drop the kernel albedo cache. Best-effort."""
    path = _kernel_albedo_path(scene)
    if not os.path.isfile(path):
        return False
    try:
        os.remove(path)
        return True
    except Exception as e:
        _log.debug(f"[HeatSim] WARNING: failed to delete kernel albedo cache {path}: {e}")
        return False


# ---------------------------------------------------------------------------
# Sky-visibility cache (bent normal + AO) for the Direct Kernel
# ---------------------------------------------------------------------------
#
# Per-object per-vertex Bent Normal (3 floats) + Sky AO (1 scalar). Used
# by the Direct Kernel to attenuate the SH9 sky term against per-vertex
# hemispherical visibility. Layout: one npz with keys
# ``<obj_name>__bent_normal`` (N, 3) float32 and ``<obj_name>__sky_ao``
# (N,) float32. Invalidated by Reset Simulation, source switch, and
# whenever animated geometry triggers a sky-visibility rebake.


def _sky_visibility_path(scene: bpy.types.Scene) -> str:
    archive_dir = _archive_dir_for_blend()
    if archive_dir is None:
        archive_dir = _archive_dir_fallback(getattr(scene, "name", "default"))
    return os.path.join(archive_dir, SKY_VISIBILITY_FILENAME)


def write_sky_visibility_cache(
    scene: bpy.types.Scene,
    per_object: dict[str, dict[str, np.ndarray]],
) -> Optional[str]:
    """Persist ``{obj_name: {"bent_normal": (N,3), "ao": (N,)}}`` to disk."""
    if not per_object:
        return None
    archive_dir = _archive_dir_for_blend()
    if archive_dir is None:
        archive_dir = _archive_dir_fallback(getattr(scene, "name", "default"))
    try:
        os.makedirs(archive_dir, exist_ok=True)
    except Exception as e:
        _log.debug(f"[HeatSim] WARNING: cannot create archive dir for sky-vis cache: {e}")
        return None
    path = _sky_visibility_path(scene)
    arrays: dict[str, np.ndarray] = {}
    for name, payload in per_object.items():
        bn = payload.get("bent_normal")
        ao = payload.get("ao")
        if bn is None or ao is None:
            continue
        arrays[f"{name}__bent_normal"] = np.asarray(bn, dtype=np.float32).reshape(-1, 3)
        arrays[f"{name}__sky_ao"] = np.asarray(ao, dtype=np.float32).reshape(-1)
    if not arrays:
        return None
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp_path, path)
        return path
    except Exception as e:
        _log.debug(f"[HeatSim] WARNING: failed to write sky-vis cache {path}: {e}")
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return None


def read_sky_visibility_cache(
    scene: bpy.types.Scene,
) -> Optional[dict[str, dict[str, np.ndarray]]]:
    """Load the sky-visibility cache, or None if absent."""
    path = _sky_visibility_path(scene)
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as npz:
            files = list(npz.files)
            out: dict[str, dict[str, np.ndarray]] = {}
            for k in files:
                if "__" not in k:
                    continue
                obj_name, _, kind = k.partition("__")
                arr = np.array(npz[k], dtype=np.float32)
                if kind == "bent_normal":
                    out.setdefault(obj_name, {})["bent_normal"] = arr
                elif kind == "sky_ao":
                    out.setdefault(obj_name, {})["ao"] = arr
            # Keep only entries that have both arrays.
            return {k: v for k, v in out.items() if "bent_normal" in v and "ao" in v} or None
    except Exception as e:
        _log.debug(f"[HeatSim] WARNING: failed to read sky-vis cache {path}: {e}")
        return None


def delete_sky_visibility_cache(scene: bpy.types.Scene) -> bool:
    """Drop the sky-visibility cache. Best-effort."""
    path = _sky_visibility_path(scene)
    if not os.path.isfile(path):
        return False
    try:
        os.remove(path)
        return True
    except Exception as e:
        _log.debug(f"[HeatSim] WARNING: failed to delete sky-vis cache {path}: {e}")
        return False


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_archive(
    scene: bpy.types.Scene,
    per_object_temps: dict[str, np.ndarray],
    metadata: dict[str, Any],
    *,
    mode: str = "replace",
    replace_existing: Optional[bool] = None,
) -> Optional[str]:
    """Write `temperatures.npz` + `manifest.json` and stamp lookup custom
    properties on each named object in `per_object_temps`.

    `per_object_temps`: maps `obj.name` -> 2D ndarray of shape (T, N), where
    T is timesteps and N is vertex count for that object. 1D arrays are
    promoted to (1, N).

    `mode`:
        - "replace"     overwrite the file with `per_object_temps` only
        - "merge"       per-key replace: new keys replace, missing keys keep
                        their old values (used by the disabled-objects pass
                        in full-run mode)
        - "append_time" per-key concatenate along axis 0 (time): new arrays
                        are appended after existing ones (used by incremental
                        runs to grow the cumulative history)

    `replace_existing` is the legacy bool kwarg. `True` -> "replace",
    `False` -> "merge". Mapped onto `mode` for backward compat.

    Returns the absolute archive path, or None on failure (e.g. disk-full).
    """
    if not per_object_temps:
        return None

    if replace_existing is not None:
        mode = "replace" if replace_existing else "merge"

    if mode not in ("replace", "merge", "append_time"):
        raise ValueError(f"write_archive: unknown mode {mode!r}")

    archive_dir = _archive_dir_for_blend()
    if archive_dir is None:
        archive_dir = _archive_dir_fallback(getattr(scene, "name", "default"))
        _log.debug(
            f"[HeatSim] WARNING: .blend is unsaved; storing temperatures under "
            f"{archive_dir}. Save the .blend before closing to keep results."
        )

    try:
        os.makedirs(archive_dir, exist_ok=True)
    except Exception as e:
        _log.debug(f"[HeatSim] ERROR: cannot create archive dir {archive_dir}: {e}")
        return None

    archive_path = os.path.join(archive_dir, ARCHIVE_FILENAME)
    manifest_path = os.path.join(archive_dir, MANIFEST_FILENAME)

    # Pre-load existing arrays for merge/append modes.
    arrays: dict[str, np.ndarray] = {}
    if mode in ("merge", "append_time") and os.path.isfile(archive_path):
        try:
            # Materialize via np.array so we can close the file before we replace it.
            with np.load(archive_path, allow_pickle=False) as old:
                for k in old.files:
                    arrays[k] = np.array(old[k])
        except Exception as e:
            _log.debug(f"[HeatSim] WARNING: failed to read existing archive for {mode}: {e}")

    for name, arr in per_object_temps.items():
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim == 1:
            a = a[None, :]
        if mode == "append_time" and name in arrays:
            old = arrays[name]
            if old.ndim == 1:
                old = old[None, :]
            if int(old.shape[1]) != int(a.shape[1]):
                raise ValueError(
                    f"write_archive(mode='append_time'): vertex count for "
                    f"{name!r} drifted ({old.shape[1]} -> {a.shape[1]}); "
                    f"topology must be stable across incremental clicks."
                )
            arrays[name] = np.concatenate([old, a], axis=0)
        else:
            arrays[name] = a

    # Close any cached handle on the archive path before overwriting (Windows
    # mmap'd files can't be replaced while open).
    _close_archive(archive_path)

    # `np.savez(path, ...)` auto-appends `.npz` to a string path, so we open
    # the tmp file ourselves and hand it the file object — that bypasses the
    # rewrite and lets `os.replace` atomically rename the exact path we wrote.
    tmp_path = archive_path + ".tmp"
    try:
        with open(tmp_path, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp_path, archive_path)
    except Exception as e:
        _log.debug(f"[HeatSim] ERROR: failed to write archive {archive_path}: {e}")
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return None

    # Manifest sidecar (best-effort; not load-bearing).
    try:
        sidecar = dict(metadata)
        sidecar["objects"] = {name: list(arr.shape) for name, arr in arrays.items()}
        sidecar["written_at_utc"] = _now_utc_iso()
        sidecar["blend_path"] = bpy.data.filepath or ""
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2)
    except Exception as e:
        _log.debug(f"[HeatSim] WARNING: failed to write manifest: {e}")

    # Stamp lookup props on objects we just wrote (don't touch unrelated objs).
    for name in per_object_temps.keys():
        obj = scene.objects.get(name)
        if obj is not None:
            _stamp_object_props(obj, archive_path, name)

    return archive_path


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def _read_legacy(obj: bpy.types.Object) -> Optional[np.ndarray]:
    """Backwards-compat: read from the old in-blend custom property if no
    archive resolves. Always allocates (legacy values are Python lists)."""
    legacy = obj.get("heatsim_temperature_data")
    if legacy is None:
        return None
    arr = np.asarray(legacy, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def read_object_history(obj: bpy.types.Object) -> Optional[np.ndarray]:
    """Return (T, N) history for `obj`, mmap-backed when from an archive.
    Resolution order: archive (URI -> abspath) -> legacy custom property.
    Returns None if nothing resolves."""
    path = _resolve_archive_path(obj)
    if path is not None:
        try:
            npz = _open_archive(path)
            key = obj.get("heatsim_data_key", "") or obj.name
            if key not in npz.files:
                if obj.name in npz.files:
                    key = obj.name
                else:
                    return _read_legacy(obj)
            return np.asarray(npz[key])
        except Exception as e:
            _log.debug(
                f"[HeatSim] WARNING: failed to read archive at {path}: {e}; "
                f"falling back to legacy property."
            )
    return _read_legacy(obj)


def read_object_timestep(
    obj: bpy.types.Object,
    t: int,
    *,
    clamp_past_end: bool = True,
) -> Optional[np.ndarray]:
    """Hot path used by the frame-change handler. Returns the (N,) row at
    timestep `t`.

    `clamp_past_end=True` (default) clamps `t` to `[0, history.shape[0] - 1]`,
    so timeline scrubbing past the end shows the last simulated state.

    `clamp_past_end=False` returns None when `t >= history.shape[0]`, leaving
    it to the caller to render an alternative (e.g. the per-vertex initial
    map for not-yet-simulated frames in animate-mode incremental runs).
    Negative t is still clamped to 0 either way.
    """
    history = read_object_history(obj)
    if history is None:
        return None
    if history.ndim == 1:
        return np.asarray(history)
    n = int(history.shape[0])
    t_int = int(t)
    if t_int < 0:
        t_int = 0
    if t_int >= n:
        if clamp_past_end:
            t_int = n - 1
        else:
            return None
    return np.asarray(history[t_int, :])


def has_temperature_data(obj: bpy.types.Object) -> bool:
    """True if either the archive resolves to a file or the legacy custom
    property is set. Used by visualization code that wants a presence test
    without paying the cost of actually loading the data."""
    if _resolve_archive_path(obj) is not None:
        return True
    return "heatsim_temperature_data" in obj


def get_num_timesteps(obj: bpy.types.Object) -> int:
    """Read the metadata custom prop; fall back to history.shape[0]."""
    v = obj.get("heatsim_num_timesteps")
    if v is not None:
        try:
            return int(v)
        except Exception:
            pass
    h = read_object_history(obj)
    if h is None:
        return 0
    return int(h.shape[0]) if h.ndim == 2 else 1
