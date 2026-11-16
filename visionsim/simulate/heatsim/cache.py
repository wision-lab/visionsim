from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

CACHE_SCHEMA_VERSION = 2
# Bump when bake/solver semantics change, even if the archive layout is unchanged.
# Revision 16 verifies CG residuals and retries failed solves in double precision.
BAKE_PIPELINE_VERSION = 16
_log = logging.getLogger("rich")


def read_bake_uv(path: Path, signature: str, loop_count: int) -> np.ndarray | None:
    """Read one validated layout for this geometry; reject damaged cache entries."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            uv = data["uv"]
            if (
                str(data["signature"].item()) != signature
                or uv.shape != (loop_count, 2)
                or uv.dtype != np.float32
                or not np.isfinite(uv).all()
                or np.any(uv < -1e-6)
                or np.any(uv > 1 + 1e-6)
            ):
                raise ValueError("signature or UV coordinates do not match")
            return uv.reshape(-1)
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Invalid thermal UV cache {path}: {exc}") from exc


def publish_bake_uv(path: Path, signature: str, uv: np.ndarray) -> np.ndarray:
    """Publish a complete layout once; every competing writer uses the winner.

    Blender unwrapping can vary slightly between processes. Replacing an existing
    layout would change the sample addresses after another worker had solved them.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        coordinates = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
        np.savez(temp_path, signature=np.asarray(signature), uv=coordinates)
        read_bake_uv(temp_path, signature, len(coordinates))
        try:
            os.link(temp_path, path)
        except FileExistsError:
            pass
        result = read_bake_uv(path, signature, len(coordinates))
        if result is None:
            raise RuntimeError(f"Thermal UV cache disappeared during publication: {path}")
        return result
    finally:
        temp_path.unlink(missing_ok=True)


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
        {
            "schema": CACHE_SCHEMA_VERSION,
            "pipeline": BAKE_PIPELINE_VERSION,
            "p": str(blend_path),
            "source": source_digest,
            "c": solver_cfg,
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transient_scene_digest(scene: Any, initial: Any) -> str:
    """Fingerprint live thermal inputs before attaching another render worker.

    The saved blend's hash alone misses RPC edits and future animation keys.
    Ignore generated thermal UVs/AOV branches, which are outputs of preparation,
    while retaining authored geometry, shading, motion, and external image data.
    """
    import bpy

    from visionsim.simulate.heatsim.names import ATLAS_UV_LAYER_NAME, BAKE_UV_LAYER_NAME

    def properties(value):
        result = {}
        for prop in value.bl_rna.properties:
            key = prop.identifier
            if prop.is_readonly or key == "rna_type":
                continue
            item = getattr(value, key)
            if prop.type in {"BOOLEAN", "INT", "FLOAT", "STRING", "ENUM"}:
                if getattr(prop, "is_array", False):
                    item = np.asarray(item).tolist()
                elif isinstance(item, set):
                    item = sorted(item)
                result[key] = item
            elif prop.type == "POINTER" and isinstance(item, bpy.types.ID):
                result[key] = (item.bl_rna.identifier, item.name_full)
        try:
            custom = value.items()
        except (AttributeError, TypeError):
            custom = ()
        result["custom"] = {}
        if custom:
            # Thermal preparation adds cache bookkeeping and render attributes.
            result["custom"] = {
                str(k): v.to_dict() if hasattr(v, "to_dict") else v.to_list() if hasattr(v, "to_list") else v
                for k, v in custom
                if not str(k).startswith("heatsim_")
                and k
                not in {"heat_sim_flux_uv", "heat_sim_irradiance_image", "heat_sim_albedo_uv", "heat_sim_albedo_image"}
                and (isinstance(v, (int, float, str, list, tuple)) or hasattr(v, "to_dict") or hasattr(v, "to_list"))
            }
        return result

    def animation(value):
        data = getattr(value, "animation_data", None)
        if data is None:
            return None
        return {
            "settings": properties(data),
            "drivers": [curve(c) for c in data.drivers],
            "nla": [(properties(t), [properties(s) for s in t.strips]) for t in data.nla_tracks],
        }

    def curve(value):
        result = {
            "settings": properties(value),
            "keys": [properties(k) for k in value.keyframe_points],
            "samples": [properties(k) for k in value.sampled_points],
            "modifiers": [properties(m) for m in value.modifiers],
        }
        if value.driver is not None:
            result["driver"] = properties(value.driver)
            result["variables"] = [(properties(v), [properties(t) for t in v.targets]) for v in value.driver.variables]
        return result

    images = {}
    trees = {}
    tree_keys = {}
    uv_trees = set()

    def tree(value, key=None):
        if value is None:
            return None
        pointer = value.as_pointer()
        if pointer in tree_keys:
            return tree_keys[pointer]
        key = key or f"nodegroup/{value.name_full}"
        tree_keys[pointer] = key
        trees[key] = None
        # Follow authored output dependencies; temperature AOV nodes are generated
        # after solving and must not prevent attachment to the same scene.
        retained = set()

        def visit(node):
            if node in retained:
                return
            retained.add(node)
            for socket in node.inputs:
                for link in socket.links:
                    visit(link.from_node)

        for node in value.nodes:
            if node.type in {"OUTPUT_MATERIAL", "OUTPUT_WORLD", "OUTPUT_LIGHT", "GROUP_OUTPUT"}:
                visit(node)
        nodes = []
        for node in sorted(retained, key=lambda n: n.name):
            record = properties(node)
            record["inputs"] = [properties(socket) for socket in node.inputs]
            record["animation"] = animation(node.id_data)
            if getattr(node, "node_tree", None) is not None:
                record["group"] = tree(node.node_tree)
                if record["group"] in uv_trees:
                    uv_trees.add(key)
            if node.type in {"TEX_IMAGE", "UVMAP", "ATTRIBUTE"} or (
                node.type == "TEX_COORD" and any(l.to_node in retained for l in node.outputs["UV"].links)
            ):
                uv_trees.add(key)
            image = getattr(node, "image", None)
            if image is not None and image.name_full not in images:
                if image.source not in {"FILE", "GENERATED"}:
                    raise ValueError("Transient attachment cannot fingerprint animated image inputs")
                pixels = np.empty(len(image.pixels), dtype=np.float32)
                image.pixels.foreach_get(pixels)
                images[image.name_full] = hashlib.sha256(pixels.tobytes()).hexdigest()
                if image.source == "FILE" and image.packed_file is None:
                    path = Path(bpy.path.abspath(image.filepath, library=image.library))
                    images[image.name_full] += file_digest(path)
            nodes.append(record)
        trees[key] = {
            "nodes": nodes,
            "links": sorted(
                (l.from_node.name, l.from_socket.identifier, l.to_node.name, l.to_socket.identifier)
                for l in value.links
                if l.from_node in retained and l.to_node in retained
            ),
        }
        return key

    def shading(value):
        if value is None:
            return None
        return (
            properties(value),
            animation(value),
            tree(getattr(value, "node_tree", None), f"{value.bl_rna.identifier}/{value.name_full}/nodes"),
        )

    objects = []
    for obj in sorted(scene.objects, key=lambda o: o.name):
        if obj.type == "CAMERA":
            continue
        record = {
            "object": properties(obj),
            "animation": animation(obj),
            "modifiers": [properties(m) for m in obj.modifiers],
            "constraints": [properties(c) for c in obj.constraints],
        }
        record["node_groups"] = [tree(m.node_group) for m in obj.modifiers if m.type == "NODES"]
        if hasattr(obj, "heat_sim_material"):
            record["thermal"] = properties(obj.heat_sim_material)
        if obj.type == "MESH":
            mesh = obj.data
            record["materials"] = [shading(s.material) for s in obj.material_slots]
            uses_uv = any(material is not None and material[2] in uv_trees for material in record["materials"])
            record["mesh"] = {
                "vertices": [tuple(v.co) for v in mesh.vertices],
                "faces": [(tuple(p.vertices), p.material_index) for p in mesh.polygons],
                "uv": [
                    (uv.name, [tuple(v.uv) for v in uv.data])
                    for uv in mesh.uv_layers
                    if uses_uv and uv.name not in {ATLAS_UV_LAYER_NAME, BAKE_UV_LAYER_NAME}
                ],
                "animation": animation(mesh),
                "shape_keys": None
                if mesh.shape_keys is None
                else (
                    animation(mesh.shape_keys),
                    [(properties(k), [tuple(v.co) for v in k.data]) for k in mesh.shape_keys.key_blocks],
                ),
            }
        elif obj.type == "LIGHT":
            record["light"] = shading(obj.data)
        objects.append(record)
    actions = []
    for action in sorted(bpy.data.actions, key=lambda a: a.name):
        if getattr(action, "is_action_layered", False):
            curves = [
                c for layer in action.layers for strip in layer.strips for bag in strip.channelbags for c in bag.fcurves
            ]
        else:
            curves = list(action.fcurves)
        actions.append((properties(action), [curve(c) for c in curves]))
    payload = {
        "pipeline": BAKE_PIPELINE_VERSION,
        "scene": properties(scene),
        "objects": objects,
        "actions": actions,
        "world": shading(scene.world),
        "trees": trees,
        "images": images,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode())
    for name in ("verts", "faces", "t0", "alpha", "density", "c", "eps", "boundary_mask"):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(getattr(initial, name)).tobytes())
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
        expected_counts (dict[str, int] | None, optional): Expected sample count per object; a mismatch
            invalidates the cache. Defaults to None.
        per_object: Mapping of object name to a ``(timesteps, vertices)`` array.
        meta: JSON-serializable metadata stored alongside the arrays.

    Returns:
        The path to the written ``.npz`` archive.
    """
    out_dir = Path(cache_root) / key
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "temperatures.npz"
    meta = {
        **meta,
        "schema": CACHE_SCHEMA_VERSION,
        "pipeline": BAKE_PIPELINE_VERSION,
        "objects": sorted(per_object),
        "timesteps": next(iter(per_object.values())).shape[0] if per_object else 0,
    }
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
    cache_root: Path,
    key: str,
    expected_counts: dict[str, int] | None = None,
) -> dict[str, np.ndarray] | None:
    """Read per-object temperature histories, or return ``None`` on a cache miss.

    Args:
        cache_root: Root directory for thermal caches.
        key: Cache key from :func:`cache_key`.
        expected_counts (dict[str, int] | None, optional): Expected sample count per object; a mismatch
            invalidates the cache. Defaults to None.

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
            if meta.get("pipeline") != BAKE_PIPELINE_VERSION:
                reject("bake pipeline revision mismatch")
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
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as exc:
        # An interrupted write leaves a truncated archive, whose BadZipFile is not an
        # OSError; count it as a cache miss like the other corruption cases.
        reject(str(exc))
        return None
