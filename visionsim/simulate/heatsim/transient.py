"""Chronological HeatSim animation with independently scheduled geometry and lighting."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from visionsim.simulate.config import ThermalConfig
from visionsim.simulate.heatsim import adapter, cache

TRANSIENT_FIELDS = {
    "mode",
    "substeps_per_frame",
    "laplacian_update_every_steps",
    "irradiance_bake_mode",
    "irradiance_bake_every_n_frames",
    "simulation_start_frame",
    "simulation_end_frame",
    "cache_root",
}


def bake_due(step: int, config: ThermalConfig) -> bool:
    """Schedules are anchored to the simulation origin, independent of rendered frames."""
    if step == 0 or config.irradiance_bake_mode == "EVERY_STEP":
        return True
    return (
        config.irradiance_bake_mode == "EVERY_N_FRAMES"
        and step % (config.substeps_per_frame * config.irradiance_bake_every_n_frames) == 0
    )


def read_frame(session: dict, frame: int) -> dict:
    manifest = session["manifest"]
    if not manifest["frame_start"] <= frame <= manifest["frame_end"]:
        raise ValueError(f"Frame {frame} lies outside the prepared thermal simulation")
    history = cache.read_temperatures(session["root"], f"frame-{frame}", manifest["counts"])
    if history is None or any(v.shape[0] != 1 for v in history.values()):
        raise RuntimeError(f"Missing or invalid transient temperatures at frame {frame}")
    return history


def prepare(
    scene,
    config: ThermalConfig,
    defaults: dict,
    solver_cfg: dict,
    assignment,
    identity: dict,
    trajectory: str | None = None,
):
    """Prepare a fresh trajectory or attach an explicitly shared, completed run.

    Reuse is explicit: no automatic cache hit can hide edits to an animated scene.
    Only frame states are stored; thermal substeps remain bounded in memory.
    """
    from visionsim.simulate.heatsim.geometry import release_meshes
    from visionsim.simulate.heatsim.solver import HeatSimFEM

    start = scene.frame_start if config.simulation_start_frame is None else config.simulation_start_frame
    end = scene.frame_end if config.simulation_end_frame is None else config.simulation_end_frame
    fps = float(scene.render.fps) / float(scene.render.fps_base)
    if end < start or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid transient frame range or frame rate")
    cfg = json.loads(json.dumps(vars(config), default=str))
    stamp = {"config": cfg, "identity": identity, "frame_start": start, "frame_end": end, "fps": fps}
    original_frame, original_subframe = scene.frame_current, scene.frame_subframe
    base = Path(str(config.cache_root)) if config.cache_root is not None else Path(identity["cache_root"])
    previous_bake_root = scene.get("heatsim_bake_cache_root")
    plan = None
    try:
        scene["heatsim_bake_cache_root"] = str(base)
        scene.frame_set(start)
        objects = adapter.gather_meshes(scene)
        if not objects:
            raise ValueError("Transient simulation requires participating meshes")
        plan = adapter.build_atlas_plan(scene, objects, cfg)
        with adapter.use_atlas_geometry(scene, plan):
            initial = adapter._combine(objects, {}, defaults, solver_cfg, assignment, plan)
            topology = {o.name: (len(o.data.vertices), adapter._mesh_triangles(o.data)[0].copy()) for o in objects}
        if initial is None:
            raise ValueError("Transient simulation has no thermal samples")
        counts = {name: n for name, _, n, _ in initial.layout}
        stamp["counts"] = counts
        stamp["atlas_digest"] = plan.digest
        stamp["scene_digest"] = cache.transient_scene_digest(scene, initial)
        if trajectory is not None:
            root = Path(trajectory)
            manifest = json.loads((root / "manifest.json").read_text())
            if manifest.get("version") != 1 or any(manifest.get(k) != v for k, v in stamp.items()):
                raise ValueError("Transient trajectory does not match this scene and configuration")
            session = {"root": root, "manifest": manifest}
            read_frame(session, start)
            read_frame(session, end)
            return plan, session

        base.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="transient-", dir=base))
        dt = 1.0 / (fps * config.substeps_per_frame)
        fem = HeatSimFEM(
            SimpleNamespace(
                device=solver_cfg["device"],
                RHO=defaults["density_kg_m3"] / 1e9,
                C=defaults["specific_heat_J_kgK"],
                K=defaults["thermal_diffusivity_mm2_s"],
            ),
            SimpleNamespace(sim_radiation=True, sim_convection=False, add_tikhonov_reg=False),
        )
        temperature = initial.t0.copy()
        flux = {}
        stats = {"steps": 0, "bakes": 0, "laplacian_rebuilds": 0}
        preview_range = [float("inf"), float("-inf")]
        total_steps = (end - start) * config.substeps_per_frame
        for step in range(total_steps + 1):
            frame = start + step // config.substeps_per_frame
            subframe = (step % config.substeps_per_frame) / config.substeps_per_frame
            scene.frame_set(frame, subframe=subframe)
            if [o.name for o in adapter.gather_meshes(scene)] != [o.name for o in objects]:
                raise RuntimeError("Transient thermal object membership changed")
            adapter.refresh_atlas_geometry(scene, plan)
            with adapter.use_atlas_geometry(scene, plan):
                for obj in objects:
                    count, faces = topology[obj.name]
                    if len(obj.data.vertices) != count or not np.array_equal(
                        adapter._mesh_triangles(obj.data)[0], faces
                    ):
                        raise RuntimeError(f"Transient mesh topology changed: {obj.name!r}")
                if bake_due(step, config):
                    vertex_objects = [o for o in objects if o.name not in plan.texels]
                    flux = adapter._compute_irradiance(scene, vertex_objects, solver_cfg, defaults)
                    flux.update(adapter._compute_texel_irradiance(scene, objects, plan, solver_cfg, defaults))
                    stats["bakes"] += 1
                combined = adapter._combine(objects, flux, defaults, solver_cfg, assignment, plan)
            if combined is None or combined.layout != initial.layout:
                raise RuntimeError("Transient thermal sample identity changed")
            rebuild = step % config.laplacian_update_every_steps == 0
            pins = np.flatnonzero(~combined.boundary_mask)
            states = fem.simulate_for_pose(
                verts_np=combined.verts,
                faces_np=None,
                boundary_verts_mask=combined.boundary_mask,
                u_prev=temperature,
                irradiance_map=combined.irradiance,
                thermal_diffusivity_map=combined.alpha,
                density_map=combined.density,
                specific_heat_map=combined.c,
                emissivity_map=combined.eps,
                num_substeps=int(step > 0),
                dt=dt,
                dirichlet_indices=pins,
                dirichlet_values=combined.t0[pins],
                rebuild_matrices=rebuild,
            )
            if step:
                temperature = states[-1]
            else:
                temperature[pins] = combined.t0[pins]
            stats["steps"] += int(step > 0)
            stats["laplacian_rebuilds"] += int(rebuild)
            if step % config.substeps_per_frame == 0:
                history = adapter._split_history(temperature[None, :], combined)
                cache.write_temperatures(root, f"frame-{frame}", history, {"time_s": step * dt, "frame": frame})
                lo, hi = adapter.global_temperature_range(history, defaults["initial_temperature_K"])
                preview_range = [min(lo, preview_range[0]), max(hi, preview_range[1])]

        manifest = {
            **stamp,
            "version": 1,
            "stats": stats,
            "preview_range": preview_range,
            "dt_s": dt,
            "times_s": [(f - start) / fps for f in range(start, end + 1)],
        }
        # The manifest is the completion marker. Partial runs can never be attached.
        temporary = root / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(root / "manifest.json")
        return plan, {"root": root, "manifest": manifest}
    except Exception:
        if plan is not None:
            release_meshes(plan.meshes)
        raise
    finally:
        if previous_bake_root is None:
            scene.pop("heatsim_bake_cache_root", None)
        else:
            scene["heatsim_bake_cache_root"] = previous_bake_root
        scene.frame_set(original_frame, subframe=original_subframe)
