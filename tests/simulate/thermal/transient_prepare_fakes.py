"""Blender-free fakes for driving ``transient.prepare`` from tests."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

from visionsim.simulate.config import ThermalConfig
from visionsim.simulate.heatsim import adapter, cache, solver, transient


class _Scene(dict):
    frame_start = 1
    frame_end = 1
    frame_current = 1
    frame_subframe = 0.0
    render = SimpleNamespace(fps=24, fps_base=1.0)

    def frame_set(self, frame, subframe=0.0):
        self.frame_current = frame
        self.frame_subframe = subframe


class _FEM:
    def __init__(self, *args, **kwargs):
        pass

    def simulate_for_pose(self, **kwargs):
        steps = kwargs["num_substeps"]
        points = len(kwargs["u_prev"])
        return np.zeros((steps, points)) if steps else np.zeros((0, points))


def install_fakes(monkeypatch, *, blocker=None):
    objects = [SimpleNamespace(name="obj", data=SimpleNamespace(vertices=[None]))]
    initial = SimpleNamespace(
        layout=[("obj", 0, 1, "VERTEX")],
        verts=np.zeros((1, 3)),
        faces=np.zeros((1, 3), dtype=np.int32),
        t0=np.array([295.0]),
        irradiance=np.zeros(1),
        alpha=np.ones(1),
        density=np.ones(1),
        c=np.ones(1),
        eps=np.zeros(1),
        boundary_mask=np.ones(1, dtype=bool),
    )
    plan = SimpleNamespace(
        digest="plan", texels={}, meshes={}, density=1500.0, tile_min=16, tile_max=64, soft_max=500_000
    )
    geometry = SimpleNamespace(release_meshes=lambda meshes: None)

    monkeypatch.setitem(sys.modules, "visionsim.simulate.heatsim.geometry", geometry)
    monkeypatch.setattr(adapter, "gather_meshes", lambda scene: objects)
    monkeypatch.setattr(adapter, "build_atlas_plan", lambda scene, objects, cfg: plan)
    monkeypatch.setattr(adapter, "use_atlas_geometry", lambda scene, plan: nullcontext())
    monkeypatch.setattr(adapter, "refresh_atlas_geometry", lambda scene, plan: None)
    monkeypatch.setattr(adapter, "_combine", lambda *args: initial)
    monkeypatch.setattr(adapter, "_compute_irradiance", lambda *args: {})
    monkeypatch.setattr(adapter, "_compute_texel_irradiance", lambda *args: {})
    monkeypatch.setattr(adapter, "_mesh_triangles", lambda mesh: (np.zeros((1, 3), dtype=np.int32),) * 3)
    monkeypatch.setattr(adapter, "_split_history", lambda history, combined: {"obj": np.full((1, 1), 295.0)})
    monkeypatch.setattr(adapter, "global_temperature_range", lambda history, initial: (295.0, 295.0))
    monkeypatch.setattr(cache, "transient_scene_digest", lambda scene, initial: "scene")
    monkeypatch.setattr(solver, "HeatSimFEM", _FEM)
    monkeypatch.setattr(transient, "_reuse_blocker", lambda scene: blocker)


def run_prepare(tmp_path, change=None):
    config = ThermalConfig(
        mode="TRANSIENT",
        simulation_start_frame=1,
        simulation_end_frame=1,
        cache_root=tmp_path,
        **(change or {}),
    )
    defaults = {
        "initial_temperature_K": config.initial_temperature_K,
        "thermal_diffusivity_mm2_s": config.thermal_diffusivity_mm2_s,
        "density_kg_m3": config.density_kg_m3,
        "specific_heat_J_kgK": config.specific_heat_J_kgK,
        "emissivity": config.emissivity,
        "irradiance_scale": config.irradiance_scale,
    }
    solver_cfg = {
        "sim_time_s": config.sim_time_s,
        "timestep_s": config.timestep_s,
        "device": config.device,
        "bake_samples": config.bake_samples,
        "irradiance_texture_size": config.irradiance_texture_size,
    }
    return transient.prepare(
        _Scene(),
        config,
        defaults,
        solver_cfg,
        None,
        {"cache_root": str(tmp_path)},
    )
