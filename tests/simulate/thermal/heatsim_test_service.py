"""Composable Blender primitives for the heatsim tests.

This module is imported inside a spawned Blender process: ``HeatsimTestService`` is a
``BlenderService`` subclass, and such a service can only be instantiated within Blender's
Python runtime (``BlenderService.__init__`` raises otherwise). Tests reach it via
``BlenderClient.spawn(service=SERVICE_PATH)`` and drive multi-step scenarios by calling
several ``exposed_*`` primitives in sequence, asserting on the returned data.

The exposed primitives are coarse on purpose: ``build_scene`` constructs a scene,
``solve`` runs the thermal solve, ``bake_albedo``/``bake_irradiance`` bake maps,
``configure_thermal`` stages the TEXEL atlas, and ``render_temperature``/
``rgb_thermal_loop`` produce pixels. Tests call them in sequence to build the scenario
they need.

Return values must be plain, detached Python data (``dict``/``list``/``float``/...). rpyc
cannot inventory numpy arrays or bpy objects, so flatten those on the way out.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from visionsim.simulate.blender import BlenderService, bpy, np

# Dotted ``module:ClassName`` path passed to ``BlenderClient.spawn(service=...)``.
SERVICE_PATH = "tests.simulate.thermal.heatsim_test_service:HeatsimTestService"


def call_service(client: Any, name: str, *args: Any, **kwargs: Any) -> Any:
    """Call ``client.<name>`` and deep-materialize the result while connected.

    rpyc returns the outer container as a live reference and leaves nested containers as
    netrefs; those cannot be read once ``BlenderClient.spawn`` closes the connection.
    Rebuilding the tree inside the ``with`` block yields plain, detached Python values.
    """

    def detach(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): detach(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [detach(v) for v in value]
        return value

    return detach(getattr(client, name)(*args, **kwargs))


class HeatsimTestService(BlenderService):
    """Composable primitives (``exposed_*``) that the heatsim tests assemble into scenarios."""

    @staticmethod
    def _register() -> None:
        """Register the heatsim PropertyGroup once per Blender process, idempotently."""
        from visionsim.simulate.heatsim import register

        if not hasattr(bpy.types.Object, "heat_sim_material"):
            register()

    @staticmethod
    def _clear() -> None:
        """Remove every object so a primitive starts from an empty scene."""
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()

    @staticmethod
    def _add_sun(energy: float = 10.0) -> None:
        """Add an overhead sun; its default rotation emits straight down (-Z)."""
        bpy.ops.object.light_add(type="SUN")
        bpy.context.active_object.data.energy = energy

    @staticmethod
    def _add_world_light(strength: float = 1.0, color: tuple[float, float, float, float] = (0.2, 0.2, 0.2, 1.0)) -> None:
        """Give the scene a node-based world background so the sky term is non-zero."""
        world = bpy.context.scene.world
        if world is None:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world
        world.use_nodes = True
        bg = world.node_tree.nodes.get("Background")
        if bg is not None:
            bg.inputs["Color"].default_value = color
            bg.inputs["Strength"].default_value = strength

    @staticmethod
    def _enable_cycles(samples: int = 4, device: str = "CPU") -> None:
        """Switch the scene to Cycles on the requested device with a fixed sample count."""
        bpy.context.scene.render.engine = "CYCLES"
        bpy.context.scene.cycles.device = device
        bpy.context.scene.cycles.samples = samples

    @staticmethod
    def _checker_material(scale: float = 6.0, name: str = "checker_mat") -> Any:
        """Build a node material whose Base Color comes from a checker texture."""
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        bsdf = nt.nodes.get("Principled BSDF")
        checker = nt.nodes.new("ShaderNodeTexChecker")
        checker.inputs["Scale"].default_value = scale
        nt.links.new(checker.outputs["Color"], bsdf.inputs["Base Color"])
        return mat

    @staticmethod
    def _mesh(kind: str, name: str, subdivisions: int = 12, size: float = 2.0) -> Any:
        """Add a light-enabled grid/plane/cube named ``name`` and return the object."""
        if kind == "grid":
            bpy.ops.mesh.primitive_grid_add(x_subdivisions=subdivisions, y_subdivisions=subdivisions, size=size)
        elif kind == "plane":
            bpy.ops.mesh.primitive_plane_add(size=size)
        elif kind == "cube":
            bpy.ops.mesh.primitive_cube_add()
        else:
            raise ValueError(f"unsupported mesh kind: {kind}")
        obj = bpy.context.active_object
        obj.name = name
        obj.heat_simulation_enabled = True
        return obj

    @staticmethod
    def _defaults() -> dict[str, float]:
        """The thermal material defaults shared by most scenarios."""
        return {
            "initial_temperature_K": 295.0,
            "thermal_diffusivity_mm2_s": 0.17,
            "density_kg_m3": 1330.0,
            "specific_heat_J_kgK": 880.0,
            "emissivity": 0.9,
            "irradiance_scale": 100.0,
        }

    @staticmethod
    def _solver_cfg(sim_time_s: float = 0.1, timestep_s: float = 0.05, **extra: Any) -> dict[str, Any]:
        """The solver configuration shared by most scenarios."""
        return {"sim_time_s": sim_time_s, "timestep_s": timestep_s, "device": "cpu", **extra}

    @staticmethod
    def _attr_values(obj: Any, name: str) -> Any:
        """Read a mesh attribute (e.g. ``sim_temperature``) as a float64 array."""
        return np.array([d.value for d in obj.data.attributes[name].data])

    def exposed_build_scene(
        self,
        kind: str,
        name: str,
        subdivisions: int = 12,
        size: float = 2.0,
        lit: bool = True,
        material: str | None = None,
        sun_energy: float = 10.0,
        point_light: bool = False,
        dirichlet_K: float | None = None,
    ) -> dict:
        """Clear the scene and build one object, optionally lit and given a material.

        Args:
            kind: ``grid``, ``plane`` or ``cube``.
            name: object name.
            subdivisions: grid subdivisions (``grid`` only).
            size: object size.
            lit: add a sun and a node-based world background.
            material: ``None``, ``"checker"``, or ``"plain"``.
            sun_energy: sun strength when ``lit``.
            point_light: add a point light at (-0.6, 0, 0.4) instead of a sun.
            dirichlet_K: if set, mark the object a DIRICHLET_SOURCE at this temperature.

        Returns:
            dict: object summary (name, vertices, materials, enabled).
        """
        self._register()
        self._clear()
        obj = self._mesh(kind, name, subdivisions=subdivisions, size=size)

        if point_light:
            bpy.ops.object.light_add(type="POINT", location=(-0.6, 0, 0.4))
            bpy.context.active_object.data.energy = 200
        elif lit:
            self._add_sun(sun_energy)
        if lit:
            self._add_world_light(strength=1.0)

        if material == "checker":
            obj.data.materials.append(self._checker_material())
        elif material == "plain":
            plain = bpy.data.materials.new("white")
            plain.diffuse_color = (0.8, 0.8, 0.8, 1)
            plain.use_nodes = True
            obj.data.materials.append(plain)

        if dirichlet_K is not None:
            obj.heat_sim_material.thermal_role = "DIRICHLET_SOURCE"
            obj.heat_sim_material.dirichlet_temperature_K = dirichlet_K

        return {
            "name": obj.name,
            "vertices": len(obj.data.vertices),
            "has_material": bool(obj.data.materials),
            "enabled": bool(obj.heat_simulation_enabled),
        }

    def exposed_add_subsurf(self, levels: int = 2) -> dict:
        """Add a Subsurf modifier to the active object (changes its evaluated vertex count).

        Args:
            levels: subdivision levels.

        Returns:
            dict: base and evaluated vertex counts.
        """
        obj = bpy.context.active_object
        obj.modifiers.new("Subsurf", "SUBSURF").levels = levels
        dg = bpy.context.evaluated_depsgraph_get()
        return {"base_n": len(obj.data.vertices), "eval_n": len(obj.evaluated_get(dg).data.vertices)}

    def exposed_duplicate_linked(self, source: str, name: str, offset_x: float = 2.0) -> dict:
        """Add a linked duplicate of ``source`` (shares its mesh datablock) and enable it.

        Args:
            source: name of the existing object to copy.
            name: name for the duplicate.
            offset_x: location offset applied to the duplicate.

        Returns:
            dict: whether the two objects share one mesh datablock and its user count.
        """
        src = bpy.data.objects[source]
        dup = src.copy()
        dup.name = name
        dup.location.x += offset_x
        bpy.context.collection.objects.link(dup)
        dup.heat_simulation_enabled = True
        return {"shared": bool(src.data is dup.data), "users": int(src.data.users)}

    def exposed_gather_meshes(self) -> dict:
        """Run ``adapter.gather_meshes`` and report the un-sharing it performed.

        Returns:
            dict: gathered names, whether meshes are single-user, and mesh datablock names.
        """
        from visionsim.simulate.heatsim import adapter

        objects = adapter.gather_meshes(bpy.context.scene)
        return {
            "names": sorted(o.name for o in objects),
            "users": {o.name: int(o.data.users) for o in objects},
            "mesh_names": {o.name: o.data.name for o in objects},
            "vertices": {o.name: len(o.data.vertices) for o in objects},
        }

    def exposed_write_history(self, histories: dict[str, dict[str, Any]]) -> dict:
        """Write hand-crafted per-object fields via ``adapter.write_frame_attributes``.

        Args:
            histories: ``{object_name: {"vertices": n, "value": K, "timesteps": m}}``; each
                object gets a constant ``K`` field over ``n`` vertices for ``m`` steps.
                The vertex count must match the object's mesh (see ``gather_meshes``).

        Returns:
            dict: the written ``sim_temperature`` mean per object.
        """
        from visionsim.simulate.heatsim import adapter

        defaults = self._defaults()
        history = {}
        for obj_name, spec in histories.items():
            n = int(spec["vertices"])
            history[obj_name] = np.full((int(spec.get("timesteps", 2)), n), float(spec["value"]))
        adapter.write_frame_attributes(bpy.context.scene, history, timestep=-1, defaults=defaults)
        return {
            obj_name: float(self._attr_values(bpy.data.objects[obj_name], "sim_temperature").mean())
            for obj_name in histories
        }

    def exposed_solve(
        self,
        root_path: str,
        object_name: str,
        sim_time_s: float = 0.1,
        timestep_s: float = 0.05,
        irradiance_scale: float = 100.0,
        bake_samples: int = 1024,
        irradiance_texture_size: int = 512,
        write_attributes: bool = True,
        atlas: dict[str, Any] | None = None,
    ) -> dict:
        """Run ``adapter.solve_scene`` on the current scene and report the result.

        Args:
            root_path: cache root directory.
            object_name: object whose field to summarize.
            sim_time_s: simulated time.
            timestep_s: solver timestep.
            irradiance_scale: absorbed-flux scale.
            bake_samples: Cycles samples for the irradiance bake.
            irradiance_texture_size: bake texture size.
            write_attributes: write ``sim_temperature``/``emissivity`` back to the mesh.
            atlas: optional ``build_atlas_plan`` config (``render_domain`` etc.); when set,
                the plan is built and passed to the solver and its texels are reported.

        Returns:
            dict: shape/range/finiteness of the solved field, plus atlas info when requested.
        """
        from visionsim.simulate.heatsim import adapter

        self._register()
        defaults = self._defaults() | {"irradiance_scale": irradiance_scale}
        solver_cfg = self._solver_cfg(
            sim_time_s, timestep_s, bake_samples=bake_samples, irradiance_texture_size=irradiance_texture_size
        )
        cache_root = Path(root_path)

        plan = None
        if atlas is not None:
            objects = adapter.gather_meshes(bpy.context.scene)
            plan = adapter.build_atlas_plan(bpy.context.scene, objects, atlas)

        history = adapter.solve_scene(
            bpy.context.scene, defaults=defaults, solver_cfg=solver_cfg, cache_root=cache_root, atlas_plan=plan
        )
        field = np.asarray(history[object_name])
        result: dict[str, Any] = {
            "keys": sorted(history.keys()),
            "ndim": int(field.ndim),
            "steps": int(field.shape[0]),
            "finite": bool(np.isfinite(field).all()),
            "min": float(field.min()),
            "max": float(field.max()),
            "rose": bool(field.shape[0] >= 2 and field[-1].max() > field[0].max()),
        }
        if write_attributes:
            adapter.write_frame_attributes(bpy.context.scene, history, timestep=-1, defaults=defaults)
            written = self._attr_values(bpy.data.objects[object_name], "sim_temperature")
            result["sim_temperature"] = float(written.mean())
            result["written_finite"] = bool(np.isfinite(written).all())
            result["written_min"] = float(written.min())
            result["written_max"] = float(written.max())
            result["emissivity"] = float(self._attr_values(bpy.data.objects[object_name], "emissivity").mean())
        if plan is not None:
            result["in_plan"] = object_name in plan.texels
            result["texels"] = sorted(plan.texels)
        return result

    def exposed_solve_twice(self, root_path: str, object_name: str) -> dict:
        """Solve the same scene twice and report whether the fields are identical.

        Args:
            root_path: cache root directory.
            object_name: object whose field to compare.

        Returns:
            dict: determinism flag and the two key sets.
        """
        from visionsim.simulate.heatsim import adapter

        self._register()
        defaults = self._defaults()
        solver_cfg = self._solver_cfg(sim_time_s=0.15)
        cache_root = Path(root_path)
        first = adapter.solve_scene(bpy.context.scene, defaults=defaults, solver_cfg=solver_cfg, cache_root=cache_root)
        second = adapter.solve_scene(bpy.context.scene, defaults=defaults, solver_cfg=solver_cfg, cache_root=cache_root)
        return {
            "keys": sorted(first.keys()),
            "keys2": sorted(second.keys()),
            "deterministic": bool(np.array_equal(np.asarray(first[object_name]), np.asarray(second[object_name]))),
        }

    def exposed_bake_albedo(self, texture_size: int = 128, samples: int = 4) -> dict:
        """Bake a per-pixel albedo map for the active object.

        Returns:
            dict: ``baked`` is ``"ok"`` with ``shape``/``mean``/``std``, else ``None``.
        """
        from visionsim.simulate.heatsim import irradiance

        self._register()
        self._enable_cycles(samples=samples)
        baked = irradiance.bake_albedo_map(bpy.context.scene, bpy.context.active_object, texture_size)
        if baked is None:
            return {"baked": None}
        px = baked.pixels
        return {"baked": "ok", "shape": list(px.shape), "mean": float(px.mean()), "std": float(px.std())}

    def exposed_bake_vertex_albedo(self, texture_size: int = 128, samples: int = 4, stale_zeros: bool = False) -> dict:
        """Bake per-vertex albedo, optionally after stamping a stale all-zero attribute.

        Args:
            texture_size: bake texture size.
            samples: Cycles samples.
            stale_zeros: seed a zero-valued ``albedo`` attribute first, so a correct bake
                must ignore it and re-bake rather than serve the zeros.

        Returns:
            dict: ``baked`` is ``"ok"`` with ``mean``/``std``, else ``None``.
        """
        from visionsim.simulate.heatsim import irradiance

        self._register()
        self._enable_cycles(samples=samples)
        obj = bpy.context.active_object
        if stale_zeros:
            attr = obj.data.attributes.new(name="albedo", type="FLOAT", domain="POINT")
            attr.data.foreach_set("value", np.zeros(len(obj.data.vertices), dtype=np.float64))
        albedo = irradiance.bake_vertex_albedo(bpy.context.scene, obj, texture_size=texture_size)
        if albedo is None:
            return {"baked": None}
        return {"baked": "ok", "mean": float(albedo.mean()), "std": float(albedo.std())}

    def exposed_bake_irradiance(self, texture_size: int = 64, samples: int = 4, with_albedo: bool = False) -> dict:
        """Bake the Cycles irradiance map and (optionally) the vertex albedo alongside it.

        Args:
            texture_size: bake texture size.
            samples: Cycles samples.
            with_albedo: also bake per-vertex albedo so the two lengths can be compared.

        Returns:
            dict: ``flux_n``/``albedo_n`` lengths against the evaluated and base vertex counts.
        """
        from visionsim.simulate.heatsim import irradiance

        self._register()
        self._enable_cycles(samples=samples)
        obj = bpy.context.active_object
        flux = irradiance.bake_irradiance_map(bpy.context.scene, obj, texture_size, samples=samples)
        albedo = (
            irradiance.bake_vertex_albedo(bpy.context.scene, obj, texture_size=texture_size) if with_albedo else None
        )
        dg = bpy.context.evaluated_depsgraph_get()
        return {
            "base_n": len(obj.data.vertices),
            "eval_n": len(obj.evaluated_get(dg).data.vertices),
            "flux_n": None if flux is None else len(flux.vertex_flux),
            "albedo_n": None if albedo is None else len(albedo),
        }

    def exposed_prepare_bake_uv(self) -> dict:
        """Create a bake UV on the active object and report its atlas eligibility.

        Returns:
            dict: UV-layer counts before/after and whether the object reached the atlas plan.
        """
        from visionsim.simulate.heatsim import adapter, irradiance
        from visionsim.simulate.heatsim.names import BAKE_UV_LAYER_NAME

        self._register()
        obj = bpy.context.active_object
        while obj.data.uv_layers:
            obj.data.uv_layers.remove(obj.data.uv_layers[0])
        before = len(obj.data.uv_layers)
        irradiance.prepare_object_bake_uv(obj)
        plan = adapter.build_atlas_plan(
            bpy.context.scene,
            [obj],
            {"atlas_texel_density": 1500.0, "atlas_tile_min": 16, "atlas_tile_max": 512, "atlas_texel_soft_max": 500000},
        )
        return {
            "uv_before": int(before),
            "has_bake_uv": BAKE_UV_LAYER_NAME in obj.data.uv_layers,
            "in_plan": obj.name in plan.texels,
            "texels": sorted(plan.texels),
        }

    def exposed_absorbed_flux(self, object_name: str = "ThermalPlane") -> dict:
        """Compute absorbed flux for the active object via ``adapter._compute_irradiance_cycles``.

        Returns:
            dict: shape, finiteness and spread of the absorbed-flux field.
        """
        from visionsim.simulate.heatsim import adapter

        self._register()
        obj = bpy.data.objects[object_name]
        flux = adapter._compute_irradiance_cycles(bpy.context.scene, [obj], self._solver_cfg(), self._defaults())[obj]
        return {
            "shape": list(flux.shape),
            "vertices": len(obj.data.vertices),
            "finite": bool(np.all(np.isfinite(flux))),
            "mean": float(flux.mean()),
            "std": float(flux.std()),
        }

    def exposed_gray_body_radiance(self, t_hot: float = 350.0, samples: int = 16, resolution: int = 32) -> dict:
        """Render a plane at each emissivity and report the median radiance.

        Args:
            t_hot: surface temperature in Kelvin.
            samples: Cycles samples.
            resolution: square render resolution.

        Returns:
            dict: ambient/hot temperatures and ``rendered`` radiance keyed by emissivity.
        """
        from visionsim.simulate.heatsim import thermal_shader as ts
        from visionsim.simulate.heatsim.physics import AMBIENT_TEMPERATURE_K, STEFAN_BOLTZMANN_SI

        self._register()
        rendered: dict[str, float] = {}
        for eps in (0.05, 0.50, 0.98):
            self._clear()
            sc = bpy.context.scene
            sc.render.engine = "CYCLES"
            sc.cycles.device = "CPU"
            sc.cycles.samples = samples
            sc.render.resolution_x = sc.render.resolution_y = resolution
            sc.render.film_transparent = True
            bpy.ops.mesh.primitive_plane_add(size=3.0, location=(0, 0, 0))
            obj = bpy.context.active_object
            n = len(obj.data.vertices)
            for attr_name, val in (("sim_temperature", t_hot), ("emissivity", eps)):
                attr = obj.data.attributes.new(name=attr_name, type="FLOAT", domain="POINT")
                attr.data.foreach_set("value", np.full(n, val, dtype=np.float32))
            obj["heatsim_default_temperature"] = t_hot
            bpy.ops.object.camera_add(location=(0, 0, 6))
            sc.camera = bpy.context.object
            sc.camera.data.type = "ORTHO"
            sc.camera.data.ortho_scale = 4.0
            state = ts.enter_thermal_scene(sc, radiance_scale=1.0)
            try:
                out = str(Path(tempfile.mkdtemp()) / "r.exr")
                sc.render.image_settings.file_format = "OPEN_EXR"
                sc.render.image_settings.color_depth = "32"
                sc.render.filepath = out
                bpy.ops.render.render(write_still=True)
            finally:
                ts.restore_scene(sc, state)
            img = bpy.data.images.load(out)
            w, h = img.size
            px = np.array(img.pixels[:], dtype=np.float64).reshape(h, w, 4)
            lit = px[:, :, 3] > 0.5
            rendered[f"{eps:.2f}"] = float(np.median(px[:, :, 0][lit]))
        return {
            "t_amb": float(AMBIENT_TEMPERATURE_K),
            "t_hot": t_hot,
            "sigma": float(STEFAN_BOLTZMANN_SI),
            "rendered": rendered,
        }

    def exposed_write_atlas(
        self,
        root_path: str,
        tiles: dict[str, dict[str, Any]],
        texels: dict[str, list[list[int]]],
        values: dict[str, list[float]],
        atlas_size: list[int],
        digest: str = "test",
    ) -> dict:
        """Write an atlas image from explicit tiles/texels/values and sample the result.

        Args:
            root_path: output root directory.
            tiles: ``{obj: {"size": [w, h], "offset": [x, y]}}``.
            texels: ``{obj: [[x, y], ...]}``.
            values: ``{obj: [K, ...]}`` (one per texel, in the final timestep).
            atlas_size: ``[w, h]``.
            digest: plan digest string.

        Returns:
            dict: written path, size, per-texel/neighbour samples, and unwritten-corner data.
        """
        from visionsim.simulate.heatsim import adapter, atlas

        tile_specs = {
            name: atlas.TileSpec(obj_name=name, size=tuple(spec["size"]), offset=tuple(spec["offset"]))
            for name, spec in tiles.items()
        }
        layout = atlas.AtlasLayout(
            atlas_size=tuple(atlas_size), tiles=tile_specs, effective_density=500.0, rescaled=False
        )
        texel_map = {name: {"xy": np.array(coords, dtype=np.int64)} for name, coords in texels.items()}
        plan = adapter.AtlasPlan(layout=layout, texels=texel_map, digest=digest)
        history = {name: np.array([vals]) for name, vals in values.items()}

        out_path = adapter.write_atlas(history, plan, Path(root_path))
        img = bpy.data.images.load(str(out_path))
        w, h = img.size
        px = np.array(img.pixels[:], dtype=np.float64).reshape(h, w, 4)
        unwritten = np.argwhere(px[:, :, 3] == 0.0)
        unwritten_rc = [int(unwritten[0][0]), int(unwritten[0][1])] if unwritten.size else None
        # First texel of the first object, and the sample(s) the caller wants are addressed
        # by coordinates the tests already know, so report the raw corner samples too.
        return {
            "atlas": str(out_path),
            "name": out_path.name,
            "parent_prefix": out_path.parent.name.split("_")[0],
            "exists": bool(out_path.exists()),
            "size": [int(w), int(h)],
            "unwritten_count": int(unwritten.size),
            "unwritten_rc": unwritten_rc,
            "unwritten_value": [
                float(px[unwritten_rc[0], unwritten_rc[1], 0]),
                float(px[unwritten_rc[0], unwritten_rc[1], 3]),
            ]
            if unwritten_rc
            else None,
            "all_alpha_zero": bool(np.all(px[3::4] == 0.0)),
        }

    def exposed_load_atlas(self, atlas_path: str, samples: dict[str, list[int]]) -> dict:
        """Load a written atlas image and sample named pixel coordinates.

        Args:
            atlas_path: path to the EXR written by :meth:`exposed_write_atlas`.
            samples: ``{label: [row, col]}`` coordinates to read.

        Returns:
            dict: ``{label: [value, alpha]}`` samples, so tests avoid holding the image.
        """
        img = bpy.data.images.load(str(atlas_path))
        w, h = img.size
        px = np.array(img.pixels[:], dtype=np.float64).reshape(h, w, 4)
        return {
            "size": [int(w), int(h)],
            "samples": {
                label: [float(px[rc[0], rc[1], 0]), float(px[rc[0], rc[1], 3])] for label, rc in samples.items()
            },
        }

    def exposed_configure_thermal(self, root_path: str, blend_path: str, **config: Any) -> dict:
        """Save the current scene, initialize the service on it and configure TEXEL.

        Args:
            root_path: service root/output directory.
            blend_path: where to save the scene before configuring.
            **config: fields layered over the default TEXEL ``ThermalConfig``.

        Returns:
            dict: whether the object joined the atlas, its coverage/default temperature,
            the atlas image state, and material/node structure.
        """
        from dataclasses import asdict

        from visionsim.simulate.blender import BlenderService
        from visionsim.simulate.config import ThermalConfig
        from visionsim.simulate.heatsim.names import ATLAS_COVERAGE_PROP, ATLAS_IMAGE_NAME

        self._register()
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)

        service = BlenderService()
        service.exposed_initialize(blend_path, root_path)
        values = {
            "render_domain": "TEXEL",
            "atlas_texel_density": 64.0,
            "atlas_tile_min": 16,
            "atlas_tile_max": 64,
            "atlas_texel_soft_max": 500_000,
            "device": "cpu",
            "sim_time_s": 0.1,
            "timestep_s": 0.05,
            "initial_temperature_K": 295.0,
            "thermal_diffusivity_mm2_s": 0.17,
            "density_kg_m3": 1330.0,
            "specific_heat_J_kgK": 880.0,
            "emissivity": 0.9,
            "irradiance_scale": 100.0,
        } | config
        if "configure_thermal" in values:
            raise ValueError("configure_thermal is reserved")
        service.exposed_configure_thermal(asdict(ThermalConfig(**values)))

        objects = [o for o in bpy.data.objects if o.heat_simulation_enabled]
        obj = objects[0]
        plan = service._thermal_atlas_plan
        atlas_img = bpy.data.images.get(ATLAS_IMAGE_NAME)
        live_mat = obj.material_slots[0].material if obj.material_slots else None
        nodes = live_mat.node_tree.nodes if live_mat is not None else []
        return {
            "in_plan": bool(plan is not None and obj.name in plan.texels),
            "texels": sorted(plan.texels) if plan is not None else [],
            "obj_name": obj.name,
            "coverage": float(obj[ATLAS_COVERAGE_PROP]),
            "default_temperature": float(obj.get("heatsim_default_temperature", 0.0)),
            "has_vertex_attr": "sim_temperature" in obj.data.attributes,
            "atlas_loaded": atlas_img is not None,
            "atlas_packed": bool(atlas_img is not None and atlas_img.packed_file is not None),
            "aov_count": sum(1 for n in nodes if n.type == "OUTPUT_AOV"),
            "atlas_tex_count": sum(1 for n in nodes if n.bl_idname == "ShaderNodeTexImage" and n.image is atlas_img),
            "has_temperature_pass": "temperature" in service.render_layers.outputs,
        }

    def exposed_freeze(self, blend_path: str, freeze_path: str) -> dict:
        """Save the live configured scene, delete the atlas file, reopen, report survival.

        The atlas is produced by :meth:`exposed_configure_thermal` on the *live* scene (it
        is packed there), so this saves that scene directly rather than reloading the
        pre-configure blend from disk. The cache directory is the initialized blend path
        with ``.heatsim`` appended, so deriving it here avoids the caller passing a name
        that silently fails to match.

        Args:
            blend_path: the blend path ``exposed_configure_thermal`` initialized from.
            freeze_path: where to write the frozen blend.

        Returns:
            dict: whether the atlas image survived as a packed, wired datablock.
        """
        from visionsim.simulate.heatsim.names import ATLAS_IMAGE_NAME

        saved_persistent = bpy.context.scene.render.use_persistent_data
        bpy.ops.wm.save_as_mainfile(filepath=freeze_path)
        for atlas_file in Path(f"{blend_path}.heatsim").glob("atlas_*/atlas_temperature.exr"):
            atlas_file.unlink()
        bpy.ops.wm.open_mainfile(filepath=freeze_path)
        reopened = bpy.data.images.get(ATLAS_IMAGE_NAME)
        return {
            "persistent_restored": bpy.context.scene.render.use_persistent_data == saved_persistent,
            "reopened_packed": bool(reopened is not None and reopened.packed_file is not None),
            "reopened_wired": any(
                node.bl_idname == "ShaderNodeTexImage" and node.image is reopened
                for mat in bpy.data.materials
                if mat.use_nodes and mat.node_tree
                for node in mat.node_tree.nodes
            ),
        }

    def exposed_save_scene(self, scene_path: str) -> dict:
        """Save the current scene to ``scene_path`` and report its source-identity digest.

        The digest is what the cache keys on, so a clean save is a precondition for a
        cache hit in a later process.

        Args:
            scene_path: destination blend path.

        Returns:
            dict: whether a source identity could be derived from the saved scene.
        """
        from visionsim.simulate.heatsim import cache

        bpy.ops.wm.save_as_mainfile(filepath=scene_path)
        source = cache.source_identity(bpy.data)
        return {"source_is_none": source is None, "scene": scene_path}

    def exposed_cache_solve(self, root_path: str, scene_path: str | None = None, forbid_bake: bool = False) -> dict:
        """Solve with the cache, optionally reopening a saved blend and blocking the bake.

        Args:
            root_path: cache root directory.
            scene_path: if set, open this blend first (the reuse leg of the round trip).
            forbid_bake: replace the irradiance bake with a raising stub, so a cache hit
                must still succeed, then confirm ``recompute=True`` does reach the bake.

        Returns:
            dict: source identity presence, cache-hit flag and recompute behaviour.
        """
        from visionsim.simulate.heatsim import adapter, cache

        self._register()
        if scene_path is not None:
            bpy.ops.wm.open_mainfile(filepath=scene_path)
        defaults = self._defaults()
        settings = self._solver_cfg(bake_samples=4, irradiance_texture_size=64)
        cache_root = Path(root_path) / "cache"
        # ``source_identity`` reports dirty on some headless Blender builds even for a
        # freshly saved file; use the saved file's digest for this controlled scene.
        source = cache.file_digest(Path(bpy.data.filepath))

        propagated = False
        if forbid_bake:

            def forbid(*args: Any, **kwargs: Any) -> None:
                raise RuntimeError("bake was reached")

            adapter._compute_irradiance = forbid
        history = adapter.solve_scene(
            bpy.context.scene, defaults=defaults, solver_cfg=settings, cache_root=cache_root, source_digest=source
        )
        if forbid_bake:
            try:
                adapter.solve_scene(
                    bpy.context.scene,
                    defaults=defaults,
                    solver_cfg=settings,
                    cache_root=cache_root,
                    source_digest=source,
                    recompute=True,
                )
            except RuntimeError as exc:
                propagated = str(exc) == "bake was reached"
        return {
            "source_is_none": source is None,
            "solved": "Grid" in history,
            "recompute_reached_bake": bool(propagated),
        }

    def exposed_temperature_range(self) -> dict:
        """Exercise ``adapter.global_temperature_range`` on synthetic fields.

        Returns:
            dict: outlier-robust range plus the empty and near-uniform fallbacks.
        """
        from visionsim.simulate.heatsim import adapter

        final = np.full(1000, 295.0)
        final[:100] = np.linspace(296.0, 310.0, 100)
        final[-3:] = 2000.0
        hist = {"room": np.stack([np.full(1000, 295.0), final])}
        tmin, tmax = adapter.global_temperature_range(hist, default_K=295.0)
        empty = adapter.global_temperature_range({}, 295.0)
        uniform = adapter.global_temperature_range({"o": np.full((2, 4), 300.0)}, 300.0)
        pooled = adapter.global_temperature_range(
            {
                "vertex_mesh": np.stack([np.full(50, 295.0), np.full(50, 295.5)]),
                "atlas_mesh": np.stack([np.full(400, 295.0), np.full(400, 340.0)]),
            },
            default_K=295.0,
        )
        return {
            "tmin": float(tmin),
            "tmax": float(tmax),
            "empty": [float(empty[0]), float(empty[1])],
            "uniform": [float(uniform[0]), float(uniform[1])],
            "pooled": [float(pooled[0]), float(pooled[1])],
        }

    def exposed_preview_nodegroup(self, tmin: float = 295.0, tmax: float = 297.0) -> dict:
        """Build the thermal preview compositor node group and report its ramp.

        Returns:
            dict: map-range bounds and the inferno ramp's stop count/positions/colors.
        """
        from visionsim.simulate.nodes import thermal_preview_node_group

        ng = thermal_preview_node_group(tmin=tmin, tmax=tmax)
        mr = ng.nodes["TempNormalize"]
        ramp = ng.nodes["InfernoRamp"].color_ramp
        lo, hi = ramp.elements[0], ramp.elements[10]
        return {
            "mr_min": float(mr.inputs[1].default_value),
            "mr_max": float(mr.inputs[2].default_value),
            "n_stops": len(ramp.elements),
            "lo_pos": float(lo.position),
            "hi_pos": float(hi.position),
            "lo_color": [float(c) for c in lo.color],
            "hi_color": [float(c) for c in hi.color],
        }

    def exposed_thermal_props(self, emissivity: float = 0.7) -> dict:
        """Register heatsim and round-trip the per-object thermal PropertyGroup.

        Returns:
            dict: the emissivity read back and whether the enable flag exists.
        """
        self._register()
        obj = bpy.data.objects.new("o", bpy.data.meshes.new("m"))
        obj.heat_sim_material.emissivity = emissivity
        return {
            "emissivity": float(obj.heat_sim_material.emissivity),
            "has_enabled_prop": bool(hasattr(obj, "heat_simulation_enabled")),
        }

    def exposed_temperature_aov(self) -> dict:
        """Add a cube and run ``setup_temperature_aov`` on the view layer.

        Returns:
            dict: the view layer's AOV names.
        """
        from visionsim.simulate.heatsim import thermal_shader as ts

        bpy.ops.mesh.primitive_cube_add()
        vl = bpy.context.view_layer
        ts.setup_temperature_aov(bpy.context.scene, vl)
        return {"aov_names": [a.name for a in vl.aovs]}

    def exposed_shader_graphs(self, register_atlas: bool = True, aov: bool = True) -> dict:
        """Build the gray-body (and optionally AOV) material graphs and summarize wiring.

        Args:
            register_atlas: register the atlas image datablock before building the graph,
                so the Image Texture node resolves it (an unregistered graph leaves the
                texture imageless -- the VERTEX fallback).
            aov: also build the AOV material graph.

        Returns:
            dict: ``gray`` and (when requested) ``aov`` structural summaries.
        """
        from visionsim.simulate.heatsim import thermal_shader as ts
        from visionsim.simulate.heatsim.names import ATLAS_COVERAGE_PROP, ATLAS_IMAGE_NAME, ATLAS_UV_LAYER_NAME

        if register_atlas:
            img = bpy.data.images.new(ATLAS_IMAGE_NAME, width=2, height=2, alpha=True, float_buffer=True)
            img.pixels.foreach_set([310.0, 310.0, 310.0, 1.0] * 4)

        def _upstream(node: Any, links: Any, max_hops: int = 4) -> set:
            seen, frontier = set(), [node]
            for _ in range(max_hops):
                nxt = [ln.from_node for ln in links if ln.to_node in frontier]
                nxt = [n for n in nxt if n not in seen]
                if not nxt:
                    break
                seen.update(nxt)
                frontier = nxt
            return seen

        def summarize(tree: Any, label: str) -> dict:
            nodes, links = tree.nodes, tree.links
            attrs = {n.attribute_name: n for n in nodes if n.bl_idname == "ShaderNodeAttribute"}
            tex_nodes = [n for n in nodes if n.bl_idname == "ShaderNodeTexImage"]
            tex = tex_nodes[0] if tex_nodes else None
            mix = next(
                (
                    n
                    for n in nodes
                    if n.bl_idname == "ShaderNodeMix"
                    and any(
                        link.to_socket == n.inputs["B"]
                        and link.from_node.type == "MATH"
                        and link.from_node.operation == "DIVIDE"
                        for link in links
                    )
                ),
                None,
            )
            gate_sources = _upstream(mix, links) if mix is not None else set()
            factor_node = (
                next((ln.from_node for ln in links if ln.to_socket == mix.inputs["Factor"]), None)
                if mix is not None
                else None
            )
            pow4 = next((n for n in nodes if n.bl_idname == "ShaderNodeMath" and n.operation == "POWER"), None)
            divide = (
                next(
                    ln.from_node
                    for ln in links
                    if ln.to_socket == mix.inputs["B"]
                    and ln.from_node.type == "MATH"
                    and ln.from_node.operation == "DIVIDE"
                )
                if mix is not None
                else None
            )
            summary = {
                "label": label,
                "has_uv_attr": ATLAS_UV_LAYER_NAME in attrs,
                "uv_attr_geometry": ATLAS_UV_LAYER_NAME in attrs
                and attrs[ATLAS_UV_LAYER_NAME].attribute_type == "GEOMETRY",
                "has_coverage_attr": ATLAS_COVERAGE_PROP in attrs,
                "coverage_attr_object": ATLAS_COVERAGE_PROP in attrs
                and attrs[ATLAS_COVERAGE_PROP].attribute_type == "OBJECT",
                "has_sim_temp_attr": "sim_temperature" in attrs,
                "has_default_temp_attr": "heatsim_default_temperature" in attrs,
                "n_tex_nodes": len(tex_nodes),
                "tex_image_name": tex.image.name if tex is not None and tex.image is not None else None,
                "tex_non_color": tex is not None
                and tex.image is not None
                and tex.image.colorspace_settings.name == "Non-Color",
                "tex_linear": tex is not None and tex.interpolation == "Linear",
                "tex_clip": tex is not None and tex.extension == "CLIP",
                "uv_feeds_vector": any(
                    link.from_node.bl_idname == "ShaderNodeAttribute"
                    and link.from_node.attribute_name == ATLAS_UV_LAYER_NAME
                    and link.to_node in tex_nodes
                    and link.to_socket.identifier == "Vector"
                    for link in links
                ),
                "has_mix": mix is not None,
                "mix_float": mix is not None and mix.data_type == "FLOAT",
                "mix_factor_linked": mix is not None and mix.inputs["Factor"].is_linked,
                "mix_ab_linked": bool(mix is not None and mix.inputs["A"].is_linked and mix.inputs["B"].is_linked),
                "factor_multiply": factor_node is not None
                and factor_node.bl_idname == "ShaderNodeMath"
                and factor_node.operation == "MULTIPLY",
                "factor_from_tex": any(n in gate_sources for n in tex_nodes),
                "factor_has_gate": any(
                    n.bl_idname == "ShaderNodeAttribute" and n.attribute_name == ATLAS_COVERAGE_PROP
                    for n in gate_sources
                ),
                "mix_feeds_pow4": bool(
                    pow4 is not None and any(ln.from_node == mix for ln in links if ln.to_node == pow4)
                ),
                # Mix B is the filtered temperature divided by the filtered coverage: the
                # numerator must separate the texture's red channel and the divisor must be
                # the texture's alpha, or atlas-edge pixels read the wrong temperature.
                "b_divide": divide is not None,
                "b_separate_color": bool(
                    divide is not None
                    and any(
                        ln.from_node.bl_idname == "ShaderNodeSeparateColor"
                        for ln in links
                        if ln.to_node == divide and ln.to_socket == divide.inputs[0]
                    )
                ),
                "tex_feeds_separate_color": bool(
                    divide is not None
                    and any(
                        ln.from_node in tex_nodes and ln.to_node.bl_idname == "ShaderNodeSeparateColor" for ln in links
                    )
                ),
                "alpha_feeds_divide": bool(
                    divide is not None
                    and any(
                        ln.from_node in tex_nodes
                        and ln.from_socket.name == "Alpha"
                        and ln.to_node == divide
                        and ln.to_socket == divide.inputs[1]
                        for ln in links
                    )
                ),
            }
            return summary

        gray_tree = ts._build_gray_body_material(1.0).node_tree
        gray = summarize(gray_tree, "gray-body")

        result: dict[str, Any] = {"gray": gray}
        if aov:
            mat = bpy.data.materials.new("atlas_aov_mat")
            mat.use_nodes = True
            ts._append_temperature_aov_nodes(mat, "temperature")
            aov_summary = summarize(mat.node_tree, "aov")
            aov_node = next(n for n in mat.node_tree.nodes if n.type == "OUTPUT_AOV")
            reached, frontier = set(), [n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeMix"]
            for _ in range(4):
                nxt = [
                    ln.to_node for ln in mat.node_tree.links if ln.from_node in frontier and ln.to_node not in reached
                ]
                if not nxt:
                    break
                reached.update(nxt)
                frontier = nxt
            aov_summary["mix_reaches_aov"] = aov_node in reached
            aov_summary["value_linked"] = bool(aov_node.inputs["Value"].is_linked)
            result["aov"] = aov_summary
        return result

    def exposed_render_temperature(self, root_path: str, tmin: float | None = None) -> dict:
        """Build the atlas-edge scene, render the temperature pass, and report pixel range.

        Args:
            root_path: output directory for the EXR.
            tmin: optional uniform stamp temperature; defaults to the edge fixture.

        Returns:
            dict: min/max of the rendered temperature channel and pixel count.
        """
        from visionsim.simulate.compat import file_output_node
        from visionsim.simulate.heatsim import thermal_shader
        from visionsim.simulate.heatsim.names import ATLAS_COVERAGE_PROP, ATLAS_IMAGE_NAME, ATLAS_UV_LAYER_NAME

        root = Path(root_path)
        bpy.ops.mesh.primitive_plane_add(size=2)
        plane = bpy.context.active_object
        material = bpy.data.materials.new("wall")
        material.use_nodes = True
        plane.data.materials.append(material)
        uv = plane.data.uv_layers.new(name=ATLAS_UV_LAYER_NAME)
        for loop in uv.data:
            loop.uv = (0.4, 0.25)
        plane[ATLAS_COVERAGE_PROP] = 1.0

        image = bpy.data.images.new(ATLAS_IMAGE_NAME, width=2, height=2, alpha=True, float_buffer=True)
        image.colorspace_settings.name = "Non-Color"
        image.pixels.foreach_set([295.0, 295.0, 295.0, 1.0, 0.0, 0.0, 0.0, 0.0] * 2)
        image.update()
        image.pack()

        bpy.ops.object.camera_add(location=(0, 0, 2))
        camera = bpy.context.active_object
        camera.data.type = "ORTHO"
        camera.data.ortho_scale = 2
        scene = bpy.context.scene
        scene.camera = camera
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 1
        if hasattr(scene.render, "compositor_device"):
            scene.render.compositor_device = "CPU"
        scene.render.resolution_x = 16
        scene.render.resolution_y = 16
        scene.render.resolution_percentage = 100
        thermal_shader.stamp_default_temperatures(scene, default_K=tmin if tmin is not None else 295.0)
        thermal_shader.setup_temperature_aov(scene, bpy.context.view_layer)

        # Blender >= 5.0 uses a named compositing node group; older versions the scene node tree.
        if bpy.app.version >= (5, 0, 0):
            bpy.ops.node.new_compositing_node_group(name="Compositor Nodes")
            scene.compositing_node_group = bpy.data.node_groups["Compositor Nodes"]
            tree = scene.compositing_node_group
        else:
            scene.use_nodes = True
            tree = scene.node_tree
        scene.render.use_compositing = True
        tree.nodes.clear()
        layers = tree.nodes.new("CompositorNodeRLayers")
        output, sockets, _ = file_output_node(tree, root, slot_names=(("temp", "RGBA"),))
        output.format.file_format = "OPEN_EXR"
        output.format.color_mode = "RGB"
        output.format.color_depth = "32"
        tree.links.new(layers.outputs["temperature"], sockets[0])
        bpy.ops.render.render()

        loaded = bpy.data.images.load(str(next(root.glob("temp*.exr"))))
        pixels = np.empty(16 * 16 * 4, dtype=np.float32)
        loaded.pixels.foreach_get(pixels)
        temperature = pixels.reshape(-1, 4)[:, 0]
        return {"min": float(temperature.min()), "max": float(temperature.max()), "n": int(temperature.size)}

    def exposed_enter_restore(self) -> dict:
        """Enter and restore the thermal scene three times on a shared-mesh setup.

        Returns:
            dict: whether source assignments were preserved, overrides applied, and state
            fully restored after each round trip.
        """
        from visionsim.simulate.heatsim import thermal_shader as ts

        bpy.ops.mesh.primitive_cube_add()
        a = bpy.context.object
        materials = [bpy.data.materials.new(name) for name in ("First", "Second", "ObjectOverride")]
        for mat in materials[:2]:
            a.data.materials.append(mat)
        for face in a.data.polygons:
            face.material_index = face.index % 2
        b = a.copy()
        bpy.context.collection.objects.link(b)
        b.material_slots[1].link = "OBJECT"
        b.material_slots[1].material = materials[2]
        scene = bpy.context.scene
        extra = scene.view_layers.new("ExistingOverride")
        extra.material_override = materials[0]
        light = next(o for o in scene.objects if o.type == "LIGHT")

        def snapshot() -> tuple:
            return (
                tuple(a.data.materials),
                tuple(p.material_index for p in a.data.polygons),
                tuple((o.data, tuple((s.link, s.material) for s in o.material_slots)) for o in (a, b)),
                tuple(v.material_override for v in scene.view_layers),
                scene.world,
                light.hide_render,
                light.hide_viewport,
            )

        before = snapshot()
        setup_preserved = True
        override_ok = True
        for _ in range(3):
            state = ts.enter_thermal_scene(scene, radiance_scale=1.0)
            if snapshot()[:3] != before[:3]:
                setup_preserved = False
            if not all(v.material_override == ts._build_gray_body_material(1.0) for v in scene.view_layers):
                override_ok = False
            ts.restore_scene(scene, state)
        return {
            "setup_preserved": bool(setup_preserved),
            "override_ok": bool(override_ok),
            "restored": snapshot() == before,
        }

    def exposed_default_surface(self) -> dict:
        """Give bare/authored/empty-slot meshes a temperature AOV and report slot state.

        Returns:
            dict: slot counts and AOV counts per object, whether the authored material was
            kept, whether the default surface is shared, and whether a second pass is a no-op.
        """
        from visionsim.simulate.heatsim import thermal_shader

        self._register()
        for o in list(bpy.data.objects):
            bpy.data.objects.remove(o, do_unlink=True)

        def plane(name: str) -> Any:
            bpy.ops.mesh.primitive_plane_add()
            obj = bpy.context.active_object
            obj.name = name
            return obj

        bare = plane("bare")
        bare.data.materials.clear()
        authored = plane("authored")
        mat = bpy.data.materials.new("authored_m")
        mat.use_nodes = True
        authored.data.materials.append(mat)
        half = plane("half")
        half.data.materials.append(bpy.data.materials.new("half_m"))
        half.data.materials["half_m"].use_nodes = True
        half.data.materials.append(None)

        # Fixture preconditions, captured before the setup pass: 'bare' starts slotless and
        # 'half' starts with an empty slot, so the test proves it exercises those cases.
        bare_slots_before = len(bare.material_slots)
        half_empty_before = any(s.material is None for s in half.material_slots)

        sc = bpy.context.scene
        thermal_shader.setup_temperature_aov(sc, bpy.context.view_layer)

        def aov_count(obj: Any) -> int:
            total = 0
            for slot in obj.material_slots:
                if slot.material is None:
                    return -1
                total += sum(1 for node in slot.material.node_tree.nodes if node.type == "OUTPUT_AOV")
            return total

        default_name = thermal_shader._DEFAULT_SURFACE_MATERIAL_NAME
        before = [len(o.material_slots) for o in (bare, authored, half)]
        thermal_shader.setup_temperature_aov(sc, bpy.context.view_layer)
        after = [len(o.material_slots) for o in (bare, authored, half)]
        return {
            "slots": {o.name: len(o.material_slots) for o in (bare, authored, half)},
            "aov_counts": {o.name: aov_count(o) for o in (bare, authored, half)},
            "bare_slots_before": bare_slots_before,
            "half_empty_before": half_empty_before,
            "authored_kept": authored.material_slots[0].material.name == "authored_m",
            "bare_default": bare.material_slots[0].material.name == default_name,
            "default_count": sum(1 for m in bpy.data.materials if m.name.startswith(default_name)),
            "idempotent": before == after,
        }

    def exposed_coarse_wall(
        self, densities: list[list[float]], root_path: str = "/tmp/visionsim-coarse-wall-test"
    ) -> dict:
        """Solve the wall at each (density, tile) plan and compare the fields.

        Args:
            densities: list of ``[density, tile]`` pairs, coarse first.
            root_path: cache root directory.

        Returns:
            dict: per-plan atlas membership, finiteness, spread, and coarse-vs-reference RMS.
        """
        from scipy.spatial import cKDTree

        from visionsim.simulate.heatsim import adapter

        self._register()
        objects = adapter.gather_meshes(bpy.context.scene)
        defaults = self._defaults() | {"irradiance_scale": 500}
        solver = self._solver_cfg(bake_samples=32, irradiance_texture_size=64)
        cache_root = Path(root_path)
        fields = []
        in_plan = []
        finite = []
        ptp = []
        for density, tile in densities:
            plan = adapter.build_atlas_plan(
                bpy.context.scene,
                objects,
                {
                    "render_domain": "AUTO",
                    "atlas_texel_density": density,
                    "atlas_tile_min": int(tile),
                    "atlas_tile_max": int(tile),
                    "atlas_texel_soft_max": 10000,
                },
            )
            in_plan.append("wall" in plan.texels)
            history = adapter.solve_scene(
                bpy.context.scene, defaults=defaults, solver_cfg=solver, cache_root=cache_root, atlas_plan=plan
            )
            field = history["wall"][-1]
            fields.append((plan.texels["wall"]["position_mm"], field))
            finite.append(bool(np.isfinite(field).all()))
            ptp.append(float(np.ptp(field)))
        low_points, low_field = fields[0]
        ref_points, ref_field = fields[1]
        _, nearest = cKDTree(ref_points).query(low_points)
        return {
            "in_plan": in_plan,
            "low": len(low_field),
            "ref": len(ref_field),
            "wall_vertices": len(bpy.data.objects["wall"].data.vertices),
            "finite": finite,
            "ptp": ptp,
            "rms": float(np.sqrt(np.mean((low_field - ref_field[nearest]) ** 2))),
        }

    def exposed_rgb_thermal_loop(self, root_path: str, frames: int = 3) -> dict:
        """Render a lit cube for one RGB frame then two thermal frames and check stability.

        Args:
            root_path: output directory (must be writable).
            frames: total frames to render (first RGB, rest thermal).

        Returns:
            dict: pixel-identity, radiance sanity, and cleanup/propagation flags.
        """
        from types import SimpleNamespace
        from unittest.mock import patch

        from visionsim.simulate import blender as blender_module
        from visionsim.simulate.blender import BlenderService
        from visionsim.simulate.heatsim import thermal_shader as ts

        root = Path(root_path)
        cube = bpy.data.objects["Cube"]
        cube.data.materials.clear()
        for name, color in [("Red", (0.8, 0.05, 0.02, 1)), ("Blue", (0.02, 0.1, 0.8, 1))]:
            mat = bpy.data.materials.new(name)
            mat.use_nodes = True
            mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = color
            cube.data.materials.append(mat)
        for face in cube.data.polygons:
            face.material_index = face.index % 2
        blend = root / "scene.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(blend))
        service = BlenderService()
        service.exposed_initialize(blend, root)
        scene = service.scene
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 8
        if hasattr(scene.render, "compositor_device"):
            scene.render.compositor_device = "CPU"
        scene.cycles.seed = 0
        scene.cycles.use_animated_seed = False
        scene.render.use_persistent_data = False
        scene.render.resolution_x = scene.render.resolution_y = 48
        scene.render.resolution_percentage = 100
        service.exposed_include_frames(file_format="OPEN_EXR", exr_codec="ZIP", bit_depth=32)
        scene.frame_set(1)
        service.exposed_render_current_frame(allow_skips=False)

        ts.stamp_default_temperatures(scene, default_K=310)
        ts.setup_temperature_aov(scene, service.view_layer)
        service.exposed_include_thermal(preview=False)
        for frame in range(2, frames + 1):
            scene.frame_set(frame)
            service.exposed_render_current_frame(allow_skips=False)

        def pixels(path: Path) -> Any:
            img = bpy.data.images.load(str(path), check_existing=False)
            values = np.array(img.pixels[:])
            bpy.data.images.remove(img)
            return values

        rendered = sorted((root / "frames").rglob("*.exr"))
        baseline = pixels(rendered[0]) if rendered else None
        # Allow float32 render accumulation differences across CPU builds.
        identical = all(bool(np.allclose(pixels(p), baseline, rtol=1e-5, atol=1e-6)) for p in rendered[1:])
        radiance = sorted((root / "thermal_radiance").rglob("*.exr"))
        radiance_finite = bool(np.isfinite(pixels(radiance[0])).all()) if radiance else False
        radiance_max = float(pixels(radiance[0]).max()) if radiance else 0.0

        before = (
            scene.world,
            service.view_layer.material_override,
            [(o.name, o.hide_render, o.hide_viewport) for o in scene.objects if o.type == "LIGHT"],
        )
        mutes = {name: entry[0].mute for name, entry in service._outputs.items()}
        cleanup_ok = True
        propagation_ok = True
        call_counts: list[int] = []
        for failed_pass in (1, 2):
            calls: list[int] = []

            def fail_render(*, _calls: list[int] = calls, _target: int = failed_pass, **kwargs: Any) -> None:
                _calls.append(1)
                if len(_calls) == _target:
                    raise RuntimeError("injected render failure")

            proxy = SimpleNamespace(
                app=bpy.app,
                context=bpy.context,
                ops=SimpleNamespace(render=SimpleNamespace(render=fail_render)),
            )
            with patch.object(blender_module, "bpy", proxy):
                try:
                    service.exposed_render_current_frame(allow_skips=False)
                except RuntimeError as exc:
                    if str(exc) != "injected render failure":
                        propagation_ok = False
                else:
                    propagation_ok = False
            call_counts.append(len(calls))
            after = (
                scene.world,
                service.view_layer.material_override,
                [(o.name, o.hide_render, o.hide_viewport) for o in scene.objects if o.type == "LIGHT"],
            )
            if after != before:
                cleanup_ok = False
            if {name: entry[0].mute for name, entry in service._outputs.items()} != mutes:
                cleanup_ok = False

        return {
            "n_frames": len(rendered),
            "identical": bool(identical),
            "n_radiance": len(radiance),
            "radiance_finite": radiance_finite,
            "radiance_max": radiance_max,
            "cleanup_ok": bool(cleanup_ok),
            "propagation_ok": bool(propagation_ok),
            "call_counts": call_counts,
        }
