"""Composable Blender primitives for the heatsim tests.

This module is loaded inside a spawned Blender process: ``HeatsimTestService`` is a
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

# Absolute ``/path/to/service.py:ClassName`` target for ``BlenderClient.spawn(service=...)``:
# importing this module by name would first execute every parent package's ``__init__``
# inside blender's python, which commonly lacks their dependencies.
SERVICE_PATH = f"{Path(__file__).resolve()}:HeatsimTestService"


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

    def exposed_empty_private_atlas(self) -> dict:
        """A subpixel modified surface must retain a writable render-mesh field."""
        from unittest.mock import patch

        from visionsim.simulate.config import ThermalConfig
        from visionsim.simulate.heatsim import adapter, atlas
        from visionsim.simulate.heatsim.geometry import release_meshes

        self._register()
        self._clear()
        mesh = bpy.data.meshes.new("Thin ribbon")
        mesh.from_pydata([(0, 0, 0), (2, 0, 0), (2, 0.002, 0), (0, 0.002, 0)], [], [(0, 1, 2, 3)])
        obj = bpy.data.objects.new("Thin ribbon", mesh)
        bpy.context.collection.objects.link(obj)
        obj.modifiers.new("Thickness", "SOLIDIFY").thickness = 0.0002
        bpy.context.view_layer.update()
        empty = {"xy": np.empty((0, 2), dtype=int)}
        result = {}
        with patch.object(atlas, "rasterize_tile", return_value=empty):
            plan = adapter.build_atlas_plan(bpy.context.scene, [obj], {"render_domain": "AUTO"})
            try:
                result["uses_vertices"] = obj.name not in plan.texels
                with adapter.use_atlas_geometry(bpy.context.scene, plan):
                    count = len(obj.data.vertices)
                    assert count != len(mesh.vertices)
                    adapter.write_frame_attributes(
                        bpy.context.scene,
                        {obj.name: np.full((1, count), 301.0)},
                        0,
                        vars(ThermalConfig()),
                        atlas_plan=plan,
                    )
                    values = [item.value for item in obj.data.attributes["sim_temperature"].data]
                    result["field_matches_render_mesh"] = len(values) == count and all(v == 301 for v in values)
            finally:
                release_meshes(plan.meshes)
            try:
                adapter.build_atlas_plan(bpy.context.scene, [obj], {"render_domain": "TEXEL"})
            except RuntimeError as exc:
                result["forced_texel_rejects_empty"] = "no solve points" in str(exc)
        result["source_preserved"] = (
            obj.data == mesh and len(mesh.vertices) == 4 and "sim_temperature" not in mesh.attributes
        )
        return result

    def exposed_thin_chart_irradiance(self) -> dict:
        """A repaired narrow strip must retain the same uniform lighting as its neighbors."""
        from unittest.mock import patch

        from visionsim.simulate.heatsim import irradiance

        self._register()
        self._clear()
        self._enable_cycles(samples=16)
        mesh = bpy.data.meshes.new("Thin connected strip")
        vertices = [(x, y, 0) for y in (0, 1) for x in (0, 1, 1.0002, 2)]
        mesh.from_pydata(vertices, [], [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6)])
        obj = bpy.data.objects.new("Thin connected strip", mesh)
        bpy.context.collection.objects.link(obj)
        mat = bpy.data.materials.new("Opaque strip")
        mat.use_nodes = True
        mesh.materials.append(mat)
        world = bpy.data.worlds.new("Uniform incident light")
        world.use_nodes = True
        world.node_tree.nodes.get("Background").inputs["Color"].default_value = (1, 1, 1, 1)
        world.node_tree.nodes.get("Background").inputs["Strength"].default_value = 1
        bpy.context.scene.world = world
        native = irradiance.repair_bake_uv

        def inject(projected, name, source):
            bpy.ops.object.mode_set(mode="OBJECT")
            uv = []
            for loop in projected.loops:
                x, y, _ = projected.vertices[loop.vertex_index].co
                if 1.0001 < x < 1.0003:
                    x = 0.9998
                uv.extend((0.1 + 0.4 * x, 0.1 + 0.4 * y))
            projected.uv_layers.active.data.foreach_set("uv", uv)
            bpy.ops.object.mode_set(mode="EDIT")
            native(projected, name, source)

        with patch.object(irradiance, "repair_bake_uv", side_effect=inject):
            baked = irradiance.bake_irradiance_map(bpy.context.scene, obj, 64, samples=16)
        assert baked is not None
        values = baked.vertex_flux
        pixels = np.repeat(np.arange(12, dtype=float).reshape(3, 4, 1), 3, axis=2)
        centers = [((x + 0.5) / 4, (y + 0.5) / 3) for y in range(3) for x in range(4)]
        return {
            "flux": values.tolist(),
            "pixel_centers": [irradiance._bilinear_sample(pixels, 4, 3, np.array(uv)) for uv in centers],
            "source_preserved": len(mesh.vertices) == 8 and len(mesh.polygons) == 3,
        }

    def exposed_ngon_geometry(self) -> dict:
        """Extract and sample a concave six-sided face beside a different material."""
        from visionsim.simulate.heatsim import adapter, atlas

        self._register()
        self._clear()
        mesh = bpy.data.meshes.new("concave")
        mesh.from_pydata(
            [(0, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (1, 2, 0), (0, 2, 0), (3, 0, 0), (4, 0, 0), (3, 1, 0)],
            [],
            [(0, 1, 2, 3, 4, 5), (6, 7, 8)],
        )
        obj = bpy.data.objects.new("concave", mesh)
        bpy.context.collection.objects.link(obj)
        mesh.polygons[1].material_index = 1
        uv = mesh.uv_layers.new(name="original")
        for loop in mesh.loops:
            uv.data[loop.index].uv = mesh.vertices[loop.vertex_index].co[:2]
        verts, faces, _ = adapter._extract_geometry(obj)
        loop_uv, slots = adapter._extract_evaluated_face_uv_and_slots(obj, "original")
        result = {
            "triangles": len(faces),
            "area_m2": atlas.surface_area_m2(verts, faces),
            "uv_error": float(np.max(np.abs(loop_uv - verts[faces, :2] / 1000))),
            "material_counts": np.bincount(slots, minlength=2).tolist(),
        }
        plan = adapter.build_atlas_plan(bpy.context.scene, [obj], {"render_domain": "TEXEL"})
        result["ngon_samples"] = int(np.count_nonzero(plan.texels[obj.name]["position_mm"][:, 0] < 2500))
        return result

    def exposed_evaluated_thermal_uvs(self) -> dict:
        """Unwrap a bevelled corner after evaluation and verify exception restoration."""
        from visionsim.simulate.heatsim import adapter
        from visionsim.simulate.heatsim.geometry import release_meshes

        self._register()
        self._clear()
        mesh = bpy.data.meshes.new("panels")
        mesh.from_pydata(
            [(0, 0, 0), (0, 0, 3), (4, 0, 3), (4, 0, 0), (4, 2, 0), (4, 2, 3)], [], [(0, 3, 2, 1), (3, 4, 5, 2)]
        )
        obj = bpy.data.objects.new("panels", mesh)
        bpy.context.collection.objects.link(obj)
        material = self._checker_material()
        hidden_material = bpy.data.materials.new("hidden_data_material")
        mesh.materials.append(hidden_material)
        obj.material_slots[0].link = "OBJECT"
        obj.material_slots[0].material = material
        uv = mesh.uv_layers.new(name="original")
        for loop in mesh.loops:
            uv.data[loop.index].uv = (loop.index / 10, loop.index / 20)
        original_uv = np.array([tuple(p.uv) for p in uv.data])
        solidify = obj.modifiers.new("Thickness", "SOLIDIFY")
        solidify.thickness = 0.2
        bevel = obj.modifiers.new("Bevel", "BEVEL")
        bevel.width, bevel.segments = 0.05, 2
        bpy.context.view_layer.update()
        before = adapter._extract_geometry(obj)
        evaluated_uv = np.array(
            [
                tuple(p.uv)
                for p in obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.uv_layers["original"].data
            ]
        )
        cfg = {"render_domain": "AUTO", "atlas_texel_density": 0.01, "atlas_tile_min": 128, "atlas_tile_max": 128}
        try:
            adapter._build_atlas_plan(bpy.context.scene, [obj], cfg)
            baseline_overlap = False
        except ValueError as exc:
            baseline_overlap = "Thermal UV overlap" in str(exc)
        plan = adapter.build_atlas_plan(bpy.context.scene, [obj], cfg)
        try:
            try:
                with adapter.use_atlas_geometry(bpy.context.scene, plan):
                    after = adapter._extract_geometry(obj)
                    same_geometry = np.array_equal(before[1], after[1]) and np.allclose(before[0], after[0])
                    copied = obj.data != mesh and all(not m.show_render for m in obj.modifiers)
                    source_uv_preserved = np.array_equal(
                        evaluated_uv, [tuple(p.uv) for p in obj.data.uv_layers["original"].data]
                    )
                    material_preserved = obj.material_slots[0].material == material
                    raise RuntimeError("test restoration")
            except RuntimeError as exc:
                assert str(exc) == "test restoration"
            return {
                "baseline_overlap": baseline_overlap,
                "same_geometry": bool(same_geometry),
                "copied": bool(copied),
                "source_uv_preserved": bool(source_uv_preserved),
                "material_preserved": bool(material_preserved),
                "points": len(plan.texels[obj.name]["xy"]),
                "restored": obj.data == mesh
                and all(m.show_render and m.show_viewport for m in obj.modifiers)
                and np.array_equal(original_uv, [tuple(p.uv) for p in mesh.uv_layers["original"].data])
                and obj.material_slots[0].link == "OBJECT"
                and obj.material_slots[0].material == material
                and mesh.materials[0] == hidden_material,
            }
        finally:
            release_meshes(plan.meshes)

    def exposed_render_evaluated_thermal_mesh(self, root_path: str) -> dict:
        """Exercise native baking, both render passes and saving with a render-only bevel."""
        from dataclasses import asdict

        from visionsim.simulate.config import ThermalConfig
        from visionsim.simulate.heatsim.names import ATLAS_UV_LAYER_NAME

        self._register()
        self._clear()
        bpy.ops.mesh.primitive_cube_add()
        obj = bpy.context.object
        obj.name = "bevelled"
        obj.data.materials.append(self._checker_material())
        modifier = obj.modifiers.new("Render bevel", "BEVEL")
        modifier.width, modifier.segments = 0.1, 2
        modifier.show_viewport = False
        subdivision = obj.modifiers.new("Render subdivision", "SUBSURF")
        subdivision.levels, subdivision.render_levels = 1, 1
        modifier.show_viewport = True
        bpy.context.view_layer.update()
        expected_vertices = len(obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices)
        subdivision.levels = 0
        modifier.show_viewport = False
        bpy.ops.object.camera_add(location=(4, -4, 3))
        camera = bpy.context.object
        camera.rotation_euler = (-camera.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.scene.camera = camera
        self._add_world_light()
        self._enable_cycles(samples=8)
        root = Path(root_path)
        root.mkdir(parents=True, exist_ok=True)
        source = root / "source.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(source))
        self.exposed_initialize(source, root)
        self.exposed_set_resolution(width=32, height=32)
        self.exposed_cycles_settings(device_type="cpu", use_cpu=True, max_samples=8, use_denoising=False)
        BlenderService.exposed_configure_thermal(
            self,
            asdict(
                ThermalConfig(
                    sim_time_s=1,
                    timestep_s=1,
                    device="cpu",
                    render_domain="TEXEL",
                    preview=False,
                    bake_samples=1,
                    irradiance_texture_size=16,
                    atlas_tile_min=32,
                    atlas_tile_max=32,
                )
            ),
        )
        self.exposed_render_frame(1, allow_skips=False)
        obj = self.scene.objects["bevelled"]
        restored = (
            len(obj.data.vertices) == 8
            and obj.modifiers[0].show_render
            and not obj.modifiers[0].show_viewport
            and obj.modifiers[1].levels == 0
        )
        finite = True
        for kind in ("temperature", "thermal_radiance"):
            image = bpy.data.images.load(str(next((root / kind).rglob("*.exr"))))
            values = np.array(image.pixels[:])
            finite &= bool(np.isfinite(values).all() and values.max() > 0)
            bpy.data.images.remove(image)
        frozen = root / "thermal.blend"
        self.exposed_save_file(frozen)
        bpy.ops.wm.open_mainfile(filepath=str(frozen))
        obj = bpy.context.scene.objects["bevelled"]
        return {
            "restored": bool(restored),
            "finite_outputs": bool(finite),
            "saved_final_mesh": len(obj.data.vertices) == expected_vertices
            and ATLAS_UV_LAYER_NAME in obj.data.uv_layers,
            "saved_modifiers_disabled": not any(m.show_render or m.show_viewport for m in obj.modifiers),
        }

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
            "irradiance_scale": 1.0,
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
        irradiance_scale: float = 1.0,
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

    def exposed_solve_twice(
        self, root_path: str, object_name: str, bake_samples: int = 1024, irradiance_texture_size: int = 512
    ) -> dict:
        """Solve the same scene twice and report whether the fields are identical.

        Args:
            root_path: cache root directory.
            object_name: object whose field to compare.
            bake_samples: Cycles samples for the irradiance bake.
            irradiance_texture_size: bake texture size.

        Returns:
            dict: determinism flag and the two key sets.
        """
        from visionsim.simulate.heatsim import adapter

        self._register()
        defaults = self._defaults()
        solver_cfg = self._solver_cfg(
            sim_time_s=0.15, bake_samples=bake_samples, irradiance_texture_size=irradiance_texture_size
        )
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

    def exposed_opaque_receiver_fixture(self, layered: bool = False) -> dict:
        """Bake mixed-material coplanar patches under known uniform radiance 1."""
        from unittest.mock import patch

        from visionsim.simulate.heatsim import irradiance
        from visionsim.simulate.heatsim.adapter import _sample_bilinear
        from visionsim.simulate.heatsim.names import BAKE_UV_LAYER_NAME

        self._clear()
        self._enable_cycles(samples=64)
        self._add_world_light(color=(1, 1, 1, 1))
        scene = bpy.context.scene
        scene.cycles.use_adaptive_sampling = False
        scene.cycles.seed = 71
        kinds = (
            "black",
            "gray",
            "metal",
            "alpha",
            "transmission",
            "subsurface",
            "linked_alpha",
            "linked_transmission",
            "linked_subsurface",
            "mixed",
            "translucent",
            "volume",
        )
        if layered:
            kinds = (
                "layered_add",
                "layered_mix",
                "layered_nested",
                "layered_glossy",
                "layered_transparent",
                "layered_transmission",
                "layered_subsurface",
                "layered_linked_alpha",
                "layered_normal_mismatch",
                "layered_missing",
                "layered_unknown",
                "layered_volume",
            )
        vertices, faces, materials = [], [], []
        for i, kind in enumerate(kinds):
            x = i * 3
            faces.append(tuple(range(len(vertices), len(vertices) + 4)))
            vertices.extend(((x, 0, 0), (x + 1, 0, 0), (x + 1, 1, 0), (x, 1, 0)))
            mat = bpy.data.materials.new(kind)
            mat.use_nodes = True
            nodes, links = mat.node_tree.nodes, mat.node_tree.links
            surface = nodes.get("Principled BSDF")
            output = nodes.get("Material Output")
            color = {"black": 0.0, "gray": 0.18, "metal": 0.6}.get(kind, 0.2)
            surface.inputs["Base Color"].default_value = (color, color, color, 1)
            if kind == "metal":
                surface.inputs["Metallic"].default_value = 1
            socket_names = {
                "alpha": "Alpha",
                "transmission": "Transmission Weight" if "Transmission Weight" in surface.inputs else "Transmission",
                "subsurface": "Subsurface Weight" if "Subsurface Weight" in surface.inputs else "Subsurface",
            }
            if kind in socket_names:
                surface.inputs[socket_names[kind]].default_value = 0.5
            elif kind.startswith("linked_"):
                name = kind.removeprefix("linked_")
                value = nodes.new("ShaderNodeValue")
                value.outputs[0].default_value = 1 if name == "alpha" else 0
                links.new(value.outputs[0], surface.inputs[socket_names[name]])
            elif kind == "mixed":
                mix = nodes.new("ShaderNodeMixShader")
                transparent = nodes.new("ShaderNodeBsdfTransparent")
                links.new(surface.outputs[0], mix.inputs[1])
                links.new(transparent.outputs[0], mix.inputs[2])
                links.new(mix.outputs[0], output.inputs["Surface"])
            elif kind == "translucent":
                links.new(nodes.new("ShaderNodeBsdfTranslucent").outputs[0], output.inputs["Surface"])
            elif kind == "volume":
                links.new(nodes.new("ShaderNodeVolumeAbsorption").outputs[0], output.inputs["Volume"])
            if layered:
                second = nodes.new("ShaderNodeBsdfGlossy")
                second.inputs["Color"].default_value = (0.3, 0.3, 0.3, 1)
                combine = nodes.new("ShaderNodeMixShader" if kind == "layered_mix" else "ShaderNodeAddShader")
                branches = [socket for socket in combine.inputs if socket.type == "SHADER"]
                links.new(surface.outputs[0], branches[0])
                links.new(second.outputs[0], branches[1])
                links.new(combine.outputs[0], output.inputs["Surface"])
                if kind == "layered_add":
                    bump = nodes.new("ShaderNodeBump")
                    for branch in (surface, second):
                        links.new(bump.outputs["Normal"], branch.inputs["Normal"])
                elif kind == "layered_nested":
                    mix = nodes.new("ShaderNodeMixShader")
                    links.new(combine.outputs[0], mix.inputs[1])
                    links.new(nodes.new("ShaderNodeBsdfDiffuse").outputs[0], mix.inputs[2])
                    links.new(mix.outputs[0], output.inputs["Surface"])
                elif kind == "layered_glossy":
                    links.new(second.outputs[0], output.inputs["Surface"])
                elif kind == "layered_transparent":
                    links.new(nodes.new("ShaderNodeBsdfTransparent").outputs[0], branches[1])
                elif kind in {"layered_transmission", "layered_subsurface"}:
                    surface.inputs[socket_names[kind.removeprefix("layered_")]].default_value = 0.5
                elif kind == "layered_linked_alpha":
                    value = nodes.new("ShaderNodeValue")
                    value.outputs[0].default_value = 1
                    links.new(value.outputs[0], surface.inputs["Alpha"])
                elif kind == "layered_normal_mismatch":
                    links.new(nodes.new("ShaderNodeNewGeometry").outputs["Normal"], second.inputs["Normal"])
                elif kind == "layered_missing":
                    links.remove(branches[1].links[0])
                elif kind == "layered_unknown":
                    links.new(nodes.new("ShaderNodeEmission").outputs[0], branches[1])
                elif kind == "layered_volume":
                    links.new(nodes.new("ShaderNodeVolumeAbsorption").outputs[0], output.inputs["Volume"])
            materials.append(mat)
        mesh = bpy.data.meshes.new("MixedReceiver")
        mesh.from_pydata(vertices, [], faces)
        obj = bpy.data.objects.new("MixedReceiver", mesh)
        scene.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        for i, mat in enumerate(materials):
            mesh.materials.append(mat)
            mesh.polygons[i].material_index = i
        # A second user exercises restoration of shared meshes and object slots.
        shared = obj.copy()
        scene.collection.objects.link(shared)
        shared.location.y = 3
        shared.hide_render = True
        shared.material_slots[0].link = "OBJECT"
        shared.material_slots[0].material = materials[1]
        irradiance.prepare_object_bake_uv(obj)

        def snapshot():
            return (
                tuple(bpy.data.materials),
                tuple(bpy.data.node_groups),
                tuple((m, len(m.node_tree.nodes), len(m.node_tree.links)) for m in materials),
                tuple((o, tuple((s.link, s.material) for s in o.material_slots)) for o in (obj, shared)),
                tuple(p.material_index for p in mesh.polygons),
                tuple((uv.name, uv.active_render) for uv in mesh.uv_layers),
                mesh.uv_layers.active_index,
            )

        before = snapshot()
        eligible = [kind for kind, mat in zip(kinds, materials) if irradiance._opaque_principled_surface(mat)]
        incident_eligible = [kind for kind, mat in zip(kinds, materials) if irradiance._opaque_incident_surface(mat)]
        uv = mesh.uv_layers[BAKE_UV_LAYER_NAME]
        centers = np.array([np.mean([uv.data[i].uv[:] for i in p.loop_indices], axis=0) for p in mesh.polygons])

        def sample(baked, scale=1):
            # Face centers avoid mixing adjacent padded UV charts at vertices.
            return _sample_bilinear(baked.pixels, baked.width, baked.height, centers) * scale

        # Control uses precisely the old bake path on identical geometry/UVs.
        with patch.object(irradiance, "_prepare_opaque_bake_material"):
            old_flux = sample(irradiance.bake_irradiance_map(scene, obj, 256, samples=64), np.pi)
            old_color = sample(irradiance.bake_albedo_map(scene, obj, 256))
        flux = sample(irradiance.bake_irradiance_map(scene, obj, 256, samples=64), np.pi)
        color = sample(irradiance.bake_albedo_map(scene, obj, 256))
        restored = snapshot() == before
        with patch.object(
            irradiance, "_prepare_image_nodes_for_bake", side_effect=RuntimeError("injected bake failure")
        ):
            failed = irradiance.bake_irradiance_map(scene, obj, 32, samples=4)
        return {
            "eligible": eligible,
            "incident_eligible": incident_eligible,
            "patches": {
                kind: {
                    "incident": float(flux[i]),
                    "albedo": float(color[i]),
                    "old_incident": float(old_flux[i]),
                    "old_albedo": float(old_color[i]),
                }
                for i, kind in enumerate(kinds)
            },
            "restored": restored,
            "failure_restored": failed is None and snapshot() == before,
        }

    def exposed_opaque_receiver_concave(self, layered: bool = False) -> dict:
        """One black mesh forms both a bowl and its center measurement patch."""
        import math

        from visionsim.simulate.heatsim import irradiance
        from visionsim.simulate.heatsim.names import BAKE_UV_LAYER_NAME

        self._clear()
        self._enable_cycles(samples=256)
        self._add_world_light(color=(1, 1, 1, 1))
        scene = bpy.context.scene
        scene.cycles.max_bounces = scene.cycles.diffuse_bounces = 32
        scene.cycles.seed = 71
        n, s = 128, 0.002
        vertices = [(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)]
        for radius, z in ((0.01, 0), (1, 0), (1, 1)):
            vertices.extend(
                (radius * math.cos(2 * math.pi * i / n), radius * math.sin(2 * math.pi * i / n), z) for i in range(n)
            )
        faces = [(0, 1, 2, 3)]
        for i in range(n):
            j = (i + 1) % n
            faces.extend(((4 + i, 4 + n + i, 4 + n + j, 4 + j), (4 + n + j, 4 + n + i, 4 + 2 * n + i, 4 + 2 * n + j)))
        mesh = bpy.data.meshes.new("BlackBowl")
        mesh.from_pydata(vertices, [], faces)
        obj = bpy.data.objects.new("BlackBowl", mesh)
        scene.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        mat = bpy.data.materials.new("BlackBowl")
        mat.use_nodes = True
        surface = mat.node_tree.nodes.get("Principled BSDF")
        surface.inputs["Base Color"].default_value = (0, 0, 0, 1)
        (surface.inputs.get("Specular IOR Level") or surface.inputs["Specular"]).default_value = 0
        if layered:
            nodes, links = mat.node_tree.nodes, mat.node_tree.links
            glossy = nodes.new("ShaderNodeBsdfGlossy")
            glossy.inputs["Color"].default_value = (0, 0, 0, 1)
            add = nodes.new("ShaderNodeAddShader")
            links.new(surface.outputs[0], add.inputs[0])
            links.new(glossy.outputs[0], add.inputs[1])
            links.new(add.outputs[0], nodes.get("Material Output").inputs["Surface"])
        mesh.materials.append(mat)
        # Let production build/cache its UVs, then devote the atlas to the tiny
        # patch. Degenerate wall UVs still leave the walls in light transport.
        irradiance.prepare_object_bake_uv(obj)
        uv = mesh.uv_layers[BAKE_UV_LAYER_NAME]
        for corner in uv.data:
            corner.uv = (0, 0)
        for index, coord in zip(mesh.polygons[0].loop_indices, ((0, 0), (1, 0), (1, 1), (0, 1))):
            uv.data[index].uv = coord
        probe = irradiance.bake_irradiance_map(scene, obj, 32, samples=256)
        # A global-white control must visibly disagree with the black-bowl truth.
        surface.inputs["Base Color"].default_value = (1, 1, 1, 1)
        white = irradiance.bake_irradiance_map(scene, obj, 32, samples=256)
        return {
            "expected": math.pi / 2,
            "measured": float(probe.pixels[2:-2, 2:-2].mean() * math.pi),
            "global_white": float(white.pixels[2:-2, 2:-2].mean() * math.pi),
        }

    def exposed_opaque_receiver_transparent_path(self) -> dict:
        """A transparent crossing must end the depth-zero receiver override."""
        from visionsim.simulate.heatsim import irradiance

        self._clear()
        self._enable_cycles(samples=32)
        self._add_world_light(color=(1, 1, 1, 1))
        scene = bpy.context.scene
        scene.cycles.use_adaptive_sampling = scene.cycles.use_denoising = False
        bpy.ops.mesh.primitive_plane_add(size=4)
        receiver = bpy.context.object
        mat = bpy.data.materials.new("Receiver")
        mat.use_nodes = True
        surface = mat.node_tree.nodes.get("Principled BSDF")
        surface.inputs["Base Color"].default_value = (0, 0, 0, 1)
        (surface.inputs.get("Specular IOR Level") or surface.inputs["Specular"]).default_value = 0
        irradiance._prepare_opaque_bake_material(mat, incident=True)
        receiver.data.materials.append(mat)
        bpy.ops.mesh.primitive_plane_add(size=4, location=(0, 0, 1))
        window = bpy.context.object
        glass = bpy.data.materials.new("Transparent")
        glass.use_nodes = True
        nodes, links = glass.node_tree.nodes, glass.node_tree.links
        links.new(nodes.new("ShaderNodeBsdfTransparent").outputs[0], nodes.get("Material Output").inputs["Surface"])
        window.data.materials.append(glass)
        bpy.ops.object.camera_add(location=(0, 0, 3))
        scene.camera = bpy.context.object
        scene.camera.data.type = "ORTHO"
        scene.camera.data.ortho_scale = 0.5
        scene.render.resolution_x = scene.render.resolution_y = 8
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_depth = "32"
        measured = {}
        with tempfile.TemporaryDirectory() as root:
            for present in (False, True):
                window.hide_render = not present
                scene.render.filepath = str(Path(root) / f"{present}.exr")
                bpy.ops.render.render(write_still=True)
                image = bpy.data.images.load(scene.render.filepath)
                image.colorspace_settings.name = "Non-Color"
                pixels = np.array(image.pixels[:]).reshape(8, 8, 4)
                measured[str(present)] = float(pixels[:, :, :3].mean())
                bpy.data.images.remove(image)
        return measured

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

    def exposed_cached_bake_uv(self, scene_path: str, object_name: str, forbid_projection: bool = False) -> dict:
        """Reopen an unprepared source and verify layout reuse across Blender processes."""
        from unittest.mock import patch

        from visionsim.simulate.heatsim import irradiance
        from visionsim.simulate.heatsim.names import BAKE_UV_LAYER_NAME

        bpy.ops.wm.open_mainfile(filepath=scene_path)
        obj = bpy.data.objects[object_name]
        rgb = obj.data.uv_layers.active
        original = tuple(tuple(loop.uv) for loop in rgb.data)
        if forbid_projection:
            with patch.object(irradiance, "repair_bake_uv", side_effect=AssertionError("regenerated cached UVs")):
                irradiance.prepare_object_bake_uv(obj)
        else:
            irradiance.prepare_object_bake_uv(obj)
        signature = obj.data["heatsim_bake_uv_signature"]
        return {
            "uv": [list(loop.uv) for loop in obj.data.uv_layers[BAKE_UV_LAYER_NAME].data],
            "signature": signature,
            "saved": (Path(scene_path + ".heatsim") / "bake_uv" / f"{signature}.npz").is_file(),
            "rgb_preserved": tuple(tuple(loop.uv) for loop in rgb.data) == original,
        }

    def exposed_bake_uv_reuse(self, object_name: str = "ThermalPlane") -> dict:
        """Verify shared bake coordinates survive repeat preparation and track geometry edits."""
        from visionsim.simulate.heatsim import irradiance
        from visionsim.simulate.heatsim.names import BAKE_UV_LAYER_NAME

        obj = bpy.data.objects[object_name]
        source_name = obj.data.uv_layers.active.name
        source = np.array([tuple(loop.uv) for loop in obj.data.uv_layers[source_name].data])
        irradiance.prepare_object_bake_uv(obj)
        first_key = obj.data.get("heatsim_bake_uv_signature")
        obj.data.uv_layers[BAKE_UV_LAYER_NAME].data[0].uv.x += 0.0007
        expected = np.array([tuple(loop.uv) for loop in obj.data.uv_layers[BAKE_UV_LAYER_NAME].data])
        irradiance.prepare_object_bake_uv(obj)
        reused = np.array([tuple(loop.uv) for loop in obj.data.uv_layers[BAKE_UV_LAYER_NAME].data])
        obj.data.vertices[0].co.x += 0.1
        obj.data.update()
        irradiance.prepare_object_bake_uv(obj)
        geometry_key = obj.data.get("heatsim_bake_uv_signature")
        obj.scale.x *= 2
        bpy.context.view_layer.update()
        irradiance.prepare_object_bake_uv(obj)
        scale_key = obj.data.get("heatsim_bake_uv_signature")
        obj.location.x += 5
        bpy.context.view_layer.update()
        irradiance.prepare_object_bake_uv(obj)
        after = np.array([tuple(loop.uv) for loop in obj.data.uv_layers[source_name].data])
        return {
            "reused_coordinates": bool(np.array_equal(expected, reused)),
            "regenerated_after_geometry_edit": geometry_key != first_key,
            "regenerated_after_scale_edit": scale_key != geometry_key,
            "reused_after_translation": obj.data.get("heatsim_bake_uv_signature") == scale_key,
            "source_uv_unchanged": bool(np.array_equal(source, after)),
        }

    def exposed_bake_uv_folded_ngon(self) -> dict:
        """Retail ceiling regression: a nonplanar pentagon needs separate triangle UV corners."""
        from visionsim.simulate.heatsim import adapter
        from visionsim.simulate.heatsim.geometry import release_meshes

        self._register()
        self._clear()
        mesh = bpy.data.meshes.new("nonplanar pentagon")
        mesh.from_pydata(
            [
                (0, 0, 0),
                (-5.67032, 0, 0),
                (-5.67032, 0.000001, -0.086817),
                (-5.682873, -0.012608, -0.074264),
                (0, 0.000001, -0.061711),
            ],
            [],
            [(0, 1, 2, 3, 4)],
        )
        obj = bpy.data.objects.new("nonplanar pentagon", mesh)
        bpy.context.collection.objects.link(obj)
        source_uv = mesh.uv_layers.new(name="RGB")
        for i, loop in enumerate(source_uv.data):
            loop.uv = (i / 5, i % 2)
        before_uv = np.array([tuple(p.uv) for p in source_uv.data])
        original_triangles, _, _ = adapter._mesh_triangles(mesh)
        plan = adapter.build_atlas_plan(
            bpy.context.scene,
            [obj],
            {
                "render_domain": "TEXEL",
                "atlas_tile_min": 512,
                "atlas_tile_max": 512,
            },
        )
        try:
            with adapter.use_atlas_geometry(bpy.context.scene, plan):
                triangles, _, _ = adapter._mesh_triangles(obj.data)
                uv, _ = adapter._face_uv_and_slots_from_mesh(obj.data, "RGB")
                result = {
                    "same_triangles": sorted(map(tuple, np.sort(triangles, axis=1)))
                    == sorted(map(tuple, np.sort(original_triangles, axis=1))),
                    "rgb_uv_preserved": bool(np.array_equal(uv, before_uv[triangles])),
                    "thermal_points": len(plan.texels[obj.name]["xy"]),
                }
        finally:
            release_meshes(plan.meshes)
        result["source_preserved"] = obj.data == mesh and len(mesh.polygons) == 1 and mesh.polygons[0].loop_total == 5
        return result

    def exposed_bake_uv_degenerate_faces(self) -> list[int]:
        """Separate float32 collinearity noise from a real surface with missing UVs."""
        from visionsim.simulate.heatsim import uv_utils

        self._clear()
        mesh = bpy.data.meshes.new("UV degeneracy")
        mesh.from_pydata(
            [
                (0, 0, 0),
                (1, 0, 0),
                (0.5, 1e-8, 0),
                (2, 0, 0),
                (3, 0, 0),
                (2, 1, 0),
                (4, 0, 0),
                (4.01, 0, 0),
                (4, 1e-10, 0),
                (5, 0, 0),
                (6, 0, 0),
                (5, 1, 0),
                (7, 0, 0),
                (8, 0, 0),
                (7, 1, 0),
            ],
            [],
            [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11), (12, 13, 14)],
        )
        uv = mesh.uv_layers.new(name="Bake")
        coordinates = np.zeros((15, 2))
        # A physically collapsed bevel can still have a finite UV footprint.
        # It must not make a healthy chart look as though two surfaces overlap.
        coordinates[:3] = ((0.2, 0.2), (0.8, 0.2), (0.2, 0.8))
        coordinates[9:12] = ((0, 0), (1, 0), (0, 1))
        # Packing shared chart boundaries can round by one float32 coordinate step.
        coordinates[12:] = ((1, 1), (0, 1 - 6e-8), (1 - 6e-8, 0))
        uv.data.foreach_set("uv", coordinates.reshape(-1))
        obj = bpy.data.objects.new("UV degeneracy", mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        sync = bpy.context.scene.tool_settings.use_uv_select_sync
        bpy.context.scene.tool_settings.use_uv_select_sync = False
        try:
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            return sorted(uv_utils._invalid_bake_uv_faces(mesh))
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")
            bpy.context.scene.tool_settings.use_uv_select_sync = sync

    def exposed_bake_uv_warped_face(self) -> dict:
        """Unfold a warped sofa-like quad across the renderer's diagonal."""
        from visionsim.simulate.heatsim import adapter, uv_utils

        self._clear()
        source = bpy.data.meshes.new("Warped quad")
        source.from_pydata(
            [(0, 0, 0), (-0.138, -0.816, 0.354), (-0.125, -0.014, -0.053), (-0.424, 0.185, 0.772)],
            [],
            [(0, 1, 2, 3)],
        )
        rgb = source.uv_layers.new(name="RGB")
        for loop, uv in zip(rgb.data, [(0, 0), (1, 0), (1, 1), (0, 1)]):
            loop.uv = uv
        before = (tuple(v.co[:] for v in source.vertices), tuple(loop.uv[:] for loop in rgb.data))
        mesh = source.copy()
        obj = bpy.data.objects.new("Private warped UV projection", mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        old_sync = bpy.context.scene.tool_settings.use_uv_select_sync
        bpy.context.scene.tool_settings.use_uv_select_sync = False
        try:
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            uv_utils._unfold_bake_uv_faces(mesh, {0}, source)
            bpy.ops.uv.select_all(action="SELECT")
            bpy.ops.uv.pack_islands(margin=0.002, margin_method="FRACTION", merge_overlap=False)
            invalid = uv_utils._invalid_bake_uv_faces(mesh, source)
            bpy.ops.object.mode_set(mode="OBJECT")
            vertices = np.array([v.co[:] for v in source.vertices])
            triangles, loops, _ = adapter._mesh_triangles(source)
            uv = np.array([loop.uv[:] for loop in mesh.uv_layers.active.data])[loops]
            positions = vertices[triangles]
            lengths = np.linalg.norm(np.roll(positions, -1, axis=1) - positions, axis=2)
            uv_lengths = np.linalg.norm(np.roll(uv, -1, axis=1) - uv, axis=2)
            scale = uv_lengths / lengths
            return {
                "invalid_faces": len(invalid),
                "edge_scale_ratio": float(scale.max() / scale.min()),
                "source_preserved": before
                == (tuple(v.co[:] for v in source.vertices), tuple(loop.uv[:] for loop in rgb.data)),
            }
        finally:
            if bpy.context.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.meshes.remove(mesh)
            bpy.context.scene.tool_settings.use_uv_select_sync = old_sync

    def exposed_bake_uv_repair(self, kind: str) -> dict:
        """Inject an invalid generated layout on a private mesh and exercise repair."""
        from unittest.mock import patch

        from visionsim.simulate.heatsim import adapter, atlas, irradiance, uv_utils
        from visionsim.simulate.heatsim.names import BAKE_UV_LAYER_NAME

        self._clear()
        mesh = bpy.data.meshes.new("UV repair source")
        mesh.from_pydata(
            [(x + 2 * i, y, i) for i in range(4) for x, y in [(0, 0), (1, 0), (1, 1), (0, 1)]],
            [],
            [tuple(range(4 * i, 4 * i + 4)) for i in range(4)],
        )
        obj = bpy.data.objects.new("UV repair fixture", mesh)
        bpy.context.collection.objects.link(obj)
        obj.scale = (1.5, 0.75, 1)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.context.view_layer.update()
        rgb = mesh.uv_layers.new(name="RGB")
        for i, loop in enumerate(rgb.data):
            loop.uv = (i / 40, i / 50)
        mat = bpy.data.materials.new("Wide RGB texture")
        mat.use_nodes = True
        texture = mat.node_tree.nodes.new("ShaderNodeTexImage")
        texture.image = bpy.data.images.new("Wide UV repair RGB", width=256, height=16)
        mat.node_tree.nodes.active = texture
        mesh.materials.append(mat)
        if kind == "hidden":
            mesh.polygons[0].hide = True

        def state():
            return (
                tuple(tuple(v.co) for v in mesh.vertices),
                tuple(tuple(loop.uv) for loop in rgb.data),
                tuple(edge.use_seam for edge in mesh.edges),
                tuple(face.hide for face in mesh.polygons),
                tuple(mesh.materials),
                uv_utils.get_uv_state(obj),
            )

        initial = state()
        resources = (len(bpy.data.objects), len(bpy.data.meshes), len(bpy.data.materials))
        original_sync = bpy.context.scene.tool_settings.use_uv_select_sync
        bpy.context.scene.tool_settings.use_uv_select_sync = True
        result = {}
        native_repair = irradiance.repair_bake_uv

        def inject(projected, name, source_mesh):
            bpy.ops.object.mode_set(mode="OBJECT")
            uv = np.zeros((16, 2), dtype=np.float32)
            square = np.array([[0, 0], [2, 0], [2, 1], [0, 1]])
            for i in range(4):
                if kind == "collapsed":
                    continue
                if kind == "stretched":
                    uv[4 * i : 4 * i + 4] = square * (0.1, 0.001) + (0.22 * i, 0.1)
                elif kind == "subpixel" and i < 2:
                    uv[4 * i : 4 * i + 4] = square * 0.0001 + (0.123, 0.456)
                else:
                    uv[4 * i : 4 * i + 4] = square * 0.1 + ((0.22 * i if kind == "subpixel" else 0), 0.1)
            projected.uv_layers.active.data.foreach_set("uv", uv.reshape(-1))
            vertices = np.array([v.co[:] for v in projected.vertices]) * 1000
            faces, loops, _ = adapter._mesh_triangles(projected)
            if kind == "subpixel":
                atlas.rasterize_tile(vertices, faces, uv[loops], (64, 64), reject_overlaps=True)
                result["raster_missed_subpixel_overlap"] = True
            bpy.ops.object.mode_set(mode="EDIT")
            result["invalid_before"] = len(uv_utils._invalid_bake_uv_faces(projected))
            if kind == "failure":
                with patch.object(uv_utils, "_invalid_bake_uv_faces", return_value={0}):
                    native_repair(projected, name, source_mesh)
            else:
                native_repair(projected, name, source_mesh)
            result["invalid_after"] = len(uv_utils._invalid_bake_uv_faces(projected))

        try:
            with patch.object(irradiance, "repair_bake_uv", side_effect=inject):
                try:
                    irradiance.prepare_object_bake_uv(obj)
                except RuntimeError as exc:
                    result["error"] = str(exc)
            if kind != "failure":
                coordinates = np.array([loop.uv[:] for loop in mesh.uv_layers[BAKE_UV_LAYER_NAME].data])
                with patch.object(irradiance, "repair_bake_uv", side_effect=AssertionError("regenerated UVs")):
                    irradiance.prepare_object_bake_uv(obj)
                result["reused"] = bool(
                    np.array_equal(
                        coordinates, np.array([loop.uv[:] for loop in mesh.uv_layers[BAKE_UV_LAYER_NAME].data])
                    )
                )
            else:
                result["unpublished"] = (
                    BAKE_UV_LAYER_NAME not in mesh.uv_layers and "heatsim_bake_uv_signature" not in mesh
                )
            uv_utils.set_uv_state(obj, initial[-1])
            result["source_preserved"] = state() == initial
            result["resources_restored"] = resources == (
                len(bpy.data.objects),
                len(bpy.data.meshes),
                len(bpy.data.materials),
            )
            result["sync_restored"] = bpy.context.scene.tool_settings.use_uv_select_sync
            result["object_mode"] = bpy.context.mode == "OBJECT"
        finally:
            bpy.context.scene.tool_settings.use_uv_select_sync = original_sync
        return result

    def exposed_bake_uv_physical_layout(self) -> dict:
        """Compare scaled and applied geometry, RGB image aspects, and failed projection cleanup."""
        from unittest.mock import patch

        from mathutils import Matrix

        from visionsim.simulate.heatsim import irradiance
        from visionsim.simulate.heatsim.names import BAKE_UV_LAYER_NAME

        self._clear()
        bpy.ops.mesh.primitive_plane_add(size=2)
        obj = bpy.context.object
        obj.shape_key_add(name="Basis")
        obj.shape_key_add(name="Detail").data[0].co.z += 0.01
        parent = bpy.data.objects.new("Scaled parent", None)
        bpy.context.collection.objects.link(parent)
        parent.scale = (10, 5, 1)
        obj.parent = parent
        obj.scale = (2, 2, 2.5)
        bpy.context.view_layer.update()
        mat = bpy.data.materials.new("Wide RGB image")
        mat.use_nodes = True
        texture = mat.node_tree.nodes.new("ShaderNodeTexImage")
        texture.image = bpy.data.images.new("Wide RGB", width=1024, height=64)
        mat.node_tree.nodes.active = texture
        obj.data.materials.append(mat)

        def source_state():
            return (
                tuple(tuple(v.co) for v in obj.data.vertices),
                tuple(tuple(row) for row in obj.matrix_world),
                tuple(tuple(loop.uv) for loop in obj.data.uv_layers["UVMap"].data),
                tuple(tuple(p.co) for key in obj.data.shape_keys.key_blocks for p in key.data),
                obj.data.texspace_location[:],
                obj.data.texspace_size[:],
            )

        def layout(target):
            irradiance.prepare_object_bake_uv(target)
            return np.array([tuple(loop.uv) for loop in target.data.uv_layers[BAKE_UV_LAYER_NAME].data])

        initial = source_state()
        resources = (tuple(bpy.data.objects), tuple(bpy.data.meshes), tuple(bpy.data.shape_keys))
        scaled = layout(obj)
        restored = source_state() == initial and resources == (
            tuple(bpy.data.objects),
            tuple(bpy.data.meshes),
            tuple(bpy.data.shape_keys),
        )
        applied = bpy.data.objects.new("Applied dimensions", obj.data.copy())
        bpy.context.collection.objects.link(applied)
        applied.data.transform(obj.matrix_world, shape_keys=True)
        applied.matrix_world = Matrix.Identity(4)
        physical = layout(applied)
        texture.image = bpy.data.images.new("Square RGB", width=64, height=64)
        obj.data.pop("heatsim_bake_uv_signature", None)
        square = layout(obj)
        # For a flat patch, equal singular values mean equal physical distance per UV unit.
        xyz = np.array([tuple(obj.matrix_world @ obj.data.vertices[loop.vertex_index].co) for loop in obj.data.loops])
        jacobian = np.stack((xyz[1] - xyz[0], xyz[3] - xyz[0]), axis=1) @ np.linalg.inv(
            np.stack((scaled[1] - scaled[0], scaled[3] - scaled[0]), axis=1)
        )
        singular = np.linalg.svd(jacobian, compute_uv=False)
        resources = (tuple(bpy.data.objects), tuple(bpy.data.meshes), tuple(bpy.data.shape_keys))
        before_uv = square.copy()
        obj.data.pop("heatsim_bake_uv_signature", None)
        failed = None
        with patch.object(irradiance.math, "radians", side_effect=RuntimeError("injected UV failure")):
            try:
                layout(obj)
            except RuntimeError:
                failed = np.array([tuple(loop.uv) for loop in obj.data.uv_layers[BAKE_UV_LAYER_NAME].data])
        return {
            "scaled_vs_applied_error": float(np.max(np.abs(scaled - physical))),
            "wide_vs_square_error": float(np.max(np.abs(scaled - square))),
            "physical_spacing_ratio": float(singular.max() / singular.min()),
            "source_restored": bool(restored and source_state() == initial),
            "failure_restored": bool(
                np.array_equal(failed, before_uv)
                and resources == (tuple(bpy.data.objects), tuple(bpy.data.meshes), tuple(bpy.data.shape_keys))
                and obj.data.get("heatsim_bake_uv_signature") is None
            ),
        }

    def exposed_absorbed_flux(
        self, object_name: str = "ThermalPlane", bake_samples: int = 1024, irradiance_texture_size: int = 512
    ) -> dict:
        """Compute absorbed flux for the active object via ``adapter._compute_irradiance_cycles``.

        Args:
            object_name: object whose flux to compute.
            bake_samples: Cycles samples for the irradiance bake.
            irradiance_texture_size: bake texture size.

        Returns:
            dict: shape, finiteness and spread of the absorbed-flux field.
        """
        from visionsim.simulate.heatsim import adapter

        self._register()
        obj = bpy.data.objects[object_name]
        solver_cfg = self._solver_cfg(bake_samples=bake_samples, irradiance_texture_size=irradiance_texture_size)
        flux = adapter._compute_irradiance_cycles(bpy.context.scene, [obj], solver_cfg, self._defaults())[obj]
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

    def exposed_thermal_reflection_pixels(self, root_path: str) -> dict:
        """Render a hot sphere reflected in a cool opaque plane, plus equilibrium controls."""
        from visionsim.simulate.heatsim import thermal_shader as ts
        from visionsim.simulate.heatsim.materials import PRESETS, MaterialEntry, SceneAssignment
        from visionsim.simulate.heatsim.physics import STEFAN_BOLTZMANN_SI

        self._clear()
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 512
        scene.cycles.seed = 29
        scene.cycles.use_adaptive_sampling = scene.cycles.use_denoising = False
        scene.render.use_compositing = False
        scene.render.film_transparent = False
        scene.render.resolution_x = scene.render.resolution_y = 64
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_depth = "32"
        scene.render.image_settings.color_mode = "RGB"

        def surface(obj, name, temperature, eps):
            mat = bpy.data.materials.new(name)
            obj.data.materials.append(mat)
            obj["heatsim_default_temperature"] = temperature
            attr = obj.data.attributes.new("emissivity", "FLOAT", "POINT")
            attr.data.foreach_set("value", [eps] * len(attr.data))
            return obj

        bpy.ops.mesh.primitive_plane_add(size=20)
        plane = surface(bpy.context.object, "Mirror", 295, 0.84)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.4, location=(0, 2, 2))
        hot = surface(bpy.context.object, "Hot", 350, 1.0)
        # This hot object must never be visible through the plane in the equilibrium cases.
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.4, location=(0, 0, -1))
        surface(bpy.context.object, "Below", 400, 1.0)
        bpy.ops.object.camera_add(location=(0, -4, 4))
        camera = bpy.context.object
        camera.rotation_euler = (-camera.location).to_track_quat("-Z", "Y").to_euler()
        scene.camera = camera
        camera.data.type = "ORTHO"
        camera.data.ortho_scale = 4
        root = Path(root_path)
        root.mkdir(parents=True, exist_ok=True)
        rendered = {}
        for name, specular, roughness, temperature in (
            ("diffuse", 0, 0.5, 350),
            ("sharp", 1, 0, 350),
            ("rough", 1, 0.4, 350),
            ("equilibrium_sharp", 1, 0, 295),
            ("equilibrium_rough", 1, 0.6, 295),
        ):
            hot["heatsim_default_temperature"] = temperature
            # Test energy preservation on a much more reflective surface too.
            if name.startswith("equilibrium"):
                plane.data.attributes["emissivity"].data.foreach_set("value", [0.05] * 4)
            assignment = SceneAssignment(
                "fixture",
                "",
                None,
                {"Mirror": MaterialEntry(PRESETS["glass"], "FEM_PARTICIPANT", None, 1.0, "", specular, roughness)},
            )
            state = ts.enter_thermal_scene(scene, radiance_scale=1, assignment=assignment)
            if name.startswith("equilibrium"):
                # Remove emitter-sampling variance from this analytic energy control.
                for material in bpy.data.materials:
                    material.cycles.emission_sampling = "NONE"
            path = root / f"{name}.exr"
            try:
                scene.render.filepath = str(path)
                bpy.ops.render.render(write_still=True)
            finally:
                ts.restore_scene(scene, state)
            img = bpy.data.images.load(str(path))
            pixels = np.array(img.pixels[:]).reshape(64, 64, 4)[:, :, 0]
            centre = pixels[24:40, 24:40]
            ring = centre.copy()
            ring[5:11, 5:11] = np.nan
            rendered[name] = {
                "peak": float(np.percentile(centre, 99)),
                "ring_mean": float(np.nanmean(ring)),
                "centre_mean": float(centre.mean()),
                "centre_min": float(centre.min()),
                "centre_max": float(centre.max()),
            }
            bpy.data.images.remove(img)
        return {"rendered": rendered, "ambient": STEFAN_BOLTZMANN_SI * 295**4}

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
            "irradiance_scale": 1.0,
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
        from pathlib import Path

        from visionsim.simulate.heatsim import cache

        bpy.ops.wm.save_as_mainfile(filepath=scene_path)
        # ``source_identity`` reports dirty on some headless Blender builds even for a freshly
        # saved file (see `cache_solve`), so fall back to the saved file's digest, which is
        # what the cache actually keys on.
        source = cache.source_identity(bpy.data) or cache.file_digest(Path(bpy.data.filepath))
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

    def exposed_enter_restore(self, assigned: bool = False) -> dict:
        """Enter and restore the thermal scene three times on a shared-mesh setup.

        Returns:
            dict: whether source assignments were preserved, overrides applied, and state
            fully restored after each round trip.
        """
        from visionsim.simulate.heatsim import thermal_shader as ts
        from visionsim.simulate.heatsim.materials import PRESETS, MaterialEntry, SceneAssignment

        bpy.ops.mesh.primitive_cube_add()
        a = bpy.context.object
        materials = [bpy.data.materials.new(name) for name in ("First", "Second", "ObjectOverride")]
        for mat in materials[:2]:
            a.data.materials.append(mat)
        for face in a.data.polygons:
            face.material_index = face.index % 2
            face.use_smooth = face.index % 2 == 0
        a.modifiers.new("Keep bevel geometry", "BEVEL")
        normal = a.modifiers.new("Smooth normals", "WEIGHTED_NORMAL")
        normal.show_viewport = False
        b = a.copy()
        bpy.context.collection.objects.link(b)
        b.modifiers["Smooth normals"].show_render = False
        b.modifiers["Smooth normals"].show_viewport = True
        b.material_slots[1].link = "OBJECT"
        b.material_slots[1].material = materials[2]
        assignment = None
        if assigned:
            assignment = SceneAssignment(
                "fixture",
                "",
                None,
                {
                    mat.name: MaterialEntry(PRESETS[key], "FEM_PARTICIPANT", None, 1.0, "")
                    for mat, key in zip(materials, ("glass", "fabric", "aluminium_polished"))
                },
            )
            assignment.materials[f"{b.name}::{materials[0].name}"] = MaterialEntry(
                PRESETS["rubber"], "FEM_PARTICIPANT", None, 1.0, ""
            )
            bpy.ops.mesh.primitive_plane_add()
            empty = bpy.context.object
            empty_copy = empty.copy()
            bpy.context.collection.objects.link(empty_copy)
        scene = bpy.context.scene
        scene.cycles.blur_glossy = 0.75
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
                (scene.cycles.max_bounces, scene.cycles.diffuse_bounces, scene.cycles.glossy_bounces),
                tuple(empty.data.materials) if assigned else (),
                scene.cycles.blur_glossy,
                tuple(p.use_smooth for p in a.data.polygons),
                tuple((m.show_render, m.show_viewport) for o in (a, b) for m in o.modifiers),
            )

        before = snapshot()
        setup_preserved = True
        override_ok = True
        thermal_defaults_ok = True
        restored = True
        for _ in range(3):
            state = ts.enter_thermal_scene(scene, radiance_scale=1.0, assignment=assignment)
            thermal_defaults_ok &= scene.cycles.blur_glossy == 0
            thermal_defaults_ok &= not any(p.use_smooth for p in a.data.polygons)
            thermal_defaults_ok &= all(
                not m.show_render and not m.show_viewport
                for o in (a, b)
                for m in o.modifiers
                if m.type == "WEIGHTED_NORMAL"
            )
            thermal_defaults_ok &= all(o.modifiers["Keep bevel geometry"].show_render for o in (a, b))
            if snapshot()[:2] != before[:2] or (not assigned and snapshot()[2] != before[2]):
                setup_preserved = False
            if assigned:
                override_ok &= all(v.material_override is None for v in scene.view_layers)
                override_ok &= a.material_slots[0].material["heatsim_thermal_specular_fraction"] == 0.95
                override_ok &= a.material_slots[1].material["heatsim_thermal_specular_fraction"] == 0.0
                override_ok &= b.material_slots[0].material["heatsim_thermal_specular_fraction"] == 0.0
                override_ok &= b.material_slots[1].material["heatsim_thermal_specular_fraction"] == 0.98
            else:
                override_ok &= all(v.material_override == ts._build_gray_body_material(1.0) for v in scene.view_layers)
            ts.restore_scene(scene, state)
            restored &= snapshot() == before
        if assigned:
            from unittest.mock import patch

            build = ts._build_gray_body_material
            calls = []

            def fail_setup(*args, **kwargs):
                calls.append(1)
                if len(calls) == 3:
                    assert scene.cycles.blur_glossy == 0
                    assert not any(p.use_smooth for p in a.data.polygons)
                    raise RuntimeError("injected material failure")
                return build(*args, **kwargs)

            with patch.object(ts, "_build_gray_body_material", side_effect=fail_setup):
                try:
                    ts.enter_thermal_scene(scene, radiance_scale=1.0, assignment=assignment)
                except RuntimeError as exc:
                    assert str(exc) == "injected material failure"
                else:
                    raise AssertionError("material failure was swallowed")
        return {
            "setup_preserved": bool(setup_preserved),
            "override_ok": bool(override_ok),
            "thermal_defaults_ok": bool(thermal_defaults_ok),
            "restored": bool(restored and snapshot() == before),
        }

    def exposed_uniform_smoothed_surface(self, root_path: str) -> dict:
        """Render a smoothed mesh in a uniform thermal environment.

        Args:
            root_path: Output directory for the control render.

        Returns:
            dict: Maximum radiance error and whether the source smoothing was restored.
        """
        from visionsim.simulate.heatsim import thermal_shader as ts
        from visionsim.simulate.heatsim.physics import STEFAN_BOLTZMANN_SI

        self._clear()
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 128
        scene.cycles.seed = 29
        scene.cycles.min_light_bounces = 16
        scene.cycles.use_adaptive_sampling = scene.cycles.use_denoising = False
        scene.render.use_compositing = scene.render.film_transparent = False
        scene.render.resolution_x = scene.render.resolution_y = 64
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_depth = "32"
        scene.render.image_settings.color_mode = "RGB"
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8)
        sphere = bpy.context.object
        for face in sphere.data.polygons:
            face.use_smooth = True
        bpy.ops.object.camera_add(location=(0, -4, 0))
        camera = bpy.context.object
        camera.rotation_euler = (-camera.location).to_track_quat("-Z", "Y").to_euler()
        camera.data.type = "ORTHO"
        camera.data.ortho_scale = 3
        scene.camera = camera
        ts.stamp_default_temperatures(scene, default_K=295)
        path = Path(root_path) / "uniform_smoothed_surface.exr"
        path.parent.mkdir(parents=True, exist_ok=True)
        state = ts.enter_thermal_scene(scene, radiance_scale=1)
        # Isolate normal-related energy loss from emitter-sampling variance.
        for material in bpy.data.materials:
            material.cycles.emission_sampling = "NONE"
        try:
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
        finally:
            ts.restore_scene(scene, state)
        image = bpy.data.images.load(str(path))
        pixels = np.array(image.pixels[:]).reshape(64, 64, 4)[:, :, 0]
        expected = STEFAN_BOLTZMANN_SI * 295**4
        error = float(np.max(np.abs(pixels - expected)))
        bpy.data.images.remove(image)
        return {"max_radiance_error": error, "smoothing_restored": all(p.use_smooth for p in sphere.data.polygons)}

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
        import json
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
        cube = scene.objects["Cube"]
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 8
        if hasattr(scene.render, "compositor_device"):
            scene.render.compositor_device = "CPU"
        scene.cycles.seed = 0
        scene.cycles.use_animated_seed = False
        scene.cycles.blur_glossy = 0.75
        for face in cube.data.polygons:
            face.use_smooth = face.index % 2 == 0
        cube.modifiers.new("RGB bevel", "BEVEL")
        normal = cube.modifiers.new("RGB normals", "WEIGHTED_NORMAL")
        normal.show_viewport = False
        scene.render.use_persistent_data = False
        scene.render.resolution_x = scene.render.resolution_y = 48
        scene.render.resolution_percentage = 100
        service.exposed_include_frames(file_format="OPEN_EXR", exr_codec="ZIP", bit_depth=32)
        scene.frame_set(1)
        service.exposed_render_current_frame(allow_skips=False)

        ts.stamp_default_temperatures(scene, default_K=310)
        ts.setup_temperature_aov(scene, service.view_layer)
        sidecar = root / "scene.thermal.json"
        sidecar.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "materials": {"Red": {"preset": "glass"}, "Blue": {"preset": "aluminium_polished"}},
                }
            )
        )
        service.exposed_include_thermal(preview=False, assignments=str(sidecar))
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
            tuple((s.link, s.material) for s in cube.material_slots),
            tuple(p.material_index for p in cube.data.polygons),
            scene.cycles.blur_glossy,
            tuple(p.use_smooth for p in cube.data.polygons),
            tuple((m.show_render, m.show_viewport) for m in cube.modifiers),
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
                tuple((s.link, s.material) for s in cube.material_slots),
                tuple(p.material_index for p in cube.data.polygons),
                scene.cycles.blur_glossy,
                tuple(p.use_smooth for p in cube.data.polygons),
                tuple((m.show_render, m.show_viewport) for m in cube.modifiers),
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
