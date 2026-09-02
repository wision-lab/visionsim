from __future__ import annotations

import subprocess


def test_temperature_aov_registered(executable):
    code = (
        "import bpy;"
        "from visionsim.simulate.heatsim import thermal_shader as ts;"
        "bpy.ops.mesh.primitive_cube_add();"
        "vl=bpy.context.view_layer;"
        "ts.setup_temperature_aov(bpy.context.scene, vl);"
        "assert any(a.name=='temperature' for a in vl.aovs);"
        "print('THERMAL_AOV_OK')"
    )
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True, check=False)
    assert "THERMAL_AOV_OK" in out.stdout, out.stderr


def test_atlas_shader_group_samples_atlas_and_mixes_by_alpha(executable):
    """Both temperature-source consumers (gray-body radiance and the AOV chain) must gain
    the atlas UV -> Image Texture(Non-Color, Linear, CLIP) -> Mix-by-alpha extension, wired
    from the SAME shared chain (one node group's worth of logic, not duplicated per-material
    node trees) - see docs/superpowers/specs/2026-08-04-thermal-atlas-design.md §4.6.
    """
    code = r"""
import bpy
from visionsim.simulate.heatsim import thermal_shader as ts
from visionsim.simulate.heatsim.constants import ATLAS_COVERAGE_PROP, ATLAS_IMAGE_NAME, ATLAS_UV_LAYER_NAME

# Register the atlas image datablock so the Image Texture node picks it up at build time.
img = bpy.data.images.new(ATLAS_IMAGE_NAME, width=2, height=2, alpha=True, float_buffer=True)
img.pixels.foreach_set([310.0, 310.0, 310.0, 1.0] * 4)

def _check(nodes, links, label):
    attrs = {n.attribute_name: n for n in nodes if n.bl_idname == 'ShaderNodeAttribute'}
    assert ATLAS_UV_LAYER_NAME in attrs, f'{label}: missing atlas UV attribute node'
    assert attrs[ATLAS_UV_LAYER_NAME].attribute_type == 'GEOMETRY'
    assert ATLAS_COVERAGE_PROP in attrs, f'{label}: missing atlas coverage gate attribute node'
    assert attrs[ATLAS_COVERAGE_PROP].attribute_type == 'OBJECT'
    assert 'sim_temperature' in attrs and 'heatsim_default_temperature' in attrs

    tex_nodes = [n for n in nodes if n.bl_idname == 'ShaderNodeTexImage']
    assert len(tex_nodes) == 1, f'{label}: expected exactly one Image Texture node'
    tex = tex_nodes[0]
    assert tex.image is not None and tex.image.name == ATLAS_IMAGE_NAME
    assert tex.image.colorspace_settings.name == 'Non-Color'
    assert tex.interpolation == 'Linear'
    assert tex.extension == 'CLIP'
    # UV attribute feeds the Image Texture's Vector input.
    uv_node = attrs[ATLAS_UV_LAYER_NAME]
    assert any(
        link.from_node == uv_node and link.to_node == tex and link.to_socket.identifier == 'Vector'
        for link in links
    ), f'{label}: atlas UV attribute not wired into the Image Texture Vector input'

    mix_nodes = [n for n in nodes if n.bl_idname == 'ShaderNodeMix']
    assert len(mix_nodes) == 1, f'{label}: expected exactly one Mix node'
    mix = mix_nodes[0]
    assert mix.data_type == 'FLOAT'
    assert mix.inputs['Factor'].is_linked, f'{label}: Mix Factor not wired'
    assert mix.inputs['A'].is_linked and mix.inputs['B'].is_linked, f'{label}: Mix A/B not wired'

    # Mix Factor traces back (within 2 hops) to the Image Texture's Alpha output -- the
    # atlas validity signal must gate the mix, not just feed some unrelated math chain.
    factor_link = next(link for link in links if link.to_socket == mix.inputs['Factor'])
    gate_node = factor_link.from_node
    assert gate_node.bl_idname == 'ShaderNodeMath' and gate_node.operation == 'MULTIPLY'
    def _upstream(node, max_hops=4):
        seen, frontier = set(), [node]
        for _ in range(max_hops):
            nxt = [ln.from_node for ln in links if ln.to_node in frontier]
            nxt = [n for n in nxt if n not in seen]
            if not nxt:
                break
            seen.update(nxt)
            frontier = nxt
        return seen

    gate_sources = _upstream(gate_node)
    assert tex in gate_sources, f'{label}: Mix Factor does not trace back to the atlas Alpha'
    assert attrs[ATLAS_COVERAGE_PROP] in gate_sources, f'{label}: Mix Factor missing the object-level gate'

    # Mix B traces back to a channel split of the Image Texture's Color output (the R channel).
    b_link = next(link for link in links if link.to_socket == mix.inputs['B'])
    sep = b_link.from_node
    assert sep.bl_idname == 'ShaderNodeSeparateColor'
    assert any(link.from_node == tex and link.to_node == sep for link in links)

# -- Gray-body radiance material --------------------------------------------
mat = ts._build_gray_body_material(1.0)
_check(mat.node_tree.nodes, mat.node_tree.links, 'gray-body')
# The Mix Result must feed the POWER(4) chain (T_eff -> sigma*T^4), not a stale
# pre-atlas temp_effective node.
pow4 = next(n for n in mat.node_tree.nodes if n.bl_idname == 'ShaderNodeMath' and n.operation == 'POWER')
mix = next(n for n in mat.node_tree.nodes if n.bl_idname == 'ShaderNodeMix')
assert any(
    link.from_node == mix and link.to_node == pow4 for link in mat.node_tree.links
), 'gray-body: Mix result not wired into the T^4 chain'

# -- AOV material --------------------------------------------------------
mat2 = bpy.data.materials.new('atlas_aov_mat')
mat2.use_nodes = True
ts._append_temperature_aov_nodes(mat2, 'temperature')
_check(mat2.node_tree.nodes, mat2.node_tree.links, 'aov')
aov = next(n for n in mat2.node_tree.nodes if n.type == 'OUTPUT_AOV')
mix2 = next(n for n in mat2.node_tree.nodes if n.bl_idname == 'ShaderNodeMix')
_links2 = mat2.node_tree.links
_frontier, _reached = [mix2], set()
for _ in range(4):
    _nxt = [ln.to_node for ln in _links2 if ln.from_node in _frontier and ln.to_node not in _reached]
    if not _nxt:
        break
    _reached.update(_nxt)
    _frontier = _nxt
assert aov in _reached, 'aov: Mix result does not reach the OutputAOV'

print('ATLAS_SHADER_OK')
"""
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True, check=False)
    assert "ATLAS_SHADER_OK" in out.stdout, out.stdout + "\n" + out.stderr


def test_atlas_shader_group_falls_back_when_no_atlas_image(executable):
    """No ``HeatSim_Temperature_Atlas`` image registered (render_domain=VERTEX, the atlas is
    never built) -> the Image Texture node has no image, but the graph must still build (no
    exceptions) and every node this test can reach must exist -- the byte-identical VERTEX
    guarantee is enforced by the mix factor being multiplied by the OBJECT-level gate, which
    is never stamped (and so defaults to 0) whenever ``write_frame_attributes`` is called
    without an ``atlas_plan`` -- this test only guards the structural half (no crash, no
    dangling/unlinked sockets) since the zero-default-attribute behavior itself is core
    Blender semantics, not something under test here.
    """
    code = r"""
import bpy
from visionsim.simulate.heatsim import thermal_shader as ts

mat = ts._build_gray_body_material(2.0)
tex = next(n for n in mat.node_tree.nodes if n.bl_idname == 'ShaderNodeTexImage')
assert tex.image is None
mix = next(n for n in mat.node_tree.nodes if n.bl_idname == 'ShaderNodeMix')
assert mix.inputs['Factor'].is_linked and mix.inputs['A'].is_linked and mix.inputs['B'].is_linked

mat2 = bpy.data.materials.new('novatlas_aov')
mat2.use_nodes = True
ts._append_temperature_aov_nodes(mat2, 'temperature')
aov = next(n for n in mat2.node_tree.nodes if n.type == 'OUTPUT_AOV')
assert aov.inputs['Value'].is_linked

print('NO_ATLAS_FALLBACK_OK')
"""
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True, check=False)
    assert "NO_ATLAS_FALLBACK_OK" in out.stdout, out.stdout + "\n" + out.stderr


def test_enter_restore_round_trip(executable):
    """enter_thermal_scene then restore_scene must leave materials, lights, and world unchanged."""
    code = (
        "import bpy;"
        "from visionsim.simulate.heatsim import thermal_shader as ts;"
        # Build a simple scene: cube with a named material + a point light.
        "bpy.ops.object.select_all(action='SELECT');"
        "bpy.ops.object.delete();"
        "bpy.ops.mesh.primitive_cube_add();"
        "cube = bpy.context.active_object;"
        "mat = bpy.data.materials.new('TestMat');"
        "cube.data.materials.append(mat);"
        "bpy.ops.object.light_add(type='POINT');"
        "light = bpy.context.active_object;"
        # Record original state.
        "orig_mat_name = cube.material_slots[0].material.name;"
        "orig_light_hide = light.hide_render;"
        "orig_light_hide_vp = light.hide_viewport;"
        "scene = bpy.context.scene;"
        "orig_world = scene.world;"
        # Round-trip.
        "state = ts.enter_thermal_scene(scene, radiance_scale=1.0);"
        "ts.restore_scene(scene, state);"
        # Verify materials restored.
        "restored_name = cube.material_slots[0].material.name if cube.material_slots[0].material else None;"
        "assert restored_name == orig_mat_name, f'material: expected {orig_mat_name!r}, got {restored_name!r}';"
        # Verify light visibility restored (enter_thermal_scene mutates both flags).
        "assert light.hide_render == orig_light_hide, f'light hide_render changed';"
        "assert light.hide_viewport == orig_light_hide_vp, f'light hide_viewport changed';"
        # Verify world restored.
        "assert scene.world is orig_world, f'world not restored: {scene.world!r} vs {orig_world!r}';"
        "print('ROUND_TRIP_OK')"
    )
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True, check=False)
    assert "ROUND_TRIP_OK" in out.stdout, out.stderr


def test_meshes_without_materials_get_a_temperature_carrying_surface(executable):
    """A mesh with no material (or an empty slot) has no shader to write the value
    AOV, so it renders 0 K however well it was simulated. It must be given one."""
    code = r"""
import bpy
from visionsim.simulate.heatsim import thermal_shader

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)

def plane(name):
    bpy.ops.mesh.primitive_plane_add()
    o = bpy.context.active_object
    o.name = name
    return o

bare = plane('bare')
bare.data.materials.clear()
assert len(bare.material_slots) == 0

authored = plane('authored')
m = bpy.data.materials.new('authored_m')
m.use_nodes = True
authored.data.materials.append(m)

half = plane('half')
half.data.materials.append(bpy.data.materials.new('half_m'))
half.data.materials['half_m'].use_nodes = True
half.data.materials.append(None)
assert any(s.material is None for s in half.material_slots)

sc = bpy.context.scene
thermal_shader.setup_temperature_aov(sc, bpy.context.view_layer)

def aov_count(obj):
    n = 0
    for slot in obj.material_slots:
        mat = slot.material
        assert mat is not None, f'{obj.name}: still has an empty slot'
        n += sum(1 for node in mat.node_tree.nodes if node.type == 'OUTPUT_AOV')
    return n

for o in (bare, authored, half):
    assert len(o.material_slots) > 0, f'{o.name}: no material slot'
    assert aov_count(o) > 0, f'{o.name}: no temperature AOV on its material'

# The stand-in must not disturb the authored material.
assert authored.material_slots[0].material.name == 'authored_m'

# It is shared, not one material per object.
default_name = thermal_shader._DEFAULT_SURFACE_MATERIAL_NAME
assert bare.material_slots[0].material.name == default_name
assert sum(1 for m in bpy.data.materials if m.name.startswith(default_name)) == 1

# Idempotent: a second pass must not add more slots.
before = [len(o.material_slots) for o in (bare, authored, half)]
thermal_shader.setup_temperature_aov(sc, bpy.context.view_layer)
assert [len(o.material_slots) for o in (bare, authored, half)] == before

print('DEFAULT_SURFACE_OK')
"""
    out = subprocess.run(
        [str(executable), "-b", "--python-expr", code],
        capture_output=True, text=True,
     check=False)
    assert "DEFAULT_SURFACE_OK" in out.stdout, out.stdout + "\n" + out.stderr


def test_radiance_matches_gray_body_for_per_vertex_emissivity(executable):
    """Rendered radiance must equal eps*sigma*T^4 + (1-eps)*sigma*T_amb^4, per surface.

    Two defects made this false. The gray-body shader baked ONE emissivity constant into
    the mix factor and assigned that single material to every mesh, so the per-vertex
    ``emissivity`` attribute the solve writes was never read and every surface radiated as
    if eps=0.9. And the thermal world emitted ``_AMBIENT_K`` (295.372 -- a temperature)
    where the reflected ``(1 - eps)`` term needs a radiance, ``sigma*T_amb^4`` (431.6).

    The second was a ~4% error while eps was pinned at 0.9 and invisible; it dominates at
    low emissivity, where a surface is mostly a mirror. Both matter for LWIR: the preset
    library spans eps 0.05 (polished aluminium) to 0.98 (skin), and a low-emissivity
    surface reading close to ambient IS the physical behaviour that makes polished metal
    hard to measure with a thermal camera.
    """
    code = r"""
import bpy, numpy as np, tempfile, os
from visionsim.simulate.heatsim import thermal_shader as ts

SIGMA = 5.670374419e-8
T_AMB = ts._AMBIENT_K
T_HOT = 350.0

def render_with(eps):
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = 16
    sc.render.resolution_x = sc.render.resolution_y = 32
    sc.render.film_transparent = True
    bpy.ops.mesh.primitive_plane_add(size=3.0, location=(0, 0, 0))
    o = bpy.context.active_object; me = o.data; n = len(me.vertices)
    for name, val in (("sim_temperature", T_HOT), ("emissivity", eps)):
        a = me.attributes.new(name=name, type='FLOAT', domain='POINT')
        a.data.foreach_set("value", np.full(n, val, dtype=np.float32))
    o["heatsim_default_temperature"] = T_HOT
    bpy.ops.object.camera_add(location=(0, 0, 6)); sc.camera = bpy.context.object
    sc.camera.data.type = 'ORTHO'; sc.camera.data.ortho_scale = 4.0
    state = ts.enter_thermal_scene(sc, radiance_scale=1.0)
    try:
        out = os.path.join(tempfile.mkdtemp(), "r.exr")
        sc.render.image_settings.file_format = 'OPEN_EXR'
        sc.render.image_settings.color_depth = '32'
        sc.render.filepath = out
        bpy.ops.render.render(write_still=True)
    finally:
        ts.restore_scene(sc, state)
    img = bpy.data.images.load(out); w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float64).reshape(h, w, 4)
    lit = px[:, :, 3] > 0.5
    return float(np.median(px[:, :, 0][lit]))

for eps in (0.05, 0.50, 0.98):
    got = render_with(eps)
    want = eps * SIGMA * T_HOT**4 + (1.0 - eps) * SIGMA * T_AMB**4
    rel = abs(got - want) / want
    assert rel < 0.02, f"eps={eps}: rendered {got:.2f}, gray body predicts {want:.2f} ({rel:.1%} off)"

# And the emissivity must actually be read per surface, not collapsed to one constant.
lo, hi = render_with(0.05), render_with(0.98)
assert hi - lo > 300.0, f"emissivity barely changed radiance: {lo:.1f} vs {hi:.1f}"
print("GRAY_BODY_EMISSIVITY_OK")
"""
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True, check=False)
    assert "GRAY_BODY_EMISSIVITY_OK" in out.stdout, out.stdout + out.stderr
