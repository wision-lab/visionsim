"""Which meshes block a shortwave shadow ray (visionsim.simulate.heatsim.occluders)."""

from __future__ import annotations

import subprocess


def test_transparent_and_shadowless_meshes_do_not_occlude(executable):
    """Glass panes, Cycles light portals and shadow-disabled props must not
    block shortwave shadow rays; opaque, alpha-cutout and partly-transmissive
    geometry must keep casting shadows."""
    code = r"""
import bpy
from visionsim.simulate.heatsim import irradiance_kernel as ik
from visionsim.simulate.heatsim import occluders

def group_shader_output(group):
    # `node_group.interface` is Blender 4.0+; 3.6 uses `node_group.outputs`. The behaviour
    # under test (glass inside a node group still reads as clear) is version-independent,
    # so support both rather than skipping the 3.6 CI leg.
    if hasattr(group, "interface"):
        group.interface.new_socket('Shader', in_out='OUTPUT', socket_type='NodeSocketShader')
    else:
        group.outputs.new('NodeSocketShader', 'Shader')

def mesh(name):
    bpy.ops.mesh.primitive_plane_add()
    o = bpy.context.active_object
    o.name = name
    return o

def make_mat(name, build):
    # Build a material whose Surface output is driven by build's shader.
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    tree = m.node_tree
    out = next(n for n in tree.nodes if n.type == 'OUTPUT_MATERIAL')
    # Drop the default Principled so only what `build` returns feeds the output.
    for n in [n for n in tree.nodes if n.type == 'BSDF_PRINCIPLED']:
        tree.nodes.remove(n)
    tree.links.new(build(tree).outputs[0], out.inputs['Surface'])
    return m

def add(tree, bl_idname):
    return tree.nodes.new(bl_idname)

def glass(tree):
    return add(tree, 'ShaderNodeBsdfGlass')

def transparent(tree):
    return add(tree, 'ShaderNodeBsdfTransparent')

def principled(weight):
    def build(tree):
        n = add(tree, 'ShaderNodeBsdfPrincipled')
        for key in ('Transmission Weight', 'Transmission'):
            if key in n.inputs:
                n.inputs[key].default_value = weight
        return n
    return build

def cutout_leaf(tree):
    # Opaque leaf mixed with a Transparent BSDF - standard alpha cutout.
    mix = add(tree, 'ShaderNodeMixShader')
    tree.links.new(transparent(tree).outputs[0], mix.inputs[1])
    tree.links.new(principled(0.0)(tree).outputs[0], mix.inputs[2])
    return mix

def transparent_plus_diffuse(tree):
    mix = add(tree, 'ShaderNodeMixShader')
    tree.links.new(transparent(tree).outputs[0], mix.inputs[1])
    tree.links.new(add(tree, 'ShaderNodeBsdfDiffuse').outputs[0], mix.inputs[2])
    return mix

def driven_transmission(tree):
    # Transmission fed by a texture - unknowable statically, must be opaque.
    n = add(tree, 'ShaderNodeBsdfPrincipled')
    key = next(k for k in ('Transmission Weight', 'Transmission') if k in n.inputs)
    tree.links.new(add(tree, 'ShaderNodeTexNoise').outputs[0], n.inputs[key])
    return n

def check(name, build, should_occlude):
    o = mesh(name)
    if build is not None:
        o.data.materials.append(make_mat(name + '_m', build))
    got = occluders.casts_shadow(o)
    assert got == should_occlude, f'{name}: expected occlude={should_occlude}, got {got}'
    return o

check('no_slots', None, True)
check('glass', glass, False)
check('portal', transparent, False)
check('clear_principled', principled(1.0), False)
check('frosted', principled(0.5), True)
check('opaque', principled(0.0), True)
check('cutout_leaf', cutout_leaf, True)
check('decal', transparent_plus_diffuse, True)
check('driven', driven_transmission, True)

# Glass wrapped in a node group must still read as clear (review finding).
group = bpy.data.node_groups.new('GlassGroup', 'ShaderNodeTree')
g_out = group.nodes.new('NodeGroupOutput')
group_shader_output(group)
group.links.new(group.nodes.new('ShaderNodeBsdfGlass').outputs[0], g_out.inputs[0])

def grouped_glass(tree):
    n = tree.nodes.new('ShaderNodeGroup')
    n.node_tree = group
    return n

check('grouped_glass', grouped_glass, False)

# An opaque BSDF hidden inside a group must still occlude.
ogroup = bpy.data.node_groups.new('OpaqueGroup', 'ShaderNodeTree')
og_out = ogroup.nodes.new('NodeGroupOutput')
group_shader_output(ogroup)
ogroup.links.new(ogroup.nodes.new('ShaderNodeBsdfDiffuse').outputs[0], og_out.inputs[0])

def grouped_opaque(tree):
    n = tree.nodes.new('ShaderNodeGroup')
    n.node_tree = ogroup
    return n

check('grouped_opaque', grouped_opaque, True)

# Only the ACTIVE Group Output feeds the instance. A leftover inactive one wired
# to an opaque shader must not turn a clear pane into a shadow caster.
stale = bpy.data.node_groups.new('StaleGroup', 'ShaderNodeTree')
group_shader_output(stale)
live_out = stale.nodes.new('NodeGroupOutput')
live_out.is_active_output = True
stale.links.new(stale.nodes.new('ShaderNodeBsdfGlass').outputs[0], live_out.inputs[0])
dead_out = stale.nodes.new('NodeGroupOutput')
dead_out.is_active_output = False
stale.links.new(stale.nodes.new('ShaderNodeBsdfDiffuse').outputs[0], dead_out.inputs[0])

def grouped_stale(tree):
    n = tree.nodes.new('ShaderNodeGroup')
    n.node_tree = stale
    return n

check('grouped_stale_output', grouped_stale, False)

# An empty material slot renders opaque, so [None, glass] must keep its shadow.
empty_and_glass = mesh('empty_and_glass')
empty_and_glass.data.materials.append(None)
empty_and_glass.data.materials.append(make_mat('eg_glass', glass))
assert occluders.casts_shadow(empty_and_glass), 'None slot + glass must occlude'

# Volume-only material (Surface unlinked) has no surface to block a ray.
volbox = mesh('volbox')
vm = bpy.data.materials.new('vol_m')
vm.use_nodes = True
vt = vm.node_tree
vout = next(n for n in vt.nodes if n.type == 'OUTPUT_MATERIAL')
for n in [n for n in vt.nodes if n.type == 'BSDF_PRINCIPLED']:
    vt.nodes.remove(n)
vt.links.new(vt.nodes.new('ShaderNodeVolumePrincipled').outputs[0], vout.inputs['Volume'])
volbox.data.materials.append(vm)
assert not occluders.casts_shadow(volbox), 'volume-only material must not occlude'

# An unconnected leftover Principled must not make a glass pane opaque.
stray = mesh('stray')
m = bpy.data.materials.new('stray_m')
m.use_nodes = True
tree = m.node_tree
out = next(n for n in tree.nodes if n.type == 'OUTPUT_MATERIAL')
tree.links.new(tree.nodes.new('ShaderNodeBsdfGlass').outputs[0], out.inputs['Surface'])
stray.data.materials.append(m)
assert not occluders.casts_shadow(stray), 'orphaned Principled must not force opacity'

# One opaque slot is enough to keep the whole object casting shadows.
mixed = mesh('mixed')
mixed.data.materials.append(make_mat('mixed_glass', glass))
mixed.data.materials.append(make_mat('mixed_opaque', principled(0.0)))
assert occluders.casts_shadow(mixed), 'mixed slots must occlude'

noshadow = check('noshadow_base', principled(0.0), True)
noshadow.visible_shadow = False
assert not occluders.casts_shadow(noshadow), 'visible_shadow=False must not occlude'

# The BVH collector must agree with _casts_shadow.
collected = ik._collect_scene_meshes_world(bpy.context.scene)
expected = sum(
    1 for o in bpy.context.scene.objects
    if o.type == 'MESH' and not o.hide_render and o.visible_get() and occluders.casts_shadow(o)
)
assert len(collected) == expected, (len(collected), expected)
print('OCCLUDER_FILTER_OK')
"""
    out = subprocess.run(
        [str(executable), "-b", "--python-expr", code],
        capture_output=True, text=True,
     check=False)
    assert "OCCLUDER_FILTER_OK" in out.stdout, out.stdout + "\n" + out.stderr
