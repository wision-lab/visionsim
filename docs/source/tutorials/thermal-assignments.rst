Assign thermal materials to a Blender scene
===========================================

A Blender material describes how a surface looks in an RGB render. Thermal
simulation also needs properties such as diffusivity and emissivity. A thermal
assignment sidecar maps each Blender material name to one of VisionSim's
thermal presets, and can give selected surfaces a fixed temperature. The
renderer reads this JSON file when you pass ``--config.thermal.assignments``.
You can write the file yourself or have a coding agent draft it for review.

Inventory the scene
-------------------

From a VisionSim source checkout, run the helper inside Blender::

    blender -b scene.blend --python scripts/thermal_assign.py -- \
        dump --output scene.materials.json

The inventory lists material names exactly as Blender stores them, the objects
using each material, image texture names, and an approximate share of mesh
surface area. Use the area share to review large walls and floors first. An
emission node is a clue about the RGB shader; it does not by itself mean that
the surface should be held at a fixed thermal temperature.

Create and edit the sidecar
---------------------------

Create a scaffold with one entry for every inventoried material::

    python scripts/thermal_assign.py template scene.materials.json \
        --output scene.thermal.json

Each entry starts with ``"preset": null``. Replace that with a preset such as
``plaster``, ``wood``, ``glass`` or ``metal_painted``. The full preset list and
example physical values are in the :doc:`thermal rendering guide
<../sections/blender/thermal>`. A normal material can be as simple as::

    "Wood Panel": {"preset": "wood"}

For a surface whose temperature you deliberately prescribe, add a source role
and a temperature in Kelvin::

    "Warm Plate": {
      "preset": "metal_painted",
      "role": "DIRICHLET_SOURCE",
      "dirichlet_K": 320.0
    }

Keep the material names exactly as they appear in the inventory. A source
stays at its assigned temperature throughout the heat solve; ordinary
materials start at their initial temperature and evolve. The sidecar accepts
source temperatures from 280 to 500 K. Add an optional ``reason`` or
``confidence`` to record why you chose a preset; these notes do not affect the
simulation.

You can also use a coding agent to help edit the sidecar. Give it the material
inventory and preset library, then review its choices before rendering.

Review the draft yourself, especially large-area materials, polished versus
painted metals, and any fixed-temperature sources. Presets are starting points
for a scene, not measurements of a particular asset.

Thermal reflections
-------------------

Thermal rendering uses the assigned surface finish, independently of its RGB
shader. Glass is opaque in this model. Its preset uses mostly specular
reflection; polished metal has a narrow glossy lobe and rough metal a broader
one. Matte presets retain diffuse reflection. With no sidecar, reflection
remains diffuse.

Two optional per-material values override a preset::

    "Brushed trim": {
      "preset": "stainless_steel",
      "specular_fraction_ir": 0.90,
      "roughness_ir": 0.30,
      "reason": "Estimated brushed finish; no measured LWIR BRDF for this asset."
    }

Both values must lie in [0, 1]. ``specular_fraction_ir`` divides the reflected
energy between glossy and diffuse lobes; emissivity still sets the total
reflection budget, ``1 - emissivity``. ``roughness_ir`` controls the width of
the white Cycles MULTI_GGX lobe, from sharp (0) to broad (1). For example,
emissivity 0.84 and specular fraction 0.95 give 0.152 specular and 0.008 diffuse
reflectance. These controls change the rendered radiance, not the heat input
or temperature integration. The thermal pass permits at least 16 diffuse and
glossy bounces and restores RGB materials, overrides and bounce limits afterward.

The current reflection-shape estimates are:

.. list-table::
   :header-rows: 1

   * - Preset / finish
     - Specular fraction
     - Cycles roughness
   * - Smooth glass
     - 0.95
     - 0.05
   * - Polished aluminium
     - 0.98
     - 0.08
   * - Buffed stainless steel
     - 0.95
     - 0.18
   * - Rough aluminium
     - 0.80
     - 0.35
   * - Rolled steel
     - 0.80
     - 0.30
   * - Cast iron
     - 0.60
     - 0.40
   * - Oxidised copper
     - 0.50
     - 0.40

These are **estimated LWIR appearance controls**, not measured 8–14 µm BRDF
fits. Blender roughness is not a roughness height in micrometres. Brushing is
approximated isotropically; this model does not include directional brushing,
angle-dependent Fresnel emissivity, wavelength-dependent reflection or infrared
transmission. Ordinary glass is specular despite high emissivity (see
`LBNL-46590, printed page 22 <https://windows.lbl.gov/sites/default/files/46590.pdf#page=23>`_).
`NIST/PNNL infrared measurements <https://pmc.ncbi.nlm.nih.gov/articles/PMC6193458/>`_
show how strongly metal scattering depends on finish; their Table 2 averages
are not 8–14 µm band averages. The existing scalar emissivities retain their
documented approximations.

Check and render
----------------

Check that every material name matches, presets exist, and any fixed source has
a valid temperature::

    python scripts/thermal_assign.py check scene.materials.json scene.thermal.json

The checker reports unknown or missing names and invalid values as errors. It
warns about ``null`` presets, which use the sidecar fallback or the object's
thermal defaults. Correct the errors, review the warnings, then render::

    vsim blender.render-animation scene.blend out/ \
        --config.include-thermal \
        --config.thermal.assignments scene.thermal.json

Keep the reviewed sidecar with the scene so the assignments used for a render
are reproducible. The inventory and template helper are authoring tools; the
render command only needs the final sidecar.
