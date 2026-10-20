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
