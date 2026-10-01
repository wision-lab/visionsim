Thermal rendering
=================

VisionSim simulates heat on the surfaces of a Blender scene and renders the
result as temperature and thermal-camera images. Enable it alongside the usual
RGB and other outputs with ``--config.include-thermal``. VisionSim bakes the
scene's lighting with Cycles, integrates a temperature field for the requested
simulation time, and renders that final field from each requested camera pose.
A render animation therefore shows the same solved thermal state from different
views; it does not advance the heat simulation between frames.

The scene's world, lights, emissive surfaces, materials and geometry affect the
Cycles lighting bake. The heat solve uses surface sample points and a robust
point-cloud Laplacian [1]_. The point-cloud approach tolerates typical scene
meshes, though scene scale, surface detail and sensible thermal material values
still matter.

Getting started
---------------

Save a Blender scene with visible, renderable meshes, then run::

    vsim blender.render-animation scene.blend out/ --config.include-thermal

The defaults produce a temperature EXR, an inferno-colour preview PNG, and a
thermal-radiance EXR in addition to normal render outputs. For a CPU-only
machine, add ``--config.thermal.device cpu``. To skip the second render when
you only need temperatures, add ``--config.thermal.no-radiance``. The default
``render-domain AUTO`` chooses how densely to sample each surface; large walls
and floors will often use thermal atlas texels.

For example, to assign thermal materials, run a 60-second static heat solve,
and increase the bake's spatial resolution::

    vsim blender.render-animation scene.blend out/ \
        --config.include-thermal \
        --config.thermal.assignments scene.thermal.json \
        --config.thermal.sim-time-s 60 \
        --config.thermal.timestep-s 0.1 \
        --config.thermal.irradiance-texture-size 1024

``sim-time-s`` is elapsed *simulation* time, not animation duration. Adjust it
to let heating and conduction develop before rendering. It does not move
objects or change illumination during the solve.

Reading the outputs
-------------------

``temperature/``
    Single-channel OpenEXR containing visible surface temperature in Kelvin.
    Read its floating-point values for measurement; 300 K is about 27 °C.
    A fixed-temperature source reports its assigned temperature.

``previews/temperature/``
    PNG with the temperature EXR mapped through the inferno palette. The colour
    range follows the solved scene's temperatures, using the central 1st-to-99th
    percentile range and a minimum span so small differences remain visible.
    The same colour in two scenes need not mean the same Kelvin value. Disable
    with ``--config.thermal.no-preview``.

``thermal_radiance/``
    Three-channel OpenEXR from a second Cycles render. The thermal material
    emits according to temperature and emissivity and diffusely reflects
    thermal light from other surfaces and the world. This is a broadband
    gray-body image, not a calibrated image for a particular infrared camera
    band. Its emission scale follows :math:`\varepsilon\sigma T^4`, where
    :math:`\varepsilon` is surface emissivity and :math:`\sigma` is the
    Stefan–Boltzmann constant [2]_. The shader uses an exitance-equivalent
    :math:`\sigma T^4` magnitude, so pixel values should not be interpreted
    directly as W/(m²·sr) without a Lambertian :math:`1/\pi` conversion.

Keep the EXRs when quantitative values matter. The PNG is a visualization.
Radiance is a separate render pass, so turning it off can save substantial
render time while leaving the temperature solve and output available.

How the solve works
-------------------

The Cycles bake estimates incident light and visible albedo on each sampled
surface. VisionSim approximates absorbed heat as irradiance times
``1 - albedo``; ``irradiance-scale`` then multiplies that input. This is a
broadband approximation: visible colour is not measured infrared absorptivity.
The point-cloud solver spreads heat among nearby samples and includes radiative
exchange with a fixed ambient reference. Its proximity-based conduction can
also couple nearby, physically separate surfaces. It does not model air
convection. Material presets are practical starting values, not measurements
of a particular asset.

The scene starts at ``initial-temperature-K`` except at pinned sources. The
solver advances in ``timestep-s`` increments for ``sim-time-s`` and renders the
last temperature field. Both the heat calculation and thermal-render world use
a fixed 295 K ambient reference, separate from the configurable initial
temperature. Geometry is converted to millimetres internally; specify
diffusivity in mm²/s, density in kg/m³, specific heat in J/(kg·K), and
temperature in K.

Thermal parameters
------------------

All options below use the ``--config.thermal.`` prefix. For example,
``--config.thermal.emissivity 0.8`` changes the global emissivity. These are
scene-wide defaults: explicitly authored Blender object values can replace
material defaults, and an assignment sidecar can supply values per material
slot. Boolean options such as ``radiance`` and ``preview`` accept a ``no-``
form to turn them off. See :class:`visionsim.simulate.config.ThermalConfig`
for the configuration type.

Output and file settings
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 29 12 59

   * - Parameter
     - Default
     - What it changes
   * - ``radiance``
     - ``True``
     - Includes the thermal-camera EXR and its second Cycles render. Disable
       when temperature data alone is enough.
   * - ``preview``
     - ``True``
     - Writes the colourized temperature PNG. This does not change the solve.
   * - ``assignments``
     - ``None``
     - Path to JSON that assigns thermal presets and source roles by Blender
       material name. See `Assigning materials and heat sources`_.
   * - ``exr-codec``
     - ``ZIP``
     - Compression for temperature and radiance EXRs. ZIP preserves their
       floating-point values.
   * - ``bit-depth``
     - ``32``
     - Bits per channel in those EXRs. ``16`` reduces file size at the cost of
       precision; use ``32`` for quantitative temperature analysis.

Material defaults
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 32 12 56

   * - Parameter
     - Default
     - What it changes
   * - ``initial-temperature-K``
     - ``295.0``
     - Starting surface temperature in Kelvin. Raising it warms the initial
       field, but does not change the fixed ambient reference.
   * - ``thermal-diffusivity-mm2-s``
     - ``0.17``
     - How quickly temperature differences spread spatially. Larger values
       smooth hot spots faster; smaller values keep heating more local.
   * - ``density-kg-m3``
     - ``1330.0``
     - Density. Together with specific heat, it sets thermal inertia: higher
       density generally slows temperature change for a given input.
   * - ``specific-heat-J-kgK``
     - ``880.0``
     - Energy needed to warm a unit mass by one Kelvin. Larger values generally
       reduce and slow the temperature rise.
   * - ``emissivity``
     - ``0.9``
     - Surface emissivity from 0 to 1. It controls radiative exchange in the
       solve and the mix of emitted versus diffusely reflected thermal light
       in the radiance image. Low-emissivity metal can appear closer to its
       surroundings than to its own temperature.

Heating and integration
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 32 12 56

   * - Parameter
     - Default
     - What it changes
   * - ``irradiance-scale``
     - ``100.0``
     - Multiplies absorbed input from the lighting bake. Increase it for a
       stronger temperature rise from scene lighting. Unlike
       ``radiance-scale``, it changes the solved temperature.
   * - ``sim-time-s``
     - ``1.0``
     - Duration of the static heat solve in seconds. Longer times allow more
       heat to accumulate, diffuse and exchange with the ambient reference.
   * - ``timestep-s``
     - ``0.05``
     - Integration step in seconds. Smaller steps resolve the transient more
       finely but require more solve time.
   * - ``bake-samples``
     - ``1024``
     - Cycles samples for the irradiance bake. More samples reduce stochastic
       speckle in the heating input, at extra bake cost; they do not increase
       spatial resolution.
   * - ``irradiance-texture-size``
     - ``512``
     - Width and height of the square Cycles irradiance and albedo bakes.
       Increasing it can preserve finer lighting and texture variation, with
       roughly quadratic image-memory and bake-work growth.
   * - ``device``
     - ``cuda``
     - Compute device for the heat solve: ``cuda`` or ``cpu``. If CUDA is
       unavailable, the CUDA choice falls back to CPU.
   * - ``recompute``
     - ``False``
     - Regenerates the Cycles bakes and heat solve instead of using a reusable
       result from the scene's thermal cache.

Choosing surface samples
~~~~~~~~~~~~~~~~~~~~~~~~

The solver needs points on surfaces where temperatures can differ. ``VERTEX``
samples evaluated mesh vertices and interpolates the result over each face.
This works well for dense geometry, but a wall made from two large triangles
cannot show a detailed heating pattern between its corners. ``TEXEL`` creates
a thermal UV atlas and samples its covered texels. Its resolution can exceed
the mesh's vertex density. Both choices use the same point-cloud heat solver
and Cycles bake.

``AUTO`` chooses separately for each object. It keeps vertices when they are
dense enough and the solved field can be written back safely. It chooses texels
for sparse surfaces, locally large triangles, and evaluated meshes whose
vertices cannot be mapped safely back to the base mesh. Use ``TEXEL`` when you
want atlas sampling throughout a scene, or ``VERTEX`` when diagnosing a dense
mesh. The thermal UV layer is separate from the UVs used for RGB materials.
If an atlas is required but its surface cannot be rasterized, or if vertex
write-back is unsafe in ``VERTEX`` mode, VisionSim reports an error.

.. list-table::
   :header-rows: 1
   :widths: 32 12 56

   * - Parameter
     - Default
     - What it changes
   * - ``render-domain``
     - ``AUTO``
     - Selects ``AUTO``, ``VERTEX`` or ``TEXEL`` thermal sampling. It controls
       solve points and how the field reaches the render, not image resolution.
   * - ``atlas-texel-density``
     - ``1500.0``
     - Target thermal samples per square metre for atlas surfaces. Raising it
       can resolve smaller hot spots, but increases point count, memory and
       solve cost. Tile dimensions and the overall budget can limit it.
   * - ``atlas-tile-min``
     - ``16``
     - Minimum side length, in texels, of one object's atlas tile. A larger
       minimum gives small objects more samples, at extra memory cost.
   * - ``atlas-tile-max``
     - ``512``
     - Maximum side length of one object's tile. A lower maximum limits the
       cost of large objects but reduces their effective resolution.
   * - ``atlas-texel-soft-max``
     - ``500000``
     - Soft budget for atlas texels plus retained vertices. If exceeded,
       VisionSim reduces effective atlas density and warns. It is a planning
       limit rather than an exact cap.
   * - ``radiance-scale``
     - ``1.0``
     - Multiplies gray-body emission in the thermal-radiance render. It changes
       image brightness, not the heat solve or temperature EXR.

If a heating pattern looks noisy, raise ``bake-samples`` first. If it looks
smooth but misses small features, inspect both ``irradiance-texture-size`` and
the thermal sample density. More render pixels cannot restore a pattern absent
from the bake or solve points. Start with ``AUTO`` and change one setting at a
time while comparing the temperature EXR.

Assigning materials and heat sources
------------------------------------

Global material defaults apply when nothing more specific is authored. Blender
objects can carry a ``heat_sim_material`` property group with initial
temperature, diffusivity, density, specific heat, emissivity, ``thermal_role``
and ``dirichlet_temperature_K``. Explicitly set object values override global
defaults. The object's ``heat_simulation_enabled`` property controls whether
it joins the solve; an excluded visible object still renders with its default
temperature.

For scenes with several materials on one object, supply a JSON sidecar so a
wooden seat and metal legs need not share one thermal material. A sidecar maps
*Blender material names* to presets. The :doc:`thermal assignment tutorial
<../../tutorials/thermal-assignments>` shows how to inventory a scene, draft a
sidecar and check it before rendering. Save this as ``scene.thermal.json``::

    {
      "schema_version": 1,
      "scene": "scene.blend",
      "defaults": {"preset": "plaster"},
      "materials": {
        "Wood Panel": {"preset": "wood"},
        "Steel Fixture": {"preset": "metal_painted"},
        "Warm Plate": {
          "preset": "metal_painted",
          "role": "DIRICHLET_SOURCE",
          "dirichlet_K": 320.0
        }
      }
    }

Pass it using ``--config.thermal.assignments scene.thermal.json``. Material
names must match the blend file exactly. ``schema_version`` must be ``1``;
``scene`` is an informational label. The optional ``defaults.preset`` applies
when a material entry is missing or has no preset, including unnamed slots.
When neither supplies a preset, the object's authored values or the generic
fallback values in ``ThermalConfig`` apply. A null preset does not exclude a
surface from the simulation.

A ``FEM_PARTICIPANT`` starts at its initial temperature and evolves during the
solve. A ``DIRICHLET_SOURCE`` stays pinned at ``dirichlet_K`` and can warm or
cool nearby participants through point-cloud coupling. A 320 K hotplate is
about 47 °C. Use a fixed source when its temperature is known or deliberately
prescribed. The sidecar accepts 280–500 K for ``dirichlet_K``. On a Blender
object, set ``thermal_role`` to ``DIRICHLET_SOURCE`` and
``dirichlet_temperature_K`` to its fixed value; if that object value is zero,
its initial temperature is used. Sidecar entries can also include
``confidence`` and ``reason`` as authoring notes; the solver ignores them.

Preset values supply diffusivity, density, specific heat and emissivity. Some
common starting points are shown below. The full set of preset names is
``aluminium``, ``aluminium_polished``, ``asphalt``, ``brick``, ``carpet``,
``ceramic``, ``concrete``, ``copper``, ``drywall``, ``fabric``, ``foliage``,
``food``, ``glass``, ``iron``, ``leather``, ``li_ion``, ``marble``,
``metal_painted``, ``paper``, ``plaster``, ``polystyrene``, ``porcelain``,
``pvc``, ``rubber``, ``skin``, ``stainless_steel``, ``steel``, ``water`` and
``wood``. Their values are defined in
:mod:`visionsim.simulate.heatsim.materials`. They describe generic materials
rather than a measured piece of furniture or appliance. A coating can change
surface emissivity even when the metal underneath has similar thermal mass.

.. list-table:: Example preset values
   :header-rows: 1
   :widths: 25 18 18 19 12

   * - Preset
     - Diffusivity (mm²/s)
     - Density (kg/m³)
     - Specific heat (J/kg·K)
     - Emissivity
   * - ``wood``
     - 0.082
     - 897
     - 2380
     - 0.90
   * - ``plaster``
     - 0.4
     - 1200
     - 1090
     - 0.91
   * - ``glass``
     - 0.34
     - 2500
     - 840
     - 0.92
   * - ``metal_painted``
     - 4.2
     - 7930
     - 280
     - 0.92
   * - ``stainless_steel``
     - 4.0
     - 7900
     - 500
     - 0.16
   * - ``aluminium_polished``
     - 97.0
     - 2700
     - 978
     - 0.05

Within an atlas tile, material properties follow the face's material slot.
On the vertex path, properties at a vertex shared by multiple material faces
are area-weighted; a fixed-temperature role follows the dominant adjacent
material. Both sampling modes retain per-material behaviour, while a material
boundary may look sharper in the atlas.

Reusing a solve
---------------

VisionSim stores reusable thermal results beside the blend file in a
``.heatsim`` directory. For a clean, saved blend, it checks blend bytes,
external images and libraries, solver settings, sampling layout, and sidecar
contents before reusing a result. Edited or missing dependencies cause a new
bake and solve. ``--config.thermal.recompute`` forces a new one. Changing only
preview or radiance display settings does not require a new heat solve.
Thermal atlases and attributes are prepared in memory for rendering; the
source blend is not overwritten. Save an output blend explicitly if you need
to reopen that prepared scene without relying on its external cache.

References
----------

.. [1] Sharp and Crane (2020), `A Laplacian for Nonmanifold Triangle Meshes
   <https://www.cs.cmu.edu/~kmcrane/Projects/NonmanifoldLaplace/index.html>`_.
   VisionSim uses the ``robust_laplacian`` point-cloud implementation.
.. [2] NIST, `CODATA value of the Stefan–Boltzmann constant
   <https://physics.nist.gov/cgi-bin/cuu/Value?sigma>`_.
