Thermal Modality
================

The thermal modality adds heat-transfer simulation outputs to any render produced by VisionSim.
When ``--config.include-thermal`` is set, VisionSim runs a finite-element-method (FEM)
heat-transfer solve on the scene geometry before the main render loop and produces temperature
and thermal-camera outputs per frame alongside the standard RGB/depth passes.

Enabling thermal
----------------

Thermal is off by default.  Turn it on with a single flag on any render command:

.. code-block:: bash

    vsim blender.render-animation scene.blend out/ --config.include-thermal

Everything else is optional tuning, exposed under the ``--config.thermal.*`` namespace and
documented in `Parameters`_ below.

Outputs
-------

With thermal enabled, three output directories are written per frame in addition to the usual
passes:

``temperature/``
    Per-pixel surface temperature in **Kelvin**, saved as a single-channel
    ``OPEN_EXR`` file.  This is a Cycles value AOV co-rendered with the RGB pass,
    so it adds no extra render samples.  Meshes that did not participate in the FEM
    solve report a fallback, so the value is defined everywhere geometry is visible: a
    ``DIRICHLET_SOURCE`` reports its reservoir temperature; an object whose solved field
    could not be written back per-vertex (a topology-changing modifier) reports the
    **mean of its own solved field**; anything else reports ``initial-temperature-K``.

``thermal_radiance/``
    Gray-body thermal-camera image produced by a **second Blender render** that
    replaces all scene materials with emission shaders driven by the solved
    surface temperature.  The 3-channel ``OPEN_EXR`` is proportional to the
    Stefan-Boltzmann radiated power (``ε × σ × T⁴``), where ``ε`` is a **fixed
    scene-wide constant of 0.9**, not the per-material emissivity — see the note under
    ``emissivity`` below.  This is the most expensive part of thermal; disable it with
    ``--config.thermal.no-radiance`` when you only need the temperature map.

``previews/temperature/``
    An **inferno-colormap PNG** derived from the temperature map, for quick
    visual inspection.  The colormap spans the **global temperature range of the
    solved scene** (its 1st to 99th percentile, so a few outlier-hot texels cannot
    flatten everything else; the range is also floored at the initial temperature and
    widened to span at least 1 K), so a scene with only a small
    temperature rise still uses the full colormap instead of being crushed to one
    end.  Controlled by ``--config.thermal.preview`` (default ``True``).

.. admonition:: Note

    ``temperature/`` and ``thermal_radiance/`` carry physically meaningful
    magnitudes (Kelvin and radiated power).  Load them with a library that
    preserves EXR float precision (e.g. ``OpenEXR``, ``imageio`` with the EXR
    plugin, or ``cv2``); the ``previews/`` PNGs are for display only and are not
    quantitative.

Rendering with thermal
----------------------

Thermal composes with the normal render commands — you can request it alongside any other
modality.  A few common invocations:

.. code-block:: bash

    # temperature + radiance + preview, GPU solve (defaults)
    vsim blender.render-animation scene.blend out/ --config.include-thermal

    # temperature map only — skip the (expensive) gray-body radiance render
    vsim blender.render-animation scene.blend out/ \
        --config.include-thermal \
        --config.thermal.no-radiance

    # force a CPU solve (no CUDA required)
    vsim blender.render-animation scene.blend out/ \
        --config.include-thermal \
        --config.thermal.device cpu

    # make the scene heat up more (larger temperature rise)
    vsim blender.render-animation scene.blend out/ \
        --config.include-thermal \
        --config.thermal.irradiance-scale 1000

Animated geometry
-----------------

Everything above is a **static** solve: one temperature field is computed and held constant for
every frame of the render. Setting ``--config.thermal.animated`` switches to a **per-frame
transient solve** instead, so geometry that moves or deforms over the timeline produces a genuine
thermal *animation* rather than a single frozen field. The motivating example is a hot liquid
pouring into a cup: frame by frame, the liquid's heat diffuses into the cup and the cup's surface
visibly warms up over the sequence.

.. code-block:: bash

    vsim blender.render-animation cup_pour.blend out/ \
        --config.include-thermal \
        --config.thermal.animated \
        --config.thermal.substeps-per-frame 4

Animated mode distinguishes two kinds of objects, selected per object via the ``thermal_role``
material override (see `Per-object material overrides`_):

``FEM_PARTICIPANT`` objects (default) — stable topology
    Objects whose vertex count never changes across the animation (e.g. the cup itself). Their
    temperature **evolves**: each frame's solve carries the previous frame's result forward, so
    heat accumulates and diffuses realistically over time.

``DIRICHLET_SOURCE`` objects — topology-changing sources
    Objects whose mesh is regenerated every frame with a different vertex count — the standard
    situation for a fluid simulation's surface as it pours and splashes. A per-vertex temperature
    history can't be carried forward for a mesh like this, so instead it is treated as a
    **constant-temperature source**: every frame it drives heat into the nearby FEM-participant
    objects at its fixed ``dirichlet_temperature_K``, but its own temperature never evolves and is
    not part of the solve output. Set ``thermal_role = "DIRICHLET_SOURCE"`` on the hot liquid to
    get this behavior.

.. admonition:: Note

    The fluid (or other topology-changing) mesh must already be **baked** before running an
    animated thermal solve — e.g. a Mantaflow fluid domain baked to disk. The solver only reads
    geometry at each frame; it does not run or advance the fluid simulation itself, so the mesh
    must already exist at every frame the thermal solve visits.

.. admonition:: Note

    Animated mode currently requires ``--config.thermal.domain POINTS``. Requesting
    ``animated`` together with ``domain MESH`` logs a warning and falls back to the static
    (M1) solve described above.

.. admonition:: Note

    Because the animated thermal timeline is keyed to unscaled Blender frames at
    ``dt = (1 / fps) / substeps-per-frame``, combining animated thermal with
    ``every-n-frames > 1`` or a ``keyframe-multiplier != 1.0`` render stretch is only
    approximate (and not officially supported) — the simulated timestep doesn't rescale
    to match either setting.

The four animated-mode parameters are listed in the `Solver`_ table below; they only take effect
when ``animated`` is ``True``.

Parameters
----------

All thermal parameters live under ``--config.thermal.*`` and are collected in the
``ThermalConfig`` dataclass (:mod:`visionsim.simulate.config`).  The tables below list every
parameter, its default, and the effect of changing it.  Values set here are **global defaults**;
individual objects can override the material parameters (see `Per-object material overrides`_).

Output control
~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1
    :widths: 26 12 62

    * - Parameter
      - Default
      - Effect
    * - ``radiance``
      - ``True``
      - Render the gray-body ``thermal_radiance/`` image (a second render pass).
        Set ``False`` to skip it and render roughly twice as fast.
    * - ``preview``
      - ``True``
      - Save the inferno-colormap ``previews/temperature/`` PNG.  Set ``False`` to
        skip preview generation.

Material defaults
~~~~~~~~~~~~~~~~~

These set the physical material used for every mesh that has no per-object override.

.. list-table::
    :header-rows: 1
    :widths: 30 12 58

    * - Parameter
      - Default
      - Effect
    * - ``initial-temperature-K``
      - ``295.0``
      - Starting temperature (K) of every vertex, and the fallback reported for
        meshes that do not participate in the solve.  Shifts the whole baseline.
    * - ``thermal-diffusivity-mm2-s``
      - ``0.17``
      - How fast heat spreads through the surface (mm²/s).  Higher values
        equalize temperature across the object faster (smoother field); lower
        values keep sharper local hot spots.
    * - ``density-kg-m3``
      - ``1330.0``
      - Material density.  With specific heat it sets the thermal mass: higher
        density means a slower temperature change for the same heat input.
    * - ``specific-heat-J-kgK``
      - ``880.0``
      - Heat capacity (J/kg·K).  Higher values require more energy to raise the
        temperature, so the rise is slower and smaller.
    * - ``emissivity``
      - ``0.9``
      - Surface emissivity in ``[0, 1]``.  Drives both halves of the modality: radiative
        cooling in the solve (higher ε means more radiative loss, so a lower steady-state
        temperature) and the ``thermal_radiance`` render, where a surface emits
        ``ε·σ·T⁴`` and reflects ``(1 − ε)`` of the light arriving from its surroundings
        (see the note below).

.. note::

   **What the radiance pass actually renders.**  Each surface is a gray body:

   .. math::

      L = \varepsilon\,\sigma T^4 \;+\; (1 - \varepsilon)\,L_{\text{incident}}

   It emits :math:`\varepsilon\,\sigma T^4` from its own temperature and reflects the
   remaining :math:`(1 - \varepsilon)` of whatever light reaches it.  Emissivity is read
   **per vertex** from the material assignment, so different materials in one scene radiate
   differently.

   :math:`L_{\text{incident}}` is **not** a constant.  It is path-traced over the
   hemisphere, so it gathers the radiance of every other surface in view — each of which is
   emitting its own :math:`\varepsilon\,\sigma T^4` — plus the world background wherever
   nothing blocks it.  A cool surface facing a hot one therefore picks up that neighbour,
   weighted by how much of its hemisphere the neighbour covers.  The familiar closed form

   .. math::

      L = \varepsilon\,\sigma T^4 \;+\; (1 - \varepsilon)\,\sigma T_{\text{amb}}^4

   is the special case in which nothing else is in view and the gather reduces to the world
   term alone.  The world is a blackbody enclosure at ambient, emitting
   :math:`\sigma T_{\text{amb}}^4`.

   This is the contrast an LWIR camera actually sees, and it is why the preset library's
   range matters: ``aluminium_polished`` (ε = 0.05) and ``skin`` (ε = 0.98) at the same
   physical temperature are far apart in the image.  It is also why a low-emissivity surface
   reads closer to *its surroundings* than to its own temperature — it is mostly mirroring
   them, which is exactly what makes polished metal hard to measure with a real thermal
   camera.

   The reflection is **Lambertian, not specular**: the radiance is a cosine-weighted average
   over the hemisphere, so a low-ε surface picks up the correct amount of energy from a hot
   neighbour but does not show a mirror image of it.  Bounce counts are inherited from the
   ``.blend`` rather than set by the thermal pass.

Material presets
~~~~~~~~~~~~~~~~

The material knobs above are raw physical properties, so any material is a matter of
supplying the right four numbers.  The **preset library** packages those numbers under a
short key; a sidecar refers to a material by that key rather than repeating the values
(see `Automatic material assignment (sidecars)`_).

The table below shows 16 of the 29 presets, with the exact values the
library uses.  **The full list of keys is the closed vocabulary a sidecar's** ``preset``
**field accepts** — read it from
:mod:`visionsim.simulate.heatsim.materials`'s ``_PRESET_TABLE``, or at runtime::

    from visionsim.simulate.heatsim import materials
    print(materials.preset_keys())

Thermal **diffusivity** is the dominant knob for how heat *spreads* — metals conduct it
across the whole object quickly, while glass, plastics and textiles keep it localized.
**Emissivity** is what an LWIR camera sees: note ``aluminium_polished`` at 0.05 against
``skin`` at 0.98, a 20x range.

.. list-table::
    :header-rows: 1
    :widths: 26 20 18 20 16

    * - ``preset`` key
      - ``thermal-diffusivity-mm2-s``
      - ``density-kg-m3``
      - ``specific-heat-J-kgK``
      - ``emissivity``
    * - ``aluminium``
      - ``97``
      - ``2700``
      - ``978``
      - ``0.2``
    * - ``aluminium_polished``
      - ``97``
      - ``2700``
      - ``978``
      - ``0.05``
    * - ``stainless_steel``
      - ``4``
      - ``7900``
      - ``500``
      - ``0.16``
    * - ``copper``
      - ``111``
      - ``8960``
      - ``385``
      - ``0.15``
    * - ``glass``
      - ``0.34``
      - ``2500``
      - ``840``
      - ``0.92``
    * - ``marble``
      - ``1.2``
      - ``2700``
      - ``880``
      - ``0.94``
    * - ``concrete``
      - ``0.5``
      - ``2300``
      - ``880``
      - ``0.92``
    * - ``plaster``
      - ``0.4``
      - ``1200``
      - ``1090``
      - ``0.91``
    * - ``drywall``
      - ``0.31``
      - ``800``
      - ``1090``
      - ``0.9``
    * - ``wood``
      - ``0.082``
      - ``897``
      - ``2380``
      - ``0.9``
    * - ``pvc``
      - ``0.17``
      - ``1330``
      - ``880``
      - ``0.93``
    * - ``fabric``
      - ``0.09``
      - ``300``
      - ``1300``
      - ``0.95``
    * - ``carpet``
      - ``0.06``
      - ``200``
      - ``1300``
      - ``0.9``
    * - ``water``
      - ``0.143``
      - ``997``
      - ``4182``
      - ``0.96``
    * - ``foliage``
      - ``0.15``
      - ``700``
      - ``3000``
      - ``0.96``
    * - ``skin``
      - ``0.11``
      - ``1050``
      - ``3470``
      - ``0.98``


Solver
~~~~~~

.. list-table::
    :header-rows: 1
    :widths: 28 14 58

    * - Parameter
      - Default
      - Effect
    * - ``irradiance-scale``
      - ``100.0``
      - Scales the absorbed-heat input that drives the temperature rise.  This is
        the main knob for **how hot the scene gets** — larger values produce a
        larger temperature rise.  (If the blend file carries authored thermal
        scene settings, that authored value takes precedence over this flag.)
    * - ``sim-time-s``
      - ``1.0``
      - Total simulated time of the static solve (seconds).  Longer times let the
        scene heat closer to steady state.
    * - ``timestep-s``
      - ``0.05``
      - Solver timestep (seconds).  Smaller steps are more accurate and stable at
        higher cost; the solve runs ``sim-time-s / timestep-s`` steps.
    * - ``domain``
      - ``POINTS``
      - FEM domain.  **Leave this on** ``POINTS`` **for scene content.**  It solves on a
        surface point cloud and does not care about mesh connectivity, so it tolerates
        the non-manifold, mixed quad/ngon, duplicated-vertex geometry that ordinary
        archviz assets are full of.  ``MESH`` solves on the mesh connectivity directly,
        which is only appropriate for clean, **fully triangulated, manifold** meshes —
        see ``laplacian-backend`` below for why.
    * - ``laplacian-backend``
      - ``ROBUST``
      - Discrete-Laplacian construction.  ``ROBUST`` builds a point-cloud Laplacian that
        tolerates non-manifold and low-quality meshes.  ``IGL`` uses the libigl
        **cotangent** Laplacian, which is defined per triangle: on quads or ngons it is
        either undefined or silently poor, and degenerate (zero-area, sliver) triangles
        make the cotangent weights blow up.  Prefer ``ROBUST`` unless you know the mesh
        is triangulated and clean.
    * - ``irradiance-source``
      - ``DIRECT_KERNEL``
      - Where absorbed flux comes from.  ``DIRECT_KERNEL`` is the analytic path: per-light
        form factors, Embree shadow rays and a 9-coefficient SH sky.  It is fast, but it
        counts **only objects of type** ``LIGHT`` plus the world sky, and models no
        indirect bounce.  ``CYCLES_BAKE`` bakes DIFFUSE DIRECT+INDIRECT per object with
        Cycles instead, so **emissive geometry, indirect bounce, portals and HDRI
        transport** all contribute.  Prefer it whenever a scene's real light source is not
        a lamp object — an interior daylit through emissive window planes or light
        portals, or lit by emissive fixture panels, which is how such scenes are commonly
        built.  The symptom of using the kernel there is a room that renders flat at its
        initial temperature while its RGB render is well lit; switching source can change
        the interior temperature spread by an order of magnitude.
    * - ``bake-samples``
      - ``1024``
      - Cycles samples for the irradiance bake (``CYCLES_BAKE`` only).  Adaptive sampling
        is **disabled** for the bake, so this is a true per-texel sample count rather than
        a cap.  It is set here rather than inherited from the blend on purpose: production
        blends are usually tuned for a *look*, often with a loose adaptive-sampling
        threshold, and adaptive sampling terminates texels well below the nominal cap.
        That is fine for a rendered image and not fine for a physical input, because baked
        irradiance noise propagates into the temperature field.  A steady-state surface
        sits at ``T ~ (E/(eps*sigma))**0.25``, so a relative error in irradiance appears as
        roughly a quarter of that in temperature — a bake left noisy at the single-digit
        percent level shows up as several-Kelvin blotching.  Raising this only helps as
        ``1/sqrt(N)``, so quadrupling the samples buys a little under half the noise; if a
        bake is visibly blotchy, check first whether adaptive sampling or a shading-heavy
        material is the real cause.  Denoising does **not** apply to a bake, so sample
        count is the only lever here.
    * - ``irradiance-texture-size``
      - ``512``
      - Resolution of the Cycles bakes, in pixels per side.  Governs the albedo bake and,
        under ``CYCLES_BAKE``, the irradiance bake.  This is the bake's **spatial detail**,
        as distinct from ``bake-samples``, which is its **noise**: more samples give a
        smoother bake at the same resolution and cannot recover detail the resolution never
        captured.  A large surface unwrapped into one 512 px tile gets few texels per square
        metre however many samples you use, so raise this for scenes with big floors, walls
        or ceilings.  Cost is quadratic in this value.
    * - ``device``
      - ``cuda``
      - Compute device for the solve.  Falls back to ``cpu`` automatically when
        CUDA is unavailable.
    * - ``animated``
      - ``False``
      - Enable the per-frame transient solve described in `Animated geometry`_,
        instead of the static single-shot solve above.  Requires
        ``domain = POINTS``; with ``domain = MESH`` it logs a warning and falls
        back to the static solve unchanged.
    * - ``substeps-per-frame``
      - ``4``
      - Solver substeps computed within each rendered Blender frame in animated
        mode.  More substeps make the per-frame integration more stable and
        accurate, at extra solve cost; the internal timestep is
        ``(1 / fps) / substeps-per-frame``.
    * - ``frame-start`` / ``frame-end``
      - scene frame range
      - First/last frame of the animated solve.  Defaults to the blend file's
        own ``frame_start``/``frame_end``.  Narrowing this range solves (and
        caches) only the portion of the timeline you actually render.
    * - ``every-n-frames``
      - ``1``
      - Solve every Nth frame instead of every frame, to cut solve cost on long
        sequences.  Frames that are skipped hold the most recently solved
        field rather than triggering a fresh solve.

Render domain and the temperature atlas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The solver produces a temperature per solve node; the render has to turn that into a
temperature per *pixel*.  ``render-domain`` chooses how.

``VERTEX`` (the default) writes the solved field back as a per-vertex
``sim_temperature`` attribute and lets Blender interpolate it across faces.  That is
exact when the mesh is dense, and coarse when it is not: a large floor slab modelled as a
handful of vertices can only ever show a bilinear ramp, however detailed the underlying
solve.  Architectural assets are full of such surfaces — big, flat, and cheap in geometry.

``TEXEL`` instead packs every participating object into a shared **temperature atlas** —
one tile per object in UV space — and the shader samples that image.  Render resolution is
then set by texel density rather than vertex density, so the same 4-vertex slab can carry
thousands of independent temperature samples.  This is what makes large flat surfaces
(floors, walls, ceilings, table tops) render with real spatial structure.

.. list-table::
    :header-rows: 1
    :widths: 28 14 58

    * - Parameter
      - Default
      - Effect
    * - ``render-domain``
      - ``VERTEX``
      - ``VERTEX`` interpolates the per-vertex field across faces.  ``TEXEL`` builds the
        atlas described above.  Use ``TEXEL`` for scenes with large, coarsely-tessellated
        surfaces; it costs an atlas build plus one EXR per solve.
    * - ``atlas-texel-density``
      - ``1500.0``
      - Target texels per m² of surface area.  Tile side is
        ``ceil(sqrt(area_m2 * density))``, so this sets the effective spatial resolution
        of the thermal field.  Total texels are ``Σ area × density``, so atlas size
        grows **linearly** with this value (each object's tile *side* grows as its square
        root).  Raising it sharpens detail at proportional memory cost.
    * - ``atlas-tile-min`` / ``atlas-tile-max``
      - ``16`` / ``512``
      - Clamp on a single object's tile side in texels.  The minimum keeps tiny objects
        from collapsing to a single texel; the maximum stops one large object from
        dominating the atlas.
    * - ``atlas-texel-soft-max``
      - ``500000``
      - Warn-only budget covering atlas texels **plus the vertices of every object that
        stayed on the per-vertex path** (``Σ side² + retained_vertex_count``).  If the
        request exceeds it, density is rescaled uniformly downward once and a warning is
        emitted; a single corrective pass, so it is not an exact bound.

**Which objects join the atlas.**  Membership is automatic, not a per-object switch.  An
object joins when its own vertex density is below ``atlas-texel-density`` (its vertices are
too sparse to sample it well), and it joins **unconditionally** when a topology-changing
modifier — Subdivision, Solidify, Bevel, Geometry Nodes — makes its base and evaluated
vertex counts differ.  In that case the per-vertex path cannot write a result back onto the
base mesh at all, so the atlas is the only representation that works.  Objects whose native
density already exceeds the target keep the vertex path, since atlasing them would throw
resolution away.

.. note::

   Objects that are demoted from the atlas *and* cannot take a per-vertex write-back fall
   back to a **constant fill** at the mean of their solved field — the whole object renders
   as one flat temperature.  If a surface looks uniformly warm where you expect a gradient,
   that is the cause, and the solve emits a warning naming the object.

Recommended starting point
~~~~~~~~~~~~~~~~~~~~~~~~~~

For the interior scenes this modality targets:

.. code-block:: shell

    visionsim blender.render-animation scene.blend outdir/ \
        --config.include-thermal \
        --config.thermal.assignments assets/thermal/scene.thermal.json \
        --config.thermal.domain POINTS \
        --config.thermal.render-domain TEXEL \
        --config.thermal.irradiance-source CYCLES_BAKE \
        --config.thermal.bake-samples 1024 \
        --config.thermal.sim-time-s 500 --config.thermal.timestep-s 1.0

``domain POINTS`` because scene geometry is rarely clean enough for a cotangent operator;
``render-domain TEXEL`` because such scenes are full of large, coarsely-tessellated
surfaces; and ``irradiance-source CYCLES_BAKE`` because they are typically lit by emissive
geometry rather than lamp objects, which the analytic kernel does not see at all.

The shipped defaults (``VERTEX``, ``DIRECT_KERNEL``) are deliberately the *conservative*
choices, so that turning the modality on changes as little as possible about an arbitrary
blend.  They are the safe starting point, not the best one for any particular scene — treat
the block above as the setting to reach for once you know your scene has these properties,
and the tables above as the reasoning for departing from it.


Radiance render and file formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1
    :widths: 24 14 62

    * - Parameter
      - Default
      - Effect
    * - ``radiance-scale``
      - ``1.0``
      - Brightness multiplier for the ``thermal_radiance`` image only.  A display
        scale — it does not change the temperature solve.
    * - ``exr-codec``
      - ``DWAA``
      - EXR compression codec for the ``temperature`` and ``thermal_radiance``
        files (e.g. ``ZIP``, ``PIZ``, ``DWAA``, ``NONE``).
    * - ``bit-depth``
      - ``32``
      - EXR channel bit depth, ``16`` or ``32``.  Use ``16`` for smaller files
        where full float precision is not required.

Per-object material overrides
-----------------------------

The `Material defaults`_ above apply to the whole scene.  To give a specific object a different
material, add a ``heat_sim_material`` property group to it in the blend file (registered at
``bpy.types.Object.heat_sim_material``).  Any field that is set overrides the corresponding
global default **for that object only**; objects without the property group fall back to the
globals.

The per-object fields are:

``initial_temperature_K``
    Starting temperature (K) for the object's vertices.

``thermal_diffusivity_mm2_s``
    Thermal diffusivity in mm²/s.

``density_kg_m3``
    Material density (kg/m³).

``specific_heat_J_kgK``
    Specific heat capacity (J/kg·K).

``emissivity``
    Surface emissivity in ``[0, 1]``, used for the radiation boundary condition in the
    solve.  It is written to the mesh as a per-vertex attribute but is **not** read by
    the gray-body radiance shader (see the note below).

``thermal_role``
    Either ``"FEM_PARTICIPANT"`` (default — full transient solve) or
    ``"DIRICHLET_SOURCE"`` (vertex temperatures pinned to a constant value every
    step, i.e. a fixed-temperature heat source or sink).

``dirichlet_temperature_K``
    Constant temperature applied when ``thermal_role = "DIRICHLET_SOURCE"``.
    Falls back to ``initial_temperature_K`` when set to 0.

.. admonition:: Animated scenes

    These two fields are the mechanism behind `Animated geometry`_.  Give the
    topology-changing object — a pouring liquid, or any mesh whose vertex count
    changes frame to frame — ``thermal_role = "DIRICHLET_SOURCE"`` and a
    ``dirichlet_temperature_K``; leave stable-topology objects like the cup at
    the default ``"FEM_PARTICIPANT"`` so their temperature evolves across the
    animation instead of staying fixed.

Automatic material assignment (sidecars)
----------------------------------------

`Per-object material overrides`_ are authored inside the blend file, one object at a time.  That
does not scale to a published dataset of scenes whose objects have no thermal information and whose
materials are named by artists (often not in English).  For that case VisionSim reads a committed
**assignment sidecar**: a JSON file that maps each Blender *material name* to a thermal **preset**,
and per material a role and (for heat sources) a reservoir temperature.  Pass it with:

.. code-block:: bash

    vsim blender.render-animation scene.blend out/ \
        --config.include-thermal \
        --config.thermal.assignments assets/thermal/scene.thermal.json

When ``--config.thermal.assignments`` is omitted, behaviour is exactly as documented above
(scene-wide `Material defaults`_ plus any in-blend `Per-object material overrides`_).  When it is
supplied, thermal properties are resolved **per material slot**: a stool's metal legs and wooden
seat get different :math:`\alpha, \rho, c, \varepsilon`, and an emissive lamp element becomes a
pinned heat source while its glass shade solves normally.  A material the sidecar does not name, or
names with an unknown preset, falls back to the scene defaults.

Slot-to-vertex resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Blender stores materials on faces; the solver wants one value per vertex.  Almost every vertex is
interior to a single material, so resolution is a lookup.  At a **seam** — a vertex touching faces
of two materials — continuous properties (:math:`\alpha, \rho, c, \varepsilon`, initial
temperature) are the **area-weighted mean** of the incident faces, while categorical facts (is the
vertex a pinned heat source, and at what temperature) take the **dominant** incident material by
area.  A vertex cannot be "partly pinned", so the pin is a majority vote, not a blend.

.. note::

    Per-slot resolution needs the object's base-mesh vertex count to match the evaluated geometry.
    An object carrying a **topology-changing modifier** (Subdivision Surface, Array, …) changes that
    count, so it falls back to object-level (scene-default or ``heat_sim_material``) properties for
    the whole object, with a warning naming it.  The static per-slot dataset path does not use such
    modifiers; this only affects hand-authored scenes that do.

Authoring a sidecar
~~~~~~~~~~~~~~~~~~~~

A sidecar is a small JSON document.  Author it however suits the scene — by hand for a
handful of materials, or with a script or a language model for a scene with a hundred
artist-named ones.  Only the schema below matters; the renderer just reads the committed
JSON.

.. code-block:: json

    {
      "schema_version": 1,
      "scene": "kitchen1.blend",
      "defaults": { "preset": "plaster" },
      "materials": {
        "PARED blanca":  { "preset": "plaster" },
        "encimera":      { "preset": "marble" },
        "grifo":         { "preset": "stainless_steel" },
        "cortina":       { "preset": "fabric" },
        "madera puerta": { "preset": "wood", "confidence": 0.6,
                           "reason": "door slab; grain texture, no metal shader" }
      }
    }

**Top level**

``schema_version`` (required)
    Must be ``1``.  A missing or different value is an error, not a warning — the loader
    refuses a sidecar it cannot interpret rather than guessing.

``scene``
    The blend file this was authored against.  Informational.

``defaults``
    Optional scene-wide fallback, e.g. ``{"preset": "plaster"}``.  Used for any material the
    ``materials`` block does not name.  Omit it (or use ``{}``) to fall back to the
    scene-wide `Material defaults`_ instead.

``materials``
    Maps a Blender **material name** (exactly as it appears in the blend) to one entry.

**Per-material entry**

``preset`` (the only field that usually matters)
    One of the keys in the preset library — see `Material presets`_, or call
    ``materials.preset_keys()`` for the authoritative list.  Use ``null`` to say
    "deliberately unassigned", which falls back to ``defaults`` and then to the global
    knobs.  An unknown key is rejected rather than silently guessed.

``role``
    ``"FEM_PARTICIPANT"`` (default) — the material solves normally.
    ``"DIRICHLET_SOURCE"`` — the material is *pinned* at a fixed temperature and acts as a
    heat source rather than solving.  Use it for a lamp element,
    a hotplate, a radiator: something whose temperature you are asserting rather than
    simulating.

``dirichlet_K``
    The fixed temperature for a ``DIRICHLET_SOURCE``, in Kelvin.  Ignored for a
    ``FEM_PARTICIPANT``.  Accepted range is 280–500 K; a value outside it is dropped **with
    a warning**, and the role degrades to ``FEM_PARTICIPANT``.  That degrade is deliberate:
    keeping the role while dropping the temperature would pin the object at *ambient*,
    silently turning an intended heat source into a heat **sink**, which is a worse and much
    harder-to-spot failure than simply letting it solve normally.

``confidence`` / ``reason``
    Free-form provenance, ignored by the loader.  Useful when a sidecar is reviewed later:
    they record how sure the author was and why.

.. note::

    The sidecars shipped in ``assets/thermal/`` use no ``DIRICHLET_SOURCE`` entries — every
    material is a ``FEM_PARTICIPANT``, so no object is held at a fixed temperature and every
    surface solves from the same ambient start.  Pin a source only when you actually want to
    assert a temperature.

Optional helper
^^^^^^^^^^^^^^^

``scripts/thermal_assign.py`` (outside the installed package) is handy for larger scenes.  It
can inventory a scene's materials — names, slot areas, node graphs, emission — as JSON, and
render an HTML review sheet for an existing sidecar sorted by surface area:

.. code-block:: bash

    # inventory the scene's materials (runs inside Blender)
    blender -b scene.blend --python scripts/thermal_assign.py -- dump --output scene.materials.json

    # review an existing sidecar against that inventory
    python scripts/thermal_assign.py report scene.materials.json \
        assets/thermal/scene.thermal.json --output scene.review.html

Check the high-coverage materials first — they dominate the result.  Watch for render helpers
such as a *daylight portal*: they carry an emission node and read as a heat source, which is
rarely what you want.

Solve caching
-------------

The FEM solve is **lazy and cached**.  On the first render of a given blend file with a given set
of solver parameters, VisionSim runs the full solve and writes the per-object temperature
histories to::

    <blend_file>.heatsim/<cache_key>/temperatures.npz

where ``<cache_key>`` is a 16-character digest derived from the blend file path, its modification
timestamp, the solver configuration, the sorted list of participating object names, and the
SHA-256 of the material sidecar (if one is given).  Subsequent renders that use the same blend
file and the same parameters **skip the solve entirely** and load the cached result.  Changing any
solver parameter, editing the sidecar, or touching the blend produces a new cache key and triggers
a fresh solve.

.. note::

   The key includes the blend's modification timestamp, so *any* write to the ``.blend`` — even
   one that changes nothing relevant to the solve — invalidates every cached solve for it.

To prime the cache ahead of a render run — useful when the solve is expensive and you want
rendering to start immediately — run the solve on its own:

.. code-block:: bash

    vsim blender.heatsim-solve scene.blend \
        --config.thermal.device cpu

Because the cache key is anchored to the source blend file, a primed solve is reused by all later
``vsim blender.render-animation`` calls against the same blend at the same solver settings — even
in a different output directory.

.. note::

   ``heatsim-solve`` always runs the **static** solve; it accepts ``--config.thermal.animated``
   but ignores it. The animated path writes a different cache under a different key, so an
   animated render is not served by a primed static solve. The subcommand also accepts the rest
   of the render config (``--config.include-*``, ``--config.frames.*`` …) and ignores all of it —
   only ``--config.thermal.*`` has any effect.

Using thermal from the API
--------------------------

The same parameters are available when driving VisionSim programmatically.  ``ThermalConfig`` holds
every setting shown above and is attached to ``RenderConfig.thermal`` (gated by
``RenderConfig.include_thermal``); the render job forwards it to the two service calls that set up
thermal on the Blender side:

* :meth:`~visionsim.simulate.blender.BlenderService.exposed_prepare_thermal` — runs (or loads from
  cache) the FEM solve and prepares the scene for thermal rendering.
* :meth:`~visionsim.simulate.blender.BlenderService.exposed_include_thermal` — wires the
  temperature AOV, the inferno preview, and the gray-body radiance pass into the compositor.
* :meth:`~visionsim.simulate.blender.BlenderService.exposed_heatsim_solve` — the standalone
  solve-and-cache entry point behind ``vsim blender.heatsim-solve``.

See :mod:`visionsim.simulate.config` for the full ``ThermalConfig`` field reference.
