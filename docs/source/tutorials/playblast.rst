Fast Previews with Playblast
============================

Sometimes you don't need a full render. While iterating on camera moves or a layout, Cycles ground truth is overkill: you want a rough preview you can flip through in seconds. ``blender.render-playblast`` renders the animation with Blender's viewport/OpenGL renderer instead, which is far faster but only produces color frames.

|

Playblast vs. Full Render
-------------------------

The two modes side by side, both rendered from ``loft.blend``:

.. list-table::
    :class: borderless

    * - .. figure:: ../_static/playblast-preview.gif

            Playblast (viewport render)

      - .. figure:: ../_static/playblast-full-preview.gif

            Full render (Cycles)

The playblast gives up shading fidelity: the walls and paintings are untextured and there are no ray-traced shadows or reflections, which is the tradeoff you accept for iteration speed.

Render time comparison: the playblast took 8.2 seconds, the full render took 15 minutes 31 seconds. That is roughly a 113x speedup, and the gap grows on heavier scenes.

|

Running a Playblast
-------------------

The CLI mirrors :doc:`render-animation <../quick-start>`, with the same positional arguments and frame range options:

.. code-block:: bash

    visionsim blender.render-playblast scene.blend output/

By default the preview is encoded to ``output/playblast/playblast.mp4``. Pass ``--no-video`` to instead get a PNG sequence (``output/playblast/0001.png``, ...) along with a ``transforms.db`` metadata database.

You can also drive it from Python through :meth:`render_playblast <visionsim.simulate.blender.BlenderService.exposed_render_playblast>`:

.. code-block:: python

    from visionsim.simulate.blender import BlenderClient

    with BlenderClient.spawn(background=False) as client:
        client.initialize("scene.blend", "output/")
        client.render_playblast(video=False)

.. warning::
    Playblasts need a GL context, so the Blender instance must run with ``background=False``. This requires a display; see the limitations below.

|

Limitations
-----------

A playblast is a preview, not a dataset. Keep these constraints in mind:

Playblast rendering needs a GL context. Viewport rendering runs in a live window, so Blender cannot run in background mode. ``render-playblast`` forces ``background=False``, which means a display is required. On a headless machine, start a virtual X server first::

    DISPLAY="" WAYLAND_DISPLAY="" xvfb-run -a --server-args="-screen 0 1920x1080x24" \
        visionsim blender.render-playblast scene.blend output/

Blender 4.2 or newer is required. The viewport renderer used here is unavailable on older versions, and the command raises a ``RuntimeError`` if yours is too old.

No ground truth annotations are produced. Depths, normals, flows, segmentations, materials, diffuse/specular passes and point maps are all silently dropped. The ``include_*`` fields of :class:`RenderConfig <visionsim.simulate.config.RenderConfig>` have no effect here; only the color preview is written.

Viewport settings come from the blend-file. Engine, shading, anti-aliasing and sampling are whatever the viewport is set to. ``RenderConfig`` options that only apply to Cycles, such as ``device_type``, ``max_samples`` or ``use_denoising``, are ignored.

Playblasts always run in a single job. ``config.jobs`` and ``config.autoscale`` are ignored, because the output is a single shared file that isn't safe to write from multiple processes. Parallelizing across scenes with :meth:`BlenderClients.pool <visionsim.simulate.blender.BlenderClients.pool>` still works.

The scene is mutated in place. Like the other render methods, the frame range, output filepath and image format are overwritten on the loaded scene and not restored. Reopen the blend-file if you need the originals back.
