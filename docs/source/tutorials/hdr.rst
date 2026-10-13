Rendering HDR Sequences
=======================

This tutorial shows how to render high-dynamic-range sequences suitable for downstream sensor emulation. HDR images store linear intensity values, unlike display-referred formats like PNG/JPEG.

Here, "HDR" refers specifically to the rendered **image** outputs — frames and composites — which are HDR when saved as linear ``.exr`` or ``.hdr`` instead of 8-bit PNG/JPEG. Auxiliary ground truths that are also stored as ``.exr`` (depth, normals, flow, ...) carry non-color data by definition and are not what this page is about.

|

HDR File Formats
----------------

HDR outputs can be saved as either ``.exr`` (OpenEXR) or ``.hdr`` (Radiance HDR):

.. list-table::
    :header-rows: 1

    * - Format
      - Extension
      - Color Depth
      - Color Mode
      - Compression
    * - OpenEXR
      - ``.exr``
      - 16 or 32 bit float
      - ``RGB`` or ``RGBA``
      - ``NONE``, ``PXR24``, ``ZIP``, ``PIZ``, ``RLE``, ``ZIPS`` (lossless); ``DWAA``, ``DWAB`` (lossy)
    * - Radiance HDR
      - ``.hdr``
      - 32 bit float (always)
      - ``RGB`` only
      - lossy RLE

The default ``DWAA`` codec for OpenEXR provides a good balance of compression and quality. For lossless archival use ``ZIP`` or ``PIZ``.

|

Using the Python API
--------------------

The :meth:`include_frames <visionsim.simulate.blender.BlenderService.exposed_include_frames>` method defaults to ``PNG`` with 8-bit color depth. Configure it for HDR output as follows:

.. code-block:: python

    # OpenEXR with lossy default codec (DWAA) at 32-bit float
    client.include_frames(file_format="OPEN_EXR", color_mode="RGB", bit_depth=32)

    # OpenEXR with lossless compression
    client.include_frames(file_format="OPEN_EXR", color_mode="RGB", bit_depth=32, exr_codec="ZIP")

    # Radiance HDR (always 32-bit float, RGB only)
    client.include_frames(file_format="HDR", color_mode="RGB")

Composite outputs can also be configured for HDR:

.. code-block:: python

    client.include_composites(file_format="OPEN_EXR", color_mode="RGB", bit_depth=16)
    client.include_composites(file_format="HDR", color_mode="RGB")

|

Using the CLI
-------------

The same configuration is available through the :meth:`blender.render-animation command <visionsim.cli.blender.render_animation>`. Render options are grouped under a ``config`` subcommand, so frame settings are passed as ``--config.frames.*``:

.. code-block:: bash

    # OpenEXR at 32-bit float, lossless ZIP compression
    visionsim blender.render-animation scene.blend ./output \\
        --config.frames.file-format OPEN_EXR \\
        --config.frames.bit-depth 32 \\
        --config.frames.exr-codec ZIP

    # Radiance HDR (always 32-bit float, RGB only)
    visionsim blender.render-animation scene.blend ./output \\
        --config.frames.file-format HDR \\
        --config.frames.color-mode RGB

Composites work the same way (provided the compositor setup does not do any tonemapping already) but must first be enabled:

.. code-block:: bash

    visionsim blender.render-animation scene.blend ./output \\
        --config.include-composites \\
        --config.composites.file-format OPEN_EXR \\
        --config.composites.bit-depth 32

.. note::

    ``bit-depth`` and ``exr-codec`` only apply to ``OPEN_EXR``. Radiance HDR is always 32-bit float and ``RGB``, so pairing it with ``bit-depth`` has no effect.

.. tip::

    Some sensor emulation might require HDR inputs. When running the downstream :doc:`sensor emulation pipeline <../sections/emulation>`, ensure the rendered frames are in ``.hdr`` or ``.exr`` format.