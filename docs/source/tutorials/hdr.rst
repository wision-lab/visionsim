Rendering HDR Sequences
=======================

This tutorial shows how to render high-dynamic-range sequences suitable for downstream sensor emulation. HDR images store linear intensity values, unlike display-referred formats like PNG/JPEG.

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

The same configuration is available through the :meth:`render-animation command <visionsim.cli.blender.render_animation>`:

.. code-block:: bash

    visionsim render-animation scene.blend ./output \\
        --frames.file-format OPEN_EXR \\
        --frames.bit-depth 32

    visionsim render-animation scene.blend ./output \\
        --frames.file-format HDR \\
        --frames.color-mode RGB

.. tip::

    Some sensor emulation might require HDR inputs. When running the downstream :doc:`sensor emulation pipeline <../sections/emulation>`, ensure the rendered frames are in ``.hdr`` or ``.exr`` format.