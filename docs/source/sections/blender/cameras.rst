Camera Types and Intrinsics
===========================

Every cell rendered with :func:`blender.render-animation <visionsim.cli.blender.render_animation>` records the camera intrinsics and extrinsics alongside each frame, as listed under :doc:`groundtruths`. This page covers what the metadata can describe and what a render can change.

|

Supported Camera Models
-----------------------

The :class:`Camera <visionsim.dataset.models.Camera>` model that backs the ``transforms.json`` schema currently supports OPENCV-style pinhole cameras only. It stores the pinhole intrinsics:

``fl_x``, ``fl_y``
    Focal lengths in pixels.

``cx``, ``cy``
    Principal point in pixels.

``w``, ``h``
    Image size in pixels.

``c``, ``fps``
    Output channels (optional, e.g. 4 for RGBA) and framerate (optional).

Blender itself only renders perspective cameras: :meth:`camera_info <visionsim.simulate.blender.BlenderService.exposed_camera_info>` raises for any camera data type other than ``PERSP`` (``visionsim/simulate/blender.py:1712``). A plain render therefore never sets or measures the camera model or any distortion coefficient.

.. warning::

    Other lens types are not yet fully supported. The schema carries a ``camera_model`` field with ``OPENCV`` and ``OPENCV_FISHEYE`` values, plus Brown-Conrady coefficients ``k1`` through ``k3`` and ``p1``/``p2`` (``k1`` through ``k4`` for fisheye), but the renderer neither sets nor measures them: :meth:`camera_info <visionsim.simulate.blender.BlenderService.exposed_camera_info>` raises for any type other than ``PERSP``. They describe a physical camera for downstream consumers rather than something a render produces.

The format is a Nerfstudio superset. Intrinsics are either global or defined for every frame, but never both, and ``camera_model`` is global only (it is the one field that may not be per-frame). The schema and coordinate conventions are described under :doc:`../datasets`.

|

.. seealso::

    - :doc:`../../tutorials/varying-focal-lengths` for animating the lens and what it does to the recorded intrinsics.
    - :doc:`keyframing` for the camera trajectory workflow.
    - :doc:`groundtruths` for the full list of recorded annotations.
    - :doc:`../datasets` for the ``transforms.json`` schema and coordinate conventions.
