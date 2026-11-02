Varying Focal Lengths
=====================

The focal length decides how much of the scene a camera sees and how strongly the depth of a scene collapses. This page shows how to keyframe it, because the render client cannot set it for you yet.

|

Keyframing the Lens
-------------------

Camera poses can be keyframed through the API, using :meth:`position_camera <visionsim.simulate.blender.BlenderService.exposed_position_camera>` to set a pose and :meth:`set_camera_keyframe <visionsim.simulate.blender.BlenderService.exposed_set_camera_keyframe>` to key it; see :doc:`../sections/blender/keyframing` for the trajectory workflow. Only the pose can be set this way, so the focal length has to be keyframed in Blender itself.

Keyframe the camera's ``lens`` (or ``angle``) there and Blender interpolates it per frame, exactly as it interpolates the trajectory, so both are evaluated together when a frame is rendered. Each rendered frame then records the focal length in effect at that frame, and the resulting metadata ends up with one camera instance per focal length, since every frame carries its own ``fl_x`` and ``fl_y``.

The `contrazoom scene <https://drive.google.com/drive/folders/1gRxhL3rbGDTfgKytre8WkbBu-QDJFy15?usp=sharing>`_ does this. It is a dolly zoom: the camera moves along its view axis while the lens counter-zooms, so the subject stays roughly the same size in the image while the background compresses. The scene animates the lens from 200 mm to 35 mm over frames 0 through 49 at 30 fps, moving the camera closer to the subject over the same range so that the ratio of focal length to subject distance stays roughly fixed.

Authoring that is a Blender step, and how to keyframe a lens there is out of scope for this page: it is an ordinary keyframe, and the only part that matters here is that it is saved with the blend. Rendering it needs nothing special either, since animations play by default and both the location and the lens interpolate, so every frame records the focal length in effect at that frame:

.. code-block:: bash

    visionsim blender.render-animation contrazoom.blend contrazoom/ --frame-start=0 --frame-end=49

.. video:: ../_static/varying-focal-lengths/contrazoom-preview.mp4
   :loop:
   :autoplay:
   :nocontrols:
   :playsinline:
   :align: center
   :width: 80%

The preview plays at half the rate it was rendered at, which makes the zoom easier to follow.

|

Focal Length in the Metadata
----------------------------

:func:`dataset.convert <visionsim.cli.dataset.convert>` turns the rendered ``transforms.db`` into a standard ``transforms.json``. An animated lens gives per-frame intrinsics: the global fields (``w``, ``h``, ``cx``, ``cy``, ``fps``) stay at the top level and every frame carries its own ``fl_x`` and ``fl_y``:

.. code-block:: json

    {
      "w": 800,
      "h": 800,
      "cx": 400.0,
      "cy": 400.0,
      "fps": 30.0,
      "frames": [
        {
          "file_path": "frames/0000/001.png",
          "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, -3.1], [0, 0, 1, 0], [0, 0, 0, 1]],
          "fl_x": 4444.4,
          "fl_y": 4444.4
        },
        {
          "file_path": "frames/0000/050.png",
          "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, -0.6], [0, 0, 1, 0], [0, 0, 0, 1]],
          "fl_x": 777.8,
          "fl_y": 777.8
        }
      ]
    }

A static lens would keep a single global focal length at the top level instead. The schema is described under :doc:`../sections/datasets`.

|

.. seealso::

    - :doc:`../sections/blender/keyframing` for the general trajectory workflow.
    - :doc:`../sections/blender/cameras` for the camera intrinsics the metadata records.
    - :doc:`../sections/datasets` for the ``transforms.json`` schema and coordinate conventions.
