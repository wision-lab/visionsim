Stereo and Multi-Rig Setups
===========================

This tutorial shows how to render stereoscopic (left/right eye) or multi-camera rig
sequences using the ``camera_offset`` parameter. Stereo rendering is useful for
downstream sensor emulation that requires depth perception (e.g., binocular vision
models, disparity estimation).

|

Understanding Camera Offsets
----------------------------

The ``camera_offset`` moves the camera in its **local coordinate frame**.
For more details on the camera coordinate system see the
:ref:`coordinate conventions <sections/datasets:Coordinate Conventions>` section.
In this frame:

- **X** : left (negative) / right (positive)
- **Y** : down (negative) / up (positive)
- **Z** : backward (negative) / forward (positive)

For a typical stereo pair, a human interpupillary distance (IPD) of around **6.5 cm**
is standard. Each eye is offset by half the IPD along the local X axis:

.. code-block:: text

    Left eye:  camera_offset = (-0.0325,  0.0, 0.0)
    Right eye: camera_offset = ( 0.0325,  0.0, 0.0)

|

Using the Python API
--------------------

The :meth:`offset_camera <visionsim.simulate.blender.BlenderService.exposed_offset_camera>` method
offsets the camera in its local coordinate frame for the current frame.
After offsetting, call :meth:`set_camera_keyframe <visionsim.simulate.blender.BlenderService.exposed_set_camera_keyframe>`
to bake the new position into the animation:

.. code-block:: python

    # Left eye: offset 3.25 cm to the left
    client.offset_camera((-0.0325, 0.0, 0.0))
    client.set_camera_keyframe(frame)

    # Right eye: offset 3.25 cm to the right
    client.offset_camera((0.0325, 0.0, 0.0))
    client.set_camera_keyframe(frame)

|

Using the CLI
-------------

The ``--camera-offset`` flag applies the offset to every rendered frame automatically.
Render each eye view into a separate output directory:

.. code-block:: bash

    # Render left eye view
    visionsim render-animation scene.blend ./left_eye \\
        --camera-offset -0.0325 0 0

    # Render right eye view
    visionsim render-animation scene.blend ./right_eye \\
        --camera-offset 0.0325 0 0

.. note::

    The offset is applied in **local camera coordinates**, so it works correctly
    regardless of the camera's world-space orientation. The left/right direction
    always follows the camera's own +X axis.

|

Multi-Rig Setups
----------------

For multi-camera rigs with more than two cameras (e.g., a 3-camera array), simply
run ``render-animation`` once per camera position with the appropriate offset:

.. code-block:: bash

    # Three-camera rig at 6.5 cm spacing
    visionsim render-animation scene.blend ./cam_left   --camera-offset -0.065 0 0
    visionsim render-animation scene.blend ./cam_center --camera-offset  0.0   0 0
    visionsim render-animation scene.blend ./cam_right  --camera-offset  0.065 0 0

|

Combining Stereo Views
----------------------

After rendering both eyes, you can use the :class:`Dataset <visionsim.dataset.Dataset>` API
to inspect or merge the output directories:

.. code-block:: python

    from visionsim.dataset import Dataset

    left  = Dataset.from_path("./left_eye/frames")
    right = Dataset.from_path("./right_eye/frames")

    # Both datasets contain the same number of frames with corresponding indices
    for (left_img, _), (right_img, _) in zip(left, right):
        # left_img and right_img have shape (H, W, C)
        ...
