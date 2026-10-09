Stereo and Multi-Rig Setups
===========================

This tutorial shows how to render stereoscopic (left/right eye) or multi-camera rig sequences using a camera offset — ``--config.camera-offset`` on the command line, or :meth:`offset_camera <visionsim.simulate.blender.BlenderService.exposed_offset_camera>` in the Python API. Stereo rendering is useful for downstream sensor emulation that requires depth perception (e.g., binocular vision models, disparity estimation).

|

Understanding Camera Offsets
----------------------------

The ``camera_offset`` moves the camera in its **local coordinate frame**. For more details on the camera coordinate system see the :ref:`coordinate conventions <sections/datasets:Coordinate Conventions>` section. In this frame:

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

The :meth:`offset_camera <visionsim.simulate.blender.BlenderService.exposed_offset_camera>` method **moves the camera** by a vector in its local coordinate frame; it is not a parameter that is re-applied on every frame. To bake a shifted trajectory you must first re-establish the original position for each frame, then apply the offset and re-key:

.. code-block:: python

    # Re-bake the original trajectory first, so offsets do not accumulate.
    for frame in client.common_animation_range():
        client.set_current_frame(frame)
        client.set_camera_keyframe(frame)

    # Then, for each eye, offset from that clean base and re-key.
    for frame in client.common_animation_range():
        client.set_current_frame(frame)
        client.offset_camera((-0.0325, 0.0, 0.0))  # left eye, 3.25 cm
        client.set_camera_keyframe(frame)

.. warning::

    Calling :meth:`offset_camera <visionsim.simulate.blender.BlenderService.exposed_offset_camera>` without first restoring the un-offset position makes the shift **cumulative**: running the loop again compounds the offset, and offsetting an already-offset camera keys from the wrong base. This is why the two loops above are separate, and why using the CLI with ``--config.camera-offset`` is the safer option for the common case — it performs exactly this two-pass rebake internally.

|

Using the CLI
-------------

The ``--config.camera-offset`` flag applies the offset to every rendered frame automatically. Render each eye view into a separate output directory:

.. code-block:: bash

    # Render left eye view
    visionsim blender.render-animation scene.blend ./left_eye \\
        --config.camera-offset -0.0325 0 0

    # Render right eye view
    visionsim blender.render-animation scene.blend ./right_eye \\
        --config.camera-offset 0.0325 0 0

.. note::

    The offset is applied in **local camera coordinates**, so it works correctly
    regardless of the camera's world-space orientation. The left/right direction
    always follows the camera's own +X axis.

|

Result
------

The two eye views can be combined into a red/cyan anaglyph, which you can view with
standard red/cyan glasses. With the 6.5 cm interpupillary distance above, the pair
rendered from a kitchen scene looks like this:

.. video:: ../_static/stereo/stereo-anaglyph.mp4
   :loop:
   :autoplay:
   :nocontrols:
   :width: 75%
   :align: center

The anaglyph is built with ffmpeg's ``stereo3d`` filter. The eyes are stacked side by
side, which the filter reads as ``sbsl`` (side-by-side, left first) and writes out as a
half-width ``arcd`` anaglyph:

.. code-block:: bash

    # Stack the two eyes side by side, then fold them into one anaglyph video
    ffmpeg -framerate 50 -i left_eye/frames/0000/%03d.png \\
           -framerate 50 -i right_eye/frames/0000/%03d.png \\
           -filter_complex \\
    "[0:v][1:v]hstack=inputs=2[stacked];[stacked]stereo3d=in=sbsl:out=arcd[out]" \\
           -map "[out]" -c:v libx264 -crf 23 -pix_fmt yuv420p anaglyph.mp4

The input framerate should match the scene's own, so the video runs at the speed the
animation was authored at. The ``-crf`` trades file size against quality; the encoder
keeps the result an order of magnitude smaller than the same sequence as a gif.

Other ``stereo3d`` output modes work the same way — ``arcg`` (gray), ``arch`` (half
color), or ``arcd`` (Dubois, used above) trade color accuracy against ghosting.

|

Multi-Rig Setups
----------------

For multi-camera rigs with more than two cameras (e.g., a 3-camera array), simply
run ``blender.render-animation`` once per camera position with the appropriate offset:

.. code-block:: bash

    # Three-camera rig with 6.5 cm between neighboring cameras (13 cm baseline)
    visionsim blender.render-animation scene.blend ./cam_left   --config.camera-offset -0.065 0 0
    visionsim blender.render-animation scene.blend ./cam_center --config.camera-offset  0.0   0 0
    visionsim blender.render-animation scene.blend ./cam_right  --config.camera-offset  0.065 0 0

    # For a 6.5 cm total baseline between the outer cameras, halve the offsets:
    visionsim blender.render-animation scene.blend ./cam_left   --config.camera-offset -0.0325 0 0
    visionsim blender.render-animation scene.blend ./cam_center --config.camera-offset  0.0    0 0
    visionsim blender.render-animation scene.blend ./cam_right  --config.camera-offset  0.0325 0 0

|

Combining Stereo Views
----------------------

After rendering both eyes, you can use the :class:`Dataset <visionsim.dataset.Dataset>` API to inspect or merge the output directories:

.. code-block:: python

    from visionsim.dataset import Dataset

    left  = Dataset.from_path("./left_eye/frames")
    right = Dataset.from_path("./right_eye/frames")

    # Check first: a mismatch means one eye was not rendered over the same frame range.
    assert len(left) == len(right), f"frame count mismatch: {len(left)} vs {len(right)}"

    # Both datasets contain the same number of frames with corresponding indices
    for (left_img, _), (right_img, _) in zip(left, right):
        # left_img and right_img have shape (H, W, C)
        ...
