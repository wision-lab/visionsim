Quick Start
===========

Installation & Dependencies 
---------------------------

First, you'll need:

* `Blender <https://www.blender.org/download/>`_ >= 3.6, to render new views. 
* `FFmpeg <https://ffmpeg.org/download.html>`_, for visualizations. 


Make sure Blender and ffmpeg are on your PATH.

Then you can **install the latest stable release** via `pip <https://pip.pypa.io>`_::
    
    pip install visionsim
    visionsim post-install

|

Generating a dataset
--------------------

We will create a small scale dataset of a furnished living room as if it was captured by a realistic 25fps conventional RGB camera, a 1.6kHz single photon camera, an event camera, an indirect time-of-flight camera and an inertial measurement unit, for 4 seconds. To achieve this, we will:

- Render a small dataset of 201 ground truth frames, plus depth maps, using the :func:`blender.render-animation <visionsim.cli.blender.render_animation>` CLI,
- Interpolate this dataset 32-fold using the :func:`interpolate.dataset <visionsim.cli.interpolate.dataset>` CLI,
- And emulate different cameras and an IMU using the :func:`emulate.rgb <visionsim.cli.emulate.rgb>` / :func:`emulate.spad <visionsim.cli.emulate.spad>` / :func:`emulate.events <visionsim.cli.emulate.events>` / :func:`emulate.itof <visionsim.cli.emulate.itof>` / :func:`emulate.imu <visionsim.cli.emulate.imu>` CLIs.

|

Rendering Ground Truth
----------------------

First, download the test scene, here we will be using the ultramodern living room, which you can download from `here <https://drive.google.com/drive/folders/1gRxhL3rbGDTfgKytre8WkbBu-QDJFy15?usp=sharing>`_. This blender file has been modified to include a camera animation. Specifically, the camera travels through a furnished living room at a constant height of 1.61 m, panning across the sofa, the tables and the curtains, and the animation lasts 600 frames at 50 fps. Here, we assume your blender file is pre-animated, but you can control camera movements and keyframes manually too. 


To create the dataset, we render the 201 frames between 150 and 350, leaving the animation unstretched: 

.. literalinclude:: ../../examples/quickstart.sh
   :language: bash
   :lines: 3-5

.. note::
    The :func:`blender.render-animation <visionsim.cli.blender.render_animation>` CLI has a lot of options which enable changing render settings and resolution, parallelization, and for generating different types of ground truth annotations such as depth and segmentation maps. You can see all options by running the following::

        visionsim blender.render-animation --help

    ``--config.keyframe-multiplier`` stretches every keyframe in the file, so ``2.0`` plays the animation back at half speed and ``0.5`` at double speed. The command above leaves it at its default of ``1.0``, which keeps the camera moving at the speed it was keyed at.

    If the above command does not work, you might have to change some settings, notably the ``device-type``. For instance on older GPUs that do not support Optix you can do ``--config.device-type=cuda`` to use CUDA.
    
    Finer grain control can be had using the :class:`BlenderClient API <visionsim.simulate.blender.BlenderClient>`.

.. warning:: This might take a while: 201 Cycles renders at 512x512. 

All the rendered frames will be in ``quickstart/ultramodern-gt/frames``. Each data directory holds its own ``transforms.db`` metadata file describing the camera trajectory and intrinsics, and ``--config.include-depths`` adds a sibling ``quickstart/ultramodern-gt/depths`` directory holding an aligned depth map for every frame. See :doc:`sections/datasets` for what these metadata files carry. The :func:`dataset.convert <visionsim.cli.dataset.convert>` CLI turns any of these ``.db`` files into a Nerfstudio-style ``transforms.json`` if you prefer to work with JSON, and :func:`dataset.merge <visionsim.cli.dataset.merge>` combines them when the frames, depths and other annotations share a camera.

If you need a single frame instead of a sequence, :func:`blender.render-frame <visionsim.cli.blender.render_frame>` takes the same positional arguments and render options as ``render-animation``, plus a ``--frame`` index:

.. code-block:: bash

    visionsim blender.render-frame scene.blend output/ --frame 42 --config.include-depths

The output layout matches ``render-animation``, so the frame lands in ``output/frames/`` with a ``transforms.db`` holding that frame's camera pose. Animations stay enabled, so the result is identical to what ``render-animation`` produces for the same index.

Let's create a quick preview of this dataset by animating every 2nd frame into a video, so it plays back in realtime:

.. literalinclude:: ../../examples/quickstart.sh
   :language: bash
   :lines: 6-11

``preview.mp4`` animates the camera's pass through the living room (the clips below are that same pass encoded for the browser, your own render will look better). ``preview-depths.mp4`` previews the same depth annotation that the iToF emulation later reads, colorized per frame:

.. grid:: 2
    :gutter: 2

    .. grid-item::

        .. video:: _static/quick-start/ultramodern-gt-preview.mp4
            :loop:
            :autoplay:
            :nocontrols:
            :playsinline:
            :align: center
            :caption: Rendered RGB

    .. grid-item::

        .. video:: _static/quick-start/ultramodern-depth-preview.mp4
            :loop:
            :autoplay:
            :nocontrols:
            :playsinline:
            :align: center
            :caption: Ground truth depth

.. note::
    The depth preview flickers, and this is an artifact of how the renderer produces it rather than something in the data. ``--config.depths.preview`` normalizes each frame independently so the colormap stretches to fill the near/far range of that frame alone, which makes a depth map that changes little between frames appear to change a lot. The underlying ``.exr`` maps in ``quickstart/ultramodern-gt/depths`` hold absolute distances and are unaffected.

    To normalize across the whole sequence instead, colorize after rendering with :func:`transforms.colorize-depths <visionsim.cli.transforms.colorize_depths>`, which estimates a single depth range from every frame and reuses it:

    .. code-block:: bash

        visionsim transforms.colorize-depths \
            --input-dir=quickstart/ultramodern-gt/depths \
            --output-dir=quickstart/ultramodern-gt/depths-colorized

|

Interpolating Frames
--------------------

You can optionally interpolate an existing dataset to get intermediate frames that have not been rendered. This quickly increases the effective framerate of the data, at the cost of artifacts if adjacent frames are too "far" apart. The following will interpolate a dataset by a factor of 32x:

.. literalinclude:: ../../examples/quickstart.sh
   :language: bash
   :lines: 12-14

You can preview this new dataset like above, just use a step of 64 (=2x32) to ensure playback is at the same speed. 

.. note::
    The new dataset contains 6,401 frames, not 32x201=6,432. Consider interpolating two frames by a factor of 2x: you create a frame between them, giving 3 frames total, the first original, the interpolated one, and the second original. In general, M original frames interpolated N-times yields NM-N+1 frames.  

.. warning::
    Interpolation introduces artifacts when adjacent frames in the original dataset are too different from one another. This and its implications are discussed further in the :doc:`sections/interpolation` section. Interpolation helps bridge the gap from 1,000fps to 10,000fps, not from 10fps to 100fps. 

.. warning::
    Interpolation is faster than rendering new frames but still slow: the 32x pass above takes minutes on an RTX 3080.

|

Emulating Sensor Data
---------------------

To simulate real cameras, we must convert these perfect ground truth frames and their annotations into realistic measurements. Each of the five sensors below reads a different property of the same rendered sequence.

Conventional Camera
^^^^^^^^^^^^^^^^^^^

Motion blur is emulated by :func:`emulate.rgb <visionsim.cli.emulate.rgb>`. The ``--chunk-size`` flag averages the interpolated frames in groups of 64, and ``--readout-std`` sets the standard deviation of the Gaussian read noise, left at 0 here to keep the example clean. Since the interpolated dataset runs at 1,600fps, 64 frames per output frame simulate a 25fps RGB camera:

.. literalinclude:: ../../examples/quickstart.sh
   :language: bash
   :lines: 15-18

Single Photon Camera
^^^^^^^^^^^^^^^^^^^^

We can emulate a single-photon camera using :func:`emulate.spad <visionsim.cli.emulate.spad>`, at the same framerate as the interpolated dataset like so:

.. literalinclude:: ../../examples/quickstart.sh
   :language: bash
   :lines: 19-21

The emulator linearizes the sRGB frames back to linear scene radiance and samples each pixel independently, treating it as a Bernoulli trial whose success probability is :math:`1 - e^{-\varphi}` for incident intensity :math:`\varphi`. 

Event Camera
^^^^^^^^^^^^

An event camera (emulated using :func:`emulate.events <visionsim.cli.emulate.events>`) reports per-pixel brightness changes as asynchronous events rather than frames. It reads the ground truth frames (or interpolated ones), but we must pass the correct frame rate using ``--fps``:

.. literalinclude:: ../../examples/quickstart.sh
   :language: bash
   :lines: 22-24

Each event is a tuple of ``(t, x, y, p)``: the timestamp in microseconds, the pixel location and the polarity, where ``p = 1`` marks a brightness increase and ``p = -1`` a decrease. Events are written one per line to ``events.txt``. The ``--preview-step`` flag additionally accumulates events into visualization frames, with ON events in blue and OFF events in red; previews are off unless it is set, and the command above passes 1 to render one per input frame.

Indirect Time-of-Flight Camera
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Indirect time-of-flight (iToF) sensors recover depth by correlating a modulated light source against the returning signal, so they read the depth maps as well as the albedo frames. The renderer writes depths to a sibling directory with its own metadata rather than merging them into the albedo dataset, so both are passed explicitly:

.. literalinclude:: ../../examples/quickstart.sh
   :language: bash
   :lines: 25-29

The scheme, capture count and frequency define the acquisition. Each frame is written as a ``.npy`` array of ``(n_captures, H, W)`` raw taps, and the :doc:`iToF tutorial <tutorials/itof>` covers the full set of coding schemes and shows how to decode them back into depth. The ramp schemes are unambiguous only over ``c/4f``, so the CLI warns when the scene is deeper than the unambiguous range of the selected scheme.

Inertial Measurement Unit
^^^^^^^^^^^^^^^^^^^^^^^^^

An IMU measures acceleration and angular velocity rather than light, so :func:`emulate.imu <visionsim.cli.emulate.imu>` reads only the camera poses in the rendered dataset, not its frames:

.. literalinclude:: ../../examples/quickstart.sh
   :language: bash
   :lines: 30-32

``--dt`` is the time between consecutive poses, 0.02s for the 50fps render above, and the output is a CSV with one row per pose:

+------------+------------------------------------------------------------------+
| Column     | Meaning                                                          |
+============+==================================================================+
| ``t``      | Timestamp in seconds, counting up from zero.                     |
+------------+------------------------------------------------------------------+
| ``acc_*``  | Accelerometer reading: acceleration plus noise and bias.         |
+------------+------------------------------------------------------------------+
| ``gyro_*`` | Gyroscope reading: angular velocity plus noise and bias.         |
+------------+------------------------------------------------------------------+
| ``bias_*`` | The slowly varying accelerometer and gyroscope biases.           |
+------------+------------------------------------------------------------------+

The readings are what a real IMU would report, so integrating them back into a trajectory reproduces the camera's motion only approximately: the biases are carried forward at every step, which makes the estimate drift away from the ground truth the longer it runs. The :doc:`IMU section <sections/sensors/imu>` derives the model and shows how large that drift gets.

|

Results
-------

Four seconds of the same living room, as seen by each camera, plus the trajectories integrated from its IMU readings:

.. grid:: 3
    :gutter: 2

    .. grid-item::

        .. video:: _static/quick-start/ultramodern-rgb25fps-preview.mp4
            :loop:
            :autoplay:
            :nocontrols:
            :playsinline:
            :align: center
            :caption: Conventional Camera

    .. grid-item::

        .. video:: _static/quick-start/ultramodern-spc1600fps-preview.mp4
            :loop:
            :autoplay:
            :nocontrols:
            :playsinline:
            :align: center
            :caption: Single Photon Camera

    .. grid-item::

        .. video:: _static/quick-start/ultramodern-dvs50fps-preview.mp4
            :loop:
            :autoplay:
            :nocontrols:
            :playsinline:
            :align: center
            :caption: Event Camera

    .. grid-item::

        .. video:: _static/quick-start/ultramodern-itof-preview.mp4
            :loop:
            :autoplay:
            :nocontrols:
            :playsinline:
            :align: center
            :caption: iToF, first tap

    .. grid-item::

        .. video:: _static/sensors/imu-preview.mp4
            :loop:
            :autoplay:
            :nocontrols:
            :playsinline:
            :align: center
            :caption: IMU, integrated trajectories

|

.. seealso::

    - :doc:`tutorials/sensors` for the sensor tutorials that continue from here.
