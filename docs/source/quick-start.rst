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

We will create a small scale dataset of a toy lego truck as if it was captured by a realistic 25fps conventional RGB camera, a 4kHz single photon camera, an event camera and an indirect time-of-flight camera, for 4 seconds. To achieve this, we will:

- Render a small dataset of 500 ground truth frames, plus depth maps, using the :func:`blender.render-animation <visionsim.cli.blender.render_animation>` CLI,
- Interpolate this dataset 32-fold using the :func:`interpolate.dataset <visionsim.cli.interpolate.dataset>` CLI,
- And emulate different cameras using the :func:`emulate.rgb <visionsim.cli.emulate.rgb>` / :func:`emulate.spad <visionsim.cli.emulate.spad>` / :func:`emulate.events <visionsim.cli.emulate.events>` / :func:`emulate.itof <visionsim.cli.emulate.itof>` CLIs.

|

Rendering Ground Truth
----------------------

First, download the test scene, here we will be using the lego truck (of the `NeRF <https://www.matthewtancik.com/nerf>`_ fame) which you can download from `here <https://drive.google.com/drive/folders/1gRxhL3rbGDTfgKytre8WkbBu-QDJFy15?usp=sharing>`_. This blender file has been modified to include an HDR skybox and a camera animation. Specifically, the camera moves along a circular orbit at Z=1 with radius=5 that points towards the origin and lasts 100 frames. Here, we assume your blender file is pre-animated, but you can control camera movements and keyframes manually too. 


To create the lego dataset, we'll slow down the camera movement by a factor of 5x and render 500 RGB frames: 

.. literalinclude:: ../../examples/quickstart.sh
   :language: bash
   :lines: 3-4

.. note::
    The :func:`blender.render-animation <visionsim.cli.blender.render_animation>` CLI has a lot of options which enable changing render settings and resolution, parallelization, and for generating different types of ground truth annotations such as depth and segmentation maps. You can see all options by running the following::

        visionsim blender.render-animation --help

    If the above command does not work, you might have to change some settings, notably the ``device-type``. For instance on older GPUs that do not support Optix you can do ``--config.device-type=cuda`` to use CUDA.
    
    Finer grain control can be had using the :class:`BlenderClient API <visionsim.simulate.blender.BlenderClient>`.

.. warning:: This might take a while, with blender 4.2 on a RTX 3080 it takes about 18 minutes. 

All the rendered frames will be in ``quickstart/lego-gt/frames``. Each data directory holds its own ``transforms.db`` metadata file describing the camera trajectory and intrinsics, and ``--config.include-depths`` adds a sibling ``quickstart/lego-gt/depths`` directory holding an aligned depth map for every frame. See :doc:`sections/datasets` for what these metadata files carry. The :func:`dataset.convert <visionsim.cli.dataset.convert>` CLI turns any of these ``.db`` files into a Nerfstudio-style ``transforms.json`` if you prefer to work with JSON, and :func:`dataset.merge <visionsim.cli.dataset.merge>` combines them when the frames, depths and other annotations share a camera.

Let's create a quick preview of this dataset by animating every 5th frame into a video, so it plays back in realtime:

.. literalinclude:: ../../examples/quickstart.sh
   :language: bash
   :lines: 5-10

``preview.mp4`` animates a turntable-style loop of a lego truck (shown here as a GIF made with `gifski <https://gif.ski/>`_, yours will look better). ``preview-depths.mp4`` previews the same depth annotation that the iToF emulation later reads, colorized per frame:

.. list-table::
    :class: borderless

    * - .. figure:: _static/lego-gt-preview.gif

            Rendered RGB

      - .. figure:: _static/lego-depth-preview.gif

            Ground truth depth

.. note::
    The depth preview flickers, and this is an artifact of how the renderer produces it rather than something in the data. ``--config.depths.preview`` normalizes each frame independently so the colormap stretches to fill the near/far range of that frame alone, which makes a depth map that changes little between frames appear to change a lot. The underlying ``.exr`` maps in ``quickstart/lego-gt/depths`` hold absolute distances and are unaffected.

    To normalize across the whole sequence instead, colorize after rendering with :func:`transforms.colorize-depths <visionsim.cli.transforms.colorize_depths>`, which estimates a single depth range from every frame and reuses it:

    .. code-block:: bash

        visionsim transforms.colorize-depths \
            --input-dir=quickstart/lego-gt/depths \
            --output-dir=quickstart/lego-gt/depths-colorized

|

Interpolating Frames
--------------------

You can optionally interpolate an existing dataset to get intermediate frames that have not been rendered. This quickly increases the effective framerate of the data, at the cost of artifacts if adjacent frames are too "far" apart. The following will interpolate a dataset by a factor of 32x:

.. literalinclude:: ../../examples/quickstart.sh
   :language: bash
   :lines: 11-13

You can preview this new dataset like above, just use a step of 160 (=5x32) to ensure playback is at the same speed. 

.. note::
    The new dataset contains 15,969 frames, not 32x500=16,000. Consider interpolating two frames by a factor of 2x: you create a frame between them, giving 3 frames total, the first original, the interpolated one, and the second original. In general, M original frames interpolated N-times yields NM-N+1 frames.  

.. warning::
    Interpolation introduces artifacts when adjacent frames in the original dataset are too different from one another. This and its implications are discussed further in the :doc:`sections/interpolation` section. Interpolation helps bridge the gap from 1,000fps to 10,000fps, not from 10fps to 100fps. 

.. warning::
    Interpolation is faster than rendering new frames but still slow: the 32x pass above takes about 10 minutes on an RTX 3080.

|

Emulating Sensor Data
---------------------

To simulate real cameras, we must convert these perfect ground truth frames and their annotations into realistic measurements. Each of the four sensors below reads a different property of the same rendered sequence.

Conventional Camera
^^^^^^^^^^^^^^^^^^^

Motion blur is emulated by :func:`emulate.rgb <visionsim.cli.emulate.rgb>`. The ``--chunk-size`` flag averages the interpolated frames in groups of 160, and ``--readout-std`` sets the standard deviation of the Gaussian read noise, left at 0 here to keep the example clean. Since the interpolated dataset runs at 4,000fps, 160 frames per output frame simulate a 25fps RGB camera:

.. literalinclude:: ../../examples/quickstart.sh
   :language: bash
   :lines: 14-17

Single Photon Camera
^^^^^^^^^^^^^^^^^^^^

We can emulate a single-photon camera using :func:`emulate.spad <visionsim.cli.emulate.spad>`, at the same framerate as the interpolated dataset like so:

.. literalinclude:: ../../examples/quickstart.sh
   :language: bash
   :lines: 18-20

The emulator linearizes the sRGB frames back to linear scene radiance and samples each pixel independently, treating it as a Bernoulli trial whose success probability is :math:`1 - e^{-\varphi}` for incident intensity :math:`\varphi`. 

Event Camera
^^^^^^^^^^^^

An event camera (emulated using :func:`emulate.events <visionsim.cli.emulate.events>`) reports per-pixel brightness changes as asynchronous events rather than frames. It reads the ground truth frames (or interpolated ones), but we must pass the correct frame rate using ``--fps``:

.. literalinclude:: ../../examples/quickstart.sh
   :language: bash
   :lines: 21-23

Each event is a tuple of ``(x, y, t, p)``: the pixel location, the timestamp in microseconds and the polarity, where ``p = 1`` marks a brightness increase and ``p = -1`` a decrease. Events are written one per line to ``events.txt``. The ``--preview-step`` flag additionally accumulates events into visualization frames, with ON events in blue and OFF events in red; previews are off unless it is set, and the command above passes 1 to render one per input frame.

Indirect Time-of-Flight Camera
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Indirect time-of-flight (iToF) sensors recover depth by correlating a modulated light source against the returning signal, so they read the depth maps as well as the albedo frames. The renderer writes depths to a sibling directory with its own metadata rather than merging them into the albedo dataset, so both are passed explicitly:

.. literalinclude:: ../../examples/quickstart.sh
   :language: bash
   :lines: 24-28

The scheme, capture count and frequency define the acquisition. Each frame is written as a ``.npy`` array of ``(n_captures, H, W)`` raw taps, and the :doc:`iToF tutorial <tutorials/itof>` covers the full set of coding schemes and shows how to decode them back into depth. The ramp schemes are unambiguous only over ``c/4f``, so the CLI warns when the scene is deeper than the unambiguous range of the selected scheme.

|

Results
-------

Four seconds of the same lego truck, as seen by each sensor:

.. list-table::
    :class: borderless
    :widths: 50 50

    * - .. figure:: _static/lego-rgb25fps-preview.gif

            Conventional Camera

      - .. figure:: _static/lego-spc4kHz-preview.gif

            Single Photon Camera

    * - .. figure:: _static/lego-dvs125fps-preview.gif

            Event Camera

      - .. figure:: _static/lego-itof-preview.gif

            iToF, first tap

.. note::
    The iToF panel has no background. The scene is lit by an `HDRI <https://en.wikipedia.org/wiki/Reflection_mapping>`_ skybox, so everything behind the truck is emitted light from the environment rather than geometry. RGB and SPAD read that light and render the backdrop; depth and iToF need a surface to return a distance from, and there is none out there.
