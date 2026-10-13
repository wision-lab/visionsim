Emulating a Conventional Camera
===============================

This tutorial covers conventional RGB emulation: turning a dense sequence of ground truth frames into the output of a rolling sensor with a finite exposure. See :doc:`the sensor section <../sections/sensors/rgb>` for the imaging model, the meaning of full well capacity and the noise terms.

|

Prerequisites
-------------

You need an interpolated frame sequence: a render whose frames are close enough in time that averaging a run of them reproduces the motion blur of a slower shutter. The :doc:`../quick-start` guide renders 201 frames and interpolates them 32-fold into ``quickstart/ultramodern-interp/``. The :doc:`large-dataset` tutorial produces a longer sequence the same way.

Interpolation is what makes the blur meaningful. Averaging frames of the original render only samples the scene at the render's own frame rate, so the result is a stack of distinct poses rather than the continuous exposure a real camera integrates. Interpolating first fills in the sub-frame motion, and only then does averaging a chunk trace the sweep of the shutter.

The emulator reads PNG or EXR frames from a directory and writes its own frames out, one directory per run. The input and output directories must differ.

|

Emulating Frames
----------------

Run the emulator over an interpolated sequence:

.. code-block:: bash

    visionsim emulate.rgb \
        --input-dir=quickstart/ultramodern-interp/ \
        --output-dir=quickstart/ultramodern-rgb25fps/ \
        --chunk-size=64 --readout-std=0

``--chunk-size`` is the number of consecutive input frames averaged into one output frame, so it sets the exposure length; ``--readout-std`` is the standard deviation of the Gaussian read noise added after the accumulation. The remaining flags are ``--factor`` to scale the linear intensity, ``--fwc`` to override the full well capacity (it defaults to the chunk size), ``--duplicate`` to repeat each input frame ``duplicate`` times when the chunk size is too small to average a realistic signal, and ``--pattern`` to select the input files. ``--force`` overwrites an output directory that already exists. Frames are written under the output directory, and ``transforms.json`` records the resulting sequence.

The API takes the loaded frames directly:

.. literalinclude:: ../../../examples/sensors/rgb.py

Each call to :func:`emulate_rgb_from_sequence <visionsim.emulate.rgb.emulate_rgb_from_sequence>` returns one frame, so the example builds two lists of four: the first varies the exposure length at zero noise, the second varies the read noise at a fixed exposure.

.. note::

    Frames are read as sRGB and divided by 255 to get the linear intensity the model expects. Pass normalized float images in the same range, not the raw 8-bit arrays.

|

Controlling Blur and Noise
--------------------------

The chunk size and the read noise are the two knobs that move the output between a clean long exposure and a noisy short one.

The chunk size sets how long the shutter stays open, and therefore how far a moving edge smears. At the 1,600 fps of the interpolated quickstart data, a chunk of 64 frames integrates 40 ms and sits at 25 fps, while a chunk of 256 integrates 160 ms and blurs four times as far. Too small a chunk approaches the interpolated frame rate and recovers the sharp, noiseless input.

Varying the exposure length at zero read noise, from a 10-frame chunk on the left to 640 frames on the right:

.. grid:: 4
    :gutter: 2

    .. grid-item::

        .. image:: ../_static/sensors/rgb/ultramodern-exposure-1.webp

    .. grid-item::

        .. image:: ../_static/sensors/rgb/ultramodern-exposure-2.webp

    .. grid-item::

        .. image:: ../_static/sensors/rgb/ultramodern-exposure-3.webp

    .. grid-item::

        .. image:: ../_static/sensors/rgb/ultramodern-exposure-4.webp

The read noise sets the grain of the result. At a fixed chunk size, raising ``--readout-std`` on its own is a way to see the noise floor move independently of the blur. Here it is lowered from left to right, with the exposure held constant:

.. grid:: 4
    :gutter: 2

    .. grid-item::

        .. image:: ../_static/sensors/rgb/ultramodern-readnoise-1.webp

    .. grid-item::

        .. image:: ../_static/sensors/rgb/ultramodern-readnoise-2.webp

    .. grid-item::

        .. image:: ../_static/sensors/rgb/ultramodern-readnoise-3.webp

    .. grid-item::

        .. image:: ../_static/sensors/rgb/ultramodern-readnoise-4.webp
