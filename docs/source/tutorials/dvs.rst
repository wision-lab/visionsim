Emulating Event Data
====================

This tutorial covers event camera emulation: turning a sequence of intensity frames into an asynchronous stream of brightness-change events. See :doc:`the sensor section <../sections/sensors/dvs>` for the pixel model, the contrast threshold and the noise and mismatch terms it reproduces.

An event camera reports per-pixel changes in log-intensity rather than absolute intensity, so it compares each frame against the previous one. The signal it produces is therefore set by the frame rate of the input: the same scene sampled at 1,000 fps yields events an order of magnitude finer in time than at 100 fps, and a scene sampled too slowly loses the motion between consecutive frames entirely.

|

Prerequisites
-------------

You need a frame sequence whose frame rate is high relative to the motion in the scene. The :doc:`../quick-start` renders a sequence and the :doc:`../sections/interpolation` section turns it into a much denser one; a rate of at least 500 fps is a reasonable starting point for handheld motion. The emulator reads PNG or EXR frames, or a dataset with a ``transforms.json`` or ``transforms.db``, and writes its output to a separate directory.

The frame rate has to be known: the emulator assigns each frame a timestamp from it and measures the intensity change over the intervals in between. It is taken from ``--fps``, or inferred from the dataset metadata when the input is a rendered dataset.

|

Emulating Events
----------------

Run the emulator over a rendered sequence:

.. code-block:: bash

    visionsim emulate.events \
        --input-dir=quickstart/ultramodern-interp/ \
        --output-dir=quickstart/ultramodern-dvs1600fps/ --fps=1600 --preview-step=1

The frame rate goes in ``--fps``, here the 1600 fps of the interpolated sequence, and ``--preview-step`` renders a visualization every N input frames; without it no previews are written. When the input is a dataset whose cameras carry an ``fps``, ``--fps`` can be omitted, and ``--pattern`` selects the input frames when the directory has no metadata at all.

|

Output
------

The output directory contains:

* ``events.txt``: one event per line as ``t,x,y,p``, where ``t`` is the timestamp in microseconds, ``x`` and ``y`` the pixel column and row, and ``p`` the polarity: ``1`` for an ON event (brightness increased) and ``-1`` for an OFF event.
* ``params.json``: the sensor parameters and frame rate the run used, so the stream can be interpreted without guessing how it was produced.
* ``preview/``: when ``--preview-step`` is set, visualization frames in which ON events are drawn in blue and OFF events in red, accumulated over that many input frames.

Because the sensor reacts between samples, not at them, events are not all timestamped at the frame boundary: they are distributed uniformly across the interval since the previous frame, in as many steps as the largest per-pixel event count. Each frame therefore contributes events with sub-frame timestamps up to the frame's own timestamp.

Passing ``--only-preview`` writes the previews and skips ``events.txt`` entirely, which is a cheap way to inspect the emulation before committing to a full run.

.. video:: ../_static/quick-start/ultramodern-dvs1600fps-preview.mp4
   :loop:
   :autoplay:
   :nocontrols:
   :playsinline:
   :width: 80%
   :align: center
   :caption: ON events in blue and OFF events in red, one preview frame per input frame.

|

Sensor Parameters
-----------------

The defaults model a realistic sensor with noise; the interesting knobs are the thresholds that decide when a pixel fires and the terms that make it fire when it should not.

.. code-block:: bash

    visionsim emulate.events --input-dir=quickstart/ultramodern-interp/ \
        --output-dir=quickstart/ultramodern-dvs1600fps/ --fps=1600 --preview-step=1 \
        --pos-thres=0.2 --neg-thres=0.2 --sigma-thres=0.03 \
        --cutoff-hz=200 --leak-rate-hz=1.0 --shot-noise-rate-hz=10.0 \
        --refractory-period-s=0.001 --seed=0

The contrast thresholds and the photoreceptor decide when a pixel fires: ``--pos-thres`` and ``--neg-thres`` are the ON and OFF values in log-intensity units, ``--sigma-thres`` is the standard deviation of the per-pixel mismatch they are drawn from, so pixels do not all agree about what counts as a change, ``--cutoff-hz`` is the photoreceptor's 3 dB bandwidth, which low-passes the log-intensity and sets the shortest change a pixel can follow, and ``--refractory-period-s`` is the dead time after each event.

The noise terms decide when it fires anyway: ``--leak-rate-hz`` models junction leakage, ``--shot-noise-rate-hz`` the photon-driven rate, and ``--seed`` fixes the noise so a run is reproducible. The remaining flags, like ``--blur-sigma`` and ``--photoreceptor-noise``, are documented on :func:`emulate.events <visionsim.cli.emulate.events>`.

Two flags model alternative pixel designs: setting ``--scidvs`` switches to the SCIDVS pixel, whose adaptive high-gain photoreceptor amplifies small changes before they reach the threshold, so it responds to lower-contrast motion than the standard pixel, while combining ``--cs-lambda-pixels`` with ``--cs-tau-p-ms`` enables centre-surround inhibition instead, where a resistive-diffuser surround subtracts from the centre to suppress redundant events. Both are off unless set.

|

Threshold and Mismatch
----------------------

The contrast threshold and the per-pixel threshold mismatch are the two knobs that move the stream between a sparse set of true edges and one salted with events that have nothing driving them.

The threshold sets how far the log-intensity has to change before a pixel fires, so raising it thins the stream: pixels stop reporting slow or small changes and only the sharpest edges survive. Here the ON and OFF thresholds run from 0.1 on the left to 0.8 on the right, with every noise term off so the threshold is the only thing changing:

.. grid:: 4
    :gutter: 2

    .. grid-item::

        .. image:: ../_static/sensors/dvs/ultramodern-threshold-1.webp

    .. grid-item::

        .. image:: ../_static/sensors/dvs/ultramodern-threshold-2.webp

    .. grid-item::

        .. image:: ../_static/sensors/dvs/ultramodern-threshold-3.webp

    .. grid-item::

        .. image:: ../_static/sensors/dvs/ultramodern-threshold-4.webp

Each pixel is given its own threshold, drawn from a normal distribution centred on the nominal one, and ``--sigma-thres`` is that distribution's standard deviation. Widening it spreads the thresholds further, so more pixels land well below the nominal value and fire on changes their neighbours ignore, salting the whole image rather than tracing edges. At a fixed 0.2 threshold, raising it from 0 to 0.4 turns a clean stream into one full of random events:

.. grid:: 4
    :gutter: 2

    .. grid-item::

        .. image:: ../_static/sensors/dvs/ultramodern-mismatch-1.webp

    .. grid-item::

        .. image:: ../_static/sensors/dvs/ultramodern-mismatch-2.webp

    .. grid-item::

        .. image:: ../_static/sensors/dvs/ultramodern-mismatch-3.webp

    .. grid-item::

        .. image:: ../_static/sensors/dvs/ultramodern-mismatch-4.webp

|

Python API
----------

:class:`EventEmulator <visionsim.emulate.dvs.EventEmulator>` drives the same model from Python, one frame at a time:

.. code-block:: python

    from pathlib import Path

    import imageio.v3 as iio
    import numpy as np
    from natsort import natsorted

    from visionsim.emulate.dvs import EventEmulator

    frames = natsorted(Path("quickstart/ultramodern-gt/frames").glob("*.png"))
    emulator = EventEmulator(seed=0)
    emulator.use_preset("noisy")            # or "clean" for an ideal sensor

    events = []
    fps = 50.0

    for i, path in enumerate(frames):
        # The model expects grayscale intensities, manually convert RGB to luma
        r, g, b = np.transpose(iio.imread(path).astype(float), (2, 0, 1))[:3]
        luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
        frame_events = emulator.generate_events(luma, i / fps)
        if frame_events is not None:
            events.append(frame_events)

Calling ``generate_events`` with one ``[H, W]`` intensity frame and its timestamp in seconds returns an ``[N, 4]`` array of ``[timestamp, x, y, polarity]``, or ``None`` when the frame produced no events. The very first call initialises the pixel state and always returns ``None``, because there is no previous sample to compare against. Timestamps have to increase monotonically between calls.

The ``use_preset`` helper fills in a named set of parameters, ``"clean"`` for a sensor with no noise or mismatch and ``"noisy"`` for realistic levels, overwriting whatever the constructor set. Parameters can also be assigned directly, for instance ``emulator.pos_thres = 0.15``. The per-pixel threshold mismatch is sampled from the nominal thresholds and ``sigma_thres`` on the first frame, so apply any preset or override before feeding the emulator its first frame. Calling :meth:`~visionsim.emulate.dvs.EventEmulator.reset` clears the frame buffers and counters so the next frame starts a new stream.

.. note::

    The emulator takes grayscale input. The CLI converts colour frames to luma with the weights above, and log-encoded HDR float data is passed through as-is with ``hdr=True``.

|

.. seealso::

    - :doc:`../sections/datasets` for the ``transforms.db``/``transforms.json`` the emulator reads.
    - :doc:`rgb` for conventional-camera emulation over the same interpolated frames.
