Event Camera
============

Event cameras, also known as Dynamic Vision Sensors (DVS), are bio-inspired sensors that operate fundamentally differently from conventional cameras. Instead of capturing intensity frames at a fixed rate, each pixel in an event camera independently and asynchronously signals when it detects a change in brightness.

These sensors offer several advantages over conventional cameras, including very high temporal resolution (in the order of microseconds), high dynamic range (up to 140 dB), and low power consumption.

.. image:: ../../_static/lego-dvs125fps-preview.gif
   :align: center

| 

Sensor Modeling
---------------

The DVS sensor models the response of individual pixels to changes in the log-intensity of the scene. Let :math:`I(x, y, t)` be the intensity of light at pixel :math:`(x, y)` at time :math:`t`. The log-intensity is defined as:

.. math::
    L(x, y, t) = \ln(I(x, y, t))

An event is triggered at pixel :math:`(x, y)` and time :math:`t` if the change in log-intensity since the last event at that pixel exceeds a contrast threshold :math:`C`:

.. math::
    \Delta L = L(x, y, t) - L(x, y, t_{last})

An event is generated if :math:`|\Delta L| \ge C`. The polarity :math:`p \in \{+1, -1\}` of the event indicates whether the brightness increased (ON event) or decreased (OFF event):

.. math::
    p = \begin{cases} +1 & \text{if } \Delta L \ge C \\ -1 & \text{if } \Delta L \le -C \end{cases}

In practice, the thresholds for ON and OFF events may differ (:math:`C_{pos}` and :math:`C_{neg}`), and they are subject to noise and manufacturing variations, which can be modeled as a Gaussian distribution.

Our sensor modeling is based on `v2e <https://github.com/SensorsINI/v2e>`_, and incorporates several non-idealities as described by Hu et al. [1]_:

* **Threshold Jitter:** Variations in the contrast threshold across pixels.
* **Refractory Period:** A minimum time interval between consecutive events from the same pixel.
* **Leakage Current:** Asynchronous events triggered by sensor imperfections even in the absence of brightness changes.
* **Shot Noise:** Random events triggered by photon noise.
* **Photoreceptor Bandwidth:** Limited temporal response of the sensor, modeled as a low-pass filter.

| 

Emulation
---------

To emulate an event camera from a sequence of frames, you can use the CLI::

    $ visionsim emulate.events --input-dir=path/to/frames --output-dir=output/dvs --fps=1000

Parameters such as thresholds and noise rates can be adjusted:

.. code-block:: bash

    $ visionsim emulate.events --input-dir=path/to/frames --output-dir=output/dvs --fps=1000 \
        --pos-thres=0.2 --neg-thres=0.2 --sigma-thres=0.03 \
        --cutoff-hz=200 --leak-rate-hz=1.0 --shot-noise-rate-hz=10.0

The output directory will contain:

* ``events.txt``: A text file where each line is an event in the format ``t x y p`` (time in microseconds, x-coordinate, y-coordinate, polarity).
* ``frames/``: A set of visualization frames showing ON events in blue and OFF events in red.
* ``params.json``: The parameters used for the emulation.

.. note::
    For this to give meaningful results, the input framerate should be high enough to capture the dynamics of the scene. We recommend using a framerate of at least 500 fps. These can be acheived using :doc:`../interpolation`.  

|

References
----------

.. [1] `Hu et al. (2021), "v2e: From Video Frames to Real DVS Events" <https://arxiv.org/abs/2006.07722>`_
