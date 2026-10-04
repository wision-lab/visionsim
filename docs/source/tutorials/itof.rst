Emulating iToF Measurements
===========================

This tutorial covers indirect time-of-flight (iToF) emulation: turning rendered depth and albedo frames into raw per-tap sensor measurements, and decoding those measurements back into depth. See :doc:`the sensor section <../sections/sensors/itof>` for the underlying model.

|

Prerequisites
-------------

You need a rendered sequence with both albedo frames and depth maps. The :doc:`../quick-start` and the :doc:`large-dataset` tutorial both produce one.

The Blender renderer writes albedo and depth to sibling directories, each with its own metadata, so the emulator takes a path for each. If you already combined them into one dataset with :func:`dataset.merge <visionsim.cli.dataset.merge>`, a single ``--input-dir`` is enough and each frame's depth is read from its ``depth_file_path``.

|

Emulating Measurements
----------------------

Run the emulator over a rendered sequence:

.. code-block:: bash

    visionsim emulate.itof \
        --input-dir=renders/lego-gt/frames \
        --depth-dir=renders/lego-gt/depths \
        --output-dir=renders/itof \
        --scheme=convSin --n-captures=4 --freq=120e6

The scheme, the number of captures ``--n-captures`` and the modulation frequency ``--freq`` define the acquisition. With ``--scheme=deltaHilbertDimTwo`` (or any other Hilbert scheme) the curve parameters can be adjusted with ``--hilbert-order`` and ``--hilbert-delta``, and multi-frequency coding needs the per-tap vectors:

.. code-block:: bash

    visionsim emulate.itof --input-dir=renders/lego-gt/frames --depth-dir=renders/lego-gt/depths \
        --output-dir=renders/itof --scheme=multFreqSin --n-captures=5 \
        --freq-vec 1 1 1 2 2 --shifts-vec 0 2.094 4.189 0 1.571

Each frame writes a ``.npy`` array of shape ``(n_captures, H, W)`` holding the raw measurements of the ``n_captures`` taps. The output ``params.json`` records the acquisition parameters (scheme, captures, frequency and the unambiguous range), which is what lets you decode the arrays later without guessing how they were produced. Use ``--preview`` to also dump a colorized PNG per tap, and ``--force`` to overwrite an existing output directory.

The CLI warns when the scene contains depths beyond the unambiguous range of the selected scheme. That range is ``c/2f`` for most schemes and ``c/4f`` for the ramp schemes, and geometry past it folds back into the recovered depths while the radiometric falloff keeps following the true distance. In the quickstart scene the truck sits around 4.5-5.8 m against a 1.25 m range at ``f = 120 MHz``, so every scheme folds there; the ramp schemes fold at 0.62 m instead.

.. note::

    iToF sensors are monochromatic, so color albedo frames are converted to linear luma before the correlation is computed.

|

Decoding Measurements
---------------------

Decoding is available through the Python API, which lets you exercise a coding scheme without rendering a scene first:

.. code-block:: python

    import numpy as np
    import scipy.constants
    from visionsim.emulate.itof import decode, make_coding_functions, simulate_measurements

    freq = 120e6
    period = 1 / freq                               # one code period
    d_max = scipy.constants.c * period / 2          # range it wraps over, ~1.25 m

    modulation_codes, reference_codes = make_coding_functions("convSin", 4, 1001)
    depths = np.linspace(0.05, d_max - 0.05, 64)
    measurements = simulate_measurements(
        depths, np.full_like(depths, 0.8), modulation_codes, reference_codes, period
    )
    recovered = decode("convSin", measurements, freq)

    print(np.abs(recovered - depths).max())        # sub-millimetre for every depth

To decode emulated frames with :func:`~visionsim.emulate.itof.decoding.decode`, load them and flatten the spatial dimensions:

.. code-block:: python

    import numpy as np
    from visionsim.emulate.itof import decode

    measurements = np.load("renders/itof/0000.npy")           # (n_captures, H, W)
    k, height, width = measurements.shape
    depths = decode("convSin", measurements.reshape(k, -1), freq=120e6).reshape(height, width)

.. warning::

    The capture count must match the scheme; the :doc:`scheme table in the sensor section <../sections/sensors/itof>` lists the counts each decoder accepts. ``multFreqSin`` additionally needs the same ``freq_vec`` and ``shifts_vec`` that were used during acquisition.

When the scheme is known up front, the matching decoder can be called directly instead of going through the dispatcher: :func:`~visionsim.emulate.itof.decoding.decode_sinusoid` (``convSin``, ``deltaSin``), :func:`~visionsim.emulate.itof.decoding.decode_square` (``convSquare``), :func:`~visionsim.emulate.itof.decoding.decode_single_ramp` and :func:`~visionsim.emulate.itof.decoding.decode_double_ramp` (the ramp schemes), :func:`~visionsim.emulate.itof.decoding.decode_hilbert` (the Hilbert schemes) and :func:`~visionsim.emulate.itof.decoding.decode_mult_freq_sinusoid` (``multFreqSin``).
