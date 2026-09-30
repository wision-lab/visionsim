Emulating iToF Measurements
===========================

This tutorial covers indirect time-of-flight (iToF) emulation: turning rendered depth and albedo frames into raw per-tap sensor measurements, and decoding those measurements back into depth. See :doc:`the sensor section <../sections/sensors/itof>` for the underlying model.

|

Prerequisites
-------------

You need a rendered sequence that contains both albedo frames and depth maps, which :doc:`../quick-start` and the :doc:`large-dataset` tutorial produce: the Blender renderer writes a ``transforms.json`` that references the depth of every frame (``depth_file_path``).

|

Emulating Measurements
----------------------

Run the emulation CLI over a dataset:

.. code-block:: bash

    $ visionsim emulate.itof --input-dir=renders/frames --output-dir=renders/itof --scheme=convSin --n-captures=4 --freq=120e6

The scheme, the number of captures ``--n-captures`` and the modulation frequency ``--freq`` define the acquisition. With ``--scheme=deltaHilbertDimTwo`` (or any other Hilbert scheme) the curve parameters can be adjusted with ``--hilbert-order`` and ``--hilbert-delta``, and multi-frequency coding needs the per-tap vectors:

.. code-block:: bash

    $ visionsim emulate.itof --input-dir=renders/frames --output-dir=renders/itof \
        --scheme=multFreqSin --n-captures=5 --freq-vec 1 1 1 2 2 --shifts-vec 0 2.094 4.189 0 1.571

Every frame results in a ``.npy`` array of shape ``(n_captures, H, W)`` holding the raw measurements of the ``n_captures`` taps, and the output ``transforms.json`` records the acquisition parameters (scheme, captures, frequency and the unambiguous range) so the data stays self-describing. Use ``--preview`` to also dump a colorized PNG per tap, and ``--force`` to overwrite an existing output directory.

The CLI warns when the scene contains depths beyond the effective range of the selected scheme. That matters most for the ramp schemes, whose usable range is only ``c/4f`` rather than ``c/2f``, and for any scene deeper than the unambiguous range of the modulation frequency: beyond it the recovered depths fold back, even though the radiometric falloff keeps following the true distance.

.. note::

    iToF sensors are monochromatic, so color albedo frames are converted to linear luma before the correlation is computed.

|

Decoding Measurements
---------------------

Decoding is available through the Python API, which also makes it easy to experiment with coding schemes without rendering anything:

.. code-block:: python

    import numpy as np
    import scipy.constants
    from visionsim.emulate.itof import decode, make_coding_functions, simulate_measurements

    freq = 120e6
    max_depth = scipy.constants.c / (2 * freq)      # unambiguous range, ~1.25 m

    modulation_codes, reference_codes = make_coding_functions("convSin", 4, 1001)
    depths = np.linspace(0.05, max_depth - 0.05, 64)
    measurements = simulate_measurements(
        depths, np.full_like(depths, 0.8), modulation_codes, reference_codes, max_depth
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

    The capture count must match the scheme: ``n_captures = 3`` for the ramp schemes, ``n_captures in {3, 4, 5, 6}`` for ``deltaHilbertDimOne``, ``n_captures in {4, 5}`` for ``deltaHilbertDimTwo`` and ``n_captures = 5`` for ``deltaHilbertDimThree``. ``multFreqSin`` additionally needs the same ``freq_vec`` and ``shifts_vec`` that were used during acquisition.

When the scheme is known up front, the matching decoder can be called directly instead of going through the dispatcher: :func:`~visionsim.emulate.itof.decoding.decode_sinusoid` (``convSin``, ``deltaSin``), :func:`~visionsim.emulate.itof.decoding.decode_square` (``convSquare``), :func:`~visionsim.emulate.itof.decoding.decode_single_ramp` and :func:`~visionsim.emulate.itof.decoding.decode_double_ramp` (the ramp schemes), :func:`~visionsim.emulate.itof.decoding.decode_hilbert` (the Hilbert schemes) and :func:`~visionsim.emulate.itof.decoding.decode_mult_freq_sinusoid` (``multFreqSin``).
