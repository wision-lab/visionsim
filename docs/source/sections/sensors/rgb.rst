Conventional Camera
===================

A conventional camera integrates the light that reaches each pixel over its exposure time, converts the accumulated charge to a voltage, digitizes it, and gamma-encodes the result for display. For scene radiance :math:`\phi`, exposure time :math:`\tau`, optical response function :math:`\Gamma` and noise :math:`\mathcal{N}`, the stored intensity :math:`I` of the pixel is modeled as [1]_:

.. math::
    I = \Gamma \left(min\left(\int_{\tau} \phi dt,~\text{FWC}\right) + \mathcal{N} \right) \,.

Each term maps to a stage of that pipeline:

:math:`\phi` and :math:`\tau`
    The pixel collects :math:`\int_\tau \phi\,dt`, the scene radiance integrated over the exposure. A longer exposure collects more light, and any motion within it smears across the frame as motion blur.

:math:`\text{FWC}`
    The full well capacity, the charge a pixel holds before it saturates. Signal above it is clipped, so bright highlights lose their detail.

:math:`\mathcal{N}`
    The noise added to the accumulated signal. Two sources matter here: photon shot noise, whose variance grows with the signal because photons arrive as a Poisson process, and a zero-mean read noise from the readout electronics.

:math:`\Gamma`
    The optical response, or camera response function, mapping charge to the stored value. It includes the gamma encoding applied to make the image display-referred.

The model is implemented by :func:`emulate_rgb_from_sequence <visionsim.emulate.rgb.emulate_rgb_from_sequence>`, which turns a sequence of linear-intensity frames and the noise parameters into emulated frames. It sums the frames of one exposure into a single accumulation, draws Poisson shot noise from that sum and adds Gaussian read noise, scales by the full well capacity with highlights clipped, quantizes to ``bitdepth`` bits to model the ADC, and gamma-encodes the result to sRGB. The exposure is the number of frames summed, so the input frame rate and the frames-per-output set the shutter time together.

.. note::
    The ``fwc`` or full-well-capacity argument is not in units of electrons, since we have no physical camera model which matches an rgb linear intensity to a number of electrons, but rather is relative to the `chunk-size`. A FWC equal to the chunk size means that, if each image has a normalized intensity of 1.0, the well will fill up.

.. seealso::
    A worked example that runs the emulator from the command line and through the Python API, and shows how the exposure and noise knobs change the output, is in :doc:`../../tutorials/rgb`.

.. [1] `M.D. Grossberg and S.K. Nayar (2004), "Modeling the space of camera response functions" <https://cave.cs.columbia.edu/old/publications/pdfs/Grossberg_PAMI04.pdf>`_
