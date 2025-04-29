Conventional Camera
===================

For scene radiance :math:`\phi`, exposure time :math:`\tau`, optical response function :math:`\Gamma`, the intensity :math:`I` of the pixel can be modeled as [1]_:

.. math::
    \begin{align*}
        I = \Gamma \left(min\left(\int_{\tau} \phi dt,~\text{FWC}\right) + \mathcal{N} \right) \,.
    \end{align*}

Where ``FWC`` is the pixel's s full well capacity, and many sources of noise, including read noise and photon shot noise, which we absorb into :math:`\mathcal{N}`.

This sensor modeling is incorporated into :func:`emulate_rgb_from_sequence <visionsim.emulate.rgb.emulate_rgb_from_sequence>`, which, when given appropriate noise parameters and sequence of ground truth RGB frames, will emulate a conventional camera. 

Using the interpolated frames from the :doc:`../../quick-start` guide, we can easily generate conventional RGB frames with varying levels of noise and blur::

    $ visionsim emulate.rgb --input-dir=quickstart/lego-interp/ --output-dir=quickstart/rgb/ --chunk-size=160 --readout-std=0

Similarly, we can emulate an RGB camera using the API like so:

.. literalinclude:: ../../../../examples/sensors/rgb.py 

.. note::
    The ``fwc`` or full-well-capacity argument is not in units of electrons, since we have no physical camera model which matches an rgb linear intensity to a number of electrons, but rather is relative to the `chunk-size`. A FWC equal to the chunk size means that, if each image has a normalized intensity of 1.0, the well will fill up.

Varying the sequence length, we emulate a longer exposure time, leading to more blur:

.. list-table::
    :class: borderless

    * - .. image:: ../../_static/sensors/rgb/lego-exposure-1.png
      - .. image:: ../../_static/sensors/rgb/lego-exposure-2.png
      - .. image:: ../../_static/sensors/rgb/lego-exposure-3.png
      - .. image:: ../../_static/sensors/rgb/lego-exposure-4.png


The amount of read noise can also be changed, here it is lowered from left to right:

.. list-table::
    :class: borderless
    
    * - .. image:: ../../_static/sensors/rgb/lego-readnoise-1.png
      - .. image:: ../../_static/sensors/rgb/lego-readnoise-2.png
      - .. image:: ../../_static/sensors/rgb/lego-readnoise-3.png
      - .. image:: ../../_static/sensors/rgb/lego-readnoise-4.png


.. [1] `M.D. Grossberg and S.K. Nayar (2004), "Modeling the space of camera response functions" <https://cave.cs.columbia.edu/old/publications/pdfs/Grossberg_PAMI04.pdf>`_
