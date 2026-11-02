Emulating a Passive Single Photon Camera
========================================

This tutorial covers passive single photon camera (SPC) emulation: turning rendered frames into the one-bit measurements a single photon sensor would record, and recovering an ordinary image from a run of them. See :doc:`the sensor section <../sections/sensors/spcs/passive_spc>` for the imaging model and the estimator behind the recovery.

A *passive* single photon camera counts the photons that reach it and has no light source of its own. Its :doc:`active <../sections/sensors/spcs/active_spc>` counterpart sends out a laser pulse and times the return.

Each exposure records a single bit per pixel, so one frame carries almost no information. The sensor makes up for this with speed, recording thousands of exposures in the time a normal camera takes one. Averaging them gives the probability that each pixel caught a photon, and inverting the response function turns that back into the intensity.

|

Prerequisites
-------------

To use the emulator, you need a sequence of frames in linear intensity. PNG or JPEG frames are treated as display-encoded and automatically converted back to linear, while ``.exr`` and ``.hdr`` frames are used as-is. The :doc:`../quick-start` provides a suitable sequence, and the :doc:`hdr` section explains how to save render passes in a linear format.

Because the emulator samples each frame independently, the SPAD emulation occurs at the same frame rate as the input sequence. You can use :doc:`interpolation <../sections/interpolation>` if you require a faster frame rate.

|

Emulating Binary Frames
-----------------------

Run the emulator over a rendered sequence:

.. code-block:: bash

    visionsim emulate.spad \
        --input-dir=quickstart/ultramodern-interp/ \
        --output-dir=quickstart/ultramodern-spc1600fps/

The ``--factor`` flag sets the exposure time, the :math:`\tau` of the model below. Larger values collect more light per exposure, so pixels are likelier to record a one. The remaining flags are documented on :func:`emulate.spad <visionsim.cli.emulate.spad>`.

The output directory contains:

* ``0000.npy``, ``0001.npy``, …: the binary frames, up to ``--max-size`` per file. A pixel only needs one bit, so eight of them are packed into each byte and a frame takes an eighth of the space an 8-bit image would. ``transforms.json`` records the packed axis (``bitpack_dim``) and each frame's ``offset`` into its file, so you can unpack the frames later; see :doc:`../sections/datasets`.
* ``transforms.json``: the camera pose and intrinsics for each frame, along with the array and offset where its data lives.

|

Recovering an Image
-------------------

The average of the binary frames estimates the photon detection probability :math:`P` that a pixel detects a photon. Inverting the response function :math:`P = 1 - e^{-\phi \tau}` turns that back into the amount of light :math:`\phi` that reached the pixel:

.. math::

    \widehat{P} = \frac{1}{n} \sum_{i=1}^{n} B_i
    \qquad \Longrightarrow \qquad
    \widehat{\phi} = -\frac{1}{\tau} \ln\!\left(1 - \widehat{P}\right).

Here is the same view recovered from a growing number of consecutive exposures from the 1,600 fps capture, from a single exposure on the left to 32 on the right:

.. grid:: 3
    :gutter: 2

    .. grid-item::

        .. image:: ../_static/sensors/spc/ultramodern-reconstruction-1.webp

    .. grid-item::

        .. image:: ../_static/sensors/spc/ultramodern-reconstruction-2.webp

    .. grid-item::

        .. image:: ../_static/sensors/spc/ultramodern-reconstruction-3.webp

With one exposure the image is mostly noise. Eight exposures already show the room clearly. Thirty-two look smoother still, though slightly soft due to camera motion.

Averaging trades noise against motion blur. The recovery assumes each pixel sees the same amount of light in every frame it averages, so the scene and the camera have to hold still, and the image blurs where they do not. How many exposures to combine therefore depends on how fast the scene moves, not on how many the sensor can record.

|

Dynamic Range
-------------

The ``--factor`` flag is the exposure time :math:`\tau` of the response function, so it behaves like the exposure control on a camera. Set it too high and bright areas record a one at every exposure and recover as flat white. Set it too low and pixels almost never fire, so the recovery is mostly noise. The same view at three factors, all averaged over the same number of exposures, from ``--factor=0.02`` on the left to ``--factor=25`` on the right:

.. grid:: 3
    :gutter: 2

    .. grid-item::

        .. image:: ../_static/sensors/spc/ultramodern-factor-1.webp

    .. grid-item::

        .. image:: ../_static/sensors/spc/ultramodern-factor-2.webp

    .. grid-item::

        .. image:: ../_static/sensors/spc/ultramodern-factor-3.webp

The three panels are displayed with the same brightness scaling, so a factor that loses signal shows as a darker, noisier image rather than being stretched back to full brightness.

The right panel is past saturation. Most bits are ones, so the recovered values pile up at a single level and the contrast collapses. A useful factor keeps the recovery probability between 0 and 1.

The recovered value is the light level :math:`\phi` on a linear scale. Most programs expect colors after tone mapping instead, so display the result or train on it only after applying the same tone mapping the render used. See :doc:`hdr`.

|

Python API
----------

The same model is available as two Python functions, :func:`emulate_spc <visionsim.emulate.spc.emulate_spc>` and :func:`spc_avg_to_rgb <visionsim.emulate.spc.spc_avg_to_rgb>`, which run it one exposure at a time:

.. code-block:: python

    import imageio.v3 as iio
    import numpy as np

    from visionsim.emulate.spc import emulate_spc, spc_avg_to_rgb
    from visionsim.utils.color import linearrgb_to_srgb, srgb_to_linearrgb

    frame = iio.imread("quickstart/ultramodern-interp/0000/150_00.png").astype(float) / 255
    flux = srgb_to_linearrgb(frame)          # convert to linear light

    rng = np.random.default_rng(0)
    total = np.zeros_like(flux)
    n = 256
    for _ in range(n):
        total += emulate_spc(flux, factor=1.0, rng=rng)
    mean = total / n                          # the probability each pixel caught a photon

    intensity = spc_avg_to_rgb(mean, factor=1.0, quantile=0.99)   # recover the original image
    image = linearrgb_to_srgb(intensity)                          # back to display sRGB

The :func:`emulate_spc <visionsim.emulate.spc.emulate_spc>` function samples a linear intensity image into a binary one, setting each pixel to one with probability :math:`p = 1 - e^{-\phi \cdot \text{factor}}`. Its counterpart :func:`spc_avg_to_rgb <visionsim.emulate.spc.spc_avg_to_rgb>` takes the mean of many binary frames and returns the light level, dividing by ``factor`` and clipping ``1 - P`` to ``epsilon`` so a saturated pixel does not become infinite. The ``quantile`` parameter, when set, scales the result so the given quantile becomes the new maximum, which is how the images above were stretched for display. Leave it at ``None`` to keep the absolute light level.

Read the emulated data back through the dataset, which unpacks the bits:

.. code-block:: python

    from visionsim.dataset import Dataset

    dataset = Dataset.from_path("quickstart/ultramodern-spc1600fps")
    frame, _ = dataset[0]                    # an [H, W, 3] array of zeros and ones

|

.. seealso::

    - :doc:`../sections/datasets` for the ``transforms.json`` metadata and the ``bitpack_dim`` convention.
    - :doc:`hdr` for rendering the linear input the emulator samples.
    - :doc:`dvs` for the other sensor that trades a low-bit sample for a high frame rate.
