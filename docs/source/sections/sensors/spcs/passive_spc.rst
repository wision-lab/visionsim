Passive Single Photon Camera
============================


What's a single photon camera?
------------------------------

Single photon cameras (SPCs) are an emerging class of sensors that offer extreme sensitivity and are manufactured at scale using existing CMOS techniques. These sensors can not only detect individual photons but also time them with picosecond accuracy, enabling applications that are `not possible with traditional sensors <https://wisionlab.com/project/burst-vision-single-photon/>`_. By sensing individual photons, SPCs provide us with the most fine-grained visual information possible and provides a lot of flexibility to the authors of downstream inference tasks.

|

Image formation model
---------------------

While a conventional camera can capture many thousands of photons during a single exposure, a single photon detector can only detect one before it has to be reset:

.. video:: ../../../_static/sensors/PhotonBucket.mp4
   :loop:
   :autoplay:
   :nocontrols:
   :width: 100%
   :class: only-light

.. video:: ../../../_static/sensors/PhotonBucket-dark.mp4
   :loop:
   :autoplay:
   :nocontrols:
   :width: 100%
   :class: only-dark

Due to this, SPCs are fundamentally digital devices. They will output a binary one if one or more photons are received during it's exposure time and a zero otherwise. For a static scene with a radiant flux (photons/second) of :math:`\phi`, the number of incident photons :math:`k` on a pixel during an exposure time :math:`\tau` follows a Poisson distribution given by:

.. math:: 
    \begin{align*}
        P(k) = \frac{(\phi\tau)^ke^{-\phi\tau}}{k!} \,.
    \end{align*}

From this, we can infer that the binary pixel measurement :math:`B` will follow a Bernoulli distribution given by [1]_:

.. math:: 
    \begin{align*}
        P(B=0) &= P(k=0) = e^{-\phi \tau},\\
        P(B=1) &= P(k \ge 1)=1-e^{-\phi \tau}.
    \end{align*}

However, a SPC can run at extremely fast rates, so in practice while each measurement is extremely noisy, and quantized to a single bits-worth of information, we can acquire many thousands of measurements in the time a conventional camera takes for a single exposure. 

| 

Inverting the response function
-------------------------------

Now that we understand the basics of a SPC, how can we recover a "normal" image from one? 

The key observation is that the previous equation is invertible, so if we have a good photon detection probability estimate :math:`\widehat P`, we can also estimate :math:`\phi`. Namely, the maximum likelihood estimator of :math:`\widehat P` for a **static scene** is simply the average of neighboring binary frames [2]_: 

.. math:: 
    \begin{align*}
        \widehat{P} = \frac{1}{n} \sum_{i=1}^n B_i.
    \end{align*}

And by inverting the image formation model above we can get a good estimate of the scene flux:

.. math::
    \begin{align*}
        \widehat{\phi}=-\frac{1}{\tau} \ln \left(1-\widehat{P}\right)\,.
    \end{align*}

Finally, bear in mind that :math:`\phi` is the scene flux in photons/second, which is proportional to linear intensity, yet most images are typically tonemapped. For more information about this, consider working through `this notebook <https://github.com/cpsiff/SPAD-ICCP-Summer-School/blob/main/part_1_passive_SPADs/activity_hdr.ipynb>`_. 

|

.. [1] Some sources of noise such as dark counts and non-ideal quantum efficiency can be absorbed into the value of :math:`\phi`.
.. [2] `Yang et al. (2011), "Bits from Photons: Oversampled Image Acquisition Using Binary Poisson Statistics". <https://arxiv.org/abs/1106.0954>`_