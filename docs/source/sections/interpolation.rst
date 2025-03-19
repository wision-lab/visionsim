Interpolation
=============

Interpolation refers to filling in the gaps between observations, in our case we use ``frame`` -based video interpolation methods, along with pose interpolation, to speed up rendering.   

Methods
-------

Many ``frame`` -based interpolation methods exist, currently we support the following:

* `RIFE (ECCV 2022) <https://github.com/hzwer/ECCV2022-RIFE>`_
* ... more to come!

To interpolate the camera trajectory, we simply linearly interpolate the camera position and spherically interpolate camera rotations.

Artifacts
---------

Video interpolation is an inherently ambiguous task -- it is inconceivable to re-create a whole movie by interpolating it's first and last frame -- yet, when input frames are close enough, the problem becomes significantly simpler. We use this key insight to our advantage, we do not need to render ground truth frames at the framerate of a single photon camera, we can instead render them at framerate that is sufficient to capture most scene motion, and then interpolate the rest of the way.       

What framerate to render at depends entirely on the scene, it's geometry and textures and how fast the relative motion is. To illustrate this point, we've taken the scene for the :doc:`quickstart guide <../quick-start>` and rendered it at different framerates and interpolated it to 400 fps:

.. list-table::
    :class: borderless

    * - .. figure:: ../_static/lego0025-interp.gif

            6.25 fps interpolated 64x

      - .. figure:: ../_static/lego0050-interp.gif

            12.5 fps interpolated 32x

    * - .. figure:: ../_static/lego0100-interp.gif
                
            25 fps interpolated 16x

      - .. figure:: ../_static/lego0200-interp.gif

            50 fps interpolated 8x

As you can see, *for this scene*, we need to render at a minimum of 50fps for artifacts to become imperceptible. Here we've interpolated the 50fps render only 8x, but we can interpolate it much more without creating additional artifacts since adjacent frames are similar enough.