==========
Simulation
==========

We use `Blender <https://www.blender.org/>`_ to generate ground truth intensity frames at any arbitrary frame rate. 

Keyframe Stretching
===================

An artist designing a 3D animation, does so by defining a set of *keyframes*. Each one holds a timestamp and set of attributes that need to be set to particular values at the timestamp. Upon rendering, Blender will interpolate attribute values between these different keyframes. For instance, we could say that at frame #1 the camera's focal length is `50` mm and then at frame #20 it will be `100` mm, then, if we render frame #10 we should expect a focal length of `75` mm.   

Keyframes are tied to frame numbers not relative animation time, so while this works well when the artist knows the framerate of the final render, if they animate for `30` fps but render it at `120` fps, the animation will be four times too fast. 

This is where keyframe stretching comes in, it allows for a decoupling between rendering framerate and animation framerate. The ``keyframe-multiplier`` option of ``blender.render`` controls this stretch factor. In the above example, we would set ``--keyframe-multiplier=4.0`` to get the desired animation, or we could reuse existing animations and render them at single photon camera framerates using a higher multiplier.  
 

Setup a Trajectory
==================


Programmatically
----------------

You can use the location/viewing-points or tnb settings to explicitly pass in a path for the camera to follow. Using something like ``--location-points=trajectory.json`` or  ``--location-points=[[0,0,0], [1,1,1], [2,2,2]]`` will load the points saved in the json file (or read the string as json) and move the camera along a spline connecting them all. The ``location-points`` argument defines the spline the camera will follow, while the ``viewing-points`` argument defines the spline the camera should look at. No roll along the optical axis is permitted in this mode. If ``--tnb`` is set, the camera will use the trajectory's `Frenet-Serret frame <https://en.wikipedia.org/wiki/Frenet%E2%80%93Serret_formulas>`_ to orient itself. Otherwise, the camera will point towards the spline defined by the ``--viewing-locations`` argument.   

.. warning::

    The TNB frame is glitchy, and will likely soon be deprecated. Using Blender to define complex camera motion is preferable. 



Using Blender
-------------

You can use a blender-defined camera animation too. To do this, first animate the camera in blender, there's a few ways to do this, namely:

* Make the camera follow a path 
* Directly keyframe the camera  

We won't go into details as to how to do this, there's an abundance of blender tutorials already. You can look at `this tutorial <https://www.youtube.com/watch?v=a7qyW1G350g>`_ or `this one <https://www.youtube.com/watch?v=K02hlKyoWNI>`_ for some examples.

Then, make sure the camera is not freed from its constraints/parents and keyframes by setting ``--no-unbind-camera`` and enabling ``--use-animations`` (enabled by default). You'll likely want to also specify at which frame to start/stop capture and you can do this with the ``frame-start`` and ``frame-step`` options as well as define the number of frames to capture with ``num-frames`` or the sequence end with ``frame-end``. Finally, you can also slow down the animation using the ``keyframe-multiplier`` argument.
