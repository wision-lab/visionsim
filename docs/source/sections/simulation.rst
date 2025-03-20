World Simulation
================

Architecture
------------

We use `Blender <https://www.blender.org/>`_ to generate ground truth intensity frames at any arbitrary frame rate. To enable scalable simulations we employ a client/server architecture that uses remote procedure calls (RPCs) as the communication medium. A few minimal dependencies are injected within Blender's runtime which enables it to run as a server, listening for requests to set up scenes, move around cameras, render data, etc. Specifically, doing so from within Blender's runtime enables us to access all its features through the ``bpy`` package and expose them via RPCs. 

To orchestrate many rendering servers a registry server is spawned and listens for UDP-based broadcast messages from rendered servers that are available. This rendezvous server communicates with the main client to inform it about available workers, enabling the user to connect to the server instances and distribute work amongst the worker pool, or even use multiple render servers to render out a single animation. Servers manage their own logging and exception handling, and, in the event of some unexpected crash, we ensure that the dataset will not get corrupted and all process resources are freed. 

.. figure:: ../_static/rpc-architecture.svg
   :width: 65% 

   Client/Server Rendering Architecture

This client/server architecture can drastically cut down on processing time and enables the use of a distributed set of machines with heterogeneous hardware. Even when worker processes are on the same machine, as is typically the case when using :meth:`BlenderClients.spawn <spsim.simulate.blender.BlenderClients.spawn>` or :meth:`BlenderClients.pool <spsim.simulate.blender.BlenderClients.pool>`, render times can be shorter as the system's resources will be better utilized. See the tutorials for examples of :doc:`parallel <../tutorials/parallel-sim>` and :doc:`distributed <../tutorials/distributed-sim>` simulation.

|

Keyframe Stretching
-------------------

An artist designing a 3D animation, does so by defining a set of *keyframes*. Each one holds a timestamp and set of attributes that need to be set to particular values at the timestamp. Upon rendering, Blender will interpolate attribute values between these different keyframes. For instance, we could say that at frame #1 the camera's focal length is 50mm and then at frame #20 it will be 100mm, then, if we render frame #10 we should expect a focal length of 75mm.   

Keyframes are tied to frame numbers not relative animation time, so while this works well when the artist knows the framerate of the final render, if they animate for 30fps but render it at 120fps, the animation will be four times too fast. 

This is where keyframe stretching comes in, it allows for a decoupling between rendering framerate and animation framerate. The ``keyframe-multiplier`` option of ``blender.render-animation`` controls this stretch factor. In the above example, we would set ``--keyframe-multiplier=4.0`` to get the desired animation, or we could reuse existing animations and render them at single photon camera framerates using a higher multiplier.  
 
|

Setup a Trajectory
------------------


Programmatically
^^^^^^^^^^^^^^^^

You can use the location/viewing-points or tnb settings to explicitly pass in a path for the camera to follow. Using something like ``--location-points=trajectory.json`` or  ``--location-points=[[0,0,0], [1,1,1], [2,2,2]]`` will load the points saved in the json file (or read the string as json) and move the camera along a spline connecting them all. The ``location-points`` argument defines the spline the camera will follow, while the ``viewing-points`` argument defines the spline the camera should look at. No roll along the optical axis is permitted in this mode. If ``--tnb`` is set, the camera will use the trajectory's `Frenet-Serret frame <https://en.wikipedia.org/wiki/Frenet%E2%80%93Serret_formulas>`_ to orient itself. Otherwise, the camera will point towards the spline defined by the ``--viewing-locations`` argument.   

.. warning::

    The TNB frame is deprecated. Using Blender to define complex camera motion is preferable. 



Using Blender's UI
^^^^^^^^^^^^^^^^^^

You can use a blender-defined camera animation too. To do this, first animate the camera in blender, there's a few ways to do this, namely:

* Make the camera follow a path 
* Directly keyframe the camera  

We won't go into details as to how to do this, there's an abundance of blender tutorials already. You can look at `this tutorial <https://www.youtube.com/watch?v=a7qyW1G350g>`_ or `this one <https://www.youtube.com/watch?v=K02hlKyoWNI>`_ for some examples.

Then, make sure the camera is not freed from its constraints/parents and keyframes by setting ``--no-unbind-camera`` and enabling ``--use-animations`` (enabled by default). You'll likely want to also specify at which frame to start/stop capture and you can do this with the ``frame-start`` and ``frame-step`` options as well as define the number of frames to capture with ``num-frames`` or the sequence end with ``frame-end``. Finally, you can also slow down the animation using the ``keyframe-multiplier`` argument.
