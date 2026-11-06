Architecture
============

We use `Blender <https://www.blender.org/>`_ to generate ground truth intensity frames at any arbitrary frame rate. To enable scalable simulations we employ a client/server architecture that uses remote procedure calls (RPCs) as the communication medium. A few minimal dependencies are injected within Blender's runtime which enables it to run as a server, listening for requests to set up scenes, move around cameras, render data, etc. Specifically, doing so from within Blender's runtime enables us to access all its features through the ``bpy`` package and expose them via RPCs. 

To orchestrate many rendering servers, each spawned server reserves its own TCP port and the client connects to it directly, checking readiness on that port before rendering starts. Servers started outside that path can still be found through an optional rpyc registry that listens for UDP broadcasts, which is what :meth:`BlenderServer.discover <visionsim.simulate.blender.BlenderServer.discover>` and :meth:`BlenderClient.auto_connect <visionsim.simulate.blender.BlenderClient.auto_connect>` use; spawning reuses an existing registry rather than requiring one. Servers manage their own logging and exception handling, and, in the event of an unexpected crash, ensure the dataset is not corrupted and all process resources are freed. 

.. figure:: ../../_static/blender/rpc-architecture.svg
   :width: 65%
   :align: center
   :class: only-light

.. figure:: ../../_static/blender/rpc-architecture-dark.svg
   :width: 65%
   :align: center
   :class: only-dark

   Client/Server Rendering Architecture

This client/server architecture can drastically cut down on processing time and enables the use of a distributed set of machines with heterogeneous hardware. Even when worker processes are on the same machine, as is typically the case when using :meth:`BlenderClients.spawn <visionsim.simulate.blender.BlenderClients.spawn>` or :meth:`BlenderClients.pool <visionsim.simulate.blender.BlenderClients.pool>`, render times can be shorter as the system's resources will be better utilized. See the :doc:`Render API <../../sections/blender/rendering>` for examples of parallel distributed simulation.

|

.. seealso::

    - :doc:`../../tutorials/extending` for subclassing ``BlenderService``.


