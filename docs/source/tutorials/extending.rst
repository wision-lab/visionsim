Extending the Simulator 
=======================

The :class:`BlenderService <visionsim.simulate.blender.BlenderService>` API, as typically accessed through a :class:`BlenderClient <visionsim.simulate.blender.BlenderClient>` instance, handles most rendering tasks. When you need finer control, subclass ``BlenderService`` to reach Blender's internal ``bpy`` API. 

Here we create a custom ``ExtendedService`` which allows for axis-aligned bounding box (AABB) calculations, and listing out any missing textures: 

.. literalinclude:: ../../../examples/blender/extended_service.py 

Pass the service's location to :meth:`BlenderClient.spawn <visionsim.simulate.blender.BlenderClient.spawn>` as either a dotted ``module:ClassName`` path or a ``/path/to/service.py:ClassName`` file. The spawned Blender gets the same command as the default service, ``--python-use-system-env`` included:

.. code-block:: python 

    with BlenderClient.spawn(service="my_package.services:ExtendedService", timeout=30) as client:
        client.initialize("cube.blend", "renders/")
        print(client.scene_aabb())

.. note::

    A dotted path is imported inside Blender, which first executes the ``__init__`` of every package on the way to the module. The blender runtime might not have your project's dependencies installed and would otherwise fail, even when the service itself needs none of them. The file form is run as a script and never imports a package, exactly like handing the file to ``blender --python``, so prefer it for a service that lives inside one. Give it an absolute path: the spawned Blender inherits the spawning process's working directory.

Pass ``factory_startup=True`` when the render must not depend on the machine's Blender preferences and startup file, and ``env=...`` to modify the environment the blender subprocess runs in.

The same arguments are available on :meth:`BlenderServer.spawn <visionsim.simulate.blender.BlenderServer.spawn>`, :meth:`BlenderClients.spawn <visionsim.simulate.blender.BlenderClients.spawn>` and :meth:`BlenderClients.pool <visionsim.simulate.blender.BlenderClients.pool>` if you need to manage the process yourself or spawn several. Omitted, ``service`` defaults to the base :class:`BlenderService <visionsim.simulate.blender.BlenderService>`.

To connect to a service you started yourself, launch it manually and use :meth:`BlenderClient.auto_connect <visionsim.simulate.blender.BlenderClient.auto_connect>`: 

.. code-block:: bash

    blender --background --python-use-system-env --python examples/blender/extended_service.py

.. code-block:: python 

    with BlenderClient.auto_connect(timeout=30) as client:
        client.initialize("cube.blend", "renders/")
        print(client.scene_aabb())

``examples/blender/imu_preview_service.py`` and ``examples/blender/imu_preview.py`` are a
fuller example of the same split: the service exposes a method that draws camera
trajectories into the loaded scene, and the script spawns it through the same ``service``
file target and drives it as a client. The :doc:`IMU tutorial <imu>` walks through what it
renders.

|

.. seealso::

    - :doc:`../development` for the editable install and the test suite.
    - :doc:`../sections/blender/architecture` for how the client/server split works.

