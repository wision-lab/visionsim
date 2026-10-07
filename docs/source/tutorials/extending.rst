Extending the Simulator 
=======================

The :class:`BlenderService <visionsim.simulate.blender.BlenderService>` API, as typically accessed through a :class:`BlenderClient <visionsim.simulate.blender.BlenderClient>` instance, handles most rendering tasks. When you need finer control, subclass ``BlenderService`` to reach Blender's internal ``bpy`` API. 

Here we create a custom ``ExtendedService`` which allows for axis-aligned bounding box (AABB) calculations, and listing out any missing textures: 

.. literalinclude:: ../../../examples/blender/extended_service.py 

Pass its dotted ``module:ClassName`` path to :meth:`BlenderClient.spawn <visionsim.simulate.blender.BlenderClient.spawn>`. The spawned Blender gets the same command as the default service, ``--python-use-system-env`` included:

.. code-block:: python 

    with BlenderClient.spawn(service="my_package.services:ExtendedService", timeout=30) as client:
        client.initialize("cube.blend", "renders/")
        print(client.scene_aabb())

The same ``service`` argument is available on :meth:`BlenderServer.spawn <visionsim.simulate.blender.BlenderServer.spawn>` and :meth:`BlenderClients.spawn <visionsim.simulate.blender.BlenderClients.spawn>` if you need to manage the process yourself or spawn several. Omitted, it defaults to the base :class:`BlenderService <visionsim.simulate.blender.BlenderService>`.

To connect to a service you started yourself, launch it manually and use :meth:`BlenderClient.auto_connect <visionsim.simulate.blender.BlenderClient.auto_connect>`: 

.. code-block:: console 
    
    $ blender --background --python-use-system-env --python examples/blender/extended_service.py

.. code-block:: python 

    with BlenderClient.auto_connect(timeout=30) as client:
        client.initialize("cube.blend", "renders/")
        print(client.scene_aabb())
