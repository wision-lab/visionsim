Extending the Simulator 
=======================

While the :class:`BlenderService <visionsim.simulate.blender.BlenderService>` API, as typically accessed through a :class:`BlenderClient <visionsim.simulate.blender.BlenderClient>` instance, provides a streamlined way to interact with Blender and facilitates most common operations, sometimes more fine-grained control is needed. For this, you can subclass ``BlenderService`` to access Blender's internal ``bpy`` API. 

Here we create a custom ``ExtendedService`` which allows for axis-aligned bounding box (AABB) calculations, and listing out any missing textures: 

.. literalinclude:: ../../../examples/blender/extended_service.py 

Currently, in order to use this new rendering service, the user must spin it up manually (as opposed to using :meth:`BlenderClient.spawn <visionsim.simulate.blender.BlenderClient.spawn>`):

.. code-block:: console 
    
    $ blender --background --python-use-system-env --python examples/blender/extended_service.py

And then connect to that render service, either directly using the appropriate connection settings, or using the :meth:`BlenderClient.auto_connect <visionsim.simulate.blender.BlenderClient.auto_connect>`: 

.. code-block:: python 

    with BlenderClient.auto_connect(timeout=30) as client:
        client.initialize("cube.blend", "renders/")
        print(client.scene_aabb())

