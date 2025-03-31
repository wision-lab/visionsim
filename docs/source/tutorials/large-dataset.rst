Large Scale Datasets 
====================

In this tutorial we will build up a CLI tool to create a large dataset from many blender scenes. Specifically, for each scene we will uniformly sample multiple sub-trajectories (making sure none overlap) and, for each, we'll render the ground truth RGB, depth maps, etc. 

.. note:: This example requires extra dependencies, notably `tyro <https://brentyi.github.io/tyro/>`_ for CLI creation. 

Here, we assume each scene is setup correctly and has an animation range of [1-600]. The scenes used for this example can be `found here <https://drive.google.com/drive/folders/1gRxhL3rbGDTfgKytre8WkbBu-QDJFy15?usp=sharing>`_. 

For clarity, we'll refer to a single blend-file as a scene, and a sequence will refer to a rendered portion of a scene. So if we use ``sequences_per_scene=10`` and we're rendering from 20 scenes, we will have 200 sequences which will be saved roughly like so::
    
    DATSETS-DIR
    └── renders
        ├── SCENE-NAME
        │   ├── SEQUENCE-ID
        │   │   ├── frames/
        │   │   ├── depths/
        │   │   ├── normals/
        │   │   ├── segmentations/
        │   │   ├── flows/
        │   │   └── transforms.json
        │   ├── SEQUENCE-ID/...
        │   └── ...
        └── SCENE-NAME/...  

.. seealso:: For more about the dataset schema see :doc:`../sections/datasets`.

.. admonition:: TODO

   This example is currently missing interpolation or sensor emulation, and will be extended soon.

|

To enable easy configuration and CLI parsing, we create a render configuration class which stores all important parameters such as render device, dimensions, and types of ground truth to use:

.. literalinclude:: ../../../scripts/mkdataset.py
   :pyobject: RenderConfig

| 

We can then define a render function which will be called for each sequence. This function is very similar to the :func:`render-animation <visionsim.tasks.blender.render_animation>` CLI, except it can be used with :meth:`BlenderClients.pool <visionsim.simulate.blender.BlenderClients.pool>` which enables us to render many sequences at the same time:

.. literalinclude:: ../../../scripts/mkdataset.py
   :pyobject: render

|

Finally, putting it all together:

.. literalinclude:: ../../../scripts/mkdataset.py
   :pyobject: create_datasets

This CLI can be used, for instance, like so::

   CUDA_VISIBLE_DEVICES=0 python scripts/mkdataset.py create-datasets \
      --scenes-dir=scenes/ --datasets-dir=datasets/ --sequences-per-scene=1 \ 
      --render-config.width=800 --render-config.height=800 \
      --render-config.depths --render-config.normals \
      --render-config.flows --render-config.segmentations \ 
      --render-config.keyframe-multiplier=2.0 --render-config.jobs=5 

For brevity, there's a few things that have been omitted in this tutorial, for the full source, see ``scripts/mkdataset.py``.
