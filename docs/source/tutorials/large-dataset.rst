Large Scale Datasets 
====================

In this tutorial we will build up a CLI tool to create a large dataset from many blender scenes. Specifically, for each scene we will uniformly sample multiple sub-trajectories (making sure none overlap) and, for each, we'll render the ground truth RGB, depth maps, etc. The final product will look something like this: 

.. video:: https://pages.cs.wisc.edu/~sjungerman/visionsim/gt-annotations.mp4
   :loop:
   :width: 100%
   :align: center

.. note:: For brevity, there's a few things that have been omitted in this tutorial, for the full source, see ``scripts/mkdataset.py``. Notably, extra dependencies may be needed. 

Here, we assume each scene is setup correctly and has an animation range of [1-600]. The scenes used for this example can be `found here <https://drive.google.com/drive/folders/1gRxhL3rbGDTfgKytre8WkbBu-QDJFy15?usp=sharing>`_, and the final dataset can be `found here <https://github.com/WISION-Lab/datasets?tab=readme-ov-file#visionsim-50-dataset-pre-release>`_.

For clarity, we'll refer to a single blend-file as a scene, and a sequence will refer to a rendered portion of a scene. So if we use ``sequences_per_scene=10`` and we're rendering from 20 scenes, we will have 200 sequences which will be saved roughly like so::
    
    DATASETS-DIR
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

To enable easy configuration and CLI parsing, we re-use the render configuration class used in the :func:`render-animation <visionsim.cli.blender.render_animation>` CLI which stores all important parameters such as render device, dimensions, and types of ground truth to use:

.. literalinclude:: ../../../visionsim/simulate/config.py
   :pyobject: RenderConfig

| 

We can then reuse the same render-job as :func:`render-animation <visionsim.cli.blender.render_animation>`, except we'll use it in conjunction with a :meth:`BlenderClients.pool <visionsim.simulation.blender.BlenderClients.pool>` in order to render multiple scenes at once (as opposed to a single scene being rendered with multiple jobs). Putting it all together we have:

.. literalinclude:: ../../../scripts/mkdataset.py
   :pyobject: create_datasets

This CLI can be used, for instance, like so::

   CUDA_VISIBLE_DEVICES=0 python scripts/mkdataset.py create-datasets \
      --scenes-dir=scenes/ --datasets-dir=datasets/ --sequences-per-scene=1 \ 
      --render-config.width=800 --render-config.height=800 \
      --render-config.depths --render-config.normals \
      --render-config.flows --render-config.segmentations \ 
      --render-config.keyframe-multiplier=2.0 --render-config.jobs=5 

Which will render the dataset show above at a framerate of 100fps (original fps of 50 time 2x keyframe multiplier).
