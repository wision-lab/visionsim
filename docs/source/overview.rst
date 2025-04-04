Overview
========

This framework is made up of four layers and accessible as both a CLI and library. These layers are as follows:

- **World Simulation:** Using existing high-quality assets and rendering engines render ground truth RGB images, depth maps, segmentation maps, normal maps, etc. 

- **Interpolation:** Using simulated data or data :doc:`from an existing dataset <tutorials/preexisting-ds>` optionally interpolate it to yield higher framerate datasets. This step can greatly help reduce the computational cost of emulating high speed sensors.

- **Sensor Emulation:** Apply realistic sensor modeling to the ground truth data to emulate different sensor modalities such as single photon cameras (both passive and active), event cameras and IMUs. 

- **Data Format and Loading:** Finally, all this data must be stored alongside all applicable metadata in a way that enables easy iteration and random access which is crucial for any deep learning applications.

.. TODO: Add examples for these use-cases

These layers can be used independently and as needed, making them very flexible. For instance, a VFX artist can use the world simulation layer to render out animations faster than what Blender can do by itself, or a computer vision researcher can uplift existing datasets, adding new sensor modalities to them by simply skipping the first step.
 