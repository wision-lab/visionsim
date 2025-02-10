
Welcome to SpSim!
=================================

Spsim is a python library and CLI that provide tools to simulate high speed sensor data as captured by single photon cameras or conventional RGB cameras.  


What's a single photon camera?
------------------------------

Single photon cameras (SPCs) are an emerging class of sensors that offer extreme sensitivity and are manufactured at scale using existing CMOS techniques. These sensors can not only detect individual photons but also time them with picosecond accuracy, enabling applications that are `not possible with traditional sensors <https://wisionlab.com/project/burst-vision-single-photon/>`_. By sensing individual photons, SPCs provide us with the most fine-grained visual information possible and provides a lot of flexibility to the authors of downstream inference tasks.

However, today there exists no easy way to simulate SPCs under a wide range of settings efficiently and at scale, which severely impedes rapid development of new techniques and limiting their adoption. This work aims to provide such tools to the community in order to facilitate the development of new SPC-based techniques and provide a standardized way to do so.


How does SpSim work?
--------------------

The single photon simulation engine today is made up of four layers and accessible as both a CLI and library. These layers are as follows:

- Ground Truth Simulation Layer: Generate ground truth data using existing high-quality rendering engines via either custom plugins or render scripts. The output of this layer will consist of clean, blur and noise free, RGB images, depth maps, as well as segmentation maps, normal maps, and other data modalities where applicable. These should be stored in a common format described below, or something that the next later can ingest.

- Interpolation Layer: The simulated data from layer#1, or in fact, data from existing datasets such as the `XVFI <https://github.com/JihyongOh/XVFI>`_, can be interpolated to yield higher framerate datasets. The reasoning for this layer is two fold: i) Simulations can be slow and expensive to run, especially when the single photon sensors run at hundreds of thousands of frames per second, filling the gaps between adjacent frames is fast and relatively cheap to compute. Similarly, existing datasets can be uplifted to single photon regimes using interpolation techniques. ii) While interpolation can produce some artifacts, these are minor when the frames we interpolate between are close and are further muddled as we use the interpolated frames to emulate different camera modalities.

- Emulation Layer: At this stage we have access to high speed interpolated data which we'll need to further process into the desired modality. We'll refer to this process as `emulation` and reserve the term `simulation` for layer#1. Specifically we are interested in the following:
  
  - Passive Single Photon Cameras: These can be emulated by Bernoulli sampling the interpolated intensity frames.
  - Active Single Photon Cameras: By defining a few added parameters such as laser power, repetition rate, wavelength, etc, we can use the interpolated depth, normals and intensity to construct arrays of histograms with preset (non negligible) field-of-views that will have realistic blur, depth edges, etc.
  - Conventional RGB Cameras: Many interpolated intensity frames can be averaged together to create motion blur. We then add Gaussian read noise and quantize to a certain bit-depth. More complex effects such as tonemapping, optical aberrations, etc, can be performed here as well.
  - Potentially others to come!

- Data Storage Layer: Finally, all this data must be stored alongside all applicable metadata in a way that enables easy iteration and random access which is crucial for any deep learning applications. This data format is further detailed below.



.. toctree::
   :hidden:

   quick-start
   simulation
   interpolation
   autocomplete
   development
   spsim

..
   usage/simulation
   usage/emulation 
   usage/datasets
   concepts/architecture
   concepts/imaging-model
      FAQs <faq>
      prior-art
      changelog

    
