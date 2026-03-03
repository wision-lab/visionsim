Types of Ground Truth 
=====================

Many types of ground truth annotations can be generated, some during the main rendering pass, and some as a post-processing step. Currently the following types are supported:

- **Composite Frames**: Any existing composition workflow will be kept and optionally saved as composite frames. 
- **RGB Frames:** High quality, blur and noise free RGB images. These *have* to be generated due to a Blender limitation. If they are not needed, you can set the :meth:`max number of samples to 1 <visionsim.simulate.blender.BlenderService.exposed_cycles_settings>` to minimize the overhead associated with frame generation.   
- **Depth Maps:** Scene distance for each pixel, a.k.a Z-buffers. These are depths inherit the units set within Blender (in Properties > Scene > Units) and are saved as full-precision floating point numbers in a single-channel EXR file. 
- **Normal Maps:** Surface normal vectors for each pixel. Each normal vector is of unit length and expressed in the camera's coordinate frame, with +X pointing to the right, +Y pointing up and +Z pointing into the camera (since the camera looks in the -Z direction, using the OpenGL/Blender coordinate system). These vectors are saved as floats in a 3-channel EXR file. 
- **Optical Flow:** 2D flow vector tracking the motion of each pixel between the current and next/previous frame. The former is dubbed forward flow, and the latter backwards flow. Both of these are saved as a 4-channel (RGBA) floating point EXR file, with the forward flow packed as RG and backwards flow as BA. 
- **Segmentation Maps:** Assign a unique object ID to each pixel. These are saved as single-channel integer EXR file. Multiple objects can be assigned the same ID if they are instances or made with modifiers (i.e: array modifier).
- **Material Maps:** Assign a unique material ID to each pixel. Similar to segmentation maps, these are saved as single-channel integer EXR files. Multiple objects can be assigned the same material ID if they share the same material.
- **Diffuse Passes:** Light reflected off surfaces in a non-directional way. For CYCLES, this includes Direct, Indirect, and Color passes. For EEVEE, this includes Light and Color passes. These are saved as 3-channel EXR files.
- **Specular Passes:** Light reflected off surfaces in a directional/shiny way. For CYCLES, this refers to Glossy Direct, Indirect, and Color passes. For EEVEE, this includes Specular Light and Color passes. These are saved as 3-channel EXR files.
- **Point Maps:** World-space positions (X, Y, Z) for each pixel. These are saved as 3-channel floating point EXR files.
- **Camera Intrinsics & Extrinsics:** Camera intrinsic parameters such as focal lengths, camera model, distortion coefficients, principal point, and camera positions (extrinsics) are automatically recorded for every rendered frame and saved in a ``transforms.json`` file.

.. admonition:: Note

   The compositor nodes shown here might not match those that VisionSIM generates as these might depend on the Blender version you are using. To see what nodes are actually being used, consider :meth:`saving the blend file <visionsim.simulate.blender.BlenderService.exposed_save_file>` and inspecting it manually. 

.. warning::
    
   Not all ground truth annotations are always available. Notably, while they are all compatible with CYCLES, they are not guaranteed to work with other rendering engines.

|

Composite Frames
----------------

.. image:: ../../_static/blender/nodes/Composites-TopLevel.png
   :align: right 
   :width: 50% 
   :class: only-light

.. image:: ../../_static/blender/nodes/Composites-TopLevel-dark.png
   :align: right 
   :width: 50% 
   :class: only-dark

Composite outputs correspond to the group output node of the compositor node tree. These are saved by Blender's output settings, and not a file output node (as opposed to the other ground truth types).

|

RGB Frames
----------

.. image:: ../../_static/blender/nodes/RGB-TopLevel.png
   :align: right 
   :width: 50% 
   :class: only-light

.. image:: ../../_static/blender/nodes/RGB-TopLevel-dark.png
   :align: right 
   :width: 50% 
   :class: only-dark

To render out RGB images, at a minimum, the "Image" render layer needs to be connected to the composite output node as shown on the right. These nodes are automatically added and enabled if they aren't already in the blend file. However, if a more complex compositing setup is present, it will not be modified.   

| 

Depth Maps
----------

.. image:: ../../_static/blender/nodes/Depth-TopLevel.png
   :align: right 
   :width: 50% 
   :class: only-light

.. image:: ../../_static/blender/nodes/Depth-TopLevel-dark.png
   :align: right 
   :width: 50% 
   :class: only-dark

To generate ground truth depth maps, a Z-buffer render layer is added as well as the following compositor nodes. The depth map is saved directly as an EXR file, and if ``debug`` is enabled, it is normalized and saved as a grayscale PNG too. 

See :meth:`include_depths <visionsim.simulate.blender.BlenderService.exposed_include_depths>` for more. 

.. warning:: It is recommended to not use motion blur or depth of field when using depth maps. 

|

Normal Maps
-----------

.. image:: ../../_static/blender/nodes/Normal-TopLevel.png
   :align: right 
   :width: 50% 
   :class: only-light

.. image:: ../../_static/blender/nodes/Normal-TopLevel-dark.png
   :align: right 
   :width: 50% 
   :class: only-dark

To render normal maps, the following compositor nodes are added. However, Blender's surface normal render layer outputs normals that are in world coordinates, yet normals maps are typically expressed in the camera coordinate frame. 

This conversion is ensured by the ``Normal Debug`` node, seen below, which takes the normals in world coordinates and maps them to the camera's coordinate frame. This node group outputs both the raw normals in camera space, and a colorized version for easy debugging, which maps XYZ coordinates to RGB.

As there is no matrix multiply compositor node, three dot product nodes and a combine-XYZ node are used to perform a matrix multiplier. The rows of the camera rotation matrix are automatically updated using `drivers <https://docs.blender.org/manual/en/latest/animation/drivers/index.html>`_ (purple). 

See :meth:`include_normals <visionsim.simulate.blender.BlenderService.exposed_include_normals>` for more. 

.. image:: ../../_static/blender/nodes/NormalDebug.png
   :align: center 
   :class: only-light

.. image:: ../../_static/blender/nodes/NormalDebug-dark.png
   :align: center 
   :class: only-dark

|

Optical Flow
------------ 

.. image:: ../../_static/blender/nodes/Flow-TopLevel.png
   :align: right 
   :width: 50% 
   :class: only-light

.. image:: ../../_static/blender/nodes/Flow-TopLevel-dark.png
   :align: right 
   :width: 50% 
   :class: only-dark

The vector pass adds both forward and backward optical flow. To save these, we add the following compositor nodes, the top branch saves the optical flow directly as an EXR while the bottom branch colorizes it and saves a preview of both the forward and backward flow as PNGs. 

The ``FlowDebug`` group node, seen below, colorizes optical flow by expressing the normal vectors in polar coordinates :math:`(\theta, r)` and using an HSV colormap, with the hue being determined by :math:`\theta` and saturation by a normalized :math:`r`. 

See :meth:`include_flows <visionsim.simulate.blender.BlenderService.exposed_include_flows>` for more. 

.. image:: ../../_static/blender/nodes/FlowDebug.png
   :align: center 
   :class: only-light

.. image:: ../../_static/blender/nodes/FlowDebug-dark.png
   :align: center
   :class: only-dark

|

Segmentation Maps
-----------------

.. image:: ../../_static/blender/nodes/Segmentation-TopLevel.png
   :align: right 
   :width: 50% 
   :class: only-light

.. image:: ../../_static/blender/nodes/Segmentation-TopLevel-dark.png
   :align: right 
   :width: 50% 
   :class: only-dark

To generate segmentation maps, the following compositor nodes are added. The object index pass is enabled, which is directly saved as an EXR, or optionally colorized using the ``ColorizeIndices`` node shown below which assigns a unique color to each object index. 

See :meth:`include_segmentations <visionsim.simulate.blender.BlenderService.exposed_include_segmentations>` for more. 

.. image:: ../../_static/blender/nodes/SegmentationDebug.png
   :align: center 
   :class: only-light

.. image:: ../../_static/blender/nodes/SegmentationDebug-dark.png
   :align: center
   :class: only-dark

.. caution:: The ``From Max`` value of the Map Range node above determines which colors to sample. Internally, upon initialization this value is set to ``len(bpy.data.objects)``, however, if more objects are added after the fact then their colors might coincide. 

| 

Material Maps
-------------

To generate material ID maps, the material index pass is enabled, which is directly saved as an EXR, or optionally colorized using the same ``ColorizeIndices`` node as segmentation maps to assign a unique color to each material index for previewing.

See :meth:`include_materials <visionsim.simulate.blender.BlenderService.exposed_include_materials>` for more.

|

Diffuse Passes
--------------

Diffuse passes capture the light that is reflected uniformly in all directions. When enabled, multiple passes are captured depending on the render engine.

For CYCLES:
- **Diffuse Direct:** Light from sources that hits surfaces directly.
- **Diffuse Indirect:** Light that has bounced off other surfaces before hitting the current surface.
- **Diffuse Color:** The base color of the surfaces.

For EEVEE:
- **Diffuse Light:** The combined direct and indirect diffuse lighting.
- **Diffuse Color:** The base color of the surfaces.

See :meth:`include_diffuse_pass <visionsim.simulate.blender.BlenderService.exposed_include_diffuse_pass>` and the `Blender documentation <https://docs.blender.org/manual/en/latest/render/layers/passes.html#light>`_ for more.

|

Specular Passes
---------------

Specular passes capture the directional reflection of light, often seen as "highlights" or "shininess". When enabled, multiple passes are captured depending on the render engine.

For CYCLES (these are called "Glossy" passes):
- **Glossy Direct:** Directional light from sources.
- **Glossy Indirect:** Directional light from other surfaces (reflections).
- **Glossy Color:** The color of the specular reflection.

For EEVEE:
- **Specular Light:** The combined specular lighting.
- **Specular Color:** The color of the specular reflection.

See :meth:`include_specular_pass <visionsim.simulate.blender.BlenderService.exposed_include_specular_pass>` and the `Blender documentation <https://docs.blender.org/manual/en/latest/render/layers/passes.html#light>`_ for more.

|

Point Maps
----------

A Point Map, also referred to as a position pass, stores the world-space (X, Y, Z) coordinates of each visible surface point in the scene. Unlike depth maps, which only store the distance between the camera and the surface, point maps provide the absolute 3D position of each pixel.

This type of representation is common in modern 3D vision architectures such as VGGT [1]_, which directly predicts world-space positions. This differs from models like DUSt3R [2]_, which typically utilize camera-centric or relative point maps.

When enabled, the raw world-space coordinates are saved as 3-channel EXR files. A colorized preview is also generated by remapping world coordinates to [0, 1] by using the absolute value of the fractional part of each coordinate, where XYZ coordinates are mapped to RGB respectively.

See :meth:`include_points <visionsim.simulate.blender.BlenderService.exposed_include_points>` for more.

|

Camera Intrinsics & Extrinsics
------------------------------

Intrinsics refer to camera parameters such as focal length, width and height, and any optical distortion parameters needed to map a pixel location to a ray in 3D space. Extrinsics, on the other hand, refers to camera pose, both rotation and position. These can be accessed using :meth:`camera_info <visionsim.simulate.blender.BlenderService.exposed_camera_info>` and :meth:`camera_extrinsics <visionsim.simulate.blender.BlenderService.exposed_camera_extrinsics>`, and will be saved to disk when using methods such as :meth:`render_animation <visionsim.simulate.blender.BlenderService.exposed_render_animation>`.

.. seealso:: :doc:`../datasets`

|

.. [1] `VGGT: Visual Geometry Grounded Transformer <https://arxiv.org/abs/2503.11651>`_
.. [2] `DUSt3R: Geometric 3D Vision Made Easy with Unconstrained Image Collections <https://arxiv.org/abs/2312.14132>`_
