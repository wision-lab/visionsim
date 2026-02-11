Data Conventions
================

Coordinate Conventions
----------------------

The camera poses stored in ``transforms.json`` files follow Blender/OpenGL/NeRF camera conventions where +X is right, +Y is up, and +Z is pointing back and away from the camera, making the camera's viewing direction -Z. Another common convention is the OpenCV/COLMAP convention, which uses the same X axis, with +Y going down and +Z going into the scene.   

.. figure:: ../_static/camera-conventions.png
   :width: 85% 

   Camera coordinate conventions (`source <https://medium.com/@christophkrautz/what-are-the-coordinates-225f1ec0dd78>`_)
