Interpolate Your First Dataset with visionsim API
=============================================

Interpolating a dataset using the visionsim API is easy! Read more about interpolation in the :doc:`Interpolation Section <../sections/interpolation>`.


Load in the dataset
--------------------
First, load in the dataset. The dataset should be a directory consisting of a directory of images to be interpolated (`frames/`) and a `transforms.json` file that associates a transform matrix with each frame.

.. literalinclude:: ../../../examples/interpolation/interpolate_simple_rife.py
  :language: python
  :lines: 5-14
  :dedent:

.. note::
    To generate frames and a `transforms.json` file see :doc:`../quick-start` or download the `Lego100/` dataset


Interpolate the poses and frames
--------------------------------
Next, interpolate the transform matrices between poses by using the `interpolate_poses()` function.

.. literalinclude:: ../../../examples/interpolation/interpolate_simple_rife.py
  :language: python
  :lines: 18
  :dedent:

Similarly, interpolate between image frames using the `interpolate_frames()` function.

.. literalinclude:: ../../../examples/interpolation/interpolate_simple_rife.py
  :language: python
  :lines: 22
  :dedent:

.. note::
    We currently only support image interpolation using rife


Save interpolated poses and frames
----------------------------------
Finally, save the newly interpolated poses and frames to a new directory to be used later using the `poses_and_frames_to_json()` function.

.. literalinclude:: ../../../examples/interpolation/interpolate_simple_rife.py
  :language: python
  :lines: 26
  :dedent: