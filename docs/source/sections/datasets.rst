Data Format and Loading
=======================

Data Schemas
------------

Every dataset, or subset thereof, needs to contain a valid ``transforms.json`` file which contains camera parameters, extrinsics, and among other things, the paths to the data. This format is meant to be a superset of `NERF-like datasets <https://docs.nerf.studio/quickstart/data_conventions.html>`_ and should be familiar to a lot of users. Currently they come in two formats, the ``NPY`` format and the ``IMG`` format. CLI tools are provided to convert between the two (where applicable).

|

The ``IMG`` Format
^^^^^^^^^^^^^^^^^^

This format can be see in `the example below <Folder Structure_>`_ in the ``rgb`` dataset. Here, the raw data is simply stored as a collection of images in ``frames/``. The transforms file adheres to, at a minimum, the following schema::

    IMG_SCHEMA = {
        "fl_x": Focal length in X
        "fl_y": Focal length in Y
        "cx": Center of optical axis in pixel coordinates in X
        "cy": Center of optical axis in pixel coordinates in Y
        "w": Sensor width in pixels
        "h": Sensor height in pixels
        "c": Number of output channels (i.e: RGBA = 4)
        "frames": List of frames where each frame follows IMG_FRAME_SCHEMA
    }

    IMG_FRAME_SCHEMA = {
        "transform_matrix": 4x4 transform matrix
        "file_path": Path to color image, relative to parent dir
    }

|


.. _the-npy-format:

The ``NPY`` Format
^^^^^^^^^^^^^^^^^^

Here, we instead store the data as a ``.npy`` file which has several benefits:

- Numpy arrays in this format can be memory mapped, enabling us to manipulate larger-than-RAM arrays efficiently and easily. This is particularly important when sampling pixels as we do not need to read and decode a whole image to sample a single pixel.

- We are not constrained by an image codec, meaning we can for instance use higher precision floats (although EXRs support this), losslessly save data, and in the case of binary valued SPC data, we can bitpack it leading to larger-than-PNG compression levels.

- Dataset integrity is easier to check as it's just an npy file and it's accompanying transforms file as opposed to thousands of image files.

...and a few drawbacks:

- Not directly compatible with existing tooling, although reading from an npy file is easy.

- Can create very large single files.

The schema is very similar except now there's a single ``file_path``, as well as a few additional keys detailing if the data is bitpacked::

    NPY_SCHEMA = {
        "fl_x", "fl_y", "cx", "cy", "h", "w", "c": Same as in IMG_SCHEMA
        "bitpack": Whether npy file is bitpacked
        "bitpack_dim": Bitpacked axis
        "file_path": Path to npy file containing color images
        "frames": List of frames where each frame follows NPY_FRAME_SCHEMA
    }

    NPY_FRAME_SCHEMA = {
        "transform_matrix": 4x4 transform matrix
    }

|

Folder Structure
----------------

Datasets of a single scene will have the following structure::

    SCENE-NAME
    ├── renders
    │   ├── train
    │   │   ├── frames.npy
    │   │   └── transforms.json
    │   ├── test/
    │   └── val/
    ├── interpolated/
    ├── binary
    │   ├── train
    │   │   ├── frames.npy
    │   │   └── transforms.json
    │   ├── test/
    │   └── val/
    └── rgb
        ├── train
        │   ├── frames/
        │   │   ├── frame_000000.png
        │   │   ├── frame_000001.png
        │   │   └── ...
        │   ├── preview.mp4
        │   └── transforms.json
        ├── test/
        └── val/

.. note:: Note: Here we are only showing the training sets for brevity. Testing and validations sets will be similar in layout.

Here there are four datasets, the simulated ground truth dataset (``renders/``), it's interpolated counterpart (``interpolated/``), emulated SPC dataset (``binary/``), emulated RGB (``rgb/``). While the interpolated dataset is needed to create the last two, it is usually not needed and hence typically deleted after the other datasets have been created.

A *full* dataset refers to a folder containing a train/test/val folder, it is uniquely identified by the path of the parent folder, for example ``SCENE-NAME/binary``. More loosely speaking, a (not-full) dataset might not contain different subsets and is identified by it's path (i.e: ``SCENE-NAME/rgb/train``) or the path of it's transform file directly.

|

Data Loading
------------

Utilities for easily creating, and efficiently iterating over datasets can be found in :mod:`visionsim.dataset`. Here's how to use pytorch's dataloader:

.. code-block:: python

    from torch.utils.data import DataLoader
    from visionsim.dataset import Dataset, default_collate 

    src_dataset = Dataset.from_path(input_dir)
    loader = DataLoader(src_dataset, batch_size=256, num_workers=8, collate_fn=default_collate)

    for idxs, data, poses in loader:
        ...



