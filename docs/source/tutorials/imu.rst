Emulating IMU Measurements
==========================

This tutorial covers inertial measurement unit (IMU) emulation: turning a sequence of ground truth camera poses into accelerometer and gyroscope readings, and integrating those readings back into a pose estimate. See :doc:`the sensor section <../sections/sensors/imu>` for the measurement model, the meaning of each noise term and the drift they cause over time.

|

Prerequisites
-------------

You need ground truth poses, either a ``transforms.json`` file or a ``transforms.db`` database. Any of the rendering guides produces one, and the emulator does not need the rendered frames themselves.

|

Emulating Measurements
----------------------

Run the emulator over a dataset:

.. code-block:: bash

    visionsim emulate.imu --input-dir=transforms.db --output-file=imu.csv --dt=0.02

The ``--input-dir`` option takes either file, and ``--dt`` is the sampling period in seconds, here 50 Hz. It has to match the real spacing between consecutive poses, since the emulator differentiates the pose spline at those times instead of resampling it. The remaining flags set ``--gravity`` (in the world frame, in m/s^2), the constant parts of the biases with ``--init-bias-acc`` and ``--init-bias-gyro``, the noise parameters ``--std-acc``, ``--std-gyro``, ``--std-bias-acc`` and ``--std-bias-gyro``, and ``--seed`` for reproducible noise.

The output is a CSV table with one row per sample:

.. code-block:: text

    t,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,bias_ax,bias_ay,bias_az,bias_gx,bias_gy,bias_gz
    0.0,-0.2252,9.3807,-4.9132,0.0072,-0.0048,-0.0054,0.0,0.0,0.0,0.0,0.0,0.0
    0.02,-0.2171,9.2758,-4.9026,0.0052,-0.0115,0.0025,1.8e-06,-3.4e-06,5.5e-06,1.2e-06,6.1e-06,4.3e-06

``t`` is the timestamp in seconds, ``acc_*`` the measured acceleration in m/s^2, ``gyro_*`` the angular velocity in rad/s and ``bias_*`` the current bias on each axis. All three are expressed in the camera frame. Both the bias, which starts at its initial value and takes a random walk from there, and the white noise are added to the true readings before they are written out. Omit ``--output-file`` to print the table to stdout instead.

|

Estimating Pose
---------------

:func:`imu_integration <visionsim.emulate.imu.imu_integration>` turns readings back into a trajectory with forward Euler integration:

.. code-block:: python

    import numpy as np
    from visionsim.emulate.imu import imu_integration

    data = np.loadtxt("imu.csv", skiprows=1, delimiter=",")
    acc_reading, gyro_reading = data[:, 1:4], data[:, 4:7]

    estimated = np.array(
        list(
            imu_integration(
                acc_pos=acc_reading,
                vel_ang=gyro_reading,
                pose_init=pose_init,    # first ground truth pose, in world coordinates
                vel_init=vel_init,      # initial velocity, in world coordinates
                dt=0.02,
            )
        )
    )

It yields the pose at the start of each step, so the estimate lines up with the readings it consumed. Gravity is applied here rather than in the emulator, so it has to be passed again when the emulator was run with a non-default value. Both the noise the emulator added and the bias random walk survive into the integration, so the estimate starts on the ground truth and then drifts away from it at a rate set by the noise parameters.

|

Coordinate Frames
-----------------

The pose matrices in ``transforms.json`` and ``transforms.db`` come from Blender, and they map the OpenGL camera frame (+X right, +Y up, +Z back) straight into Blender's world axes (+Z up). They can therefore not be read as a plain rotation and translation: the rotation mixes the camera convention with the world one. :func:`emulate_imu <visionsim.emulate.imu.emulate_imu>` works entirely in Blender's convention instead, so the CLI runs every matrix through :func:`camcoord_opengl_to_blender <visionsim.utils.pose.camcoord_opengl_to_blender>` before emulating. Pass ``--no-convert-coords`` to skip that conversion and emulate the matrices exactly as they are. When you drive the API directly, do the conversion yourself.

:func:`imu_integration <visionsim.emulate.imu.imu_integration>` instead works in the world frame: ``pose_init`` and ``vel_init`` are both given in world coordinates, and the initial velocity is rotated into the camera frame internally. Mixing the two conventions, for instance by handing the emulator poses that were already converted, makes the estimate leave the ground truth immediately.

|

Ground Truth vs Integrated
--------------------------

The heart of the example is the emulate-then-integrate loop:

.. literalinclude:: ../../../examples/sensors/emulate_imu/sim_and_integrate.py
   :language: python
   :pyobject: sim_and_integrate
   :dedent:

The example loads a dataset, interpolates its poses up to an IMU-like rate, emulates a batch of noisy measurements, integrates each batch from the same initial pose and velocity, and plots the results against the ground truth. Every run starts on the ground truth and fans out as noise accumulates. The noise is scaled by ``NOISE_MULT``, which applies to all four noise parameters at once, so raising it makes the divergence visible over a shorter trajectory; ``NUM_SAMPLES`` sets how many estimates are drawn and ``PLANE`` which two axes they are plotted on. The paths, noise and plot settings all live in constants at the top of the file.

Run it from a checkout with the quick-start dataset in the working directory:

.. code-block:: bash

    python examples/sensors/emulate_imu/sim_and_integrate.py

|

Trajectory Preview
------------------

The same integration drawn in three dimensions: ``examples/blender/imu_preview.py``
integrates five trajectories from the ultramodern ground truth, draws each one as a
frustum-and-trail marker next to the ground truth, and renders the room around them.

.. video:: ../_static/sensors/imu-preview.mp4
   :loop:
   :autoplay:
   :nocontrols:
   :playsinline:
   :align: center
   :caption: The ground-truth camera in blue and five estimates in grey; each estimate starts on
       the ground truth and fans out as the bias random walk accumulates.

The script runs in the ordinary interpreter, because the integration needs scipy, and
starts ``examples/blender/imu_preview_service.py`` in its own Blender process: a
:class:`BlenderService <visionsim.simulate.blender.BlenderService>` subclass that draws
the trajectories as geometry, renders the room once for the backdrop, then renders
only the markers over a transparent film for the script to composite. Run it from a
checkout with the quick-start dataset in the working directory:

.. code-block:: bash

    python examples/blender/imu_preview.py

The marker geometry is an asset, not code: ``examples/blender/mesh_camera.json`` holds
the geometry nodes group as a Tree Clipper export, which the service imports and
applies to a cube. Reading it needs the ``tree-clipper`` package in Blender's own
python (``pip install tree-clipper``) and Blender 5.0 or newer, because Tree Clipper
uses node types that only 5.x has.

|

.. seealso::

    - :doc:`../sections/datasets` for the ``transforms.db``/``transforms.json`` the emulator reads.
