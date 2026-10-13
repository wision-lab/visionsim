Inertial Measurement Unit
=========================

.. image:: https://upload.wikimedia.org/wikipedia/commons/7/7f/SparkFun_9DoF-IMU-Breakout_LSM9DS1_13284-03.jpg
   :width: 20% 
   :align: right 

An inertial measurement unit (IMU) is chip that measures acceleration and angular velocity using a accelerometers and gyroscopes, typically one per axis. These devices are widespread, and play a crucial role in tracking the movement of devices as their measurements can be integrated to estimate the current orientation of the sensor. 

However, these suffer from various types of noise which leads to loss of precision over time, in a phenomenon dubbed drift. Carefully characterizing this drift is important as it directly correlates with the sensor's accuracy and with the amount of times IMU-derived position estimates are valid. 

From `Wikipedia <https://en.wikipedia.org/wiki/Inertial_measurement_unit>`_:

    A very wide variety of IMUs exists, depending on application types, with performance ranging:
        - From 0.1°/s to 0.001°/h for gyroscope
        - From 100 mg to 10 μg for accelerometers.
        
    To get a rough idea, this means that, for a single, uncorrected accelerometer, the cheapest (at 100 mg) loses its ability to give 50-meter accuracy after around 10 seconds, while the best accelerometer (at 10 μg) loses its 50-meter accuracy after around 17 minutes.

|

Sensor Modeling
---------------

The measurements captured by an IMU can be modeled as the ground truth acceleration and angular rates with added noise. More formally, Leutenegger et al. [1]_ defines this process like so:

    An IMU measures the rotation rate and the acceleration of the sensor with respect to an inertial frame. The measurements, namely :math:`{}_{\mathrm{B}}\tilde{\boldsymbol{\omega}}_{\mathrm{WB}}(t)` and :math:`{}_{\mathrm{B}}\tilde{\mathbf{a}}(t)`, are affected by additive white noise :math:`{\boldsymbol{\eta}}` and a slowly varying sensor bias :math:`\mathbf{b}`:

    .. math:: 
        \begin{split}
            {}_{\mathrm{B}}\tilde{\boldsymbol{\omega}}_{\mathrm{WB}}(t) &= {}_{\mathrm{B}}\boldsymbol{\omega}_{\mathrm{WB}}(t) + \mathbf{b}^g(t) + \boldsymbol{\eta}^g(t) \\
            {}_{\mathrm{B}}\tilde{\mathbf{a}}(t) &= \mathbf{R}_{\mathrm{WB}}^\top(t) \left( {}_w\mathbf{a}(t) - {}_w\mathbf{g} \right) + \mathbf{b}^a(t) + \boldsymbol{\eta}^a(t)
        \end{split}

    In our notation, the prefix B denotes that the corresponding quantity is expressed in the frame B (the IMU's frame, a.k.a the "body" frame). The pose of the IMU is described by the transformation :math:`\left\{\mathbf{R}_{\mathrm{WB}}, {}_w\mathbf{p}\right\}` which maps a point from sensor frame B to W (the world coordinate frame). The vector :math:`{}_{\mathrm{B}}\boldsymbol{\omega}_{\mathrm{WB}}(t)\in\mathbb{R}^{3}` is the instantaneous angular velocity of B relative to W expressed in coordinate frame B, while :math:`{}_w\mathbf{a}(t)\in\mathbb{R}^{3}` is the acceleration of the sensor; :math:`{}_w\mathbf{g}` is the gravity vector in world Coordinates. 

This sensor modeling is incorporated into :func:`emulate_imu <visionsim.emulate.imu.emulate_imu>`, which, when given appropriate noise parameters, gravity direction and ground truth poses, will emulate the above random walk and generate accelerometer and gyroscope readings. 

To emulate an IMU from a ``transforms.json`` file, we can also use the CLI::

    visionsim emulate.imu --input-dir=path/to/transforms.json --output-file=imu.csv --dt=0.02

Where ``dt`` is the sampling period of the IMU, set here to 50Hz. It writes one row per
sample, holding the timestamp ``t``, the accelerometer and gyroscope readings
``acc_*``/``gyro_*`` and the accumulated biases ``bias_*``; :doc:`../../tutorials/imu`
shows the emulator's output and decodes a file like this one.

| 

Estimating Pose 
---------------

Using these readings we can estimate the orientation of the IMU. 

While many methods exist, one of the simplest and most ubiquitous is to simply integrate these measurements over time. This technique, called motion integration is implemented in :func:`imu_integration <visionsim.emulate.imu.imu_integration>` and can be used like so:

.. literalinclude:: ../../../../examples/sensors/imu.py 

Using a bit of Blender's geometry nodes we can composite a few estimated trajectories back into the scene, by importing the ``mesh_camera.json`` group onto a cube:

.. node-graph:: ../../../../examples/blender/mesh_camera.json

| 

.. note::

    A worked example that runs the emulator from the command line, integrates the readings and plots the estimate against the ground truth is in :doc:`../../tutorials/imu`.


.. [1] `Leutenegger et al. (2015), "Keyframe-based visual-inertial odometry using nonlinear
        optimization". <https://www.roboticsproceedings.org/rss09/p37.pdf>`_
