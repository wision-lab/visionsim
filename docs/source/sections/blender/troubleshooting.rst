Troubleshooting
===============

Things that go wrong on the way from a blend-file to a dataset, roughly in the
order you meet them: getting a Blender the renderer can use, starting a render,
and reading the frames that come out. Each entry names the message or behaviour
you see, why it happens, and what to change.

|

Blender Is Missing or the Wrong Version
---------------------------------------

**Symptom.** A render command stops with ``RuntimeError: No blender installation
found on path!``, or Blender starts and then raises ``Blender dependencies are
missing, please run the post install script.``

**Cause.** The CLI looks for ``blender`` on ``$PATH`` unless told otherwise, and
each Blender version ships its own Python interpreter and site-packages.
Dependencies installed for one build are invisible to another, so a fresh Blender
can run the entrypoint but fail on its imports.

**Fix.** Install the package and its Blender-side dependencies:

.. code-block:: bash

    pip install visionsim
    visionsim post-install

Run ``post-install`` once per Blender version you render with. To use a build that
is not on ``$PATH``, point render commands at it with ``--config.executable`` and
pass the same path to ``post-install --executable``:

.. code-block:: bash

    visionsim post-install --executable=/opt/blender-4.2/blender
    visionsim blender.render-animation scene.blend output/ \
        --config.executable=/opt/blender-4.2/blender

See :doc:`installation <../../installation>` for the Flatpak form of
``--config.executable`` and the Python equivalents. Blender 3.6 is the floor: the
entrypoint raises ``RuntimeError: Please use newer blender version, at least
3.6.`` below it. The test suite reads ``VSIM_BLENDER`` for the same purpose, see
:doc:`development <../../development>`.

|

GPU or Optix Is Unsupported
---------------------------

**Symptom.** A Cycles render fails while setting up its device, on hardware that
has no Optix support.

**Cause.** ``RenderConfig.device_type`` defaults to ``optix``, which only recent
Nvidia GPUs expose.

**Fix.** Select another backend, for example CUDA:

.. code-block:: bash

    visionsim blender.render-animation scene.blend output/ --config.device-type=cuda

The note in :doc:`quick-start <../../quick-start>` lists ``cuda`` for older GPUs;
``cpu`` and ``metal`` are also accepted.

|

Cannot Connect to a Blender Server
----------------------------------

**Symptom.** ``TimeoutError: Unable to discover server in alloted time.`` or
``TimeoutError: Unable to connect to server in alloted time.``

**Cause.** :meth:`BlenderClient.auto_connect
<visionsim.simulate.blender.BlenderClient.auto_connect>` waits ``timeout`` seconds
(10 by default) for :meth:`BlenderServer.discover
<visionsim.simulate.blender.BlenderServer.discover>` to find a registered server
before giving up. The CLI passes ``RenderConfig.timeout`` (``--config.timeout``)
through to the spawn call, and the entrypoint accepts ``--port`` when it is
launched by hand.

**Fix.** Give discovery more time, or spawn a server yourself and watch its log:

.. code-block:: bash

    VSIM_ENTRYPOINT=$(python -c "from visionsim.simulate import blender; print(blender.__file__)")
    blender -b --python-use-system-env --python $VSIM_ENTRYPOINT -- --port 12345

Blender prints ``INFO: Started listening on 0.0.0.0:12345`` once it is up. Then
connect from Python:

.. code-block:: python

    from visionsim.simulate.blender import BlenderClient, BlenderServer

    with BlenderServer.spawn() as (procs, conns):
        with BlenderClient(*conns) as client:
            ...

|

Output Directory Already Exists
-------------------------------

**Symptom.** ``FileExistsError: Output directory already exists.`` or
``RuntimeError: Input and output directory cannot be the same!``

**Cause.** The emulation and dataset CLIs refuse to overwrite an existing output
directory, and reject an output path that points at the input.

**Fix.** Pass ``--force`` to overwrite, and choose an output path distinct from the
input:

.. code-block:: bash

    visionsim emulate.rgb \
        --input-dir=input/ --output-dir=output/ --force

|

Playblast Fails with No Display
-------------------------------

**Symptom.** ``RuntimeError: Playblast rendering requires a headful (non-background)
Blender instance with a window.`` or
``RuntimeError: Playblast rendering requires Blender >= 4.2, found 4.1.0.``

**Cause.** ``render-playblast`` forces ``background=False`` because viewport
rendering needs a live GL context, which a headless machine does not have. The
viewport renderer is also unavailable before Blender 4.2.

**Fix.** Start a virtual X server around the command, using the recipe from
:doc:`playblast <../../tutorials/playblast>`:

.. code-block:: bash

    DISPLAY="" WAYLAND_DISPLAY="" xvfb-run -a --server-args="-screen 0 1920x1080x24" \
        visionsim blender.render-playblast scene.blend output/

Upgrade Blender if the version error is the one you hit.

|

Camera Will Not Move as Authored
--------------------------------

**Symptom.** You keyframe the camera through the API, or animate it in the
blend-file, and the rendered trajectory does not match. The camera follows a path
or another object instead.

**Cause.** A parent, constraint or path on the camera overrides the keyframes you
set. Positioning is local by default, so a constraint moves the result away from
the intended pose.

**Fix.** Free the camera from its parents and constraints before positioning it
with :meth:`unbind_camera
<visionsim.simulate.blender.BlenderService.exposed_unbind_camera>`, or pass
``--config.unbind-camera`` to the CLI. If you want the authored constraint or
animation to drive the camera, leave it in place: :doc:`keyframing` shows the
``--no-unbind-camera`` and ``--use-animations`` flags for that case, and
:doc:`cameras` covers the dolly example where a keyframed location competes with a
constraint.

|

Optical Flow Is Wrong or Empty
------------------------------

**Symptom.** The ``flows`` EXRs are blank, or the flow does not line up with the
motion between frames.

**Cause.** The vector pass needs the camera to be animated so there is motion to
measure. When the active camera and its parents carry no animation, the renderer
logs ``Active camera nor it's parents are animated, camera will be static.`` and
the flow of a static camera is zero. Motion blur conflicts with flow as well:
enabling it raises ``RuntimeError: Cannot compute optical flow if motion blur is
enabled.``

**Fix.** Keyframe the camera trajectory (see :doc:`keyframing`) and render with
``--config.include-flows``. Flow is computed between consecutive rendered frames,
so it needs the next camera position to be known in advance; the API section of
:doc:`keyframing` explains why setting and keyframing the whole trajectory before
rendering works better than moving and rendering frame by frame. Leave motion blur
off.

|

Interpolation Artifacts
-----------------------

**Symptom.** Interpolated frames show ghosting, warping or flicker that the
rendered frames do not.

**Cause.** Frame interpolation assumes adjacent frames are close in time. When the
camera moves far between them, the method has to invent too much.

**Fix.** Render at a higher frame rate and interpolate by a smaller factor.
:doc:`interpolation <../interpolation>` shows the artifact level for one scene
across several render rates and notes the native rate is often a few hundred
frames per second. Interpolation bridges 1,000fps to 10,000fps, not 10fps to
100fps (see the warning in :doc:`quick-start <../../quick-start>`).
