Quickstart
==========

Below is a tutorial on how to get started with the render tools as well as the emulate and dataset creation options.

Generate Frames
------------------
To create a new dataset we'll use ``blender.render``, followed by ``emulate.spad/rgb``. Let's look at its options::

    $ spsim -h blender.render
    Usage: spsim [--core-opts] blender.render [--options] [other tasks here ...]

    Docstring:
      Render views of a .blend file while moving camera along a spline or animated trajectory

      Example:
        spsim blender.render <file.blend> <output-path> --num-frames=100 --width=800 --height=800

    Options:
      --addons=STRING            list of extra addons to enable, default: None
      --[no-]allow-skips         whether or not to skip rendering a frame if it already exists, default: True
      --autoexec                 if true, enable the execution of bundled code. default: False
      --bgcolor=STRING           background color as specified by a RGB list in [0-1] range, default: None (no override)
      --bit-depth=INT            bit depth for frames, usually 8 for pngs, default: 8
      --blend-file=STRING        path to blender file to use
      --depth                    whether or not to capture depth images, default: False
      --device=STRING            which device type to use, one of none (meaning cpu), cuda, optix. default: 'optix'
      --device-idxs=STRING       which devices to use. Ex: '[0,2]'. Default: 'all'
      --file-format=STRING       frame file format to use. Depth is always 'OPEN_EXR' thus is unaffected by this setting, default: PNG
      --frame-end=STRING         frame number to stop capture at (exclusive), default: 100
      --frame-start=INT          frame number to start capture at (inclusive), default: 0
      --frame-step=INT           step with which to capture frames, default: 1
      --height=INT               height of frame to capture, default: 800
      --keyframe-multiplier      slow down animations by this factor, default: 1.0 (no slowdown)
      --location-points=STRING   points defining the spline the camera follows. Expected to be json-str or path to json file. Default is circular obit at Z=1 with radius=5.
      --log-file=STRING          where to save log to, default: None (no log is saved)
      --normals                  whether or not to capture normals images, default: False
      --num-frames=STRING        number of frame to capture, default: 100
      --[no-]periodic            whether or not to make the splines periodic, default: True
      --[no-]render              whether or not render frames, default: True
      --root-path=STRING         location at which to save dataset
      --tnb                      if true, ignore viewing points and use trajectory's TNB frame, default: False
      --[no-]unbind-camera       free the camera from it's parents, any constraints and animations it may have. Ensures it uses the world's coordinate frame and the provided camera trajectory. default: True
      --unit-speed               whether or not to reparametrize splines so that movement is of constant speed, default: False
      --[no-]use-animation       allow any animations to play out, if false, scene will be static. default: True
      --[no-]use-motion-blur     enable realistic motion blur, default: True
      --viewing-points=STRING    points defining the spline the camera looks at. Expected to be json-str or path to json file. Default is static origin.
      --width=INT                width of frame to capture, default: 800

First, download the lego truck `original <https://www.blendswap.com/blend/11490>`_ or from `here <https://drive.google.com/file/d/1RjwxZCUoPlUgEWIUiuCmMmG0AhuV8A2Q/view?usp=drive_link>`_.

To create the lego10k dataset, we first need to create all the RGB frames it contains. By default, ``render-views`` will move the camera on a circular obit at Z=1 with radius=5 and point it towards the origin. The following will capture 10k frames on this orbit at a resolution of 800x800 and save them in ``lego10k/frames``::

    $ spsim blender.render blend_files/nerf/lego.blend lego10k --num-frames=10000 --width=800 --height=800

Warning: This takes ~7h using a single RTX3090.

All the rendered frames will be in ``lego10k/frames``. Let's create a quick preview of this dataset by animating every 100th frame::

    $ spsim ffmpeg.animate lego10k/frames --step=100 -o=preview.mp4

The file ``preview.mp4`` should show a nice turntable-style animation of a lego truck.

Emulate
-------
Now we must convert these "perfect" RGB frames into motion blurred frames, to simulate a real camera, and binary frames, to simulate a single-photon camera.

To create the RGB data with motion blur, we use `emulate.rgb`. The `chunk-size` argument determines how many frames to average together. Below we are averaging 200 frames, so if we say the 10k frames correspond to a one-second capture, this means these frames will simulate a 50fps RGB camera. The `fwc` or full-well-capacity argument is not in units of electrons, since we have no physical camera model which matches an rgb linear intensity to a number of electrons, but rather is relative to the `chunk-size`. A FWC equal to the chunck size means that, if each image has an intensity of 1.0, the well will fill up::

    $ spsim emulate.rgb lego10k/frames -o lego10k/rgb-n200 --chunk-size=200 --fwc=200

Preview RGB frames::

    $ spsim ffmpeg.animate lego10k/rgb-n200 -o=preview-rgb.mp4

Finally, we can emulate binary frames like so::

    $ spsim emulate.spad lego10k/frames/ -o lego10k/binary

Preview RGB frames::

    $ spsim ffmpeg.animate lego10k/binary -o=preview-binary.mp4

