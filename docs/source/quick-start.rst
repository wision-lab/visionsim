===========
Quick Start
===========

Installation & Dependencies 
===========================

.. TODO: 
    Remove following warning for v0.1.0 release.

.. warning::
    Until spsim goes public, it will not be accessible on PyPI. Please use the :doc:`development instructions for installation <../development>` in the meantime.  

The recommended way to get spsim is to **install the latest stable release** via `pip <https://pip.pypa.io>`_::
    
    $ pip install spsim


We currently support **Python 3.9+**. Users still on Python 3.8 or older are
urged to upgrade.



You'll also need:

* `Blender <https://www.blender.org/download/>`_ >= 3.3.1, to render new views. 
* `FFmpeg <https://ffmpeg.org/download.html>`_, for visualizations. 


Make sure Blender and ffmpeg are on your PATH.

The first time you use the renderer, it may ask you to install additional packages into blender's runtime. 


Generating a dataset
====================

To give you a taste of how spsim works, we will create a small scale dataset of a toy lego truck as if it was captured by a realistic 25fps conventional RGB camera and a 4kHz single photon camera for 4 seconds. To achieve this, we will:  

- Render a small dataset of 500 ground truth frames using the ``blender.render`` CLI,
- Interpolate this dataset 32-fold using the ``interpolate.frames`` CLI,
- And emulate a conventional camera and single photon camera from this using ``emulate.spad/rgb``. 


Rendering Ground Truth
----------------------

First, download the test scene, here we will be using the lego truck (of the `NeRF <https://www.matthewtancik.com/nerf>`_ fame) which you can download from `here <https://drive.google.com/drive/folders/1gRxhL3rbGDTfgKytre8WkbBu-QDJFy15?usp=sharing>`_ (`original <https://www.blendswap.com/blend/11490>`_).


To create the lego dataset, we'll first render 500 frames sampled on a circular obit at Z=1 with radius=5 that points towards the origin::
    
    $ spsim blender.render lego.blend lego-gt/ --num-frames=500

This might take a while, with blender 4.2 on a RTX 3080 it takes about 18 minutes.  


.. note::
    The ``blender.render`` CLI has a lot of options, you can look at them all by running the following::

        $ spsim --help blender.render


All the rendered frames will be in ``lego-gt/frames``, and alongside this directory you should see a ``transforms.json`` file which contains information about the camera trajectory used to render the data. 

Let's create a quick preview of this dataset by animating every 5th frame into a video, this allows for realtime play back without creating a video with a very fast framerate::

    $ spsim ffmpeg.animate lego-gt/frames -o=preview.mp4 --step=5 --fps=25


The file ``preview.mp4`` should show a nice turntable-style animation of a lego truck (shown here as a GIF made with `gifski <https://gif.ski/>`_, yours should look better):

.. image:: _static/lego-gt-preview.gif
    :align: center
    :width: 100%


Interpolating Frames
--------------------

You can optionally interpolate an existing dataset in order to get intermediate frames that have not been rendered. This enables us to quickly increase the effective framerate of the data at the cost of potentially introducing artifacts if frames are too "far" apart. The following will interpolate a dataset by a factor of `32x`::

    $ spsim interpolate.frames lego-gt/ -o lego-interp/ -n=32

This is much faster than rendering new frames, but can still be a bit slow. The above takes about 10 minutes on an RTX 3080.  

You can preview this new dataset like above, just use a step of `5x32=160` to ensure playback is at the same speed. 

There's a few things of note here:

* The new dataset actually contains `15,969` frames and not `16,000`, which might be a little surprising at first as one might expect `32x500=16,000` frames. But consider the case where you interpolate `2` frames by a factor of `2x`. You'll create a new frame between every existing frame pairs, which will give you 3 frames total, the first original frame, the interpolated frame, and the second original frame. In general, for `M` original frames interpolated `N`-times you'll get `NM-N+1` frames after interpolation.  
* Interpolation can introduce artifacts adjacent frames in the original dataset are too different from one another. This effect and it's implications are further discussed in the :doc:`interpolation` section. In general, interpolation is useful for higher frame rates, it can help bridge the gap between `1,000` fps to `10,000` fps, not from `10` fps to `100` fps. 


Emulating Sensor Data
---------------------

To simulate a real camera, we must convert these "perfect" RGB frames into realistic motion blurred frames with read noise and quantization artifacts.

To create the RGB data with motion blur, we use `emulate.rgb`. The `chunk-size` argument determines how many frames to average together. Below we are averaging frames from the interpolated dataset in groups of `160`, so since the interpolated dataset corresponds to a frame rate of `4,000` fps, this means these frames will simulate a 25fps RGB camera::

    $ spsim emulate.rgb lego-interp/ -o lego-rgb25fps/ --chunk-size=160 --readout-std=0

.. .. note::
..     The `fwc` or full-well-capacity argument is not in units of electrons, since we have no physical camera model which matches an rgb linear intensity to a number of electrons, but rather is relative to the `chunk-size`. A FWC equal to the chunk size means that, if each image has a normalized intensity of 1.0, the well will fill up.


Finally, we can emulate a single-photon camera, at the same framerate as the interpolated dataset like so::

    $ spsim emulate.spad lego-interp/ -o lego-spc4kHz/ --mode=img


and look at the results:


.. list-table::
    :class: borderless

    * - .. figure:: _static/lego-rgb25fps-preview.gif

            Conventional @ 25fps

      - .. figure:: _static/lego-spc4kHz-preview.gif

            Single Photon @ 4kHz
