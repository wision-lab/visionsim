# Single Photon Simulator and Tools

This package uses blender to render out scenes at high frame rates to simulate passive single photon cameras.

## Installation & Dependencies 

You'll need:

- [Blender](https://www.blender.org/download/) >= 3.3.1, to render new views. 
- [ffmpeg](https://ffmpeg.org/download.html), for visualizations. 
- python dependencies listed in `requirements.txt`. 
- install `spsim` locally using `pip install .` or using `pip install -e .[dev]` if developing.

Make sure Blender and ffmpeg are on your PATH.

The first time you use the renderer, it may ask you to install additional packages into blender's runtime. 

## CLI Tools - Overview

While many tools are available, the main one is `blender.render`.
All tools used in this repository can be accessed like so:

```
$ spsim --list
Subcommands:

  blender.render              Render views of a .blend file while moving camera along a spline or animated trajectory
  blender.to-nerf-format      Convert transform.json from blender format to nerf-style format
  colmap.generate-masks       Extract alpha channel from frames and create PNG masks that COLMAP understands
  colmap.run                  Run colmap on the provided images to get a rough pose/scene estimate
  colmap.to-nerf-format       Convert transform.json from colmap format to nerf-style format
  dataset.frames-to-npy       Convert an image folder based dataset to a NPY dataset (experimental)
  emulate.blur                Average together frames to create motion blur, similar to `emulate_rgb` but with no camera modeling
  emulate.rgb                 Simulate real camera, adding read/poisson noise and tonemapping
  emulate.spad                Perform bernoulli sampling on linearized RGB frames to yield binary frames
  ffmpeg.animate              Combine generated frames into an MP4 using ffmpeg wizardry
  ffmpeg.count-frames         Count the number of frames a video file contains using ffprobe
  ffmpeg.duration             Return duration (in seconds) of first video stream in file using ffprobe
  ffmpeg.extract              Extract frames from video file
  interpolate.frames          Interpolate between frames and poses (up to 16x) using RIFE (ECCV22)
  interpolate.video           Interpolate video by extracting all frames, performing frame-wise interpolation and re-assembling video
  transforms.colorize-depth   Convert .exr depth maps into color-coded images for visualization
  transforms.tonemap-exrs     Convert .exr linear intensity frames into tone-mapped sRGB images

```

## Autocompletion

The auto-complete functionality is provided by [pyinvoke](https://docs.pyinvoke.org/en/stable/invoke.html#shell-tab-completion), and can be activated per terminal like so:

```
source <(spsim --print-completion-script bash)
```

or by creating a file that can be sourced from your `~/.bashrc`:

```
spsim --print-completion-script bash > ~/.spsim-completion.sh

# Place in ~/.bashrc:
source ~/.spsim-completion.sh
```

The same can be done in other shells such as `zsh`, `fish`. 


## Generating the Lego10K dataset

To create a new dataset we'll use `blender.render`, followed by `emulate.spad/rgb`. Let's look at its options:

```
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
```

First, download the lego truck ([original](https://www.blendswap.com/blend/11490) or from [here](https://drive.google.com/file/d/1RjwxZCUoPlUgEWIUiuCmMmG0AhuV8A2Q/view?usp=drive_link)).

To create the lego10k dataset, we first need to create all the RGB frames it contains. By default, `render-views` will move the camera on a circular obit at Z=1 with radius=5 and point it towards the origin. The following will capture 10k frames on this orbit at a resolution of 800x800 and save them in `lego10k/frames`:

```
$ spsim blender.render blend_files/nerf/lego.blend lego10k --num-frames=10000 --width=800 --height=800
```
_Warning: This takes ~7h using a single RTX3090._

All the rendered frames will be in `lego10k/frames`. Let's create a quick preview of this dataset by animating every 100th frame:
```
$ spsim ffmpeg.animate lego10k/frames --step=100 -o=preview.mp4
```

The file `preview.mp4` should show a nice turntable-style animation of a lego truck.

Now we must convert these "perfect" RGB frames into motion blurred frames, to simulate a real camera, and binary frames, to simulate a single-photon camera. 

To create the RGB data with motion blur, we use `emulate.rgb`. The `chunk-size` argument determines how many frames to average together. Below we are averaging 200 frames, so if we say the 10k frames correspond to a one-second capture, this means these frames will simulate a 50fps RGB camera. The `fwc` or full-well-capacity argument is not in units of electrons, since we have no physical camera model which matches an rgb linear intensity to a number of electrons, but rather is relative to the `chunk-size`. A FWC equal to the chunck size means that, if each image has an intensity of 1.0, the well will fill up.

```
$ spsim emulate.rgb lego10k/frames -o lego10k/rgb-n200 --chunk-size=200 --fwc=200

# Preview RGB frames
# spsim ffmpeg.animate lego10k/rgb-n200 -o=preview-rgb.mp4
```

Finally, we can emulate binary frames like so:

```
$ spsim emulate.spad lego10k/frames/ -o lego10k/binary

# Preview RGB frames
# spsim ffmpeg.animate lego10k/binary -o=preview-binary.mp4
```

## Interpolating a dataset
You can also interpolate an existing dataset using [RIFE](https://github.com/megvii-research/ECCV2022-RIFE) like so:

```
$ spsim interpolate.frames lego10k/ -o lego80k/ -n=8
```

## Cookbook 

How to...

- Generate linear intensity images, not tone-mapped PNGs?
  
    Add the options `--file-format=open_exr --bit-depth=16` to the render command (you can use higher bit depths too).

- Move camera along a path?
    Two ways to do this:
    - You can use the location/viewing-points or tnb settings to explicitly pass in a path for the camera to follow. Using something like `--location-points=trajectory.json` or  `--location-points=[[0,0,0], [1,1,1], [2,2,2]]` will load the points saved in the json file (or read the string as json) and move the camera along a spline connecting them all. The `location-points` argument defines the spline the camera will follow, while the `viewing-points` argument defines the spline the camera should look at. No roll along the optical axis is permitted in this mode. If `--tnb` is set, the camera will use the trajectory's [Frenet-Serret frame](https://en.wikipedia.org/wiki/Frenet%E2%80%93Serret_formulas) to orient itself. Otherwise, the camera will point towards the spline defined by the `--viewing-locations` argument.   
  
    - You can use a blender-defined camera animation as well. To do this, make sure the camera is not freed from its constraints/parents and keyframes by setting `--no-unbind-camera` and enabling `--use-animations` (enabled by default). You'll likely want to also specify at which frame to start/stop capture and you can do this with the `frame-start` and `frame-step` options as well as define the number of frames to capture with `num-frames` or the sequence end with `frame-end`. Finally, you can also slow down the animation using the `keyframe-multiplier` argument.
  There's a few ways to make an animated camera, see [here](https://www.youtube.com/watch?v=a7qyW1G350g) and [here](https://www.youtube.com/watch?v=K02hlKyoWNI) for some example methods.

- Enable external addons?
    Use the `addons` argument. Ex: `--addons='sun_position,other_addon_here'`.


## Known issues
#### render-views
- TNB frame is glitchy, likely due to numerical errors.

#### ffmpeg-animate
- fps value cannot be too high (~ 200+) or too small (~ 10-) without ffmpeg complaining and dropping frames. Use step argument instead.