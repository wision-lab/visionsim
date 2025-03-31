# Single Photon Simulator and Tools

This package uses blender to render out scenes at high frame rates to simulate passive single photon cameras.

## Installation & Dependencies 

You'll need:

- [Blender](https://www.blender.org/download/) >= 3.3.1, to render new views. 
- [ffmpeg](https://ffmpeg.org/download.html), for visualizations. 
- python dependencies listed in `requirements.txt`. 
- install `visionsim` locally using `pip install .` or using `pip install -e ".[dev]"` if developing.

Make sure Blender and ffmpeg are on your PATH.
The first time you use the renderer, it may ask you to install additional packages into blender's runtime. 

## Building the Documentation

In the project root, run:
```
inv clean build-docs --preview
```
