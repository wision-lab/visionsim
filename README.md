# VisionSIM: Towards a Sensor-Realistic World Simulator

A modular and extensible framework that realistically emulates many different sensor types, alongside rich pixel-perfect ground truth annotations across low-, mid-, and high-level scene characteristics, as well as intrinsic and extrinsic camera properties.

*Warning:* This project is under heavy development and still considered unstable. 

## Installation & Dependencies 

You'll need:

- [Blender](https://www.blender.org/download/) >= 3.3.1, to render new views. 
- [ffmpeg](https://ffmpeg.org/download.html), for visualizations. 
- python dependencies listed in `requirements.txt`. 
- install `visionsim` locally using `pip install .` or using `pip install -e ".[dev]"` if developing.

Make sure Blender and ffmpeg are on your PATH.
The first time you use the renderer, it may ask you to install additional packages into blender's runtime. 

## Building the Documentation

In the project root, with visionsim installed with the dev dependencies, run:
```
inv clean build-docs --preview
```
