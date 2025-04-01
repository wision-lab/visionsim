#!/usr/bin/env bash

visionsim blender.render-animation lego.blend quickstart/lego-gt/ --keyframe-multiplier=5.0
visionsim ffmpeg.animate quickstart/lego-gt/frames -o=quickstart/preview.mp4 --step=5 --fps=25 --force
visionsim interpolate.frames quickstart/lego-gt/ -o quickstart/lego-interp/ -n=32
visionsim emulate.rgb quickstart/lego-interp/ -o quickstart/lego-rgb25fps/ --chunk-size=160 --readout-std=0
visionsim emulate.spad quickstart/lego-interp/ -o quickstart/lego-spc4kHz/ --mode=img
