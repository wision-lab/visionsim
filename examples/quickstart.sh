#!/usr/bin/env bash

visionsim blender.render-animation lego.blend quickstart/lego-gt/ --render-config.keyframe-multiplier=5.0
visionsim ffmpeg.animate --input-dir=quickstart/lego-gt/frames --outfile=quickstart/preview.mp4 --step=5 --fps=25 --force
visionsim interpolate.frames --input-dir=quickstart/lego-gt/ --output-dir=quickstart/lego-interp/ --n=32
visionsim emulate.rgb --input-dir=quickstart/lego-interp/ --output-dir=quickstart/lego-rgb25fps/ --chunk-size=160 --readout-std=0
visionsim emulate.spad --input-dir=quickstart/lego-interp/ --output-dir=quickstart/lego-spc4kHz/ --mode=img
visionsim emulate.events --input-dir=quickstart/lego-gt/ --output-dir=quickstart/lego-dvs125fps/ --fps=125
