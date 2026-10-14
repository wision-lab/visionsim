#!/usr/bin/env bash

visionsim blender.render-animation ultramodern.blend quickstart/ultramodern-gt/ \
    --frame-start=150 --frame-end=350 \
    --config.keyframe-multiplier=1.0 --config.include-depths
visionsim ffmpeg.animate \
    --input-dir=quickstart/ultramodern-gt/frames \
    --outfile=quickstart/preview.mp4 --step=2 --fps=25 --force
visionsim ffmpeg.animate \
    --input-dir=quickstart/ultramodern-gt/previews/depths \
    --outfile=quickstart/preview-depths.mp4 --step=2 --fps=25 --force
visionsim interpolate.dataset \
    --input-dir=quickstart/ultramodern-gt/frames \
    --output-dir=quickstart/ultramodern-interp/ --n=32
visionsim emulate.rgb \
    --input-dir=quickstart/ultramodern-interp/ \
    --output-dir=quickstart/ultramodern-rgb25fps/ \
    --chunk-size=64 --readout-std=0
visionsim emulate.spad \
    --input-dir=quickstart/ultramodern-interp/ \
    --output-dir=quickstart/ultramodern-spc1600fps/
visionsim emulate.events \
    --input-dir=quickstart/ultramodern-interp/ \
    --output-dir=quickstart/ultramodern-dvs1600fps/ --fps=1600 --preview-step=1
visionsim emulate.itof \
    --input-dir=quickstart/ultramodern-gt/frames \
    --depth-dir=quickstart/ultramodern-gt/depths \
    --output-dir=quickstart/ultramodern-itof/ \
    --scheme=convSin --n-captures=4 --freq=120e6 --preview
visionsim emulate.imu \
    --input-dir=quickstart/ultramodern-gt/frames \
    --output-file=quickstart/imu.csv --dt=0.02
