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
visionsim blender.render-animation ultramodern.blend quickstart/ultramodern-thermal/ \
    --frame-start=1 --frame-end=600 \
    --config.include-thermal --config.no-include-frames \
    --config.thermal.assignments=assets/thermal/ultramodern.thermal.json \
    --config.thermal.sim-time-s=600 --config.thermal.timestep-s=1 \
    --config.thermal.irradiance-texture-size=1024 \
    --config.max-samples=2048 --config.adaptive-threshold=0.0001 \
    --config.no-use-denoising
visionsim transforms.colorize-temperatures \
    --input-dir=quickstart/ultramodern-thermal/thermal_radiance \
    --output-dir=quickstart/ultramodern-thermal/radiance-colorized \
    --cmap=inferno
visionsim ffmpeg.animate \
    --input-dir=quickstart/ultramodern-thermal/radiance-colorized \
    --pattern='*.png' --outfile=quickstart/ultramodern-thermal/thermal_global.mp4
