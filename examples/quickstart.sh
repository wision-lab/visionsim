#!/usr/bin/env bash

visionsim blender.render-animation lego.blend quickstart/lego-gt/ \
    --config.keyframe-multiplier=5.0 --config.include-depths
visionsim ffmpeg.animate \
    --input-dir=quickstart/lego-gt/frames \
    --outfile=quickstart/preview.mp4 --step=5 --fps=25 --force
visionsim ffmpeg.animate \
    --input-dir=quickstart/lego-gt/previews/depths \
    --outfile=quickstart/preview-depths.mp4 --step=5 --fps=25 --force
visionsim interpolate.dataset \
    --input-dir=quickstart/lego-gt/frames \
    --output-dir=quickstart/lego-interp/ --n=32
visionsim emulate.rgb \
    --input-dir=quickstart/lego-interp/ \
    --output-dir=quickstart/lego-rgb25fps/ \
    --chunk-size=160 --readout-std=0
visionsim emulate.spad \
    --input-dir=quickstart/lego-interp/ \
    --output-dir=quickstart/lego-spc4kHz/
visionsim emulate.events \
    --input-dir=quickstart/lego-gt/frames \
    --output-dir=quickstart/lego-dvs125fps/ --fps=125 --preview-step=1
visionsim emulate.itof \
    --input-dir=quickstart/lego-gt/frames \
    --depth-dir=quickstart/lego-gt/depths \
    --output-dir=quickstart/lego-itof/ \
    --scheme=convSin --n-captures=4 --freq=120e6 --preview
