#!/usr/bin/env bash
# Commands for the rolling-ball demonstration.
visionsim blender.render-animation scene_initial_295K.blend rolling-ball/renders \
  --config.include-thermal --config.max-samples 32 \
  --config.adaptive-threshold 0.03 --config.use-motion-blur False \
  --config.thermal.mode TRANSIENT --config.thermal.device cpu \
  --config.thermal.render-domain VERTEX \
  --config.thermal.substeps-per-frame 2 --config.thermal.laplacian-update-every-steps 2 \
  --config.thermal.irradiance-scale 300 --config.thermal.bake-samples 16 \
  --config.thermal.irradiance-texture-size 256 --config.thermal.cache-root rolling-ball/cache

visionsim ffmpeg.animate --input-dir rolling-ball/renders/frames --pattern '0000/*.png' \
  --outfile rolling-ball/rgb.mp4 --fps 60 --crf 18
visionsim ffmpeg.animate --input-dir rolling-ball/renders/previews/temperature --pattern '0000/*.png' \
  --outfile rolling-ball/temperature.mp4 --fps 60 --crf 18
