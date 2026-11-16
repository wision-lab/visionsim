#!/usr/bin/env bash
# Run from the repository root with a prepared 60 fps rolling-ball scene.
# Plane and Sphere must start at 295 K as finite FEM participants.
set -euo pipefail
scene=${1:-scene_initial_295K.blend}
run_dir=${2:-renders/rolling-ball-transient}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

# Render the complete sequence.
visionsim blender.render-animation \
  "$scene" "$run_dir/renders" \
  --frame-start 1 --frame-end 180 --frame-step 1 \
  --config.executable blender --config.jobs 1 \
  --config.log-dir "$run_dir/blender-logs" --config.timeout 60 \
  --config.device-type optix --config.max-samples 32 --config.use-denoising \
  --config.adaptive-threshold 0.03 --config.use-motion-blur False --config.no-autoexec \
  --config.height 512 --config.width 640 --config.resolution-percentage 100 \
  --config.include-frames --config.include-thermal --config.no-allow-skips \
  --config.thermal.mode TRANSIENT --config.thermal.device cpu --config.thermal.render-domain VERTEX \
  --config.thermal.simulation-start-frame 1 --config.thermal.simulation-end-frame 180 \
  --config.thermal.substeps-per-frame 2 --config.thermal.laplacian-update-every-steps 2 \
  --config.thermal.irradiance-bake-mode EVERY_N_FRAMES --config.thermal.irradiance-bake-every-n-frames 1 \
  --config.thermal.irradiance-scale 300 --config.thermal.bake-samples 16 \
  --config.thermal.irradiance-texture-size 256 --config.thermal.cache-root "$run_dir/cache" \
  --config.thermal.preview --config.thermal.radiance

# Export native-resolution videos.
visionsim ffmpeg.animate --input-dir "$run_dir/renders/frames" --pattern '0000/*.png' \
  --outfile "$run_dir/rgb_native_60fps.mp4" --fps 60 --crf 18 --step 1
visionsim ffmpeg.animate --input-dir "$run_dir/renders/previews/temperature" --pattern '0000/*.png' \
  --outfile "$run_dir/temperature_native_60fps.mp4" --fps 60 --crf 18 --step 1

# Encode browser previews using the quick-start codec and quality settings.
preview_dir=docs/source/_static/thermal-transient
mkdir -p "$preview_dir"
ffmpeg -y -loglevel error -framerate 60 -pattern_type glob -i "$run_dir/renders/frames/0000/*.png" \
  -vf 'scale=320:256:flags=lanczos' -r 60 -fps_mode cfr -an \
  -c:v libx264 -crf 23 -tune animation -pix_fmt yuv420p -movflags +faststart \
  "$preview_dir/rolling-ball-rgb.mp4"
ffmpeg -y -loglevel error -framerate 60 -pattern_type glob -i "$run_dir/renders/previews/temperature/0000/*.png" \
  -vf 'scale=320:256:flags=lanczos' -r 60 -fps_mode cfr -an \
  -c:v libx264 -crf 23 -tune animation -pix_fmt yuv420p -movflags +faststart \
  "$preview_dir/rolling-ball-temperature.mp4"
