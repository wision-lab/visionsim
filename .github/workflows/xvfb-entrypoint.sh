#!/usr/bin/env bash
#
# Image entrypoint: start a virtual X display so Blender's viewport renderer (playblast) has a GL
# context, then hand off to the container's command. Containers have no display of their own, so
# this lets eg. `docker run --rm <image> visionsim blender.render-playblast ...` work headlessly.
set -euo pipefail

export DISPLAY=:99
Xvfb "$DISPLAY" -screen 0 1920x1080x24 -nolisten tcp -ac &
xvfb_pid=$!

# Xvfb is asynchronous: don't hand off until its socket exists, but fail loudly (rather than
# hanging, then silently proceeding) if it never comes up or dies during startup.
for _ in $(seq 1 100); do
    if [ -S /tmp/.X11-unix/X99 ]; then
        exec "$@"
    fi
    if ! kill -0 "$xvfb_pid" 2>/dev/null; then
        echo "Xvfb exited before $DISPLAY became available" >&2
        exit 1
    fi
    sleep 0.1
done

echo "Timed out waiting for Xvfb on $DISPLAY" >&2
exit 1
