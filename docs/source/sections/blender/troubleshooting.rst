
`vsim post-install`

VSIM_ENTRYPOINT=$(python -c "from visionsim.simulate import blender; print(blender.__file__)")
blender -b --python-use-system-env --python $VSIM_ENTRYPOINT -- --port 12345

Blender 4.4.3 (hash 802179c51ccc built 2025-04-29 15:12:13)
INFO: Started listening on 0.0.0.0:12345


from visionsim.simulate.blender import *

with BlenderServer.spawn() as (procs, conns):
    with BlenderClient(*conns) as client:
        ...
