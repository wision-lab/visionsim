import os
from importlib import metadata

from invoke import Collection, Config, Program

from . import blender, dataset, emulate, ffmpeg, interpolate, transforms


class VisionSimConfig(Config):
    """Override `prefix` attribute enabling config options like:

    Environment variables:
        VISIONSIM_MAX_THREADS instead of INVOKE_MAX_THREADS

    Configuration files:
        visionsim.(yaml, yml, json or py) instead of invoke.xxx

    For more see: https://docs.pyinvoke.org/en/latest/concepts/configuration.html
    """

    prefix = "visionsim"


ns = Collection()
ns.add_collection(Collection.from_module(dataset))
ns.add_collection(Collection.from_module(emulate))
ns.add_collection(Collection.from_module(ffmpeg))
ns.add_collection(Collection.from_module(interpolate))
ns.add_collection(Collection.from_module(blender))
ns.add_collection(Collection.from_module(transforms))

ns.configure({"max_threads": os.cpu_count()})


# Note: This is the version of the installed pkg, not the imported one.
#   They can only differ if pkg is installed with -e option.
version = metadata.version("visionsim")
program = Program(name="visionsim", version=str(version), namespace=ns, config_class=VisionSimConfig)
