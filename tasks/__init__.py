from importlib import metadata

from invoke import Collection, Program

from . import blender, colmap, dataset, emulate, ffmpeg, interpolate, transforms

ns = Collection()
ns.add_collection(Collection.from_module(colmap))
ns.add_collection(Collection.from_module(dataset))
ns.add_collection(Collection.from_module(emulate))
ns.add_collection(Collection.from_module(ffmpeg))
ns.add_collection(Collection.from_module(interpolate))
ns.add_collection(Collection.from_module(blender))
ns.add_collection(Collection.from_module(transforms))

# Note: This is the version of the installed pkg, not the imported one.
#   They can only differ if pkg is installed with -e option.
version = metadata.version("spsim")
program = Program(name="spsim", version=str(version))
