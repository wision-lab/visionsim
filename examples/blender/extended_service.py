from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

from visionsim.simulate.blender import BlenderServer, BlenderService


class ExtendedService(BlenderService):
    def exposed_scene_aabb(self):
        aabb_min = np.array([np.inf, np.inf, np.inf])
        aabb_max = -np.array([np.inf, np.inf, np.inf])

        for obj in self.scene.objects:
            bbox = np.array(obj.bound_box)
            bbox_min = obj.matrix_world @ Vector(np.min(bbox, axis=0))
            bbox_max = obj.matrix_world @ Vector(np.max(bbox, axis=0))
            aabb_min = np.minimum(aabb_min, bbox_min)
            aabb_max = np.maximum(aabb_max, bbox_max)
        return aabb_min, aabb_max

    def exposed_missing_textures(self):
        paths = []

        for image in bpy.data.images.values():
            if image.source == "FILE":
                if not (path := Path(image.filepath_from_user()).resolve()).exists():
                    paths.append(path)

        return paths


if __name__ == "__main__":
    server = BlenderServer(service=ExtendedService, port=0)
    server.start()
