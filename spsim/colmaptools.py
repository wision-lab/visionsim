# Modified from https://github.com/NVlabs/instant-ngp/blob/master/scripts/colmap2nerf.py
import json
import math
import os.path
import re
from pathlib import Path

import cv2
import more_itertools as mitertools
import numpy as np
from scipy.spatial.transform import Rotation


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def compute_sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return variance_of_laplacian(gray)


def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2 + 1e-10))


def closest_point_2_lines(oa, da, ob, db):
    # returns point closest to both rays of form o+t*d, and a weight
    # factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa + ta * da + ob + tb * db) * 0.5, denom


def _natural_sortkey(string):
    # TODO: This works fine but we could use natsort instead...
    tokenize = re.compile(r"(\d+)|(\D+)").findall
    return tuple(int(num) if num else alpha for num, alpha in tokenize(string))


def get_camera_model(text):
    cameras = np.loadtxt(Path(text) / "cameras.txt", dtype=object)

    if cameras.squeeze().ndim > 1:
        raise RuntimeError(f"Expected to find only a single camera, instead found {cameras.shape[0]}.")

    _, model, *params = cameras.squeeze()

    if model not in ("SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"):
        raise ValueError(f"Unknown camera model {model}.")

    w, h, fl_x, fl_y, cx, cy, k1, k2, p1, p2 = np.array(params + [0] * (10 - len(params))).astype(float)
    fl_y = fl_y or fl_x

    # fl = 0.5 * w / tan(0.5 * angle_x);
    angle_x = math.atan(w / (fl_x * 2)) * 2
    angle_y = math.atan(h / (fl_y * 2)) * 2

    return {
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
    }


def get_image_data(text, indices=slice(None)):
    with open(str(Path(text) / "images.txt"), "r") as f:
        lines = filter(lambda l: not l.strip().startswith("#"), f.readlines())
        data = np.loadtxt((l for i, l in enumerate(lines) if i % 2 == 0), dtype=object)

    with open(str(Path(text) / "images.txt"), "r") as f:
        lines = filter(lambda l: not l.strip().startswith("#"), f.readlines())
        points2d = (mitertools.chunked(l.split(" "), 3) for i, l in enumerate(lines) if i % 2 == 1)
        points2d = [[(float(x), float(y), int(p_id)) for x, y, p_id in line] for line in points2d]
        points2d = np.array(points2d, dtype=object)

    data = data.T[indices]
    image_ids, camera_ids, names = data[0].astype(int), data[8].astype(int), data[9].astype(str)
    quaternions, translations = data[1:5].astype(float), data[5:8].astype(float)

    # Convert quaternions and translation vectors into 4x4 transforms
    bottom = np.array([0.0, 0.0, 0.0, 1.0])
    bottom = np.tile(bottom, (len(image_ids), 1, 1))

    R = np.array(Rotation.from_quat(-quaternions[[1, 2, 3, 0]].T).as_matrix())
    t = translations.T.reshape(-1, 3, 1)
    Rt = np.concatenate([R, t], axis=2)
    transforms = np.concatenate([Rt, bottom], axis=1)
    transforms = np.linalg.inv(transforms)

    # Sort all data so the filenames are in order
    idx = sorted(np.arange(len(names)), key=lambda i: _natural_sortkey(names[i]))
    return image_ids[idx], transforms[idx], camera_ids[idx], names[idx], points2d[idx].tolist()


def reorient_to(transforms, new_up=(0, 0, 1)):
    up = transforms[:, 0:3, 1].mean(axis=0)
    up = up / np.linalg.norm(up)
    R = rotmat(up, new_up)
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1

    return np.array([R @ t for t in transforms]), R


def center_at(transforms, center=(0, 0, 0)):
    # find a central point they are all looking at
    attention_weight = 0.0
    attention_center = np.array([0.0, 0.0, 0.0])
    for transform_a in transforms:
        for transform_b in transforms:
            p, w = closest_point_2_lines(transform_a[:3, 3], transform_a[:3, 2], transform_b[:3, 3], transform_b[:3, 2])
            if w > 0.00001:
                attention_center += p * w
                attention_weight += w
    if attention_weight > 0.0:
        attention_center /= attention_weight

    offset = np.array(center) - attention_center
    transforms[:, 0:3, 3] += offset
    t = np.eye(4)
    t[:3, -1] = offset
    return transforms, t


def convert_from_colmap(
    images, text, out_file, aabb_scale=16, indices=slice(None), keep_colmap_coords=False, sharpness=False
):
    # TODO: Allow sharpness calculation for exr files
    out = get_camera_model(text)
    image_ids, transforms, camera_ids, names, points2d = get_image_data(text, indices=indices)

    # In colmap, "the local camera coordinate system of an image is defined in a way
    # that the X axis points to the right, the Y axis to the bottom, and the Z axis
    # to the front as seen from the image", in other words, COLMAP uses the OPENCV
    # coordinate frame convention. NeRF, uses the OpenGL camera convention, where +X
    # is right, +Y is up, and +Z is pointing back and away from the camera.
    #
    # So here, we flip it back by using the following rotation matrix. We would usually
    # left apply this to transform points, but since `transforms` here are the pose of
    # each camera, their x/y/z coordinate frames are encoded as the first 3 columns, not
    # as rows (as for points, where say row 2 is z), we right multiply, leaving the
    # position (4th column) intact and only flipping the camera coordinate frame.
    opencv_to_opengl = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    transforms = np.array([t @ opencv_to_opengl for t in transforms])
    post_transform = np.eye(4)

    if not keep_colmap_coords:
        # Rotate 180 around x=y line: X -> Y, Y -> X, Z -> -Z
        flipxy_negz = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        transforms = np.array([flipxy_negz @ t for t in transforms])
        transforms, R = reorient_to(transforms)
        transforms, t = center_at(transforms)
        post_transform = R @ t @ flipxy_negz

    out["aabb_scale"] = aabb_scale
    out["post_transform"] = post_transform.tolist()
    out["frames"] = [
        {
            "file_path": os.path.relpath(Path(images) / name, Path(out_file).parent),
            "sharpness": compute_sharpness(str(Path(images) / name)) if sharpness else None,
            "transform_matrix": transform.tolist(),
        }
        for name, transform in zip(names, transforms)
    ]

    with open(out_file, "w") as outfile:
        json.dump(out, outfile, indent=2)
