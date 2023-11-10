import functools
import io
import itertools
import json
import multiprocessing
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm.auto import tqdm


def extrinsic2pyramid(ax, extrinsic, color="r", focal_len_scaled=5, aspect_ratio=0.3, alpha=0.3, lw=0.3):
    # Modified from: https://github.com/demul/extrinsic2pyramid/blob/main/util/camera_pose_visualizer.py
    vertex_std = np.array(
        [
            [0, 0, 0, 1],
            [
                focal_len_scaled * aspect_ratio,
                -focal_len_scaled * aspect_ratio,
                focal_len_scaled,
                1,
            ],
            [
                focal_len_scaled * aspect_ratio,
                focal_len_scaled * aspect_ratio,
                focal_len_scaled,
                1,
            ],
            [
                -focal_len_scaled * aspect_ratio,
                focal_len_scaled * aspect_ratio,
                focal_len_scaled,
                1,
            ],
            [
                -focal_len_scaled * aspect_ratio,
                -focal_len_scaled * aspect_ratio,
                focal_len_scaled,
                1,
            ],
        ]
    )
    vertex_transformed = vertex_std @ np.array(extrinsic).T
    meshes = [
        [
            vertex_transformed[0, :-1],
            vertex_transformed[1, :-1],
            vertex_transformed[2, :-1],
        ],
        [
            vertex_transformed[0, :-1],
            vertex_transformed[2, :-1],
            vertex_transformed[3, :-1],
        ],
        [
            vertex_transformed[0, :-1],
            vertex_transformed[3, :-1],
            vertex_transformed[4, :-1],
        ],
        [
            vertex_transformed[0, :-1],
            vertex_transformed[4, :-1],
            vertex_transformed[1, :-1],
        ],
        [
            vertex_transformed[1, :-1],
            vertex_transformed[2, :-1],
            vertex_transformed[3, :-1],
            vertex_transformed[4, :-1],
        ],
    ]
    return ax.add_collection3d(Poly3DCollection(meshes, facecolors=color, linewidths=lw, edgecolors=color, alpha=alpha))


def plot_trajectory(
    trajectory,
    ax=None,
    step=200,
    pose_kwargs=None,
    path_kwargs=None,
    color="k",
    label=None,
    scale=1,
    rot=None,
    quiver_scale=1,
    direction=True,
    endpoints=True,
    full=False,
):
    if isinstance(trajectory, (str, Path)):
        with open(str(trajectory), "r") as f:
            trajectory = json.load(f)

    ax = plt.figure(figsize=(8, 8)).add_subplot(projection="3d") if ax is None else ax

    pose_kwargs_ = dict(focal_len_scaled=-1, alpha=0.05, color=color)
    pose_kwargs_.update(pose_kwargs or {})
    scale_transform = np.array(trajectory.get("scale_transform", np.eye(4))).reshape(4, 4)

    for frame in trajectory["frames"][::step]:
        T = scale_transform @ np.array(frame["transform_matrix"])
        T[:-1, 3] = T[:-1, 3] * scale
        T[:-1, 3] = rot.as_matrix() @ T[:-1, 3] if rot is not None else T[:-1, 3]
        T[:3, :3] = rot.as_matrix() @ T[:3, :3] if rot is not None else T[:3, :3]
        extrinsic2pyramid(ax, T, **pose_kwargs_)

    path_kwargs_ = dict(color=color, alpha=0.4)
    path_kwargs_.update(path_kwargs or {})
    points = [scale_transform @ np.array(frame["transform_matrix"]) for frame in trajectory["frames"]]
    points = [p[:-1, 3] for p in points]

    if rot is not None:
        points = [rot.as_matrix() @ p for p in points]
    ax.plot(*np.array(points).T * scale, label=label, **path_kwargs_)

    if direction:
        ax.quiver(
            *np.array(points)[step // 2 :: step].T * scale,
            *np.array(np.diff(points, axis=0))[step // 2 - 1 :: step].T * scale,
            length=0.25 * quiver_scale,
            arrow_length_ratio=1,
            normalize=True,
            color=color,
            alpha=path_kwargs_.get("alpha", 0.4),
        )

    if endpoints:
        ax.scatter(
            *np.array(points)[[1, -1]].T * scale,
            marker="o",
            color=color,
            alpha=path_kwargs_.get("alpha", 0.4),
            s=30 * quiver_scale,
        )
    if full:
        return ax, np.array(points)
    return ax


def plot_sparse_reconstruction(points3d_path, ax=None, transform=None, percentile=None, min_track_len=None, full=False):
    points = np.loadtxt(points3d_path, delimiter=" ", usecols=(1, 2, 3, 4, 5, 6))
    errors = np.loadtxt(points3d_path, delimiter=" ", usecols=(7,))

    with open(points3d_path, "r") as f:
        track_lengths = np.array([len(line.split(" ")) for line in f])[3:] - 8

    if min_track_len:
        points = points[track_lengths >= min_track_len]

    if percentile:
        points = points[np.percentile(errors, percentile) >= errors[track_lengths >= min_track_len]]

    if transform is None:
        # Try to load post_transforms from ../transforms.json if exists
        path = Path(points3d_path).parent / "../transforms.json"

        if path.exists():
            with open(str(path), "r") as f:
                data = json.load(f)
                post_transform = data.get("post_transform", np.eye(4))
                post_transform = np.array(post_transform).reshape(4, 4)
                scale_transform = data.get("scale_transform", np.eye(4))
                scale_transform = np.array(scale_transform).reshape(4, 4)
                transform = scale_transform @ post_transform
                print(f"Loaded transform from {path.resolve()}.")

    x, y, z, *color = points.T

    if transform is not None:
        ones = np.ones_like(x)
        points = np.stack([x, y, z, ones])
        x, y, z, ones = transform @ points
        x, y, z = x / ones, y / ones, z / ones

    ax = plt.figure(figsize=(8, 8)).add_subplot(projection="3d") if ax is None else ax
    ax.scatter(x, y, z, c=np.stack(color).T / 255)

    if full:
        return ax, (x, y, z)
    return ax


def _render_single_view(item, pickled_fig=None, savefig_kwargs=None):
    view_dir, outfile = item
    fig = pickle.loads(pickled_fig)
    ax = fig.gca()
    ax.view_init(*view_dir)  # altitude, azimuth, roll
    plt.savefig(outfile, **savefig_kwargs)


def orbit_animation(fig, dirname, view_dirs, savefig_kwargs=None, save_svgs=None):
    with io.BytesIO() as buf:
        pickle.dump(fig, buf)
        buf.seek(0)
        serialized_plot = buf.read()

    paths = [Path(dirname) / f"frame_{i:06}.png" for i in range(len(view_dirs))]
    render_single = functools.partial(
        _render_single_view, pickled_fig=serialized_plot, savefig_kwargs=savefig_kwargs or {}
    )

    if save_svgs:
        paths += [Path(save_svgs) / f"frame_{i:06}.svg" for i in range(len(view_dirs))]
        view_dirs = list(itertools.chain(view_dirs, view_dirs))

    with multiprocessing.Pool() as p:
        tasks = p.imap_unordered(render_single, zip(view_dirs, paths))
        list(tqdm(tasks, total=len(view_dirs)))
