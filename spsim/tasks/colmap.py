from pathlib import Path

from invoke import task

from spsim.tasks.common import _log_run, _raise_callback, _run, _validate_directories


def _generate_mask_single(in_file, output_dir=None):
    """Extract alpha channel as mask"""
    import numpy as np

    from spsim.io import read_img, write_img

    img, alpha = read_img(in_file, apply_alpha=True)
    mask = (alpha * 255).astype(np.uint8)
    path = output_dir / Path(in_file).with_suffix(".png").name
    return write_img(str(path), mask)


@task(
    help={
        "input_dir": "directory in which to look for frames",
        "output_dir": "directory in which to save masks",
        "pattern": "filenames of frames should match this, default: 'frame_*.png'",
    }
)
def generate_masks(
    _,
    input_dir,
    output_dir,
    pattern="frame_*.png",
):
    """Extract alpha channel from frames and create PNG masks that COLMAP understands"""
    import functools
    import multiprocessing

    from tqdm.auto import tqdm

    input_dir, output_dir, in_files = _validate_directories(input_dir, output_dir, pattern)
    in_files = list(filter(lambda p: not Path(p).is_dir(), in_files))
    generate_mask_single = functools.partial(_generate_mask_single, output_dir=output_dir)

    with multiprocessing.Pool() as p:
        tasks = p.imap(generate_mask_single, in_files)
        list(tqdm(tasks, total=len(in_files)))


@task(
    help={
        "input_dir": "directory in which to look for frames",
        "output_dir": "directory in which to save colmap results, default='.'",
        "camera_model": "one of SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV. Default: OPENCV",
        "camera_params": "intrinsic parameters, depends on the chosen model. default: ''",
        "mask_dir": "if present use these image masks when extracting features, will generate them "
        "from image alpha channel if needed. default: ''",
        "matcher": "which matcher to use. One of exhaustive, sequential, spatial, transitive, vocab_tree. "
        "Usually, sequential for videos, exhaustive for adhoc images. Default: sequential",
        "vocab_path": "vocabulary tree path, default: None",
        "mapper": "which mapper to use, either '' or 'hierarchical'. default: ''.",
        "dense_reconstruction": "if true, perform dense reconstruction after sparse one. default: False",
        "cuda": "use cuda acceleration where possible, default: True",
        "force": "if true, overwrite output file if present, default: False",
    }
)
def run(
    c,
    input_dir,
    output_dir,
    camera_model="OPENCV",
    camera_params="",
    mask_dir="",
    matcher="sequential",
    vocab_path=None,
    mapper="",
    dense_reconstruction=False,
    skip_ba=False,
    cuda=True,
    force=False,
):
    """Run colmap on the provided images to get a rough pose/scene estimate"""
    import functools
    import json
    import shutil

    from spsim.cli import StreamWatcherTqdmPbar  # Lazy import

    input_dir, output_dir, in_files = _validate_directories(input_dir, output_dir, pattern="*")
    db = Path(output_dir) / "colmap.db"
    text = Path(output_dir) / "text"
    sparse = Path(output_dir) / "sparse"
    dense = Path(output_dir) / "dense"
    log = Path(output_dir) / "log"
    mask_dir = Path(mask_dir) if mask_dir else None

    if _run(c, "colmap help", hide=True).failed:
        raise RuntimeError(f"No COLMAP installation found on path!")
    if not force and (db.exists() or text.exists() or sparse.exists() or dense.exists()):
        raise RuntimeError(
            f"COLMAP databases {db}, {text}, {sparse} or {dense} already exist. To overwrite rerun with '--force'"
        )

    db.unlink(missing_ok=True)
    shutil.rmtree(str(text), ignore_errors=True)
    shutil.rmtree(str(sparse), ignore_errors=True)
    shutil.rmtree(str(dense), ignore_errors=True)
    shutil.rmtree(str(log), ignore_errors=True)
    text.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(parents=True, exist_ok=True)
    dense.mkdir(parents=True, exist_ok=True)
    log.mkdir(parents=True, exist_ok=True)

    # Generate masks if not present at mask_dir
    if mask_dir and not mask_dir.exists():
        print(f"No masks were found. Generating them...")
        generate_masks(c, input_dir, mask_dir, pattern="*")

    # If camera params is the path to a valid json transforms file (as generated via render-views)
    # then load it up and extract ground truth camera parameters from it.
    if camera_params and camera_params.endswith(".json"):
        if Path(camera_params).exists():
            print(f"\nArgument `camera_params` points to a json file, extracting camera model...")

            with open(camera_params, "r") as f:
                camera_params_data = json.load(f).get("camera")
                params = [camera_params_data.get(k) for k in ("fx", "fy", "cx", "cy")]
                if any(v is None for v in params):
                    raise ValueError(f"Missing one of fx, fy, cx, cy in {camera_params}.")
                camera_params = ", ".join(str(i) for i in params + [0, 0, 0, 0])
                print("Using parameters", camera_params)
        else:
            raise FileNotFoundError("Supplied `camera_params` path not found!")

    # CPU only as we are estimating affine shapes.
    # See: https://github.com/colmap/colmap/issues/1110
    sift_cmd = (
        f"colmap feature_extractor --ImageReader.camera_model {camera_model.upper()} "
        f'--ImageReader.camera_params "{camera_params}" '
        f"--SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true "
        f"--ImageReader.single_camera 1 --database_path {db} --image_path {input_dir} "
        f"--SiftExtraction.use_gpu={'false' if not cuda else 'true'} "
        f"--SiftExtraction.peak_threshold=0.001 "
    )
    sift_cmd += f"--ImageReader.mask_path {mask_dir} " if mask_dir else ""

    match_cmd = (
        f"colmap {matcher}_matcher --SiftMatching.guided_matching=true --database_path {db} "
        f"--SiftMatching.use_gpu={'false' if not cuda else 'true'} "  # GPU requires attached display
        # f"--SiftMatching.max_error=20 --SiftMatching.max_distance=20.0 --SiftMatching.confidence=0.0 "
        # f"--SiftMatching.min_inlier_ratio=0.1 --SiftMatching.min_num_inliers=4"
    )
    match_cmd += f" --VocabTreeMatching.vocab_tree_path {vocab_path}" if vocab_path else ""

    # Mapper can't run on GPU either, only `hierarchical_mapper` can but results in poor reconstruction.
    mapper_cmd = f"colmap {mapper + '_' if mapper else ''}mapper "
    mapper_cmd += f"--database_path {db} --image_path {input_dir} --output_path {sparse} "
    mapper_cmd += (
        f"--Mapper.ba_refine_focal_length=0 --Mapper.ba_refine_principal_point=0 "
        f"--Mapper.ba_refine_extra_params=0 "
        f"--Mapper.min_num_matches=5 --Mapper.init_min_num_inliers=15 "
        # if camera_params
        # else ""
    )

    ba_cmd = (
        f"colmap bundle_adjuster --input_path {sparse}/0 --output_path {sparse}/0 "
        f"--BundleAdjustment.refine_principal_point {int(not bool(camera_params))}"
    )

    convert_cmd = f"colmap model_converter --input_path {sparse}/0 --output_path {text} --output_type TXT"

    undistort_cmd = (
        f"colmap image_undistorter --image_path {input_dir} --input_path {sparse}/0 "
        f"--output_path {dense} --output_type COLMAP --max_image_size 2000"
    )

    patch_match_cmd = (
        f"colmap patch_match_stereo --workspace_path {dense} "
        f"--workspace_format COLMAP --PatchMatchStereo.geom_consistency true"
    )

    fusion_cmd = (
        f"colmap stereo_fusion --workspace_path {dense} --workspace_format COLMAP "
        f"--input_type geometric --output_path {dense}/fused.ply"
    )

    poisson_mesher_cmd = f"colmap poisson_mesher --input_path {dense}/fused.ply --output_path {dense}/meshed-poisson.ply"

    delaunay_mesher_cmd = f"colmap delaunay_mesher --input_path {dense} --output_path {dense}/meshed-delaunay.ply"

    print(
        f"\nRunning colmap with:\n"
        f"  -images from: {input_dir}\n"
        f"  -output will be in: {output_dir}\n"
        f"  -logs will be saved in: {log}\n"
    )
    print(f"\nFound {len(in_files)} files in {input_dir}...")

    print("\nExtracting Features...")
    with StreamWatcherTqdmPbar(r"Processed file \[(?P<n>\d+)/\d+\]", total=len(in_files)) as pbar_watcher:
        _log_run(c, sift_cmd, log / "feature_extractor", watchers=pbar_watcher)

    print("\nMatching Features...")
    with StreamWatcherTqdmPbar(r"Matching image \[\d+/\d+\]", total=len(in_files)) as pbar_watcher:
        _log_run(c, match_cmd, log / f"{matcher}_matcher", watchers=pbar_watcher)

    print("\nRunning Mapper...")
    with StreamWatcherTqdmPbar(
        r"Registering image #\d+ \((?P<n>\d+)\)",
        on_match={
            "No good initial image pair found.": functools.partial(
                _raise_callback, err_type=RuntimeError, message="No good initial image pair found."
            )
        },
        total=len(in_files),
    ) as pbar_watcher:
        _log_run(c, mapper_cmd, log / "mapper", watchers=pbar_watcher)

    if not skip_ba:
        print("\nBundle Adjusting...")
        _log_run(c, ba_cmd, log / "bundle_adjuster")
    else:
        print("\nSkipping bundle adjustments...")
        if not camera_params:
            print("WARNING: Skipping BA is not recommended if no camera parameters are specified.")

    print("\nConverting results to text format...")
    _log_run(c, convert_cmd, log / "convert")

    print("\nConverting results to NeRF format...")
    to_nerf_format(c, input_dir, text, keep_colmap_coords=False, aabb_scale=16)

    # Early exit if needed
    if not dense_reconstruction:
        return

    print("\nStarting dense reconstruction process (this will take a while)...")
    print("\nUndistorting Images...")
    with StreamWatcherTqdmPbar(r"Undistorting image \[(?P<n>\d+)/\d+\]", total=len(in_files)) as pbar_watcher:
        _log_run(c, undistort_cmd, log / "undistorter", watchers=pbar_watcher)

    print("\nMatching Image Patches...")
    # This is twice as long as it matches based on geometric and photometric loss
    with StreamWatcherTqdmPbar(r"Processing view \d+ / \d+", total=len(in_files) * 2) as pbar_watcher:
        _log_run(c, patch_match_cmd, log / "patch_match", watchers=pbar_watcher)

    print("\nFusing Image Patches...")
    with StreamWatcherTqdmPbar(r"Fusing image \[(?P<n>\d+)/\d+\]", total=len(in_files)) as pbar_watcher:
        _log_run(c, fusion_cmd, log / "fusion", watchers=pbar_watcher)

    print("\nConverting results to a Poisson Mesh...")
    _log_run(c, poisson_mesher_cmd, log / "poisson_mesher")

    print("\nConverting results to a Delaunay Mesh...")
    with StreamWatcherTqdmPbar(r"Integrating image \[(?P<n>\d+)/\d+\]", total=len(in_files)) as pbar_watcher:
        _log_run(c, delaunay_mesher_cmd, log / "delaunay_mesher", watchers=pbar_watcher)


@task
def to_nerf_format(
    _,
    input_dir,
    text_dir,
    keep_colmap_coords=False,
    aabb_scale=16,
    indices=None,
    file_name="transforms.json",
    sharpness=False,
):
    """Convert transform.json from colmap format to nerf-style format"""
    import ast

    from spsim.colmaptools import convert_from_colmap  # Lazy import

    transforms_path = (Path(text_dir).parent / file_name).resolve()
    transforms_path.unlink(missing_ok=True)

    convert_from_colmap(
        input_dir,
        text_dir,
        transforms_path,
        aabb_scale=aabb_scale,
        indices=ast.literal_eval(indices) if indices else slice(None),
        keep_colmap_coords=keep_colmap_coords,
        sharpness=sharpness,
    )
