from __future__ import annotations

import os
from pathlib import Path


def animate(
    input_dir: str | os.PathLike,
    pattern: str = "frame_*.png",
    outfile: str | os.PathLike = "out.mp4",
    fps: int = 25,
    crf: int = 22,
    vcodec: str = "libx264",
    step: int = 1,
    multiple: int = 2,
    force: bool = False,
    bg_color: str = "black",
    strip_alpha: bool = False,
):
    """Combine generated frames into an MP4 using ffmpeg wizardry

    Args:
        input_dir: directory in which to look for frames,
        pattern: filenames of frames should match this
        outfile: where to save generated mp4
        fps: frames per second in video
        crf: constant rate factor for video encoding (0-51), lower is better quality but more memory
        vcodec: video codec to use (either libx264 or libx265)
        step: drop some frames when making video, use frames 0+step*n
        multiple: some codecs require size to be a multiple of n
        force: if true, overwrite output file if present
        bg_color: for images with transparencies, namely PNGs, use this color as a background
        strip_alpha: if true, do not pre-process PNGs to remove transparencies
    """
    import tempfile  # Lazy import

    from visionsim.cli import _run, _validate_directories

    if _run("ffmpeg -version").returncode != 0:
        raise RuntimeError("No ffmpeg installation found on path!")

    *_, _, in_files = _validate_directories(input_dir, Path(outfile).parent, pattern=pattern)

    # See: https://stackoverflow.com/questions/52804749
    strip_alpha_filter = (
        (
            f'-filter_complex "color={bg_color},format=rgb24[c];[c][0]scale2ref[c][i];'
            f'[c][i]overlay=format=auto:shortest=1,setsar=1" '
        )
        if pattern.endswith(".png") and strip_alpha
        else ""
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # There's no easy way to select out a subset of frames to use.
        # The select filter (-vf "select=not(mod(n\,step))") interferes with
        # the PNG alpha channel removal, and the concat muxer doesn't work
        # with images or leads to errors.
        # As a quick fix, we create a tmpdir with symlinks to the frames we
        # want to include and point ffmpeg to those.

        tmpdirname = Path(tmpdir)
        ext = str(Path(pattern).suffix)

        # No transformation needed, simply symlink files
        for i, p in enumerate(in_files[::step]):
            (tmpdirname / f"{i:09}{ext}").symlink_to(p, target_is_directory=False)

        cmd = (
            f"ffmpeg -framerate {fps} -f image2 -i {tmpdirname / ('%09d' + ext)} {strip_alpha_filter}"
            f"{'-y' if force else ''} -vcodec {vcodec} -crf {crf} -pix_fmt yuv420p "
        )
        if multiple:
            cmd += f"-vf scale=-{multiple}:2048 "

        cmd += f"{outfile} "
        _run(cmd)


def combine(
    matrix: str,
    outfile: str = "combined.mp4",
    mode: str = "shortest",
    color: str = "white",
    multiple: int = 2,
    force: bool = False,
):
    """Combine multiple videos into one by stacking, padding and resizing them using ffmpeg.

    Internally this task will first optionally pad all videos to length using ffmpeg's `tpad` filter,
    then `scale` all videos in a row to have the same height, combine rows together using the `hstack`
    filter before finally `scale`ing row-videos to have same width and `vstack`ing them together.

    Args:
        matrix: Way to specify videos to combine as a 2D matrix of file paths
        outfile: where to save generated mp4
        mode: if 'shortest' combined video will last as long s shortest input video. If 'static', the last frame of videos that are shorter than the longest input video will be repeated. If 'pad', all videos as padded with frames of `color` to last the same duration.
        color: color to pad videos with, only used if mode is 'pad'
        multiple: some codecs require size to be a multiple of n
        force: if true, overwrite output file if present


    Examples:
        The input videos can also be specified in a 2D array using the `--matrix` argument like so:
        $ visionsim ffmpeg.combine --matrix='[["a.mp4", "b.mp4"]]' --outfile="output.mp4"
    """
    # TODO: Allow borders and use xstack for better performance
    #   See: https://stackoverflow.com/questions/11552565/vertically-or-horizontally-stack-mosaic-several-videos-using-ffmpeg/33764934#33764934

    import ast
    import shutil
    import tempfile

    import numpy as np

    from visionsim.cli import _log, _run

    if Path(outfile).is_file() and not force:
        raise RuntimeError("Output file already exists, either specify different output path or `--force` to override.")

    if _run("ffmpeg -version").returncode != 0:
        raise RuntimeError("No ffmpeg installation found on path!")

    matrix = ast.literal_eval(matrix) if isinstance(matrix, str) else matrix
    flat_mat = [path for row in matrix for path in row]

    try:
        if any(not Path(p).is_file() for p in flat_mat):
            raise FileNotFoundError(
                "Expected video matrix to contain valid file paths or newline "
                "delimiters such as '\\n'/'\\r' or 'newline'/'enter'"
            )
    except TypeError:
        raise RuntimeError("Expected video matrix to be 2D.")

    if mode.lower() not in ("shortest", "static", "pad"):
        raise ValueError(f"Expected `mode` to be one of 'shortest', 'static', 'pad' but got {mode}.")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Keep track of new names of mp4s
        mapping: dict[str, Path] = {}
        row_paths: list[Path] = []

        # Keep track of all original dimensions
        sizes = {path: dimensions(path) for path in flat_mat}

        # Find longest video and pad all to this length
        if mode.lower() == "pad":
            max_duration = max(duration(path) for path in flat_mat)

            for path in flat_mat:
                _log.info(f"Padding {path}...")
                out_path = Path(tmpdir) / Path(path).name
                out_path = out_path.with_name(f"{out_path.stem}_padded{out_path.suffix}")
                cmd = f"ffmpeg -i {path} -vf tpad=stop=-1=color={color},trim=end={max_duration} {out_path} -y"
                _run(cmd)
                mapping[path] = out_path

        # If the matrix is not jagged, we can use ffmpeg's xstack instead
        if len(num_cols := set(len(row) for row in matrix)) == 1:
            in_paths = [mapping.get(p, p) for row in matrix for p in row]
            in_paths_str = "".join(f"-i {p} " for p in in_paths)
            filter_inputs_str = "".join(
                f"[{i}:v] setpts=PTS-STARTPTS, scale=qvga [a{i}]; " for i, _ in enumerate(in_paths)
            )
            W, H = np.meshgrid(
                ["+".join(f"w{i}" for i in range(j)) or "0" for j in range(num_cols.pop())],
                ["+".join(f"h{i}" for i in range(j)) or "0" for j in range(len(matrix))],
            )
            layout_spec = "|".join(f"{i}_{j}" for i, j in zip(W.flatten(), H.flatten()))
            placement = (
                "".join(f"[a{i}]" for i, _ in enumerate(in_paths))
                + f"xstack=inputs={len(in_paths)}:layout={layout_spec}[out]"
            )
            cmd = f'ffmpeg {in_paths_str} -filter_complex "{filter_inputs_str} {placement}" -map "[out]" -c:v libx264 {outfile}'
            _run(cmd, echo=True)
            return

        for i, row in enumerate(matrix):
            # Resize videos in each row
            max_height = max(sizes[path][1] for path in row)
            for path in row:
                if sizes[path][1] != max_height:
                    _log.info(f"Resizing {path}...")
                    in_path = mapping.get(path, path)
                    out_path = Path(tmpdir) / Path(path).name
                    out_path = out_path.with_name(f"{out_path.stem}_height_resize{out_path.suffix}")
                    _run(f"ffmpeg -i {in_path} -vf scale=-{multiple}:{max_height} {out_path} -y")
                    mapping[path] = out_path

            # Combine all videos in the row
            if len(row) >= 2:
                _log.info("Stacking rows...")
                paths = " -i ".join(str(mapping.get(p, p)) for p in row)
                out_file = Path(tmpdir) / f"row_{i:04}.mp4"
                row_paths.append(out_file)
                cmd = (
                    f"ffmpeg -i {paths} -filter_complex "
                    f"hstack=inputs={len(row)}:shortest={int(mode.lower() == 'shortest')} "
                    f"{out_file} -vsync vfr -y"
                )
                _run(cmd)
            else:
                row_paths.append(mapping.get(row[0], Path(row[0])))

        # Combine all rows
        if len(matrix) >= 2:
            # Resize row videos if needed
            row_sizes: dict[Path, tuple] = {path: dimensions(path) for path in row_paths}
            max_width: int = max(row_sizes[path][0] for path in row_paths)
            new_row_paths = []

            for path in row_paths:
                if row_sizes[path][0] != max_width:
                    _log.info(f"Resizing {path}...")
                    out_path = Path(tmpdir) / Path(path).name
                    out_path = out_path.with_name(f"{out_path.stem}_width_resize{out_path.suffix}")
                    _run(f"ffmpeg -i {path} -vf scale={max_width}:-{multiple} {out_path} -y")
                    new_row_paths.append(out_path)
                else:
                    new_row_paths.append(Path(path))

            # Join all row videos
            paths = " -i ".join(str(p) for p in new_row_paths)
            cmd = (
                f"ffmpeg -i {paths} -filter_complex "
                f"vstack=inputs={len(matrix)}:shortest={int(mode.lower() == 'shortest')} "
                f"{outfile} -vsync vfr -y"
            )
            _run(cmd)
        else:
            # We already created the video, simply move/rename it to output file
            shutil.move(row_paths[0], outfile)


def grid(
    input_dir: str | os.PathLike,
    width: int = -1,
    height: int = -1,
    pattern: str = "*.mp4",
    outfile: str = "combined.mp4",
    force: bool = False,
):
    """Make a mosaic from videos in a folder, organizing them in a grid

    Args:
        input_dir: directory containing all video files (mp4's expected),
        width: width of video grid to produce
        height: height of video grid to produce
        pattern: use files that match this pattern as inputs
        outfile: where to save generated mp4
        force: if true, overwrite output file if present
    """
    import numpy as np
    from natsort import natsorted

    files = natsorted(Path(input_dir).glob(pattern))

    if width <= 0 and height <= 0:
        candidates = [
            (w, int(len(files) / w)) for w in range(1, len(files) + 1) if int(len(files) / w) == (len(files) / w)
        ]

        print("Please select size (width x height):")
        for i, candidate in enumerate(candidates):
            print(f"{i}) {candidate}")
        selection = int(input(">  "))
        width, height = candidates[selection]
    elif width <= 0:
        width = len(files) // height
    elif height <= 0:
        height = len(files) // width

    if int(width) != width or int(height) != height:
        raise ValueError(f"Width and height should be integers, instead got {width}, {height}.")
    else:
        width, height = int(width), int(height)

    matrix = np.array([str(p) for p in files]).reshape((height, width)).tolist()
    combine(str(matrix), outfile, force=force)


def count_frames(input_file: str | os.PathLike):
    """Count the number of frames a video file contains using ffprobe

    Args:
        input_file: video file input
    """
    from visionsim.cli import _log, _run

    # See: https://stackoverflow.com/questions/2017843
    if _run("ffprobe -version").returncode != 0:
        raise RuntimeError("No ffprobe installation found on path!")

    cmd = (
        f"ffprobe -v error -select_streams v:0 -count_packets -show_entries "
        f"stream=nb_read_packets -of csv=p=0 {input_file}"
    )
    result = _run(cmd)
    _log.info(f"Video contains {int(result.stdout.strip())} frames.")
    return int(result.stdout.strip())


def duration(input_file: str | os.PathLike, /):
    """Return duration (in seconds) of first video stream in file using ffprobe


    Args:
        input_file: video file input
    """
    from visionsim.cli import _log, _run

    # See: http://trac.ffmpeg.org/wiki/FFprobeTips#Duration
    if _run("ffprobe -version").returncode != 0:
        raise RuntimeError("No ffprobe installation found on path!")

    cmd = (
        f"ffprobe -v error -select_streams v:0 -show_entries stream=duration "
        f"-of default=noprint_wrappers=1:nokey=1 {input_file}"
    )
    result = _run(cmd)
    _log.info(f"Video lasts {float(result.stdout.strip())} seconds.")
    return float(result.stdout.strip())


def dimensions(input_file: str | os.PathLike):
    """Return size (WxH in pixels) of first video stream in file using ffprobe

    Args:
        input_file: video file input
    """
    from visionsim.cli import _log, _run

    # See: http://trac.ffmpeg.org/wiki/FFprobeTips#Duration
    if _run("ffprobe -version").returncode != 0:
        raise RuntimeError("No ffprobe installation found on path!")

    cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 {input_file}"
    result = _run(cmd)
    _log.info(f"Video has size {result.stdout.strip()}.")
    return tuple(int(dim) for dim in result.stdout.strip().split("x"))


def extract(input_file: str | os.PathLike, output_dir: str | os.PathLike, pattern: str = "frames_%06d.png"):
    """Extract frames from video file

    Args:
        input_file: path to video file from which to extract frames,
        output_dir: directory in which to save extracted frames,
        pattern: filenames of frames will match this pattern
    """
    from visionsim.cli import _run

    if _run("ffmpeg -version").returncode != 0:
        raise RuntimeError("No ffmpeg installation found on path!")
    if not Path(input_file).is_file():
        raise FileNotFoundError(f"File {input_file} not found.")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _run(f"ffmpeg -i {input_file} {Path(output_dir) / pattern}")
