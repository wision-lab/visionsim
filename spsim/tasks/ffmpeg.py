from pathlib import Path

from invoke import task

from spsim.tasks.common import _run, _validate_directories


@task(
    help={
        "input_dir": "directory in which to look for frames",
        "pattern": "filenames of frames should match this, default: 'frame_*.png'",
        "outfile": "where to save generated mp4, default: 'out.mp4'",
        "fps": "frames per second in video, default: 25",
        "crf": "constant rate factor for video encoding (0-51), lower is better quality but more memory, default: 22",
        "vcodec": "video codec to use (either libx264 or libx265), default: libx264",
        "step": "drop some frames when making video, use frames 0+step*n, default: 1",
        "multiple": "some codecs require size to be a multiple of n, default: 2",
        "force": "if true, overwrite output file if present, default: False",
        "bg_color": "for images with transparencies, namely PNGs, use this color as a background, default: 'black'",
        "png_filter": "if false, do not pre-process PNGs to remove transparencies, default: True",
        "auto_tonemap": "if true and images are .exr (linear intensity), apply tonemapping first, default: True",
        "hide": "if true, hide ffmpeg output, default: False",
    }
)
def animate(
    c,
    input_dir,
    pattern="frame_*.png",
    outfile="out.mp4",
    fps=25,
    crf=22,
    vcodec="libx264",
    step=1,
    multiple=2,
    force=False,
    bg_color="black",
    png_filter=True,
    auto_tonemap=True,
    hide=False,
):
    """Combine generated frames into an MP4 using ffmpeg wizardry"""
    # TODO: Auto-tonemap for depth colorization
    import tempfile  # Lazy import

    from natsort import natsorted

    from spsim.tasks.transforms import tonemap_exrs

    if _run(c, "ffmpeg -version", hide=True).failed:
        raise RuntimeError("No ffmpeg installation found on path!")

    input_dir, output_dir, in_files = _validate_directories(input_dir, Path(outfile).parent, pattern=pattern)

    # See: https://stackoverflow.com/questions/52804749
    png_filter = (
        (
            f'-filter_complex "color={bg_color},format=rgb24[c];[c][0]scale2ref[c][i];'
            f'[c][i]overlay=format=auto:shortest=1,setsar=1" '
        )
        if (pattern.endswith(".png") or (pattern.endswith(".exr") and auto_tonemap)) and png_filter
        else ""
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        # There's no easy way to select out a subset of frames to use.
        # The select filter (-vf "select=not(mod(n\,step))") interferes with
        # the PNG alpha channel removal, and the concat muxer doesn't work
        # with images or leads to errors.
        # As a quick fix, we create a tmpdir with symlinks to the frames we
        # want to include and point ffmpeg to those.

        tmpdirname = Path(tmpdirname)
        ext = str(Path(pattern).suffix)

        if pattern.endswith(".exr") and auto_tonemap:
            # Apply tone mapping (linear -> sRGB) to all files
            # Note: `tonemap_exrs` does not rename files, so we have to do it here...
            print("Applying tonemapping to images...")
            tonemap_exrs(c, input_dir, str(tmpdirname), pattern=pattern, ext=".png", step=step)
            for i, p in enumerate(natsorted(tmpdirname.glob("*.png"))):
                p.rename(tmpdirname / f"{i:09}.png")
            ext = ".png"
        else:
            # No transformation needed, simply symlink files
            for i, p in enumerate(in_files[::step]):
                (tmpdirname / f"{i:09}{ext}").symlink_to(p, target_is_directory=False)

        cmd = (
            f"ffmpeg -framerate {fps} -f image2 -i {tmpdirname / ('%09d' + ext)} {png_filter}"
            f"{'-y' if force else ''} -vcodec {vcodec} -crf {crf} -pix_fmt yuv420p "
        )
        if multiple:
            cmd += f"-vf scale=-{multiple}:2048 "

        cmd += f"{outfile} "
        _run(c, cmd, hide=hide)


@task(
    iterable=["inputfiles"],
    help={
        "inputfiles": "video file to combine together or newline separator ('\\n', '\\r', 'newline' or 'enter'), "
        "this argument should be used once per video file.",
        "outfile": "where to save generated mp4, default: 'combined.mp4'",
        "matrix": "alternate way to specify videos to combine as a 2D matrix of file paths, default: None",
        "mode": "if 'shortest' combined video will last as long s shortest input video. If 'static', the last frame of "
        "videos that are shorter than the longest input video will be repeated. If 'pad', all videos as padded "
        "with frames of `color` to last the same duration. default: 'shortest'",
        "color": "color to pad videos with, only used if mode is 'pad'. default: 'white'",
        "multiple": "some codecs require size to be a multiple of n, default: 2",
        "force": "if true, overwrite output file if present, default: False",
    },
)
def combine(c, inputfiles, outfile="combined.mp4", matrix=None, mode="shortest", color="white", multiple=2, force=False):
    """Combine multiple videos into one by stacking, padding and resizing them using ffmpeg.

    Internally this task will first optionally pad all videos to length using ffmpeg's `tpad` filter,
    then `scale` all videos in a row to have the same height, combine rows together using the `hstack`
    filter before finally `scale`ing row-videos to have same width and `vstack`ing them together.

    Examples:
        To combine two videos in a row:
        $ spsim ffmpeg.combine -i "a.mp4" -i "b.mp4" -o "output.mp4"

        To combine two videos in a column:
        $ spsim ffmpeg.combine -i "a.mp4" -i="\n" -i "b.mp4" -o "output.mp4"

        The input videos can also be specified in a 2D array using the `--matrix` argument like so:
        $ spsim ffmpeg.combine --matrix='[["a.mp4", "b.mp4"]]' -o "output.mp4"
    """

    import ast
    import shutil
    import tempfile

    import more_itertools as mitertools
    import numpy as np

    if Path(outfile).is_file() and not force:
        raise RuntimeError("Output file already exists, either specify different output path or `--force` to override.")

    if not (len(inputfiles) != 0) ^ (matrix is not None):
        raise ValueError("Use either `matrix` or `inputfile` argument, not both.")

    if _run(c, "ffmpeg -version", hide=True).failed:
        raise RuntimeError("No ffmpeg installation found on path!")

    if inputfiles:
        matrix = list(
            mitertools.split_at(inputfiles, lambda i: i.lower() in ("n", "r", "\\n", "\\r", "enter", "newline"))
        )
    else:
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
        mapping = {}
        row_paths = []

        # Keep track of all original dimensions
        sizes = {path: dimensions(c, path) for path in flat_mat}

        # Find longest video and pad all to this length
        if mode.lower() == "pad":
            max_duration = max(duration(c, path) for path in flat_mat)

            for path in flat_mat:
                print(f"\n\nPadding {path}...")
                out_path = Path(tmpdir) / Path(path).name
                out_path = out_path.with_name(f"{out_path.stem}_padded{out_path.suffix}")
                cmd = f"ffmpeg -i {path} -vf tpad=stop=-1=color={color},trim=end={max_duration} {out_path} -y"
                _run(c, cmd)
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
            _run(c, cmd, echo=True)
            return

        for i, row in enumerate(matrix):
            # Resize videos in each row
            max_height = max(sizes[path][1] for path in row)
            for path in row:
                if sizes[path][1] != max_height:
                    print(f"\n\nResizing {path}...")
                    in_path = mapping.get(path, path)
                    out_path = Path(tmpdir) / Path(path).name
                    out_path = out_path.with_name(f"{out_path.stem}_height_resize{out_path.suffix}")
                    _run(c, f"ffmpeg -i {in_path} -vf scale=-{multiple}:{max_height} {out_path} -y")
                    mapping[path] = out_path

            # Combine all videos in the row
            if len(row) >= 2:
                print("\n\nStacking rows...")
                paths = " -i ".join(str(mapping.get(p, p)) for p in row)
                out_file = Path(tmpdir) / f"row_{i:04}.mp4"
                row_paths.append(out_file)
                cmd = (
                    f"ffmpeg -i {paths} -filter_complex "
                    f"hstack=inputs={len(row)}:shortest={int(mode.lower() == 'shortest')} "
                    f"{out_file} -vsync vfr -y"
                )
                _run(c, cmd)
            else:
                row_paths.append(mapping.get(row[0], row[0]))

        # Combine all rows
        if len(matrix) >= 2:
            # Resize row videos if needed
            row_sizes = {path: dimensions(c, path) for path in row_paths}
            max_width = max(row_sizes[path][0] for path in row_paths)
            new_row_paths = []

            for path in row_paths:
                if row_sizes[path][0] != max_width:
                    print(f"\n\nResizing {path}...")
                    out_path = Path(tmpdir) / Path(path).name
                    out_path = out_path.with_name(f"{out_path.stem}_width_resize{out_path.suffix}")
                    _run(c, f"ffmpeg -i {path} -vf scale={max_width}:-{multiple} {out_path} -y")
                    new_row_paths.append(out_path)
                else:
                    new_row_paths.append(path)

            # Join all row videos
            paths = " -i ".join(str(p) for p in new_row_paths)
            cmd = (
                f"ffmpeg -i {paths} -filter_complex "
                f"vstack=inputs={len(matrix)}:shortest={int(mode.lower() == 'shortest')} "
                f"{outfile} -vsync vfr -y"
            )
            _run(c, cmd)
        else:
            # We already created the video, simply move/rename it to output file
            shutil.move(row_paths[0], outfile)


@task(
    help={
        "input_dir": "directory containing all video files (mp4's expected)",
        "width": "width of video grid to produce, default: -1 (infer)",
        "height": "height of video grid to produce, default: -1 (infer)",
        "outfile": "where to save generated mp4, default: 'combined.mp4'",
        "force": "if true, overwrite output file if present, default: False",
    },
)
def grid(c, input_dir, width=-1, height=-1, outfile="combined.mp4", force=False):
    import numpy as np
    from natsort import natsorted

    files = natsorted(Path(input_dir).glob("*.mp4"))

    if width <= 0 and height <= 0:
        candidates = [(w, int(len(files) / w)) for w in range(1, len(files)) if int(len(files) / w) == (len(files) / w)]

        print("Please select size (width x height):")
        for i, candidate in enumerate(candidates):
            print(f"{i}) {candidate}")
        selection = int(input(">  "))
        width, height = candidates[selection]
    elif width <= 0:
        width = len(files) / height
    elif height <= 0:
        height = len(files) / width

    if int(width) != width or int(height) != height:
        raise ValueError(f"Width and height should be integers, instead got {width}, {height}.")
    else:
        width, height = int(width), int(height)

    matrix = np.array([str(p) for p in files]).reshape((height, width)).tolist()
    combine(c, [], outfile, matrix=str(matrix), force=force)


@task(help={"input_file": "video file input"})
def count_frames(c, input_file):
    """Count the number of frames a video file contains using ffprobe"""
    # See: https://stackoverflow.com/questions/2017843
    if _run(c, "ffprobe -version", hide=True).failed:
        raise RuntimeError("No ffprobe installation found on path!")

    cmd = (
        f"ffprobe -v error -select_streams v:0 -count_packets -show_entries "
        f"stream=nb_read_packets -of csv=p=0 {input_file}"
    )
    result = _run(c, cmd, hide=True)
    print(f"Video contains {int(result.stdout.strip())} frames.")
    return int(result.stdout.strip())


@task(help={"input_file": "video file input"})
def duration(c, input_file):
    """Return duration (in seconds) of first video stream in file using ffprobe"""
    # See: http://trac.ffmpeg.org/wiki/FFprobeTips#Duration
    if _run(c, "ffprobe -version", hide=True).failed:
        raise RuntimeError("No ffprobe installation found on path!")

    cmd = (
        f"ffprobe -v error -select_streams v:0 -show_entries stream=duration "
        f"-of default=noprint_wrappers=1:nokey=1 {input_file}"
    )
    result = _run(c, cmd, hide=True)
    print(f"Video lasts {float(result.stdout.strip())} seconds.")
    return float(result.stdout.strip())


@task(help={"input_file": "video file input"})
def dimensions(c, input_file):
    """Return size (WxH in pixels) of first video stream in file using ffprobe"""
    # See: http://trac.ffmpeg.org/wiki/FFprobeTips#Duration
    if _run(c, "ffprobe -version", hide=True).failed:
        raise RuntimeError("No ffprobe installation found on path!")

    cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height " f"-of csv=s=x:p=0 {input_file}"
    result = _run(c, cmd, hide=True)
    print(f"Video has size {result.stdout.strip()}.")
    return tuple(int(dim) for dim in result.stdout.strip().split("x"))


@task(
    help={
        "input_file": "path to video file from which to extract frames",
        "output_dir": "directory in which to save extracted frames",
        "pattern": "filenames of frames will match this pattern, default: 'frame_%06d.png'",
    }
)
def extract(c, input_file, output_dir, pattern="frames_%06d.png"):
    """Extract frames from video file"""
    if _run(c, "ffmpeg -version", hide=True).failed:
        raise RuntimeError("No ffmpeg installation found on path!")
    if not Path(input_file).is_file():
        raise FileNotFoundError(f"File {input_file} not found.")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _run(c, f"ffmpeg -i {input_file} {Path(output_dir) / pattern}")
