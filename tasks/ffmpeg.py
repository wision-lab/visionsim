from pathlib import Path

from invoke import task

from tasks.common import _run, _validate_directories


@task(
    help={
        "input_dir": "directory in which to look for frames",
        "pattern": "filenames of frames should match this, default: 'frame_%06d.png'",
        "outfile": "where to save generated mp4, default: 'out.mp4'",
        "fps": "frames per second in video, default: 25",
        "crf": "constant rate factor for video encoding (0-51), lower is better quality but more memory, default: 22",
        "vcodec": "video codec to use (either libx264 or libx265), default: libx264",
        "step": "drop some frames when making video, use frames 0+step*n, default: 1",
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

    from tasks.transforms import tonemap_exrs

    if _run(c, "ffmpeg -version", hide=True).failed:
        raise RuntimeError(f"No ffmpeg installation found on path!")

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
                (tmpdirname / f"{i:09}{ext}").symlink_to(p)

        cmd = (
            f"ffmpeg -framerate {fps} -f image2 -i {tmpdirname / ('%09d' + ext)} {png_filter}"
            f"{'-y' if force else ''} -vcodec {vcodec} -crf {crf} -pix_fmt yuv420p {outfile}"
        )
        _run(c, cmd, hide=hide)


@task(help={"input_file": "video file input"}, positional=["input_file"])
def count_frames(c, input_file):
    """Count the number of frames a video file contains using ffprobe"""
    # See: https://stackoverflow.com/questions/2017843
    if _run(c, "ffprobe -version", hide=True).failed:
        raise RuntimeError(f"No ffprobe installation found on path!")

    cmd = (
        f"ffprobe -v error -select_streams v:0 -count_packets -show_entries "
        f"stream=nb_read_packets -of csv=p=0 {input_file}"
    )
    result = _run(c, cmd, hide=True)
    print(f"Video contains {result.stdout.strip()} frames.")
