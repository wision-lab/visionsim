from __future__ import annotations

import platform
import re
from pathlib import Path


def _run(c, command, watchers=None, **kwargs):
    watchers = (
        [
            watchers,
        ]
        if watchers is not None and not isinstance(watchers, list)
        else watchers
    )
    return c.run(command, pty=platform.system() != "Windows", watchers=watchers, **kwargs)


def _log_run(c, command, log_path, echo=True, **kwargs):
    Path(log_path).mkdir(parents=True, exist_ok=True)
    log_out = Path(log_path).resolve() / "out.log"
    log_err = Path(log_path).resolve() / "err.log"

    with open(str(log_out), "w") as f_out:
        if echo:
            f_out.write(f"$ {command}\n")
        if platform.system() != "Windows":
            _run(c, command, out_stream=f_out, **kwargs)
        else:
            with open(str(log_err), "w") as f_err:
                _run(c, command, out_stream=f_out, err_stream=f_err, **kwargs)


def _raise_callback(*args, err_type=ValueError, message="", **kwargs):
    raise err_type(message)


def _validate_directories(input_dir, output_dir=None, pattern=None):
    import glob  # Lazy import

    from natsort import natsorted

    input_dir = Path(input_dir).resolve()

    if output_dir is not None:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise RuntimeError(f"Input directory {input_dir} does not exist.")

    if pattern:
        # Pattern might be ffmpeg-style like "frames_%06d.png", convert to "frames_*.png".
        pattern = re.sub(r"(%\d+d)", "*", pattern)
        if not (in_files := glob.glob(str(input_dir / pattern))):
            raise FileNotFoundError(f"No files matching {pattern} found in {input_dir}.")
        in_files = natsorted(in_files)
        return input_dir, output_dir, in_files
    return input_dir, output_dir
