from __future__ import annotations

import shutil
from pathlib import Path


def convert(
    input_dir: Path,
    output_dir: Path | None = None,
    force: bool = False,
):
    """Convert a `.db` database to a `.json` or vice-versa.

    Args:
        input_dir: directory in which to look for dataset
        output_dir: directory in which to save new dataset.
            If not set, save new metadata file in same directory,
            otherwise copy over all data to a new directory.
        force: if true, overwrite output file(s) if present
    """
    from visionsim.dataset import Metadata

    if output_dir:
        if input_dir.resolve() == output_dir.resolve():
            raise RuntimeError("Input and output directory cannot be the same!")
        if output_dir.exists() and not force:
            raise FileExistsError("Output directory already exists.")
        else:
            shutil.rmtree(output_dir, ignore_errors=True)

    meta = Metadata.from_path(input_dir)
    assert meta.path is not None

    rel_path = meta.path.relative_to(input_dir.resolve())
    meta_path = rel_path.with_suffix(".db" if meta.path.suffix == ".json" else ".json")

    if output_dir:
        shutil.copytree(input_dir, output_dir)
        meta.save(output_dir / meta_path)
        (output_dir / rel_path).unlink(missing_ok=True)
    else:
        meta.save(input_dir / meta_path)
