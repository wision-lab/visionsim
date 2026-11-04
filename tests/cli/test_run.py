import subprocess
import sys

import pytest

import visionsim
from visionsim.cli import _run, main


def test_run_raises_on_failure():
    with pytest.raises(subprocess.CalledProcessError):
        _run([sys.executable, "-c", "import sys; sys.exit(3)"], hide=True)


def test_run_can_ignore_failure():
    result = _run([sys.executable, "-c", "import sys; sys.exit(3)"], hide=True, check=False)

    assert result.returncode == 3


def test_main_prints_version(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["visionsim", "--version"])

    main()

    assert capsys.readouterr().out.strip() == f"visionsim {visionsim.__version__}"
