import subprocess
import sys

import pytest

from visionsim.cli import _run


def test_run_raises_on_failure():
    with pytest.raises(subprocess.CalledProcessError):
        _run([sys.executable, "-c", "import sys; sys.exit(3)"], hide=True)


def test_run_can_ignore_failure():
    result = _run([sys.executable, "-c", "import sys; sys.exit(3)"], hide=True, check=False)

    assert result.returncode == 3
