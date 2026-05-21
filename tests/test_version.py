"""Version metadata tests."""

from pathlib import Path
import re

import qtf


def test_package_version_matches_pyproject():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    match = re.search(r'^version = "([^"]+)"$', pyproject.read_text(encoding="utf-8"), re.MULTILINE)
    assert match is not None
    assert qtf.__version__ == match.group(1)
