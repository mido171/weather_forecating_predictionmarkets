from pathlib import Path

import pytest

from hkg_tmax.paths import find_project_root


@pytest.fixture
def repo_root() -> Path:
    return find_project_root(Path(__file__))
