from __future__ import annotations

import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
TEST_ROOT = PROJECT_ROOT / "test"
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]


def ensure_project_paths() -> None:
    """Make both local and namespace-package imports work."""
    for path in (SRC_ROOT, PROJECT_ROOT, TEST_ROOT, WORKSPACE_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


ensure_project_paths()
