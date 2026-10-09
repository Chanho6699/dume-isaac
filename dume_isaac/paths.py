"""Repository paths. Pure Python (no Isaac imports)."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(os.environ.get("DUME_ISAAC_CONFIG_DIR", REPO_ROOT / "configs"))
MY_SO101_DIR = REPO_ROOT / "dume_isaac" / "assets" / "my_so101"
LOG_ROOT = REPO_ROOT / "logs" / "rsl_rl"


def repo_path(path: str | os.PathLike) -> Path:
    """Resolve a path from a config file: absolute stays, relative is taken from the repo root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p
