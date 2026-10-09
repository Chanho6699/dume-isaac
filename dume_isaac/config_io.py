"""YAML loading with actionable errors. Pure Python (needs only PyYAML)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        import yaml  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the interpreter
        raise ImportError(
            f"PyYAML is required to read {path}. It ships with Isaac Lab; at home use "
            "`python3 -m pip install pyyaml` (or the dev extra)."
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data
