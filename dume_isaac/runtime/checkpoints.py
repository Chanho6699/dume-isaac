"""Resolve RSL-RL checkpoints under logs/rsl_rl/<experiment>/<run>/model_<iter>.pt. Pure Python."""

from __future__ import annotations

import re
from pathlib import Path

_MODEL_RE = re.compile(r"^model_(\d+)\.pt$")


def list_checkpoints(run_dir: Path) -> list[Path]:
    ckpts = [p for p in run_dir.glob("model_*.pt") if _MODEL_RE.match(p.name)]
    return sorted(ckpts, key=lambda p: int(_MODEL_RE.match(p.name).group(1)))  # type: ignore[union-attr]


def resolve_checkpoint(log_root: Path, checkpoint: str | None = None, load_run: str | None = None) -> Path:
    """Return a checkpoint path.

    - `checkpoint` is an existing file path -> used as is.
    - otherwise pick the run `load_run` (exact dir name or regex), default the latest run
      (run dirs are timestamp-prefixed, so name order == time order), then the file named
      `checkpoint` (e.g. model_500.pt) or the highest-iteration model_*.pt.
    """
    if checkpoint and Path(checkpoint).expanduser().is_file():
        return Path(checkpoint).expanduser().resolve()

    log_root = Path(log_root)
    if not log_root.is_dir():
        raise FileNotFoundError(f"No training logs at {log_root}. Train first (script09 / script11) or pass --checkpoint <file>.")
    runs = sorted(p for p in log_root.iterdir() if p.is_dir())
    if load_run:
        runs = [p for p in runs if p.name == load_run or re.search(load_run, p.name)]
    runs = [p for p in runs if list_checkpoints(p)]
    if not runs:
        raise FileNotFoundError(f"No run with model_*.pt under {log_root}" + (f" matching '{load_run}'" if load_run else ""))
    run = runs[-1]

    if checkpoint:
        cand = run / checkpoint
        if not cand.is_file():
            raise FileNotFoundError(f"{checkpoint} not found in {run}; available: {[p.name for p in list_checkpoints(run)]}")
        return cand.resolve()
    return list_checkpoints(run)[-1].resolve()
