"""Construct configclass objects while tolerating fields missing in the installed version.

Pure Python. Isaac Lab v2.3.2 moved observation normalization from the runner
(`empirical_normalization`) to the policy (`actor_obs_normalization` /
`critic_obs_normalization`). Building cfgs through `compat_cfg` drops keyword arguments
the installed class does not define and reports them, instead of failing at import.
"""

from __future__ import annotations

import dataclasses
from typing import Any


def compat_cfg(cls: type, **kwargs: Any) -> Any:
    try:
        known = {f.name for f in dataclasses.fields(cls)}
    except TypeError:
        return cls(**kwargs)
    dropped = sorted(set(kwargs) - known)
    if dropped:
        print(f"[dume_isaac] {cls.__name__}: ignoring fields not in this version: {dropped}")
    return cls(**{k: v for k, v in kwargs.items() if k in known})
