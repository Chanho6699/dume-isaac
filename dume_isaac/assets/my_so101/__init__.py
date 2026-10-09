"""my_so101: SO-101 asset for simulation (Brain Us SO-101 follower, TheRobotStudio SO-101 design).

Importing this package is Isaac-free. `MY_SO101_CFG` / `make_my_so101_cfg` live in
`robot_cfg.py`, which needs a running Isaac Sim app; they are loaded lazily on access.
"""

from .spec import So101Spec, load_so101_spec

__all__ = ["So101Spec", "load_so101_spec", "MY_SO101_CFG", "make_my_so101_cfg"]


def __getattr__(name: str):
    if name in ("MY_SO101_CFG", "make_my_so101_cfg"):
        from . import robot_cfg  # noqa: PLC0415  (Isaac runtime only)

        return getattr(robot_cfg, name)
    raise AttributeError(name)
