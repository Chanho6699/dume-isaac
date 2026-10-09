"""Tasks (Reach sanity task, pick-and-place teacher). Isaac-free at import time.

Env/agent configs are imported lazily through `registry` after the Isaac app starts.
"""

from .registry import TASKS, get_task

__all__ = ["TASKS", "get_task"]
