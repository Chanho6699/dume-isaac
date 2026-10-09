"""Fixed-base tabletop scene (table + my_so101 + cube + target pad).

Isaac-free at import time: only the pure `layout` module is exported here.
`scene_cfg`, `env_cfg`, `frames` need a running Isaac app.
"""

from .layout import DIFFICULTIES, TabletopLayout, load_tabletop_layout

__all__ = ["DIFFICULTIES", "TabletopLayout", "load_tabletop_layout"]
