"""Tabletop geometry resolved from configs/tabletop.yaml. Pure Python (no Isaac imports).

Each value is taken from the measured section when it is not null, otherwise from
`provisional_defaults`; `status` records which one was used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dume_isaac.config_io import load_yaml
from dume_isaac.paths import CONFIG_DIR

DIFFICULTIES = ("easy", "medium", "wide")
MEASURED = "measured"
PROVISIONAL = "provisional"

Range = tuple[float, float]


@dataclass(frozen=True)
class RandomizationLevel:
    name: str
    robot_joint_noise: float
    cube_x: Range
    cube_y: Range
    target_x: Range
    target_y: Range
    min_separation: float
    reach_x: Range
    reach_y: Range
    reach_z: Range


@dataclass(frozen=True)
class TabletopLayout:
    scene_type: str
    table_width: float   # along y
    table_depth: float   # along x
    table_height: float
    robot_base_in_table: tuple[float, float, float]
    robot_base_rot: tuple[float, float, float, float]
    workspace_x: Range
    workspace_y: Range
    workspace_z: Range
    cube_size: float
    cube_mass: float
    cube_friction: float
    target_size: tuple[float, float]  # footprint (x, y)
    status: dict[str, str]
    levels: dict[str, RandomizationLevel]

    @property
    def table_top_z_in_base(self) -> float:
        """Height of the table top in the robot base frame."""
        return -self.robot_base_in_table[2]

    @property
    def table_center_in_base(self) -> tuple[float, float, float]:
        """Table-top center in the robot base frame (base axes aligned with table axes)."""
        x, y, z = self.robot_base_in_table
        return (-x, -y, -z)

    @property
    def robot_base_in_env(self) -> tuple[float, float, float]:
        """Robot base position in the per-env frame (env origin on the ground under the table frame)."""
        x, y, z = self.robot_base_in_table
        return (x, y, self.table_height + z)

    def level(self, name: str) -> RandomizationLevel:
        if name not in self.levels:
            raise ValueError(f"unknown difficulty '{name}', choose from {list(self.levels)}")
        return self.levels[name]


def _pick(measured: Any, default: Any, key: str, status: dict[str, str]) -> Any:
    if measured is not None:
        status[key] = MEASURED
        return measured
    if default is None:
        raise ValueError(f"configs/tabletop.yaml: '{key}' has neither a measured value nor a provisional default")
    status[key] = PROVISIONAL
    return default


def _range(v: Any, key: str) -> Range:
    lo, hi = (float(x) for x in v)
    if not lo <= hi:
        raise ValueError(f"configs/tabletop.yaml: range '{key}' has min > max: {v}")
    return (lo, hi)


def load_tabletop_layout(path: str | Path | None = None) -> TabletopLayout:
    data = load_yaml(path or CONFIG_DIR / "tabletop.yaml")
    d = data.get("provisional_defaults") or {}
    st: dict[str, str] = {}

    table, dt = data.get("table") or {}, d.get("table") or {}
    base, db = data.get("robot_base") or {}, d.get("robot_base") or {}
    ws, dws = data.get("workspace_bounds") or {}, d.get("workspace_bounds") or {}
    objs, dobjs = data.get("objects") or {}, d.get("objects") or {}
    cube, dcube = objs.get("cube") or {}, dobjs.get("cube") or {}
    bin_, dbin = objs.get("bin") or {}, dobjs.get("bin") or {}

    target_size = _pick(bin_.get("size"), dbin.get("size"), "bin.size", st)
    levels = {}
    for name in DIFFICULTIES:
        lv = (data.get("randomization") or {}).get(name)
        if lv is None:
            raise ValueError(f"configs/tabletop.yaml: randomization.{name} is missing")
        levels[name] = RandomizationLevel(
            name=name,
            robot_joint_noise=float(lv["robot_joint_noise"]),
            min_separation=float(lv["min_separation"]),
            **{k: _range(lv[k], f"{name}.{k}") for k in
               ("cube_x", "cube_y", "target_x", "target_y", "reach_x", "reach_y", "reach_z")},
        )

    return TabletopLayout(
        scene_type=str(data.get("scene_type")),
        table_width=float(_pick(table.get("width"), dt.get("width"), "table.width", st)),
        table_depth=float(_pick(table.get("depth"), dt.get("depth"), "table.depth", st)),
        table_height=float(_pick(table.get("height"), dt.get("height"), "table.height", st)),
        robot_base_in_table=tuple(float(v) for v in _pick(base.get("position"), db.get("position"), "robot_base.position", st)),  # type: ignore[arg-type]
        robot_base_rot=tuple(float(v) for v in _pick(base.get("orientation"), db.get("orientation"), "robot_base.orientation", st)),  # type: ignore[arg-type]
        workspace_x=_range(_pick(ws.get("x"), dws.get("x"), "workspace_bounds.x", st), "workspace_bounds.x"),
        workspace_y=_range(_pick(ws.get("y"), dws.get("y"), "workspace_bounds.y", st), "workspace_bounds.y"),
        workspace_z=_range(_pick(ws.get("z"), dws.get("z"), "workspace_bounds.z", st), "workspace_bounds.z"),
        cube_size=float(_pick(cube.get("size"), dcube.get("size"), "cube.size", st)),
        cube_mass=float(_pick(cube.get("mass"), dcube.get("mass"), "cube.mass", st)),
        cube_friction=float(_pick(cube.get("friction"), dcube.get("friction"), "cube.friction", st)),
        # bin size is [width(y), depth(x), (height)]; the pad uses the footprint as (x, y)
        target_size=(float(target_size[1]), float(target_size[0])),
        status=st,
        levels=levels,
    )
