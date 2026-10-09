"""Fixed-base tabletop scene: ground, light, table, my_so101, blue cube.

The target pad is NOT a scene/physics entity: it is a visual-only marker driven by the
pick-place target command (see `target_pad_marker_cfg`).

Isaac runtime only. Geometry comes from configs/tabletop.yaml (measured value or
provisional default, see layout.py).

Per-env frame: origin on the ground; the robot base sits on the table top at
`layout.robot_base_in_env`. Task code works in the robot base frame.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from dume_isaac.assets.my_so101.robot_cfg import make_my_so101_cfg
from dume_isaac.assets.my_so101.spec import load_so101_spec

from .layout import load_tabletop_layout

SPEC = load_so101_spec()
LAYOUT = load_tabletop_layout()
ENV_SPACING = 2.0
TARGET_PAD_THICKNESS = 0.002  # visual marker only

_BASE = LAYOUT.robot_base_in_env
_MID = LAYOUT.level("medium")


def base_to_env(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Robot base frame -> per-env frame (base axes are aligned with the table axes)."""
    return (_BASE[0] + x, _BASE[1] + y, _BASE[2] + z)


def _mid(r: tuple[float, float]) -> float:
    return 0.5 * (r[0] + r[1])


def cube_rest_z_in_base() -> float:
    return LAYOUT.table_top_z_in_base + 0.5 * LAYOUT.cube_size


def target_z_in_base() -> float:
    return LAYOUT.table_top_z_in_base + 0.5 * TARGET_PAD_THICKNESS


@configclass
class TabletopSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    # static collider (no rigid body props) whose top surface is at table_height
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(LAYOUT.table_depth, LAYOUT.table_width, LAYOUT.table_height),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.45, 0.35)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5 * LAYOUT.table_height)),
    )
    robot: ArticulationCfg = make_my_so101_cfg(spec=SPEC, pos=_BASE, rot=LAYOUT.robot_base_rot)
    cube: RigidObjectCfg | None = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(LAYOUT.cube_size,) * 3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=LAYOUT.cube_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=LAYOUT.cube_friction, dynamic_friction=LAYOUT.cube_friction
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.2, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=base_to_env(_mid(_MID.cube_x), _mid(_MID.cube_y), cube_rest_z_in_base() + 0.002)
        ),
    )


def target_pad_marker_cfg(prim_path: str = "/Visuals/TargetPad") -> VisualizationMarkersCfg:
    """Visual-only target pad (no rigid body, no collider)."""
    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "pad": sim_utils.CuboidCfg(
                size=(LAYOUT.target_size[0], LAYOUT.target_size[1], TARGET_PAD_THICKNESS),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.2)),
            )
        },
    )


def default_target_in_base() -> tuple[float, float, float]:
    return (_mid(_MID.target_x), _mid(_MID.target_y), target_z_in_base())
