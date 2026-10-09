"""Pick-place full-state teacher env cfg (Isaac Lab v2.3.2 manager-based RL). Isaac runtime only.

Observation (policy, full state, no cameras), robot base frame:
    joint pos/vel (relative to home), TCP position, cube position, cube - TCP,
    target position, target - cube, gripper opening, last action
Target: TargetPadCommand (goal buffer + visual-only marker, no physics), sampled on reset
    at least min_separation away from the freshly reset cube.
Action: 5 arm joint position targets (scale 0.5 around home) + gripper.
    Gripper default = BinaryJointPositionAction (official Isaac Lab lift pattern: one
    open/close decision instead of a continuous jaw angle makes early exploration easier).
    `actions.gripper: continuous` in configs/pick_place.yaml switches to JointPositionAction.
Reward: staged, see mdp/rewards.py. Terminations: timeout, cube out/dropped, optional success.
Reset: scene default -> arm around home -> cube uniform (event) -> target (command).
"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from dume_isaac.envs.tabletop.env_cfg import apply_tabletop_settings
from dume_isaac.envs.tabletop.scene_cfg import (
    LAYOUT,
    SPEC,
    TabletopSceneCfg,
    cube_rest_z_in_base,
    target_pad_marker_cfg,
    target_z_in_base,
)
from dume_isaac.tasks.registry import load_task_params

from . import mdp

PARAMS = load_task_params("pick_place")
REW = PARAMS["rewards"]
THR = dict(PARAMS["thresholds"])
DEFAULT_LEVEL = LAYOUT.level(PARAMS["difficulty"])
EE_OFFSET = tuple(SPEC.ee_offset)
TARGET = "target"  # command name

GEOM = {
    "ee_offset": EE_OFFSET,
    "cube_rest_z": cube_rest_z_in_base(),
    "target_half_xy": (0.5 * LAYOUT.target_size[0], 0.5 * LAYOUT.target_size[1]),
    "gripper_open": SPEC.gripper_open,
    "gripper_closed": SPEC.gripper_closed,
}


def _ee() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=[SPEC.ee_body])


def _gripper() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=[SPEC.gripper_joint_name])


def _robot_joints() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=list(SPEC.all_joint_names), preserve_order=True)


def _state_params(**extra) -> dict:
    return {
        "object_cfg": SceneEntityCfg("cube"),
        "command_name": TARGET,
        "ee_cfg": _ee(),
        "gripper_cfg": _gripper(),
        "geom": GEOM,
        "thr": THR,
        **extra,
    }


def _gripper_action():
    if PARAMS["actions"]["gripper"] == "continuous":
        return mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[SPEC.gripper_joint_name], scale=1.0, use_default_offset=True
        )
    return mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=[SPEC.gripper_joint_name],
        open_command_expr={SPEC.gripper_joint_name: SPEC.gripper_open},
        close_command_expr={SPEC.gripper_joint_name: SPEC.gripper_closed},
    )


def _stage(stage: str) -> RewTerm:
    return RewTerm(func=mdp.staged_reward, weight=REW[stage]["weight"], params=_state_params(stage=stage, rew=REW))


@configclass
class CommandsCfg:
    target = mdp.TargetPadCommandCfg(
        resampling_time_range=(1.0e6, 1.0e6),  # only on episode reset
        debug_vis=True,                        # the visual-only pad marker
        x_range=DEFAULT_LEVEL.target_x,
        y_range=DEFAULT_LEVEL.target_y,
        min_separation=DEFAULT_LEVEL.min_separation,
        z_in_base=target_z_in_base(),
        marker_cfg=target_pad_marker_cfg(),
    )


@configclass
class ActionsCfg:
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(SPEC.arm_joint_names),
        scale=float(PARAMS["actions"]["arm_scale"]),
        use_default_offset=True,
        preserve_order=True,
    )
    gripper_action = _gripper_action()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _robot_joints()})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _robot_joints()})
        ee_pos = ObsTerm(func=mdp.ee_position_b, params={"ee_cfg": _ee(), "offset": EE_OFFSET})
        cube_pos = ObsTerm(func=mdp.object_position_b, params={"object_cfg": SceneEntityCfg("cube")})
        cube_rel_ee = ObsTerm(
            func=mdp.object_minus_ee_b,
            params={"object_cfg": SceneEntityCfg("cube"), "ee_cfg": _ee(), "offset": EE_OFFSET},
        )
        target_pos = ObsTerm(func=mdp.target_position_b, params={"command_name": TARGET})
        target_rel_cube = ObsTerm(
            func=mdp.target_minus_object_b, params={"command_name": TARGET, "object_cfg": SceneEntityCfg("cube")}
        )
        gripper = ObsTerm(
            func=mdp.gripper_opening,
            params={"asset_cfg": _gripper(), "open_pos": SPEC.gripper_open, "closed_pos": SPEC.gripper_closed},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False  # privileged full-state teacher
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(SPEC.arm_joint_names)),
            "position_range": (-DEFAULT_LEVEL.robot_joint_noise, DEFAULT_LEVEL.robot_joint_noise),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_object = EventTerm(
        func=mdp.reset_object_uniform,
        mode="reset",
        params={
            "cube_x": DEFAULT_LEVEL.cube_x,
            "cube_y": DEFAULT_LEVEL.cube_y,
            "cube_z": cube_rest_z_in_base() + 0.002,
            "object_cfg": SceneEntityCfg("cube"),
        },
    )


@configclass
class RewardsCfg:
    # math: dume_isaac/tasks/pick_place/reward_model.py (audited by tests/test_reward_audit.py)
    reach_object = _stage("reach_object")
    grasp = _stage("grasp")
    lift = _stage("lift")
    carry = _stage("carry")
    carry_fine = _stage("carry_fine")
    descend = _stage("descend")
    place = RewTerm(func=mdp.PlaceSuccessReward, weight=REW["place"]["weight"], params=_state_params())
    dropped = RewTerm(func=mdp.is_terminated_term, weight=REW["dropped"]["weight"],
                      params={"term_keys": "object_out_of_bounds"})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=REW["action_rate"]["weight"])
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=REW["joint_vel"]["weight"],
                        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(SPEC.arm_joint_names))})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_out_of_bounds = DoneTerm(
        func=mdp.object_out_of_bounds,
        params={
            "x_range": (LAYOUT.workspace_x[0] - THR["out_of_bounds_margin"], LAYOUT.workspace_x[1] + THR["out_of_bounds_margin"]),
            "y_range": (LAYOUT.workspace_y[0] - THR["out_of_bounds_margin"], LAYOUT.workspace_y[1] + THR["out_of_bounds_margin"]),
            "min_z": LAYOUT.table_top_z_in_base + THR["drop_height"],
            "object_cfg": SceneEntityCfg("cube"),
        },
    )


@configclass
class CurriculumCfg:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": REW["action_rate"]["final_weight"],
                "num_steps": REW["action_rate"]["curriculum_steps"]},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": REW["joint_vel"]["final_weight"],
                "num_steps": REW["joint_vel"]["curriculum_steps"]},
    )


@configclass
class SO101PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    scene: TabletopSceneCfg = TabletopSceneCfg(num_envs=1024, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        apply_tabletop_settings(self, PARAMS)
        if PARAMS.get("terminate_on_success"):
            self.terminations.place_success = DoneTerm(func=mdp.PlaceSuccessTermination, params=_state_params())


def make_env_cfg(difficulty: str | None = None) -> SO101PickPlaceEnvCfg:
    cfg = SO101PickPlaceEnvCfg()
    level = LAYOUT.level(difficulty or PARAMS["difficulty"])
    cfg.events.reset_robot_joints.params["position_range"] = (-level.robot_joint_noise, level.robot_joint_noise)
    cfg.events.reset_object.params.update(cube_x=level.cube_x, cube_y=level.cube_y)
    t = cfg.commands.target
    t.x_range, t.y_range, t.min_separation = level.target_x, level.target_y, level.min_separation
    return cfg
