"""Reach sanity env cfg (Isaac Lab v2.3.2 manager-based RL). Isaac runtime only.

Observation (policy, full state, no cameras):
    joint pos/vel (relative to home), TCP position, target position, target - TCP, last action
Action: 5 arm joint position targets (scale 0.5 around home); gripper held at home.
Reward: distance (L2), distance (tanh), reached bonus, action-rate / joint-vel penalties.
Termination: timeout (+ optional success, configs/reach.yaml).
Reset: arm joints uniformly around home; target resampled by the command manager.
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
from dume_isaac.envs.tabletop.scene_cfg import LAYOUT, SPEC, TabletopSceneCfg
from dume_isaac.tasks.registry import load_task_params

from . import mdp

PARAMS = load_task_params("reach")
REW = PARAMS["rewards"]
DEFAULT_LEVEL = LAYOUT.level(PARAMS["difficulty"])
EE_OFFSET = tuple(SPEC.ee_offset)
COMMAND = "ee_target"


def _ee() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=[SPEC.ee_body])


def _robot_joints() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=list(SPEC.all_joint_names), preserve_order=True)


def _ee_params(**extra) -> dict:
    return {"command_name": COMMAND, "ee_cfg": _ee(), "offset": EE_OFFSET, **extra}


@configclass
class ReachSceneCfg(TabletopSceneCfg):
    cube = None


@configclass
class CommandsCfg:
    ee_target = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=SPEC.ee_body,
        resampling_time_range=(PARAMS["command"]["resampling_time_s"],) * 2,
        debug_vis=bool(PARAMS["command"]["debug_vis"]),
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=DEFAULT_LEVEL.reach_x,
            pos_y=DEFAULT_LEVEL.reach_y,
            pos_z=DEFAULT_LEVEL.reach_z,
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
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


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _robot_joints()})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _robot_joints()})
        ee_pos = ObsTerm(func=mdp.ee_position_b, params={"ee_cfg": _ee(), "offset": EE_OFFSET})
        target_pos = ObsTerm(func=mdp.command_position_b, params={"command_name": COMMAND})
        target_rel = ObsTerm(func=mdp.command_minus_ee_b, params=_ee_params())
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False  # privileged full-state teacher
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(SPEC.arm_joint_names)),
            "position_range": (-DEFAULT_LEVEL.robot_joint_noise, DEFAULT_LEVEL.robot_joint_noise),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    distance = RewTerm(func=mdp.ee_command_distance, weight=REW["distance"]["weight"], params=_ee_params())
    distance_tanh = RewTerm(
        func=mdp.ee_command_distance_tanh,
        weight=REW["distance_tanh"]["weight"],
        params=_ee_params(std=REW["distance_tanh"]["std"]),
    )
    reached = RewTerm(
        func=mdp.ee_command_reached,
        weight=REW["reached"]["weight"],
        params=_ee_params(threshold=REW["reached"]["threshold"]),
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=REW["action_rate"]["weight"])
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=REW["joint_vel"]["weight"],
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(SPEC.arm_joint_names))},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


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
class SO101ReachEnvCfg(ManagerBasedRLEnvCfg):
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=2048, env_spacing=2.0)
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
            self.terminations.success = DoneTerm(
                func=mdp.ee_reached_command, params=_ee_params(threshold=REW["reached"]["threshold"])
            )


def make_env_cfg(difficulty: str | None = None) -> SO101ReachEnvCfg:
    cfg = SO101ReachEnvCfg()
    level = LAYOUT.level(difficulty or PARAMS["difficulty"])
    ranges = cfg.commands.ee_target.ranges
    ranges.pos_x, ranges.pos_y, ranges.pos_z = level.reach_x, level.reach_y, level.reach_z
    cfg.events.reset_robot_joints.params["position_range"] = (-level.robot_joint_noise, level.robot_joint_noise)
    return cfg
