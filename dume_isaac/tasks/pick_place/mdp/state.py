"""Torch adapter for the pick-place reward model (reward_model.py holds the math).

All positions are in the robot base frame. The target is a command (TargetPadCommand),
not a physics object. `geom` (static geometry) and `thr` (thresholds from
configs/pick_place.yaml) are plain dicts passed through term params.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

from dume_isaac.envs.tabletop.frames import ee_pos_b, world_to_base
from dume_isaac.tasks.pick_place.reward_model import StageState

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class TorchOps:
    tanh = staticmethod(torch.tanh)
    and_ = staticmethod(torch.logical_and)
    not_ = staticmethod(torch.logical_not)
    abs = staticmethod(torch.abs)
    where = staticmethod(torch.where)

    @staticmethod
    def f(b: torch.Tensor) -> torch.Tensor:
        return b.float()

    @staticmethod
    def clip(x: torch.Tensor, lo: float | None, hi: float | None) -> torch.Tensor:
        return torch.clamp(x, min=lo, max=hi)


def compute_state(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, command_name: str,
                  ee_cfg: SceneEntityCfg, gripper_cfg: SceneEntityCfg, geom: dict, thr: dict) -> StageState:
    robot = env.scene[gripper_cfg.name]
    cube = env.scene[object_cfg.name]

    ee_b = ee_pos_b(env, ee_cfg, geom["ee_offset"])
    cube_b = world_to_base(env, cube.data.root_pos_w, gripper_cfg.name)
    target_b = env.command_manager.get_command(command_name)

    ee_cube_dist = torch.norm(cube_b - ee_b, dim=1)
    height = cube_b[:, 2] - geom["cube_rest_z"]
    delta_xy = cube_b[:, :2] - target_b[:, :2]
    half = torch.tensor(geom["target_half_xy"], device=env.device) - thr["place_xy_margin"]
    in_target = (delta_xy.abs() <= half).all(dim=1)

    g = gripper_cfg.joint_ids[0]
    tgt = robot.data.joint_pos_target[:, g]
    gripper_closing = (tgt - geom["gripper_closed"]).abs() < (tgt - geom["gripper_open"]).abs()
    held = (height > thr["held_min_height"]) & (ee_cube_dist < thr["hold_radius"])

    return StageState(
        ee_cube_dist=ee_cube_dist,
        height=height,
        target_xy_dist=torch.norm(delta_xy, dim=1),
        in_target=in_target,
        held=held,
        gripper_closing=gripper_closing,
        speed=torch.norm(cube.data.root_lin_vel_w, dim=1),
    )
