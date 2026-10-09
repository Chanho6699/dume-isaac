"""Pick-place observations (robot base frame)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

from dume_isaac.envs.tabletop.frames import ee_pos_b, world_to_base

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "ee_position_b", "object_position_b", "object_minus_ee_b", "target_position_b", "target_minus_object_b",
    "gripper_opening",
]


def ee_position_b(env: ManagerBasedRLEnv, ee_cfg: SceneEntityCfg, offset: tuple[float, float, float]) -> torch.Tensor:
    return ee_pos_b(env, ee_cfg, offset)


def object_position_b(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    return world_to_base(env, env.scene[object_cfg.name].data.root_pos_w)


def object_minus_ee_b(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, ee_cfg: SceneEntityCfg,
                      offset: tuple[float, float, float]) -> torch.Tensor:
    return object_position_b(env, object_cfg) - ee_pos_b(env, ee_cfg, offset)


def target_position_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Target pad center (TargetPadCommand, already in the robot base frame)."""
    return env.command_manager.get_command(command_name)


def target_minus_object_b(env: ManagerBasedRLEnv, command_name: str, object_cfg: SceneEntityCfg) -> torch.Tensor:
    return target_position_b(env, command_name) - object_position_b(env, object_cfg)


def gripper_opening(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, open_pos: float, closed_pos: float) -> torch.Tensor:
    """Measured gripper joint mapped to 0 (closed setting) .. 1 (open setting)."""
    q = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids[0]]
    return ((q - closed_pos) / (open_pos - closed_pos)).unsqueeze(-1)
