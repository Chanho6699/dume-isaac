"""Reach observations (robot base frame)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

from dume_isaac.envs.tabletop.frames import ee_pos_b

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["ee_position_b", "command_position_b", "command_minus_ee_b"]


def ee_position_b(env: ManagerBasedRLEnv, ee_cfg: SceneEntityCfg, offset: tuple[float, float, float]) -> torch.Tensor:
    return ee_pos_b(env, ee_cfg, offset)


def command_position_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    # UniformPoseCommand is expressed in the robot root frame: (x, y, z, qw, qx, qy, qz)
    return env.command_manager.get_command(command_name)[:, :3]


def command_minus_ee_b(env: ManagerBasedRLEnv, command_name: str, ee_cfg: SceneEntityCfg,
                       offset: tuple[float, float, float]) -> torch.Tensor:
    return command_position_b(env, command_name) - ee_pos_b(env, ee_cfg, offset)
