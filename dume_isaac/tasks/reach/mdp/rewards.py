"""Reach rewards (position only)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

from .observations import command_minus_ee_b

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["ee_command_distance", "ee_command_distance_tanh", "ee_command_reached"]


def ee_command_distance(env: ManagerBasedRLEnv, command_name: str, ee_cfg: SceneEntityCfg,
                        offset: tuple[float, float, float]) -> torch.Tensor:
    return torch.norm(command_minus_ee_b(env, command_name, ee_cfg, offset), dim=1)


def ee_command_distance_tanh(env: ManagerBasedRLEnv, std: float, command_name: str, ee_cfg: SceneEntityCfg,
                             offset: tuple[float, float, float]) -> torch.Tensor:
    return 1.0 - torch.tanh(ee_command_distance(env, command_name, ee_cfg, offset) / std)


def ee_command_reached(env: ManagerBasedRLEnv, threshold: float, command_name: str, ee_cfg: SceneEntityCfg,
                       offset: tuple[float, float, float]) -> torch.Tensor:
    return (ee_command_distance(env, command_name, ee_cfg, offset) < threshold).float()
