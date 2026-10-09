"""Reach terminations (success is optional, see configs/reach.yaml terminate_on_success)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

from .rewards import ee_command_distance

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["ee_reached_command"]


def ee_reached_command(env: ManagerBasedRLEnv, threshold: float, command_name: str, ee_cfg: SceneEntityCfg,
                       offset: tuple[float, float, float]) -> torch.Tensor:
    return ee_command_distance(env, command_name, ee_cfg, offset) < threshold
