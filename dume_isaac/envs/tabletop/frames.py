"""Frame helpers shared by tabletop tasks (TCP position, world -> robot base). Isaac runtime only."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_pos_w(env: ManagerBasedRLEnv, ee_cfg: SceneEntityCfg, offset: tuple[float, float, float]) -> torch.Tensor:
    """TCP position in world: EE body position + body-frame offset (configs/my_so101.yaml)."""
    robot = env.scene[ee_cfg.name]
    idx = ee_cfg.body_ids[0]
    pos = robot.data.body_pos_w[:, idx]
    quat = robot.data.body_quat_w[:, idx]
    off = torch.as_tensor(offset, dtype=pos.dtype, device=pos.device).expand_as(pos)
    return pos + quat_apply(quat, off)


def world_to_base(env: ManagerBasedRLEnv, pos_w: torch.Tensor, robot_name: str = "robot") -> torch.Tensor:
    robot = env.scene[robot_name]
    pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, pos_w)
    return pos_b


def base_to_world(env: ManagerBasedRLEnv, pos_b: torch.Tensor, env_ids: torch.Tensor | None = None,
                  robot_name: str = "robot") -> torch.Tensor:
    robot = env.scene[robot_name]
    root_pos = robot.data.root_pos_w if env_ids is None else robot.data.root_pos_w[env_ids]
    root_quat = robot.data.root_quat_w if env_ids is None else robot.data.root_quat_w[env_ids]
    return root_pos + quat_apply(root_quat, pos_b)


def ee_pos_b(env: ManagerBasedRLEnv, ee_cfg: SceneEntityCfg, offset: tuple[float, float, float]) -> torch.Tensor:
    return world_to_base(env, ee_pos_w(env, ee_cfg, offset), ee_cfg.name)
