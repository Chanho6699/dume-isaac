"""Pick-place reset events. Only the cube is a physics object; the target is a command
(commands.py), resampled by the command manager after these events run."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

from dume_isaac.envs.tabletop.frames import base_to_world

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["reset_object_uniform"]


def uniform(r: tuple[float, float], n: int, device) -> torch.Tensor:
    return r[0] + (r[1] - r[0]) * torch.rand(n, device=device)


def reset_object_uniform(env: ManagerBasedRLEnv, env_ids: torch.Tensor, cube_x: tuple[float, float],
                         cube_y: tuple[float, float], cube_z: float, object_cfg: SceneEntityCfg) -> None:
    """Cube at a uniform (x, y) in the robot base frame, upright, at rest."""
    cube = env.scene[object_cfg.name]
    n, dev = len(env_ids), env.device
    pos_b = torch.stack([uniform(cube_x, n, dev), uniform(cube_y, n, dev), torch.full((n,), cube_z, device=dev)], dim=-1)
    quat = torch.zeros(n, 4, device=dev)
    quat[:, 0] = 1.0
    cube.write_root_pose_to_sim(torch.cat([base_to_world(env, pos_b, env_ids), quat], dim=-1), env_ids=env_ids)
    cube.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)
