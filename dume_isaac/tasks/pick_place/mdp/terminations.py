"""Pick-place terminations: cube out of workspace / dropped, optional place success."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

from dume_isaac.envs.tabletop.frames import world_to_base

from .rewards import PlaceTracker

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["object_out_of_bounds", "PlaceSuccessTermination"]


def object_out_of_bounds(env: ManagerBasedRLEnv, x_range: tuple[float, float], y_range: tuple[float, float],
                         min_z: float, object_cfg: SceneEntityCfg) -> torch.Tensor:
    p = world_to_base(env, env.scene[object_cfg.name].data.root_pos_w)
    outside = (p[:, 0] < x_range[0]) | (p[:, 0] > x_range[1]) | (p[:, 1] < y_range[0]) | (p[:, 1] > y_range[1])
    return outside | (p[:, 2] < min_z) | ~torch.isfinite(p).all(dim=1)


class PlaceSuccessTermination(PlaceTracker):
    def __call__(self, env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, command_name: str,
                 ee_cfg: SceneEntityCfg, gripper_cfg: SceneEntityCfg, geom: dict, thr: dict) -> torch.Tensor:
        _, success = self.update(env, object_cfg, command_name, ee_cfg, gripper_cfg, geom, thr)
        return success
