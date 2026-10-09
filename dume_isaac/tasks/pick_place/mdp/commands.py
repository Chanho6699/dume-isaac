"""Target pad as a command term: a per-env goal position plus a visual-only marker.

The target has no rigid body, collider, velocity or physics reset. Its position lives in
`TargetPadCommand.command` (robot base frame) and is drawn with VisualizationMarkers.

Resampled only on episode reset. Isaac Lab v2.3.2 resets the command manager AFTER the
reset events, so the cube has already been placed by `reset_object_uniform` and the target
is sampled at least `min_separation` away from it (rejection sampling). The achieved
separation is logged as metric `cube_target_xy_dist`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

from .events import uniform

__all__ = ["TargetPadCommand", "TargetPadCommandCfg"]

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class TargetPadCommand(CommandTerm):
    cfg: TargetPadCommandCfg

    def __init__(self, cfg: TargetPadCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.cube = env.scene[cfg.object_name]
        self.target_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_b[:, 2] = cfg.z_in_base
        self.metrics["cube_target_xy_dist"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return f"TargetPadCommand: x {self.cfg.x_range}, y {self.cfg.y_range}, min sep {self.cfg.min_separation}"

    @property
    def command(self) -> torch.Tensor:
        return self.target_b

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = len(ids)
        if n == 0:
            return
        cube_b, _ = subtract_frame_transforms(
            self.robot.data.root_pos_w[ids], self.robot.data.root_quat_w[ids], self.cube.data.root_pos_w[ids])
        tx, ty = uniform(self.cfg.x_range, n, self.device), uniform(self.cfg.y_range, n, self.device)
        for _ in range(20):
            bad = torch.hypot(cube_b[:, 0] - tx, cube_b[:, 1] - ty) < self.cfg.min_separation
            k = int(bad.sum())
            if k == 0:
                break
            tx[bad], ty[bad] = uniform(self.cfg.x_range, k, self.device), uniform(self.cfg.y_range, k, self.device)
        self.target_b[ids, 0] = tx
        self.target_b[ids, 1] = ty
        self.target_b[ids, 2] = self.cfg.z_in_base
        self.metrics["cube_target_xy_dist"][ids] = torch.hypot(cube_b[:, 0] - tx, cube_b[:, 1] - ty)

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "pad_marker"):
                self.pad_marker = VisualizationMarkers(self.cfg.marker_cfg)
            self.pad_marker.set_visibility(True)
        elif hasattr(self, "pad_marker"):
            self.pad_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        pos_w = self.robot.data.root_pos_w + quat_apply(self.robot.data.root_quat_w, self.target_b)
        self.pad_marker.visualize(translations=pos_w)


@configclass
class TargetPadCommandCfg(CommandTermCfg):
    class_type: type = TargetPadCommand
    asset_name: str = "robot"
    object_name: str = "cube"
    x_range: tuple[float, float] = MISSING
    y_range: tuple[float, float] = MISSING
    min_separation: float = MISSING
    z_in_base: float = MISSING
    marker_cfg: VisualizationMarkersCfg = MISSING
