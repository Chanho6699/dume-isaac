"""Staged pick-place rewards. The math lives in ../reward_model.py (shared with the
home episode-return audit, tests/test_reward_audit.py); this module evaluates it on torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from dume_isaac.tasks.pick_place.reward_model import place_step, stage_rewards

from .state import TorchOps, compute_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["staged_reward", "PlaceTracker", "PlaceSuccessReward"]


def staged_reward(env: ManagerBasedRLEnv, stage: str, rew: dict, object_cfg: SceneEntityCfg, command_name: str,
                  ee_cfg: SceneEntityCfg, gripper_cfg: SceneEntityCfg, geom: dict, thr: dict) -> torch.Tensor:
    """One unweighted stage value (reach_object, grasp, lift, carry, carry_fine, descend)."""
    st = compute_state(env, object_cfg, command_name, ee_cfg, gripper_cfg, geom, thr)
    return stage_rewards(st, rew, thr, TorchOps)[stage]


class PlaceTracker(ManagerTermBase):
    """Stateful place detector (was_lifted, consecutive placed steps) per env.

    The reward and termination managers each own one instance; both evaluate the same
    deterministic state every step, so they agree. Success statistics are written to
    `env.extras["log"]` on reset (RSL-RL logs: Episode_Metric/place_success).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n, dev = env.num_envs, env.device
        self.was_lifted = torch.zeros(n, dtype=torch.bool, device=dev)
        self.hold_count = torch.zeros(n, dtype=torch.long, device=dev)
        self.success_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self.last_episode_success = torch.zeros(n, dtype=torch.bool, device=dev)
        self.finished_episodes = 0
        self.finished_successes = 0

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        done = self.success_ever[env_ids]
        self.last_episode_success[env_ids] = done
        if done.numel() > 0:
            self.finished_episodes += int(done.numel())
            self.finished_successes += int(done.sum())
            try:
                self._env.extras.setdefault("log", {})["Episode_Metric/place_success"] = done.float().mean().item()
            except Exception:
                pass
        self.was_lifted[env_ids] = False
        self.hold_count[env_ids] = 0
        self.success_ever[env_ids] = False

    def update(self, env, object_cfg, command_name, ee_cfg, gripper_cfg, geom, thr):
        st = compute_state(env, object_cfg, command_name, ee_cfg, gripper_cfg, geom, thr)
        self.was_lifted, self.hold_count, placed, success = place_step(
            self.was_lifted, self.hold_count, st, thr, TorchOps)
        self.success_ever |= success
        return placed, success


class PlaceSuccessReward(PlaceTracker):
    def __call__(self, env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, command_name: str,
                 ee_cfg: SceneEntityCfg, gripper_cfg: SceneEntityCfg, geom: dict, thr: dict) -> torch.Tensor:
        placed, _ = self.update(env, object_cfg, command_name, ee_cfg, gripper_cfg, geom, thr)
        return placed.float()
