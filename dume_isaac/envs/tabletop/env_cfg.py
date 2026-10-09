"""Shared ManagerBasedRLEnvCfg settings for tabletop tasks. Isaac runtime only.

Tasks subclass ManagerBasedRLEnvCfg with their own MDP terms and call
`apply_tabletop_settings` in `__post_init__`. PhysX buffer sizes follow the official
Isaac Lab v2.3.2 lift task.
"""

from __future__ import annotations

from typing import Any

from .scene_cfg import ENV_SPACING, base_to_env


def apply_tabletop_settings(cfg: Any, params: dict[str, Any]) -> None:
    sim = params["sim"]
    cfg.decimation = int(sim["decimation"])
    cfg.episode_length_s = float(params["episode_length_s"])
    cfg.sim.dt = float(sim["dt"])
    cfg.sim.render_interval = cfg.decimation
    cfg.scene.env_spacing = ENV_SPACING

    cfg.sim.physx.bounce_threshold_velocity = 0.01
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
    cfg.sim.physx.friction_correlation_distance = 0.00625

    cfg.viewer.eye = base_to_env(0.65, 0.45, 0.45)
    cfg.viewer.lookat = base_to_env(0.2, 0.0, 0.0)
