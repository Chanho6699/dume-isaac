"""Reach sanity task: TCP -> sampled target position, full state, 5 arm joint position actions.

Purpose: validate actions / observations / rewards / resets / PPO pipeline before pick-place.
Env cfg: reach_env_cfg.make_env_cfg(difficulty). Isaac runtime only (not imported here).
"""
