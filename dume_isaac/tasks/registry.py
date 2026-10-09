"""Task registry. Pure Python: entry points are strings, imported only at Isaac runtime."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from dume_isaac.config_io import load_yaml
from dume_isaac.paths import CONFIG_DIR


@dataclass(frozen=True)
class TaskEntry:
    name: str
    env_cfg_entry: str      # "module:callable(difficulty) -> ManagerBasedRLEnvCfg"
    agent_cfg_entry: str    # "module:RslRlOnPolicyRunnerCfg subclass"
    experiment_name: str    # logs/rsl_rl/<experiment_name>/<run>
    train_num_envs: int     # first-run default for an RTX 5060 Laptop (8 GB); override with --num_envs
    play_num_envs: int
    sanity_num_envs: int


TASKS: dict[str, TaskEntry] = {
    "reach": TaskEntry(
        name="reach",
        env_cfg_entry="dume_isaac.tasks.reach.reach_env_cfg:make_env_cfg",
        agent_cfg_entry="dume_isaac.tasks.reach.agents.rsl_rl_ppo_cfg:SO101ReachPPORunnerCfg",
        experiment_name="so101_reach",
        train_num_envs=2048,
        play_num_envs=16,
        sanity_num_envs=64,
    ),
    "pick_place": TaskEntry(
        name="pick_place",
        env_cfg_entry="dume_isaac.tasks.pick_place.pick_place_env_cfg:make_env_cfg",
        agent_cfg_entry="dume_isaac.tasks.pick_place.agents.rsl_rl_ppo_cfg:SO101PickPlacePPORunnerCfg",
        experiment_name="so101_pick_place",
        train_num_envs=1024,
        play_num_envs=16,
        sanity_num_envs=64,
    ),
}


def get_task(name: str) -> TaskEntry:
    if name not in TASKS:
        raise KeyError(f"unknown task '{name}', choose from {list(TASKS)}")
    return TASKS[name]


def load_task_params(name: str) -> dict[str, Any]:
    """Tunable task parameters from configs/<task>.yaml."""
    return load_yaml(CONFIG_DIR / f"{get_task(name).name}.yaml")


def load_entry(entry: str) -> Any:
    module_name, attr = entry.split(":")
    return getattr(importlib.import_module(module_name), attr)
