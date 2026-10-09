#!/usr/bin/env python3
"""Stage 7a: train the Reach sanity PPO policy (RSL-RL).

    <isaaclab.sh -p> scripts/script09_train_reach_ppo.py --headless [--num_envs 2048] [--max_iterations 1000] [--seed 42]

Flow mirrors Isaac Lab v2.3.2 scripts/reinforcement_learning/rsl_rl; see
dume_isaac/runtime/rsl_rl_runner.py. Logs: logs/rsl_rl/<experiment>/<timestamp>[_run_name]/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dume_isaac.runtime.rsl_rl_runner import run_train  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_train("reach"))
