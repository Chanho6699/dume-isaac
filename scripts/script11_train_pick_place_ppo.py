#!/usr/bin/env python3
"""Stage 9a: train the pick-place full-state PPO teacher (RSL-RL).

    <isaaclab.sh -p> scripts/script11_train_pick_place_ppo.py --headless [--num_envs 1024] [--difficulty easy] [--max_iterations 3000]

Flow mirrors Isaac Lab v2.3.2 scripts/reinforcement_learning/rsl_rl; see
dume_isaac/runtime/rsl_rl_runner.py. Logs: logs/rsl_rl/<experiment>/<timestamp>[_run_name]/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dume_isaac.runtime.rsl_rl_runner import run_train  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_train("pick_place"))
