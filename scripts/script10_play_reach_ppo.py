#!/usr/bin/env python3
"""Stage 7b: play a trained Reach PPO checkpoint.

    <isaaclab.sh -p> scripts/script10_play_reach_ppo.py [--load_run <run>] [--checkpoint model_999.pt] [--num_envs 16]

Flow mirrors Isaac Lab v2.3.2 scripts/reinforcement_learning/rsl_rl; see
dume_isaac/runtime/rsl_rl_runner.py. Logs: logs/rsl_rl/<experiment>/<timestamp>[_run_name]/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dume_isaac.runtime.rsl_rl_runner import run_play  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_play("reach"))
