#!/usr/bin/env python3
"""Stages 6 and 8: random-policy env sanity gate before PPO.

Creates the RL env, runs bounded uniform random actions for many steps and reports
episodes, resets, NaN/Inf counts, reward statistics, termination reasons and (pick_place)
cube physics. Exit code 0 only if the gate passes.

    <isaaclab.sh -p> scripts/script08_random_policy.py --task reach --headless
    <isaaclab.sh -p> scripts/script08_random_policy.py --task pick_place --headless
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dume_isaac.envs.tabletop.layout import DIFFICULTIES  # noqa: E402
from dume_isaac.runtime.app import launch_app  # noqa: E402
from dume_isaac.tasks.registry import TASKS, get_task, load_entry  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="script08_random_policy.py", description="Random-policy env sanity gate.")
    p.add_argument("--task", choices=list(TASKS), required=True)
    p.add_argument("--num_envs", type=int, default=None, help="default: registry sanity_num_envs")
    p.add_argument("--steps", type=int, default=1000, help="env steps")
    p.add_argument("--action_scale", type=float, default=1.0, help="uniform actions in [-s, s]")
    p.add_argument("--difficulty", choices=DIFFICULTIES, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args, simulation_app = launch_app(build_parser(), argv)
    task = get_task(args.task)

    import torch  # noqa: PLC0415
    from isaaclab.envs import ManagerBasedRLEnv  # noqa: PLC0415

    env_cfg = load_entry(task.env_cfg_entry)(args.difficulty)
    env_cfg.scene.num_envs = args.num_envs or task.sanity_num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device
    env = ManagerBasedRLEnv(cfg=env_cfg)

    n, dim, dev = env.num_envs, env.action_manager.total_action_dim, env.device
    obs, _ = env.reset()
    print(f"\n=== random policy: task={task.name} num_envs={n} action_dim={dim} steps={args.steps} ===")
    print(f"obs['policy'] shape: {tuple(obs['policy'].shape)}")
    print(f"reward terms: {env.reward_manager.active_terms}")
    print(f"termination terms: {env.termination_manager.active_terms}")

    nan_obs = nan_rew = resets = timeouts = 0
    r_min, r_max, r_sum, r_cnt = float("inf"), float("-inf"), 0.0, 0
    reasons: Counter[str] = Counter()
    cube = env.scene["cube"] if "cube" in env.scene.keys() else None
    max_cube_speed, cube_nan = 0.0, 0

    for _ in range(args.steps):
        actions = (torch.rand(n, dim, device=dev) * 2.0 - 1.0) * args.action_scale
        obs, rew, terminated, truncated, _ = env.step(actions)
        nan_obs += int((~torch.isfinite(obs["policy"])).any(dim=1).sum())
        nan_rew += int((~torch.isfinite(rew)).sum())
        finite = rew[torch.isfinite(rew)]
        if finite.numel():
            r_min, r_max = min(r_min, float(finite.min())), max(r_max, float(finite.max()))
            r_sum, r_cnt = r_sum + float(finite.sum()), r_cnt + finite.numel()
        done = terminated | truncated
        resets += int(done.sum())
        timeouts += int(truncated.sum())
        for name in env.termination_manager.active_terms:
            reasons[name] += int(env.termination_manager.get_term(name).sum())
        if cube is not None:
            v = torch.norm(cube.data.root_lin_vel_w, dim=1)
            cube_nan += int((~torch.isfinite(v)).sum())
            max_cube_speed = max(max_cube_speed, float(v[torch.isfinite(v)].max()) if torch.isfinite(v).any() else 0.0)

    print("\n--- summary ---")
    print(f"  env steps        : {args.steps} x {n} envs")
    print(f"  episodes ended   : {resets} (timeouts {timeouts})")
    print(f"  termination terms: {dict(reasons)}")
    print(f"  NaN/Inf obs rows : {nan_obs}")
    print(f"  NaN/Inf rewards  : {nan_rew}")
    if r_cnt:
        print(f"  reward min/max/mean: {r_min:.4f} / {r_max:.4f} / {r_sum / r_cnt:.4f}")
    if cube is not None:
        print(f"  cube max speed   : {max_cube_speed:.3f} m/s, cube NaN {cube_nan}")
    if "place" in env.reward_manager.active_terms:
        t = env.reward_manager.get_term_cfg("place").func
        print(f"  place success    : {t.finished_successes}/{t.finished_episodes} episodes (random policy ~0 expected)")

    checks = {
        "no NaN/Inf in obs": nan_obs == 0,
        "no NaN/Inf in rewards": nan_rew == 0,
        "episodes reset": resets > 0,
        "cube physics sane (<10 m/s, finite)": cube is None or (cube_nan == 0 and max_cube_speed < 10.0),
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if resets == 0:
        print("  hint: --steps must exceed episode length / (sim.dt * decimation) to see timeouts")
    overall = "PASS" if all(checks.values()) else "FAIL"
    print(f"\nRANDOM POLICY GATE ({task.name}): {overall}")
    env.close()
    simulation_app.close()
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
