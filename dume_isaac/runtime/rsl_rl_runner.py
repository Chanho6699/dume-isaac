"""RSL-RL train / play flow, modeled on Isaac Lab v2.3.2
scripts/reinforcement_learning/rsl_rl/{train,play}.py, without the gym registry.

Parsers are pure Python (testable at home). Isaac / torch / rsl_rl are imported only
after `launch_app()` has started the simulator.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

from dume_isaac.envs.tabletop.layout import DIFFICULTIES
from dume_isaac.paths import LOG_ROOT
from dume_isaac.runtime.app import launch_app
from dume_isaac.tasks.registry import TaskEntry, get_task, load_entry


def build_train_parser(task_name: str, prog: str | None = None) -> argparse.ArgumentParser:
    task = get_task(task_name)
    p = argparse.ArgumentParser(prog=prog, description=f"Train PPO (RSL-RL) on SO-101 {task.name}.")
    p.add_argument("--num_envs", type=int, default=task.train_num_envs,
                   help=f"parallel envs (default {task.train_num_envs}, sized for an 8 GB laptop GPU)")
    p.add_argument("--max_iterations", type=int, default=None, help="PPO iterations (default: agent cfg)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--difficulty", choices=DIFFICULTIES, default=None, help="randomization level (default: task yaml)")
    p.add_argument("--run_name", type=str, default="", help="suffix for the log dir")
    p.add_argument("--resume", action="store_true", help="resume from --load_run / --checkpoint")
    p.add_argument("--load_run", type=str, default=None, help="run dir name or regex (default: latest)")
    p.add_argument("--checkpoint", type=str, default=None, help="checkpoint file or model_<iter>.pt name")
    return p


def build_play_parser(task_name: str, prog: str | None = None) -> argparse.ArgumentParser:
    task = get_task(task_name)
    p = argparse.ArgumentParser(prog=prog, description=f"Play a trained SO-101 {task.name} PPO policy.")
    p.add_argument("--num_envs", type=int, default=task.play_num_envs)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--difficulty", choices=DIFFICULTIES, default=None)
    p.add_argument("--load_run", type=str, default=None, help="run dir name or regex (default: latest)")
    p.add_argument("--checkpoint", type=str, default=None, help="checkpoint file or model_<iter>.pt name")
    p.add_argument("--steps", type=int, default=0, help="stop after N env steps (0 = until window closes)")
    return p


def _log_root(task: TaskEntry):
    return LOG_ROOT / task.experiment_name


def _make_cfgs(task: TaskEntry, args):
    env_cfg = load_entry(task.env_cfg_entry)(args.difficulty)
    agent_cfg = load_entry(task.agent_cfg_entry)()
    agent_cfg.experiment_name = task.experiment_name
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device if getattr(args, "device", None) else agent_cfg.device
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.device = agent_cfg.device
    try:  # Isaac Lab v2.3.x: maps deprecated fields for the installed rsl-rl-lib
        from importlib.metadata import version  # noqa: PLC0415

        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg  # noqa: PLC0415

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))
    except ImportError:
        pass
    return env_cfg, agent_cfg


def _first_obs(env):
    out = env.get_observations()
    return out[0] if isinstance(out, tuple) else out  # rsl-rl 2.x returns (obs, extras)


def run_train(task_name: str, argv: list[str] | None = None) -> int:
    task = get_task(task_name)
    parser = build_train_parser(task_name, prog=os.path.basename(sys.argv[0]))
    args, simulation_app = launch_app(parser, argv)

    import torch  # noqa: PLC0415
    from isaaclab.envs import ManagerBasedRLEnv  # noqa: PLC0415
    from isaaclab.utils.io import dump_yaml  # noqa: PLC0415
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: PLC0415
    from rsl_rl.runners import OnPolicyRunner  # noqa: PLC0415

    from dume_isaac.runtime.checkpoints import resolve_checkpoint  # noqa: PLC0415

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    env_cfg, agent_cfg = _make_cfgs(task, args)
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    agent_cfg.run_name = args.run_name

    log_root = _log_root(task)
    run_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + (f"_{args.run_name}" if args.run_name else "")
    log_dir = log_root / run_dir
    print(f"[dume_isaac] task={task.name} num_envs={args.num_envs} difficulty={args.difficulty or 'yaml default'}")
    print(f"[dume_isaac] logging to {log_dir}")

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    try:
        runner.add_git_repo_to_log(__file__)
    except Exception:
        pass
    if args.resume:
        ckpt = resolve_checkpoint(log_root, args.checkpoint, args.load_run)
        print(f"[dume_isaac] resuming from {ckpt}")
        runner.load(str(ckpt))

    dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"[dume_isaac] done. checkpoints in {log_dir}")
    env.close()
    simulation_app.close()
    return 0


def run_play(task_name: str, argv: list[str] | None = None) -> int:
    task = get_task(task_name)
    parser = build_play_parser(task_name, prog=os.path.basename(sys.argv[0]))
    args, simulation_app = launch_app(parser, argv)

    import torch  # noqa: PLC0415
    from isaaclab.envs import ManagerBasedRLEnv  # noqa: PLC0415
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: PLC0415
    from rsl_rl.runners import OnPolicyRunner  # noqa: PLC0415

    from dume_isaac.runtime.checkpoints import resolve_checkpoint  # noqa: PLC0415

    env_cfg, agent_cfg = _make_cfgs(task, args)
    ckpt = resolve_checkpoint(_log_root(task), args.checkpoint, args.load_run)
    print(f"[dume_isaac] loading checkpoint {ckpt}")

    base_env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(ckpt))
    policy = runner.get_inference_policy(device=base_env.device)

    place_term = None
    if "place" in base_env.reward_manager.active_terms:
        place_term = base_env.reward_manager.get_term_cfg("place").func

    obs = _first_obs(env)
    step, episodes, ep_return = 0, 0, torch.zeros(base_env.num_envs, device=base_env.device)
    returns: list[float] = []
    while simulation_app.is_running() and (args.steps <= 0 or step < args.steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, rew, dones, _ = env.step(actions)
        ep_return += rew
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if len(done_ids):
            returns += ep_return[done_ids].tolist()
            ep_return[done_ids] = 0.0
            episodes += len(done_ids)
        step += 1
        if step % 200 == 0 and returns:
            msg = f"[play] step {step} episodes {episodes} mean return {sum(returns[-100:]) / len(returns[-100:]):.2f}"
            if place_term is not None and place_term.finished_episodes:
                msg += f" place success {place_term.finished_successes / place_term.finished_episodes:.1%}"
            print(msg)

    env.close()
    simulation_app.close()
    return 0
