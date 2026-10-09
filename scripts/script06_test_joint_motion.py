#!/usr/bin/env python3
"""Stage 4: per-joint motion test.

home -> shoulder_pan +d/back -> shoulder_lift -> elbow_flex -> wrist_flex -> wrist_roll
-> gripper open/close -> home. For each move prints logical joint, resolved USD joint,
commanded value, measured value, error and PASS/WARN/FAIL.

This only checks that the SIMULATED articulation follows position commands. It says
nothing about the Brain Us calibration (zero, direction, limits are not verified).

    <isaaclab.sh -p> scripts/script06_test_joint_motion.py
    <isaaclab.sh -p> scripts/script06_test_joint_motion.py --headless --delta 0.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dume_isaac.assets.my_so101.spec import LOGICAL_ARM_JOINTS, load_so101_spec  # noqa: E402
from dume_isaac.runtime.app import launch_app  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="script06_test_joint_motion.py", description="SO-101 per-joint motion test.")
    p.add_argument("--delta", type=float, default=None, help="arm joint step in rad (default: yaml)")
    p.add_argument("--settle_steps", type=int, default=None)
    p.add_argument("--keep_open", action="store_true")
    return p


def motion_plan(spec, delta: float) -> list[tuple[str, str, float]]:
    """(label, logical joint, commanded value) sequence. Pure; tested at home."""
    plan = [("home", "", 0.0)]
    for j in LOGICAL_ARM_JOINTS:
        plan.append((f"{j} +{delta:g}", j, spec.home[j] + delta))
        plan.append((f"{j} back", j, spec.home[j]))
    plan.append(("gripper open", spec.gripper_joint, spec.gripper_open))
    plan.append(("gripper close", spec.gripper_joint, spec.gripper_closed))
    plan.append(("home", "", 0.0))
    return plan


def classify(err: float, pass_tol: float, warn_tol: float) -> str:
    return "PASS" if err <= pass_tol else ("WARN" if err <= warn_tol else "FAIL")


def main(argv: list[str] | None = None) -> int:
    spec = load_so101_spec()
    args, simulation_app = launch_app(build_parser(), argv)

    from dume_isaac.runtime.single_robot import hold_and_step, make_single_robot_sim, resolve_joint_ids  # noqa: PLC0415

    cfg = spec.joint_motion_test
    delta = args.delta if args.delta is not None else float(cfg.get("delta_rad", 0.25))
    settle = args.settle_steps or int(cfg.get("settle_steps", 150))
    pass_tol, warn_tol = float(cfg.get("pass_tol_rad", 0.05)), float(cfg.get("warn_tol_rad", 0.15))

    sim, robot = make_single_robot_sim(spec, args.device, float(cfg.get("sim_dt", 0.01)))
    ids, problems = resolve_joint_ids(spec, robot.joint_names)
    if problems:
        print("[joint_motion] FAIL mapping:\n  " + "\n  ".join(problems))
        simulation_app.close()
        return 1

    home = robot.data.default_joint_pos.clone()
    lo, hi = robot.data.soft_joint_pos_limits[0, :, 0], robot.data.soft_joint_pos_limits[0, :, 1]
    target = home.clone()
    rows, worst = [], "PASS"
    print(f"\n=== SO-101 joint motion test (delta {delta} rad, settle {settle} steps) ===")
    print(f"{'step':<22}{'logical':<15}{'actual':<15}{'cmd':>9}{'meas':>9}{'err':>8}{'moved':>8}  result")
    for label, logical, value in motion_plan(spec, delta):
        if logical == "":
            target = home.clone()
            idx = None
        else:
            idx = ids[logical]
            value = float(min(max(value, float(lo[idx])), float(hi[idx])))  # stay inside soft limits
            target[0, idx] = value
        before = robot.data.joint_pos[0].clone()
        hold_and_step(sim, robot, target, settle)
        q = robot.data.joint_pos[0]
        if idx is None:
            err = float((q - home[0]).abs().max())
            moved, cmd, meas, actual = 0.0, 0.0, 0.0, "(all)"
        else:
            err = abs(float(q[idx]) - value)
            moved, cmd, meas, actual = float(q[idx] - before[idx]), value, float(q[idx]), robot.joint_names[idx]
        # other joints must stay where they were commanded
        others = [i for i in range(robot.num_joints) if i != idx]
        cross = float((q[others] - target[0, others]).abs().max()) if others else 0.0
        result = classify(max(err, cross), pass_tol, warn_tol)
        worst = {"PASS": worst, "WARN": "WARN" if worst == "PASS" else worst, "FAIL": "FAIL"}[result]
        rows.append(result)
        note = f" (others max err {cross:.3f})" if cross > pass_tol else ""
        print(f"{label:<22}{logical or '-':<15}{actual:<15}{cmd:>9.3f}{meas:>9.3f}{err:>8.3f}{moved:>+8.3f}  {result}{note}")

    print(f"\nJOINT MOTION: {worst}  ({rows.count('PASS')} pass, {rows.count('WARN')} warn, {rows.count('FAIL')} fail)")
    print("Visual check (GUI): each joint moves alone; 'gripper open' visibly opens the jaw. "
          "If open/close look swapped, swap gripper.open/closed in configs/my_so101.yaml.")
    if args.keep_open and not args.headless:
        while simulation_app.is_running():
            hold_and_step(sim, robot, home, 1)
    simulation_app.close()
    return 1 if worst == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
