#!/usr/bin/env python3
"""Stage 3: SO-101 spawn smoke test (asset sanity before any RL).

Ground + light + my_so101, prints bodies / joints / default joint positions, checks the
canonical mapping, then holds the home pose for N steps and checks for NaN/Inf, joint
velocity blow-ups, base drift and sag away from home. Thresholds: configs/my_so101.yaml
smoke_test (provisional).

    <isaaclab.sh -p> scripts/script05_smoke_test_so101.py            # GUI
    <isaaclab.sh -p> scripts/script05_smoke_test_so101.py --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dume_isaac.assets.my_so101.spec import load_so101_spec, urdf_fixed_joint_into  # noqa: E402
from dume_isaac.runtime.app import launch_app  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="script05_smoke_test_so101.py", description="SO-101 spawn smoke test.")
    p.add_argument("--steps", type=int, default=None, help="physics steps (default: configs/my_so101.yaml)")
    p.add_argument("--keep_open", action="store_true", help="keep the GUI running after the checks")
    return p


def main(argv: list[str] | None = None) -> int:
    spec = load_so101_spec()
    args, simulation_app = launch_app(build_parser(), argv)

    import torch  # noqa: PLC0415

    from dume_isaac.runtime.single_robot import hold_and_step, make_single_robot_sim, resolve_joint_ids  # noqa: PLC0415

    cfg = spec.smoke_test
    steps = args.steps or int(cfg.get("steps", 300))
    sim, robot = make_single_robot_sim(spec, args.device, float(cfg.get("sim_dt", 0.01)))

    print("\n=== SO-101 smoke test ===")
    print(f"USD          : {spec.usd_path}")
    print(f"bodies ({robot.num_bodies}) : {robot.body_names}")
    print(f"joints ({robot.num_joints}) : {robot.joint_names}")
    default = robot.data.default_joint_pos.clone()
    print(f"default q    : {[round(v, 4) for v in default[0].tolist()]}")
    limits = robot.data.soft_joint_pos_limits[0]
    for name, (lo, hi) in zip(robot.joint_names, limits.tolist()):
        print(f"  limit {name:<14} [{lo:+.4f}, {hi:+.4f}]")

    results: list[tuple[str, str, str]] = []
    _, problems = resolve_joint_ids(spec, robot.joint_names)
    for body, label in ((spec.ee_body, "frames.end_effector.body"), (spec.base_body, "frames.base")):
        if body not in robot.body_names:
            problems.append(f"{label} '{body}' not in USD bodies")
    results.append(("mapping", "FAIL" if problems else "PASS", "; ".join(problems) or "all logical names resolved"))
    results.append(("joint count", "PASS" if robot.num_joints == 6 else "FAIL", f"{robot.num_joints} (expected 6)"))

    # EE frame preserved (merge_fixed_joints: false): TCP body must sit at the URDF fixed-joint offset
    fixed = urdf_fixed_joint_into(spec.urdf_path, spec.ee_body) if spec.urdf_path.is_file() else None
    if fixed and spec.ee_body in robot.body_names and fixed[1] in robot.body_names:
        want = sum(v * v for v in fixed[2]) ** 0.5
        p_ee = robot.data.body_pos_w[0, robot.body_names.index(spec.ee_body)]
        p_parent = robot.data.body_pos_w[0, robot.body_names.index(fixed[1])]
        got = float(torch.norm(p_ee - p_parent))
        ok = abs(got - want) < 0.002
        results.append(("EE frame", "PASS" if ok else "FAIL",
                        f"|{spec.ee_body} - {fixed[1]}| = {got:.4f} m (URDF {fixed[0]}: {want:.4f} m)"))

    root0 = robot.data.root_pos_w.clone()
    max_vel = 0.0
    finite = True
    chunk = 10
    for _ in range(max(1, steps // chunk)):
        hold_and_step(sim, robot, default, chunk)
        q, qd = robot.data.joint_pos, robot.data.joint_vel
        finite &= bool(torch.isfinite(q).all() and torch.isfinite(qd).all() and torch.isfinite(robot.data.body_pos_w).all())
        max_vel = max(max_vel, float(qd.abs().max()))

    drift = float(torch.norm(robot.data.root_pos_w - root0))
    home_err = float((robot.data.joint_pos - default).abs().max())
    results.append(("finite state", "PASS" if finite else "FAIL", "no NaN/Inf" if finite else "NaN/Inf found"))
    v_lim = float(cfg.get("max_joint_vel", 20.0))
    results.append(("max |joint vel|", "PASS" if max_vel < v_lim else "FAIL", f"{max_vel:.3f} rad/s (limit {v_lim})"))
    d_lim = float(cfg.get("max_base_drift", 0.001))
    results.append(("base drift", "PASS" if drift < d_lim else "FAIL", f"{drift * 1000:.2f} mm (limit {d_lim * 1000:.1f})"))
    warn, fail = float(cfg.get("max_home_error_warn", 0.1)), float(cfg.get("max_home_error_fail", 0.5))
    status = "PASS" if home_err < warn else ("WARN" if home_err < fail else "FAIL")
    results.append(("hold home", status, f"max |q - home| {home_err:.4f} rad (warn {warn}, fail {fail})"))
    print(f"final q      : {[round(v, 4) for v in robot.data.joint_pos[0].tolist()]}")

    print("\n--- results ---")
    for name, st, detail in results:
        print(f"  [{st:<4}] {name:<16} {detail}")
    overall = "FAIL" if any(st == "FAIL" for _, st, _ in results) else "PASS"
    print(f"\nSMOKE TEST: {overall}  (visually check in GUI: robot upright on the ground, not exploding)")
    print("Also scan the log above for PhysX warnings about zero mass/inertia on gripper_frame_link "
          "(massless upstream TCP link); if present, record them (fallback: configs/my_so101.yaml frames).")

    if args.keep_open and not args.headless:
        while simulation_app.is_running():
            hold_and_step(sim, robot, default, 1)
    simulation_app.close()
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
