#!/usr/bin/env python3
"""Stage 5: tabletop scene (ground, table, my_so101, blue cube, light) + visual-only target pad marker.

Holds the robot at home, lets the cube settle, resets periodically and checks that the
cube rests on the table (no fall-through, no NaN). Prints which layout values are
measured vs provisional (configs/tabletop.yaml).

    <isaaclab.sh -p> scripts/script07_run_tabletop_scene.py
    <isaaclab.sh -p> scripts/script07_run_tabletop_scene.py --headless --num_envs 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dume_isaac.envs.tabletop.layout import load_tabletop_layout  # noqa: E402
from dume_isaac.runtime.app import launch_app  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="script07_run_tabletop_scene.py", description="Tabletop scene check.")
    p.add_argument("--num_envs", type=int, default=4)
    p.add_argument("--steps", type=int, default=600, help="physics steps (0 = run until window closes)")
    p.add_argument("--reset_every", type=int, default=300)
    return p


def main(argv: list[str] | None = None) -> int:
    layout = load_tabletop_layout()
    args, simulation_app = launch_app(build_parser(), argv)

    import isaaclab.sim as sim_utils  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from isaaclab.markers import VisualizationMarkers  # noqa: PLC0415
    from isaaclab.scene import InteractiveScene  # noqa: PLC0415

    from dume_isaac.envs.tabletop.scene_cfg import (  # noqa: PLC0415
        ENV_SPACING,
        TabletopSceneCfg,
        base_to_env,
        cube_rest_z_in_base,
        default_target_in_base,
        target_pad_marker_cfg,
    )

    print("\n=== tabletop layout (measured / provisional) ===")
    for key, st in layout.status.items():
        print(f"  {key:<24} {st}")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device))
    sim.set_camera_view(list(base_to_env(0.7, 0.5, 0.45)), list(base_to_env(0.2, 0.0, 0.0)))
    scene = InteractiveScene(TabletopSceneCfg(num_envs=args.num_envs, env_spacing=ENV_SPACING))
    sim.reset()
    dt = sim.get_physics_dt()
    robot, cube = scene["robot"], scene["cube"]
    # target pad: visual marker only (no physics); in RL envs it is driven by the target command
    pad = VisualizationMarkers(target_pad_marker_cfg())
    pad.visualize(translations=robot.data.root_pos_w + torch.tensor(default_target_in_base(), device=robot.device))

    def reset():
        root = cube.data.default_root_state.clone()
        root[:, :3] += scene.env_origins
        cube.write_root_pose_to_sim(root[:, :7])
        cube.write_root_velocity_to_sim(root[:, 7:])
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
        scene.reset()

    reset()
    rest_z = cube_rest_z_in_base()
    worst_sink, finite, step = 0.0, True, 0
    while simulation_app.is_running() and (args.steps <= 0 or step < args.steps):
        if step > 0 and step % args.reset_every == 0:
            reset()
        robot.set_joint_position_target(robot.data.default_joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)
        step += 1
        if step % args.reset_every > 100:  # settled
            cube_b = cube.data.root_pos_w - robot.data.root_pos_w
            finite &= bool(torch.isfinite(cube_b).all())
            worst_sink = max(worst_sink, float((rest_z - cube_b[:, 2]).max()))
        if step % 100 == 0:
            p = (cube.data.root_pos_w[0] - robot.data.root_pos_w[0]).tolist()
            print(f"[scene] step {step} cube[0] in base frame: ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})")

    ok_sink = worst_sink < 0.005
    print("\n--- results ---")
    print(f"  [{'PASS' if finite else 'FAIL'}] finite cube state")
    print(f"  [{'PASS' if ok_sink else 'FAIL'}] cube rests on table (max sink {worst_sink * 1000:.1f} mm, limit 5 mm)")
    overall = "PASS" if finite and ok_sink else "FAIL"
    print(f"\nTABLETOP SCENE: {overall}  (visually check: robot on table top, cube blue, pad green, in reach)")
    simulation_app.close()
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
