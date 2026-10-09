"""Single SO-101 on a ground plane (Stages 3-4). Isaac runtime only: call after launch_app()."""

from __future__ import annotations

from dume_isaac.assets.my_so101.spec import So101Spec


def make_single_robot_sim(spec: So101Spec, device: str, dt: float):
    """Ground + dome light + my_so101 at the origin. Returns (sim, robot) after sim.reset()."""
    import isaaclab.sim as sim_utils  # noqa: PLC0415
    from isaaclab.assets import Articulation  # noqa: PLC0415

    from dume_isaac.assets.my_so101.robot_cfg import make_my_so101_cfg  # noqa: PLC0415

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=device))
    sim.set_camera_view([0.55, 0.45, 0.40], [0.0, 0.0, 0.15])
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/defaultGroundPlane", ground)
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
    light.func("/World/Light", light)

    robot = Articulation(make_my_so101_cfg(spec=spec, prim_path="/World/Robot"))
    sim.reset()
    robot.update(sim.get_physics_dt())
    return sim, robot


def hold_and_step(sim, robot, target, steps: int) -> None:
    """Command a joint position target (shape [1, num_joints]) and step physics."""
    dt = sim.get_physics_dt()
    for _ in range(steps):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)


def resolve_joint_ids(spec: So101Spec, joint_names: list[str]) -> tuple[dict[str, int], list[str]]:
    """logical -> index into robot.joint_names, plus a list of problems."""
    ids, problems = {}, []
    for logical in spec.joint_map:
        actual = spec.actual(logical)
        if actual in joint_names:
            ids[logical] = joint_names.index(actual)
        else:
            problems.append(f"logical joint '{logical}' -> '{actual}' not found in USD joints {joint_names}")
    return ids, problems
