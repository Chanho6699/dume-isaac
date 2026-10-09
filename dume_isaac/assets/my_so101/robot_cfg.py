"""MY_SO101_CFG: Isaac Lab v2.3.2 ArticulationCfg for the SO-101 (converted USD).

Isaac runtime only: import after the app is launched (AppLauncher). Names and
parameters come from configs/my_so101.yaml through `spec.py`.

Actuator values are provisional simulation values (upstream STS3215 MuJoCo defaults),
not identified on the Brain Us robot.
"""

from __future__ import annotations

import re

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .spec import So101Spec, load_so101_spec


def _actuators(spec: So101Spec) -> dict[str, ImplicitActuatorCfg]:
    actuators = {}
    for g in spec.actuator_groups:
        kwargs = dict(
            joint_names_expr=[re.escape(spec.actual(j)) for j in g.joints],
            stiffness=g.stiffness,
            damping=g.damping,
        )
        if g.armature is not None:
            kwargs["armature"] = float(g.armature)
        if g.effort_limit_sim is not None:
            kwargs["effort_limit_sim"] = float(g.effort_limit_sim)
        if g.velocity_limit_sim is not None:
            kwargs["velocity_limit_sim"] = float(g.velocity_limit_sim)
        actuators[g.name] = ImplicitActuatorCfg(**kwargs)
    return actuators


def make_my_so101_cfg(
    spec: So101Spec | None = None,
    prim_path: str = "{ENV_REGEX_NS}/Robot",
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    usd_path: str | None = None,
) -> ArticulationCfg:
    spec = spec or load_so101_spec()
    usd = usd_path or str(spec.usd_path)
    from pathlib import Path  # noqa: PLC0415

    if not Path(usd).is_file():
        raise FileNotFoundError(
            f"SO-101 USD not found: {usd}\n"
            "  Run Stage 1-2 first:\n"
            "    python scripts/script02_fetch_so101_source.py\n"
            "    <isaaclab.sh -p> scripts/script04_convert_so101_to_usd.py --headless"
        )
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=pos,
            rot=rot,
            joint_pos=spec.home_joint_pos_actual(),
            joint_vel={".*": 0.0},
        ),
        actuators=_actuators(spec),
        soft_joint_pos_limit_factor=1.0,
    )


def __getattr__(name: str):
    # Built on access so that a missing USD gives the actionable error above.
    if name == "MY_SO101_CFG":
        return make_my_so101_cfg()
    raise AttributeError(name)
