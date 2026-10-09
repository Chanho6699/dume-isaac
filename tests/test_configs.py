"""A. Home unit tests for the pure config layer (my_so101 spec, tabletop layout, task params, registry)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from dume_isaac.assets.my_so101.spec import LOGICAL_JOINTS, load_so101_spec  # noqa: E402
from dume_isaac.envs.tabletop.layout import DIFFICULTIES, MEASURED, PROVISIONAL, load_tabletop_layout  # noqa: E402
from dume_isaac.tasks.registry import TASKS, load_task_params  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_spec_maps_all_logical_joints():
    spec = load_so101_spec()
    assert set(spec.joint_map) == set(LOGICAL_JOINTS)
    assert len(spec.arm_joint_names) == 5
    assert spec.all_joint_names[-1] == spec.gripper_joint_name
    assert set(spec.home_joint_pos_actual()) == set(spec.all_joint_names)
    assert spec.usd_path.name == "my_so101.usd"
    assert spec.urdf_path.name == "so101_new_calib.urdf"


def test_spec_actuators_cover_joints_and_are_marked_provisional():
    spec = load_so101_spec()
    covered = [j for g in spec.actuator_groups for j in g.joints]
    assert sorted(covered) == sorted(LOGICAL_JOINTS)
    raw = yaml.safe_load((ROOT / "configs/my_so101.yaml").read_text())
    assert raw["actuators"]["status"] == "provisional simulation value"
    assert set(raw["calibration"].values()) == {"pending"}


def test_spec_rejects_bad_mapping(tmp_path):
    raw = yaml.safe_load((ROOT / "configs/my_so101.yaml").read_text())
    del raw["joints"]["wrist_roll"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="joints"):
        load_so101_spec(bad)


def test_tabletop_measured_section_still_null_and_defaults_marked_provisional():
    layout = load_tabletop_layout()
    assert layout.scene_type == "fixed_base_tabletop"
    assert set(layout.status.values()) == {PROVISIONAL}  # nothing measured yet
    assert layout.table_height > 0 and layout.cube_size > 0


def test_tabletop_measured_value_overrides_default(tmp_path):
    raw = yaml.safe_load((ROOT / "configs/tabletop.yaml").read_text())
    raw["table"]["height"] = 0.72
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(raw))
    layout = load_tabletop_layout(p)
    assert layout.table_height == 0.72 and layout.status["table.height"] == MEASURED
    assert layout.status["table.width"] == PROVISIONAL


@pytest.mark.parametrize("level", DIFFICULTIES)
def test_randomization_inside_workspace(level):
    layout = load_tabletop_layout()
    lv = layout.level(level)
    wx, wy, wz = layout.workspace_x, layout.workspace_y, layout.workspace_z
    for r in (lv.cube_x, lv.target_x, lv.reach_x):
        assert wx[0] <= r[0] <= r[1] <= wx[1]
    for r in (lv.cube_y, lv.target_y, lv.reach_y):
        assert wy[0] <= r[0] <= r[1] <= wy[1]
    assert wz[0] <= lv.reach_z[0] <= lv.reach_z[1] <= wz[1]
    # a valid cube/target pair with the required separation must exist
    span = max(abs(lv.cube_y[0] - lv.target_y[1]), abs(lv.cube_y[1] - lv.target_y[0]),
               abs(lv.cube_x[0] - lv.target_x[1]), abs(lv.cube_x[1] - lv.target_x[0]))
    assert span >= lv.min_separation


def test_difficulty_levels_widen():
    layout = load_tabletop_layout()
    width = [(lv.cube_y[1] - lv.cube_y[0]) for lv in (layout.level(d) for d in DIFFICULTIES)]
    noise = [layout.level(d).robot_joint_noise for d in DIFFICULTIES]
    assert width == sorted(width) and noise == sorted(noise)


def test_pick_place_reward_anti_hover():
    """Placing must pay more per step than every 'held' stage combined."""
    rew = load_task_params("pick_place")["rewards"]
    held = sum(rew[k]["weight"] for k in ("grasp", "lift", "carry", "carry_fine", "descend"))
    assert rew["place"]["weight"] > held
    assert rew["dropped"]["weight"] < 0


def test_pick_place_success_requires_resting_release_and_lift():
    p = load_task_params("pick_place")
    thr = p["thresholds"]
    assert thr["require_gripper_open"] and thr["require_was_lifted"]
    assert thr["place_height_tol"] < thr["held_min_height"] < thr["lift_height"]
    assert p["terminate_on_success"] is False
    assert p["actions"]["gripper"] in ("binary", "continuous")
    assert p["difficulty"] in DIFFICULTIES


def test_reach_params():
    p = load_task_params("reach")
    assert p["difficulty"] in DIFFICULTIES
    assert p["rewards"]["distance"]["weight"] < 0 < p["rewards"]["distance_tanh"]["weight"]


def _defined_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


@pytest.mark.parametrize("name", list(TASKS))
def test_registry_entries_point_to_defined_symbols(name):
    task = TASKS[name]
    for entry in (task.env_cfg_entry, task.agent_cfg_entry):
        module, attr = entry.split(":")
        path = ROOT / (module.replace(".", "/") + ".py")
        assert path.is_file(), path
        assert attr in _defined_names(path), entry
    agent_src = (ROOT / (task.agent_cfg_entry.split(":")[0].replace(".", "/") + ".py")).read_text()
    assert f'experiment_name = "{task.experiment_name}"' in agent_src
    assert 0 < task.play_num_envs <= task.train_num_envs <= 4096


def test_mdp_terms_used_by_env_cfgs_are_defined():
    """Every `mdp.<name>` referenced in an env cfg is either local or a known Isaac Lab v2.3.2 built-in."""
    builtins = {"UniformPoseCommandCfg", "JointPositionActionCfg", "BinaryJointPositionActionCfg", "joint_pos_rel",
                "joint_vel_rel", "last_action", "reset_joints_by_offset", "reset_scene_to_default", "action_rate_l2",
                "joint_vel_l2", "time_out", "modify_reward_weight", "is_terminated_term"}
    for task, cfg_file in (("reach", "reach_env_cfg.py"), ("pick_place", "pick_place_env_cfg.py")):
        tdir = ROOT / "dume_isaac/tasks" / task
        local = set().union(*(_defined_names(p) for p in (tdir / "mdp").glob("*.py")))
        tree = ast.parse((tdir / cfg_file).read_text())
        used = {n.attr for n in ast.walk(tree)
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "mdp"}
        missing = used - local - builtins
        assert not missing, f"{task}: {missing}"
