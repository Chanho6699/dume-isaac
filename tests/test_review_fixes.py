"""A. Home tests for: preserved EE frame (merge_fixed_joints=False), first-PPO sanity commands
in the checklist, visual-only target, and the pure reward model."""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from dume_isaac.assets.my_so101.spec import load_so101_spec, urdf_fixed_joint_into  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_TCP_XYZ = (-0.0079, -0.000218121, -0.0981274)  # gripper_frame_joint origin, pinned URDF


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"_dume_fix_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- 1. EE frame preserved

def test_converter_keeps_fixed_joints_and_ee_is_tcp_frame():
    spec = load_so101_spec()
    assert spec.merge_fixed_joints is False
    assert spec.ee_body == "gripper_frame_link"
    assert spec.ee_offset == (0.0, 0.0, 0.0)


def test_fallback_variant_documented_in_config():
    text = (ROOT / "configs/my_so101.yaml").read_text()
    assert "body: gripper_link" in text and "-0.0981274" in text


def _with(spec, **changes):
    return dataclasses.replace(spec, **changes)


def test_check_mapping_both_variants_against_pinned_urdf():
    spec = load_so101_spec()
    if not spec.urdf_path.is_file():
        pytest.skip("SO-101 source not fetched (run script02)")
    inspect = _script("script03_inspect_so101_urdf")
    robot = inspect.parse_urdf(spec.urdf_path)
    assert inspect.check_mapping(robot, spec) == []
    # merged variant with the URDF offset is also valid
    merged = _with(spec, ee_body="gripper_link", ee_offset=UPSTREAM_TCP_XYZ,
                   converter={**spec.converter, "merge_fixed_joints": True})
    assert inspect.check_mapping(robot, merged) == []
    # TCP frame + merging would delete the EE body
    bad = _with(spec, converter={**spec.converter, "merge_fixed_joints": True})
    assert any("merged away" in m for m in inspect.check_mapping(robot, bad))
    # TCP frame with a non-zero offset is double counting
    bad = _with(spec, ee_offset=UPSTREAM_TCP_XYZ)
    assert any("must be [0, 0, 0]" in m for m in inspect.check_mapping(robot, bad))


def test_urdf_fixed_joint_into():
    spec = load_so101_spec()
    if not spec.urdf_path.is_file():
        pytest.skip("SO-101 source not fetched (run script02)")
    name, parent, xyz = urdf_fixed_joint_into(spec.urdf_path, "gripper_frame_link")
    assert (name, parent) == ("gripper_frame_joint", "gripper_link")
    assert xyz == pytest.approx(UPSTREAM_TCP_XYZ)
    assert urdf_fixed_joint_into(spec.urdf_path, "gripper_link") is None


def test_smoke_test_checks_ee_frame():
    src = (ROOT / "scripts/script05_smoke_test_so101.py").read_text()
    assert "urdf_fixed_joint_into" in src and '"EE frame"' in src


# ---------------------------------------------------------------- 2. checklist first PPO runs

def _stage(text: str, n: int) -> str:
    return text.split(f"## Stage {n} ")[1].split("\n## ")[0]


@pytest.mark.parametrize("stage,script", [(7, "script09_train_reach_ppo.py"), (9, "script11_train_pick_place_ppo.py")])
def test_first_ppo_command_is_small(stage, script):
    sec = _stage((ROOT / "docs/school_runtime_checklist.md").read_text(), stage)
    first = next(line for line in sec.splitlines() if script in line)
    assert "--num_envs 64" in first and "--max_iterations 50" in first


def test_checklist_has_scale_up_procedure():
    text = (ROOT / "docs/school_runtime_checklist.md").read_text()
    sec = text.split("## Scaling num_envs")[1].split("\n## ")[0]
    for n in ("64", "256", "512", "1024", "2048"):
        assert f"| {n} |" in sec
    assert "nvidia-smi" in sec and "OOM" in sec


def test_baseline_num_envs_unchanged():
    from dume_isaac.tasks.registry import TASKS

    assert TASKS["reach"].train_num_envs == 2048 and TASKS["pick_place"].train_num_envs == 1024


# ---------------------------------------------------------------- 3. visual-only target

def test_target_is_not_a_physics_object():
    scene = (ROOT / "dume_isaac/envs/tabletop/scene_cfg.py").read_text()
    assert "{ENV_REGEX_NS}/Target" not in scene
    assert "kinematic_enabled" not in scene
    tree = ast.parse(scene)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TabletopSceneCfg")
    fields = {t.id for n in cls.body for t in ([n.target] if isinstance(n, ast.AnnAssign) else getattr(n, "targets", []))
              if isinstance(t, ast.Name)}
    assert "target" not in fields and {"robot", "cube", "table"} <= fields
    assert "VisualizationMarkersCfg" in scene


def test_no_physics_writes_or_scene_lookup_for_target():
    for p in (ROOT / "dume_isaac/tasks/pick_place").rglob("*.py"):
        text = p.read_text()
        assert 'scene["target"]' not in text and "scene[target_cfg" not in text, p
        assert 'SceneEntityCfg("target")' not in text, p
    cmd = (ROOT / "dume_isaac/tasks/pick_place/mdp/commands.py").read_text()
    assert "VisualizationMarkers" in cmd and "write_root" not in cmd


def test_pick_place_env_uses_target_command():
    src = (ROOT / "dume_isaac/tasks/pick_place/pick_place_env_cfg.py").read_text()
    assert "TargetPadCommandCfg" in src and "commands: CommandsCfg" in src
    assert "reset_object_uniform" in src


# ---------------------------------------------------------------- 4. pure reward model

def test_reward_model_is_pure_and_used_by_runtime_terms():
    code = ("import sys, dume_isaac.tasks.pick_place.reward_model\n"
            "print(any(k.split('.')[0] in ('torch', 'isaaclab') for k in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and out.stdout.strip() == "False", out.stderr
    rewards = (ROOT / "dume_isaac/tasks/pick_place/mdp/rewards.py").read_text()
    assert "from dume_isaac.tasks.pick_place.reward_model import place_step, stage_rewards" in rewards
    env = (ROOT / "dume_isaac/tasks/pick_place/pick_place_env_cfg.py").read_text()
    from dume_isaac.tasks.pick_place.reward_model import STAGES

    for stage in STAGES:
        assert f'{stage} = _stage("{stage}")' in env
