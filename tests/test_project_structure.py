"""Foundation-level checks. Must run without Isaac Sim / Isaac Lab / PyTorch."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRS = [
    "dume_isaac/assets/brainus_so101",
    "dume_isaac/envs/tabletop",
    "dume_isaac/tasks/pick_place",
    "dume_isaac/actor",
    "configs",
    "scripts",
    "docs",
]
REQUIRED_FILES = [
    "README.md",
    ".gitignore",
    "pyproject.toml",
    "dume_isaac/__init__.py",
    "dume_isaac/assets/brainus_so101/README.md",
    "dume_isaac/assets/brainus_so101/robot_spec.yaml",
    "configs/brainus_so101.yaml",
    "configs/tabletop.yaml",
    "scripts/check_environment.py",
    "scripts/inspect_so101_urdf.py",
    "docs/environment.md",
    "docs/roadmap.md",
    "docs/real_robot_spec.md",
    "docs/real_sim_contract.md",
]
YAML_FILES = [
    "dume_isaac/assets/brainus_so101/robot_spec.yaml",
    "configs/brainus_so101.yaml",
    "configs/tabletop.yaml",
]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_dume_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("rel", REQUIRED_DIRS)
def test_required_dirs_exist(rel):
    assert (ROOT / rel).is_dir(), rel


@pytest.mark.parametrize("rel", REQUIRED_FILES)
def test_required_files_exist(rel):
    assert (ROOT / rel).is_file(), rel


def test_package_imports_without_isaac():
    sys.path.insert(0, str(ROOT))
    try:
        import dume_isaac  # noqa: F401
        import dume_isaac.actor  # noqa: F401
        import dume_isaac.envs.tabletop  # noqa: F401
        import dume_isaac.tasks.pick_place  # noqa: F401
    finally:
        sys.path.remove(str(ROOT))


@pytest.mark.parametrize("rel", YAML_FILES)
def test_yaml_parses(rel):
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load((ROOT / rel).read_text())
    assert isinstance(data, dict) and data


def test_robot_spec_content():
    yaml = pytest.importorskip("yaml")
    spec = yaml.safe_load((ROOT / "dume_isaac/assets/brainus_so101/robot_spec.yaml").read_text())
    assert spec["arm_joints"] == 5
    assert spec["gripper_joints"] == 1
    assert spec["total_actuated_joints"] == 6
    assert spec["actuator"]["model"] == "Feetech ST-3215-C018"
    # Nothing is verified yet; flip these only with recorded evidence.
    assert all(v is False for v in spec["verification"].values())


def test_tabletop_has_no_invented_dimensions():
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load((ROOT / "configs/tabletop.yaml").read_text())
    assert cfg["scene_type"] == "fixed_base_tabletop"
    assert all(v is None for v in cfg["table"].values())
    assert cfg["objects"]["cube"]["size"] is None


def test_check_environment_import_has_no_side_effects(capsys):
    torch_loaded_before = "torch" in sys.modules
    module = _load_script("check_environment")
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
    assert callable(module.main)
    # Importing must not pull in heavy / Isaac packages.
    assert ("torch" in sys.modules) == torch_loaded_before
    assert "isaacsim" not in sys.modules
    assert "isaaclab" not in sys.modules


def test_check_environment_runs(capsys):
    module = _load_script("check_environment")
    assert module.main([]) == 0
    out = capsys.readouterr().out
    assert "=== DUM-E Isaac Environment Check ===" in out
    assert "Isaac Sim :" in out and "Isaac Lab :" in out


def test_inspect_urdf_missing_file(capsys, tmp_path):
    module = _load_script("inspect_so101_urdf")
    assert module.main([str(tmp_path / "nope.urdf")]) == 2
    assert "not found" in capsys.readouterr().err


def test_inspect_urdf_parses_minimal_urdf(capsys, tmp_path):
    # Synthetic two-link URDF for parser testing only; not an SO-101 model.
    urdf = tmp_path / "toy.urdf"
    urdf.write_text(
        '<robot name="toy">'
        '<link name="a"/><link name="b"/>'
        '<joint name="j1" type="revolute"><parent link="a"/><child link="b"/>'
        '<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="2" velocity="3"/></joint>'
        "</robot>"
    )
    module = _load_script("inspect_so101_urdf")
    assert module.main([str(urdf)]) == 0
    out = capsys.readouterr().out
    assert "Robot name : toy" in out
    assert "j1" in out and "revolute" in out and "-1" in out
