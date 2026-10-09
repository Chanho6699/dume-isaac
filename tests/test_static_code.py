"""A. Home static checks: syntax, Isaac-free imports, naming, API generation guards."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PY_FILES = sorted(p for d in ("dume_isaac", "scripts", "tests") for p in (ROOT / d).rglob("*.py"))
SCRIPTS = sorted((ROOT / "scripts").glob("script*.py"))

# Modules that must import without Isaac / torch (the "pure" layer).
PURE_MODULES = [
    "dume_isaac",
    "dume_isaac.paths",
    "dume_isaac.config_io",
    "dume_isaac.assets",
    "dume_isaac.assets.my_so101",
    "dume_isaac.assets.my_so101.spec",
    "dume_isaac.envs",
    "dume_isaac.envs.tabletop",
    "dume_isaac.envs.tabletop.layout",
    "dume_isaac.tasks",
    "dume_isaac.tasks.registry",
    "dume_isaac.tasks.agent_compat",
    "dume_isaac.tasks.reach",
    "dume_isaac.tasks.pick_place",
    "dume_isaac.actor",
    "dume_isaac.runtime",
    "dume_isaac.runtime.app",
    "dume_isaac.runtime.checkpoints",
    "dume_isaac.runtime.rsl_rl_runner",
    "dume_isaac.runtime.single_robot",
]
HEAVY = ("isaaclab", "isaacsim", "omni", "torch", "rsl_rl", "isaaclab_rl", "gymnasium")


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_python_syntax_valid_for_py311(path):
    ast.parse(path.read_text(), filename=str(path), feature_version=(3, 11))


def test_pure_modules_import_without_isaac():
    code = (
        "import importlib, sys\n"
        f"mods = {PURE_MODULES!r}\n"
        "for m in mods: importlib.import_module(m)\n"
        f"heavy = [k for k in sys.modules if k.split('.')[0] in {HEAVY!r}]\n"
        "print('HEAVY=' + ','.join(sorted(heavy)))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "HEAVY=\n" in out.stdout or out.stdout.strip() == "HEAVY=", out.stdout


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_scripts_have_no_top_level_isaac_imports(path):
    """Isaac Lab requires AppLauncher before any isaaclab/omni/torch import."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for n in names:
            assert n.split(".")[0] not in HEAVY, f"{path.name}: top-level import {n}"


def test_scripts_are_numbered_in_order():
    names = [p.name for p in SCRIPTS]
    nums = [int(n[6:8]) for n in names]
    assert nums == list(range(1, len(names) + 1)), names
    assert not [p.name for p in (ROOT / "scripts").glob("*.py") if not p.name.startswith("script")]


def test_no_stale_brainus_reference():
    stale = "brainus" + "_so101"
    hits = []
    for p in ROOT.rglob("*"):
        if ".git" in p.parts or not p.is_file() or p.suffix in (".stl", ".pyc", ".usd"):
            continue
        try:
            text = p.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if stale in text.lower() or stale.upper() + "_CFG" in text:
            hits.append(str(p.relative_to(ROOT)))
    assert not list(ROOT.rglob(stale)), "stale path exists"
    assert hits == [], hits


def test_canonical_name_is_my_so101():
    assert (ROOT / "dume_isaac/assets/my_so101").is_dir()
    assert (ROOT / "configs/my_so101.yaml").is_file()
    src = (ROOT / "dume_isaac/assets/my_so101/robot_cfg.py").read_text()
    assert "MY_SO101_CFG" in src and "def make_my_so101_cfg" in src


def test_brain_us_facts_kept():
    text = (ROOT / "dume_isaac/assets/my_so101/robot_spec.yaml").read_text()
    for fact in ("manufacturer: Brain Us", "model: SO-101 follower", "base_design: TheRobotStudio SO-101"):
        assert fact in text


def test_no_wrong_generation_isaac_api():
    """Isaac Lab v2.x namespaces only: no 1.x `omni.isaac.lab` imports, no assumed SO101_CFG."""
    for p in (ROOT / "dume_isaac").rglob("*.py"):
        text = p.read_text()
        assert "omni.isaac.lab" not in text, p
        assert "import SO101_CFG" not in text and "SO101_CFG =" not in text.replace("MY_SO101_CFG", ""), p


REQUIRED_TASK_FILES = [
    "dume_isaac/assets/my_so101/__init__.py",
    "dume_isaac/assets/my_so101/robot_cfg.py",
    "dume_isaac/assets/my_so101/source_manifest.yaml",
    "dume_isaac/envs/tabletop/scene_cfg.py",
    "dume_isaac/envs/tabletop/env_cfg.py",
    "configs/reach.yaml",
    "configs/pick_place.yaml",
    "docs/school_runtime_checklist.md",
] + [
    f"dume_isaac/tasks/{t}/{f}"
    for t, cfg in (("reach", "reach_env_cfg.py"), ("pick_place", "pick_place_env_cfg.py"))
    for f in ("__init__.py", cfg, "mdp/__init__.py", "mdp/observations.py", "mdp/rewards.py",
              "mdp/terminations.py", "mdp/events.py", "agents/rsl_rl_ppo_cfg.py")
]


@pytest.mark.parametrize("rel", REQUIRED_TASK_FILES)
def test_task_and_reward_files_exist(rel):
    assert (ROOT / rel).is_file(), rel
