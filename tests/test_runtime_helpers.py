"""A. Home tests for CLI parsers, manifest validation, checkpoints, docs, and graceful no-Isaac exits.

These never fake an Isaac run: scripts that need Isaac are only checked for argument
parsing and for exiting cleanly when Isaac Lab is unavailable.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ISAAC_PRESENT = importlib.util.find_spec("isaaclab") is not None
ISAAC_SCRIPTS = [
    "script04_convert_so101_to_usd.py",
    "script05_smoke_test_so101.py",
    "script06_test_joint_motion.py",
    "script07_run_tabletop_scene.py",
    "script08_random_policy.py",
    "script09_train_reach_ppo.py",
    "script10_play_reach_ppo.py",
    "script11_train_pick_place_ppo.py",
    "script12_play_pick_place_ppo.py",
]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_dume_rt_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- parsers

def test_train_play_parsers():
    from dume_isaac.runtime.rsl_rl_runner import build_play_parser, build_train_parser

    a = build_train_parser("reach").parse_args([])
    assert a.num_envs == 2048 and a.max_iterations is None and a.seed == 42
    a = build_train_parser("pick_place").parse_args(
        ["--num_envs", "256", "--max_iterations", "10", "--difficulty", "wide", "--seed", "3", "--resume"])
    assert (a.num_envs, a.max_iterations, a.difficulty, a.seed, a.resume) == (256, 10, "wide", 3, True)
    p = build_play_parser("pick_place").parse_args(["--checkpoint", "model_50.pt", "--load_run", "2026"])
    assert p.checkpoint == "model_50.pt" and p.load_run == "2026" and p.num_envs == 16
    with pytest.raises(SystemExit):
        build_train_parser("reach").parse_args(["--difficulty", "impossible"])


def test_script_parsers():
    assert _load_script("script08_random_policy").build_parser().parse_args(["--task", "reach"]).steps == 1000
    assert _load_script("script05_smoke_test_so101").build_parser().parse_args([]).steps is None
    assert _load_script("script07_run_tabletop_scene").build_parser().parse_args([]).num_envs == 4
    conv = _load_script("script04_convert_so101_to_usd").build_parser().parse_args([])
    assert conv.merge_fixed_joints == "false" and conv.collider_type == "convex_decomposition"
    assert conv.output.name == "my_so101.usd"


def test_joint_motion_plan_order_and_classify():
    jm = _load_script("script06_test_joint_motion")
    from dume_isaac.assets.my_so101.spec import load_so101_spec

    plan = jm.motion_plan(load_so101_spec(), 0.25)
    joints = [j for _, j, _ in plan if j]
    assert plan[0][0] == "home" and plan[-1][0] == "home"
    assert joints == ["shoulder_pan"] * 2 + ["shoulder_lift"] * 2 + ["elbow_flex"] * 2 + ["wrist_flex"] * 2 \
        + ["wrist_roll"] * 2 + ["gripper"] * 2
    assert [jm.classify(e, 0.05, 0.15) for e in (0.01, 0.1, 0.5)] == ["PASS", "WARN", "FAIL"]


# ---------------------------------------------------------------- graceful exit without Isaac

@pytest.mark.skipif(ISAAC_PRESENT, reason="Isaac Lab present: these scripts would really launch")
@pytest.mark.parametrize("script", ISAAC_SCRIPTS)
def test_isaac_scripts_exit_gracefully_without_isaac(script):
    args = ["--task", "reach"] if script.startswith("script08") else []
    out = subprocess.run([sys.executable, f"scripts/{script}", *args], cwd=ROOT,
                         capture_output=True, text=True, timeout=120)
    from dume_isaac.runtime.app import EXIT_NO_ISAAC

    if script.startswith("script04") and out.returncode == 1:
        assert "preflight" in out.stderr  # URDF not fetched on this machine: clean preflight failure
        return
    assert out.returncode == EXIT_NO_ISAAC, out.stderr
    assert "Traceback" not in out.stderr
    assert "isaaclab.sh -p" in out.stderr


@pytest.mark.parametrize("script", ISAAC_SCRIPTS)
def test_isaac_scripts_help_works_at_home(script):
    out = subprocess.run([sys.executable, f"scripts/{script}", "--help"], cwd=ROOT,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "usage" in out.stdout.lower()


# ---------------------------------------------------------------- manifest / source

def test_source_manifest_valid():
    yaml = pytest.importorskip("yaml")
    fetch = _load_script("script02_fetch_so101_source")
    m = yaml.safe_load((ROOT / "dume_isaac/assets/my_so101/source_manifest.yaml").read_text())
    assert fetch.validate_manifest(m) == []
    assert m["upstream"]["license"] == "Apache-2.0"
    assert m["local_modifications"] == "none"
    # every mesh referenced by the selected URDF must be pinned (13 meshes + URDF + LICENSE)
    assert len(m["files"]) == 15


def test_source_manifest_rejects_unofficial_or_unpinned():
    fetch = _load_script("script02_fetch_so101_source")
    bad = {"upstream": {"repository": "https://github.com/someone/fork", "commit": "main"},
           "files": [{"upstream_path": "x.urdf", "local_path": "/tmp/x.urdf", "sha256": "abc"}]}
    errors = " ".join(fetch.validate_manifest(bad))
    assert "official" in errors and "40-char" in errors and "sha256" in errors and "my_so101" in errors


def test_fetched_source_matches_manifest_if_present():
    pytest.importorskip("yaml")
    from dume_isaac.assets.my_so101.spec import load_so101_spec

    spec = load_so101_spec()
    if not spec.urdf_path.is_file():
        pytest.skip("SO-101 source not fetched on this machine (run script02)")
    assert _load_script("script02_fetch_so101_source").main(["--verify-only"]) == 0
    inspect = _load_script("script03_inspect_so101_urdf")
    assert inspect.check_mapping(inspect.parse_urdf(spec.urdf_path), spec) == []


# ---------------------------------------------------------------- checkpoints / compat

def test_resolve_checkpoint(tmp_path):
    from dume_isaac.runtime.checkpoints import resolve_checkpoint

    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(tmp_path / "none")
    old, new = tmp_path / "2026-01-01_10-00-00", tmp_path / "2026-02-01_10-00-00_x"
    for run, iters in ((old, (0, 50)), (new, (0, 50, 100, 1000))):
        run.mkdir()
        for i in iters:
            (run / f"model_{i}.pt").write_bytes(b"")
    assert resolve_checkpoint(tmp_path).name == "model_1000.pt"  # numeric, not lexical
    assert resolve_checkpoint(tmp_path, load_run="2026-01").parent.name == old.name
    assert resolve_checkpoint(tmp_path, checkpoint="model_100.pt").name == "model_100.pt"
    f = old / "model_50.pt"
    assert resolve_checkpoint(tmp_path, checkpoint=str(f)) == f.resolve()
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(tmp_path, checkpoint="model_7.pt")


def test_compat_cfg_drops_unknown_fields(capsys):
    from dume_isaac.tasks.agent_compat import compat_cfg

    @dataclasses.dataclass
    class Cfg:
        a: int = 0

    assert compat_cfg(Cfg, a=1, actor_obs_normalization=False).a == 1
    assert "actor_obs_normalization" in capsys.readouterr().out


# ---------------------------------------------------------------- docs

def test_environment_doc_has_confirmed_versions_and_pending_items():
    text = (ROOT / "docs/environment.md").read_text()
    for v in ("5.1.0", "v2.3.2", "3.11", "RTX 5060 Laptop"):
        assert v in text
    for item in ("NVIDIA driver", "CUDA runtime", "PyTorch", "Isaac Lab path", "Python executable", "Isaac launch command"):
        assert item in text


def test_checklist_has_all_stages_with_fields():
    text = (ROOT / "docs/school_runtime_checklist.md").read_text()
    sections = text.split("## Stage ")[1:]
    assert [int(s.split()[0]) for s in sections] == list(range(10))
    for s in sections:
        for field in ("**Command:**", "**Expected:**", "**PASS condition:**", "**If failed:**", "**Next:**"):
            assert field in s, (s.split("\n")[0], field)


def test_readme_separates_implemented_from_validated():
    text = (ROOT / "README.md").read_text()
    assert "Implemented locally" in text and "Runtime validated on Isaac 5.1.0" in text
    assert "validated" not in text.lower().replace("runtime validated", "").replace("validate", "")


def test_roadmap_marks_nothing_runtime_validated_yet():
    rows = [line for line in (ROOT / "docs/roadmap.md").read_text().splitlines() if line.startswith("| ") and "|" in line[2:]]
    data = [r for r in rows if r.split("|")[1].strip()[:1].isdigit()]
    assert data and all(r.rstrip(" |").split("|")[-1].strip() == "NO" for r in data)
