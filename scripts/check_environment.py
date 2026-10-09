#!/usr/bin/env python3
"""Read-only environment diagnostic for dume-isaac.

Prints Python / platform / PyTorch / CUDA / GPU / Isaac Sim / Isaac Lab information.
It never installs or modifies anything, and never dies with a traceback: every probe
is wrapped and reports its own failure.

By default Isaac packages are only *located* (importlib find_spec + package metadata),
not imported, because importing them may start Kit or require a running SimulationApp.
Pass --try-import to also attempt a plain `import isaacsim` / `import isaaclab`.

Run it with the same interpreter you use for Isaac Lab, e.g.:
    python scripts/check_environment.py
    ./isaaclab.sh -p /path/to/dume-isaac/scripts/check_environment.py
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

DETECTED = "detected"
NOT_DETECTED = "not detected"
UNKNOWN = "unknown"

ISAAC_ENV_VARS = ("ISAAC_PATH", "ISAACSIM_PATH", "CARB_APP_PATH", "EXP_PATH", "ISAACLAB_PATH")
VENV_ENV_VARS = ("VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV")
EXTRA_PACKAGES = ("torch", "torchvision", "numpy", "gymnasium", "rsl-rl-lib", "skrl", "rl-games", "pyyaml")
SO101_PATTERN = re.compile(r"SO[_-]?101|SO[_-]?ARM", re.IGNORECASE)


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def _dists_with_prefix(prefixes: tuple[str, ...]) -> dict[str, str]:
    found: dict[str, str] = {}
    try:
        for dist in importlib.metadata.distributions():
            name = (dist.metadata.get("Name") or "").strip()
            if name.lower().replace("_", "-").startswith(prefixes):
                found[name] = dist.version
    except Exception:
        pass
    return dict(sorted(found.items()))


def _find_spec(name: str):
    try:
        return importlib.util.find_spec(name)
    except Exception:
        return None


def _spec_dirs(spec) -> list[Path]:
    if spec is None:
        return []
    if spec.submodule_search_locations:
        return [Path(p) for p in spec.submodule_search_locations]
    if spec.origin and spec.origin not in ("built-in", "frozen"):
        return [Path(spec.origin).parent]
    return []


def _read_version_file(start: Path, max_up: int = 4) -> str | None:
    """Look for a VERSION file in `start` or its parents (Isaac Sim installs ship one)."""
    try:
        p = start.resolve()
        for _ in range(max_up + 1):
            vf = p / "VERSION"
            if vf.is_file():
                return vf.read_text(errors="replace").strip().splitlines()[0]
            p = p.parent
    except Exception:
        pass
    return None


def _try_import(name: str) -> tuple[bool, str | None]:
    try:
        importlib.import_module(name)
        return True, None
    except BaseException as exc:  # some Kit modules raise SystemExit
        return False, _err(exc)


# ---------------------------------------------------------------- probes


def probe_system() -> dict:
    info = {
        "python": sys.version.split()[0],
        "python_full": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "os_release": None,
        "wsl": "microsoft" in platform.release().lower(),
        "venv": {k: os.environ[k] for k in VENV_ENV_VARS if os.environ.get(k)},
    }
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                info["os_release"] = line.split("=", 1)[1].strip('"')
    except Exception:
        pass
    return info


def probe_torch() -> dict:
    info: dict = {"installed": False, "version": None, "cuda_available": None,
                  "cuda_version": None, "gpus": [], "error": None}
    if _find_spec("torch") is None:
        return info
    try:
        import torch  # noqa: PLC0415

        info["installed"] = True
        info["version"] = torch.__version__
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            for i in range(torch.cuda.device_count()):
                info["gpus"].append(torch.cuda.get_device_name(i))
    except BaseException as exc:
        info["error"] = _err(exc)
    return info


def probe_nvidia_smi() -> dict:
    info: dict = {"available": False, "gpus": [], "error": None}
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return info
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if out.returncode == 0:
            info["available"] = True
            info["gpus"] = [line.strip() for line in out.stdout.splitlines() if line.strip()]
        else:
            info["error"] = (out.stderr or out.stdout).strip() or f"exit {out.returncode}"
    except Exception as exc:
        info["error"] = _err(exc)
    return info


def probe_isaac_sim(try_import: bool) -> dict:
    spec = _find_spec("isaacsim")
    dirs = _spec_dirs(spec)
    env = {k: os.environ[k] for k in ISAAC_ENV_VARS if k != "ISAACLAB_PATH" and os.environ.get(k)}
    info: dict = {
        "module_found": spec is not None,
        "module_path": [str(d) for d in dirs],
        "dists": _dists_with_prefix(("isaacsim",)),
        "version_file": None,
        "env": env,
        "import_ok": None,
        "import_error": None,
    }
    for d in dirs:
        info["version_file"] = info["version_file"] or _read_version_file(d)
    for key in ("ISAAC_PATH", "ISAACSIM_PATH"):
        if key in env and not info["version_file"]:
            info["version_file"] = _read_version_file(Path(env[key]), max_up=0)
    if try_import and spec is not None:
        info["import_ok"], info["import_error"] = _try_import("isaacsim")

    if info["import_ok"] is False:
        info["status"] = UNKNOWN  # present but not importable from this interpreter
    elif spec is not None or info["dists"]:
        info["status"] = DETECTED
    elif env:
        info["status"] = UNKNOWN  # binary install hinted, but not on this interpreter's path
    else:
        info["status"] = NOT_DETECTED
    return info


def probe_isaac_lab(try_import: bool) -> dict:
    modules = {}
    for name in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl", "isaaclab_mimic"):
        spec = _find_spec(name)
        modules[name] = [str(d) for d in _spec_dirs(spec)] if spec is not None else None
    # Legacy (Isaac Lab 1.x / Orbit-era) namespace, reported for information only.
    legacy = _find_spec("omni.isaac.lab") if _find_spec("omni") is not None else None
    info: dict = {
        "modules": modules,
        "legacy_omni_isaac_lab": legacy is not None,
        "dists": _dists_with_prefix(("isaaclab", "isaac-lab")),
        "env": {k: os.environ[k] for k in ("ISAACLAB_PATH",) if os.environ.get(k)},
        "import_ok": None,
        "import_error": None,
    }
    found = modules["isaaclab"] is not None
    if try_import and found:
        info["import_ok"], info["import_error"] = _try_import("isaaclab")

    if info["import_ok"] is False:
        info["status"] = UNKNOWN
    elif found or info["dists"]:
        info["status"] = DETECTED
    elif info["legacy_omni_isaac_lab"] or info["env"]:
        info["status"] = UNKNOWN
    else:
        info["status"] = NOT_DETECTED
    return info


def search_so101_assets(lab: dict, max_hits: int = 20) -> dict:
    """Text-search installed Isaac Lab asset/task sources for SO-101 references. No imports."""
    info: dict = {"searched": [], "hits": [], "error": None}
    try:
        for name in ("isaaclab_assets", "isaaclab_tasks"):
            for d in lab["modules"].get(name) or []:
                root = Path(d)
                info["searched"].append(str(root))
                for py in root.rglob("*.py"):
                    try:
                        text = py.read_text(errors="replace")
                    except Exception:
                        continue
                    for lineno, line in enumerate(text.splitlines(), 1):
                        if SO101_PATTERN.search(line):
                            info["hits"].append(f"{py}:{lineno}: {line.strip()[:120]}")
                            if len(info["hits"]) >= max_hits:
                                return info
    except Exception as exc:
        info["error"] = _err(exc)
    return info


# ---------------------------------------------------------------- report


def _versions_str(d: dict) -> str:
    return ", ".join(f"{k}=={v}" for k, v in d.items()) if d else "-"


def print_report(sysinfo, torch_info, smi, sim, lab, so101) -> None:
    print("=== DUM-E Isaac Environment Check ===")
    print("(read-only diagnostic; nothing is installed or modified)\n")

    print("[System]")
    print(f"  Python      : {sysinfo['python_full']}")
    print(f"  Executable  : {sysinfo['executable']}")
    print(f"  Prefix      : {sysinfo['prefix']}")
    print(f"  Platform    : {sysinfo['platform']} ({sysinfo['machine']})")
    if sysinfo["os_release"]:
        print(f"  OS release  : {sysinfo['os_release']}")
    if sysinfo["wsl"]:
        print("  Note        : running under WSL")
    for k, v in sysinfo["venv"].items():
        print(f"  {k:<12}: {v}")

    print("\n[PyTorch / CUDA]")
    if not torch_info["installed"]:
        print("  torch       : not installed" + (f" ({torch_info['error']})" if torch_info["error"] else ""))
    else:
        print(f"  torch       : {torch_info['version']}")
        print(f"  torch CUDA  : {torch_info['cuda_version']}")
        print(f"  CUDA avail  : {torch_info['cuda_available']}")
        for i, g in enumerate(torch_info["gpus"]):
            print(f"  GPU[{i}]      : {g}")
        if torch_info["error"]:
            print(f"  error       : {torch_info['error']}")

    print("\n[nvidia-smi]  (name, driver, memory)")
    if smi["available"]:
        for g in smi["gpus"]:
            print(f"  {g}")
    else:
        print(f"  not available{' (' + smi['error'] + ')' if smi['error'] else ''}")

    print("\n[Isaac Sim]")
    print(f"  status      : {sim['status']}")
    print(f"  module path : {', '.join(sim['module_path']) or '-'}")
    print(f"  packages    : {_versions_str(sim['dists'])}")
    print(f"  VERSION file: {sim['version_file'] or '-'}")
    for k, v in sim["env"].items():
        print(f"  ${k:<11}: {v}")
    if sim["import_ok"] is not None:
        print(f"  import      : {'ok' if sim['import_ok'] else 'FAILED - ' + str(sim['import_error'])}")

    print("\n[Isaac Lab]")
    print(f"  status      : {lab['status']}")
    for name, dirs in lab["modules"].items():
        print(f"  {name:<16}: {', '.join(dirs) if dirs else 'not found'}")
    print(f"  packages    : {_versions_str(lab['dists'])}")
    if lab["legacy_omni_isaac_lab"]:
        print("  legacy      : omni.isaac.lab namespace found (Isaac Lab 1.x style)")
    for k, v in lab["env"].items():
        print(f"  ${k:<11}: {v}")
    if lab["import_ok"] is not None:
        print(f"  import      : {'ok' if lab['import_ok'] else 'FAILED - ' + str(lab['import_error'])}")

    print("\n[Other packages]")
    for name in EXTRA_PACKAGES:
        print(f"  {name:<16}: {_dist_version(name) or '-'}")

    print("\n[SO-101 asset search]  (text search in isaaclab_assets / isaaclab_tasks)")
    if not so101["searched"]:
        print("  skipped: isaaclab_assets / isaaclab_tasks not found")
    elif so101["hits"]:
        for h in so101["hits"]:
            print(f"  {h}")
    else:
        print("  no SO-101 references found -> plan: import standard SO-101 URDF to USD")
    if so101["error"]:
        print(f"  error: {so101['error']}")

    gpu = (torch_info["gpus"] or [g.split(",")[0] for g in smi["gpus"]] or ["not detected"])[0]
    if torch_info["installed"]:
        cuda = f"available (torch CUDA {torch_info['cuda_version']})" if torch_info["cuda_available"] else "not available"
    else:
        cuda = UNKNOWN + " (torch not installed)"
    sim_ver = _versions_str(sim["dists"]) if sim["dists"] else (sim["version_file"] or "")
    lab_ver = _versions_str(lab["dists"])
    if not so101["searched"]:
        so101_line = UNKNOWN + " (Isaac Lab assets not found)"
    else:
        so101_line = "references found (verify manually)" if so101["hits"] else "not found"

    print("\n=== Summary ===")
    print(f"Python    : {sysinfo['python']} ({sysinfo['executable']})")
    print(f"Platform  : {sysinfo['platform']}")
    print(f"Torch     : {torch_info['version'] or 'not installed'}")
    print(f"CUDA      : {cuda}")
    print(f"GPU       : {gpu}")
    print(f"Isaac Sim : {sim['status']}" + (f"  [{sim_ver}]" if sim_ver else ""))
    print(f"Isaac Lab : {lab['status']}" + (f"  [{lab_ver}]" if lab["dists"] else ""))
    print(f"SO-101 cfg: {so101_line}")
    print("\nCopy this whole output into docs/environment.md.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only DUM-E Isaac environment diagnostic.")
    parser.add_argument("--try-import", action="store_true",
                        help="also attempt `import isaacsim` / `import isaaclab` (may be slow)")
    args = parser.parse_args(argv)

    sysinfo = probe_system()
    torch_info = probe_torch()
    smi = probe_nvidia_smi()
    sim = probe_isaac_sim(args.try_import)
    lab = probe_isaac_lab(args.try_import)
    so101 = search_so101_assets(lab)
    print_report(sysinfo, torch_info, smi, sim, lab, so101)
    return 0


if __name__ == "__main__":
    sys.exit(main())
