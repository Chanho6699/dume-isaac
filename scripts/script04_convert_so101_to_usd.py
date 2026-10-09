#!/usr/bin/env python3
"""Stage 2: convert the pinned SO-101 URDF to USD with the Isaac Lab v2.3.2 UrdfConverter.

Converter settings come from configs/my_so101.yaml (asset.converter) and can be overridden
on the CLI. Runs inside the Isaac app (the URDF importer is a Kit extension).

    <isaaclab.sh -p> scripts/script04_convert_so101_to_usd.py --headless
    <isaaclab.sh -p> scripts/script04_convert_so101_to_usd.py --headless --force
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dume_isaac.assets.my_so101.spec import load_so101_spec  # noqa: E402
from dume_isaac.config_io import load_yaml  # noqa: E402
from dume_isaac.runtime.app import launch_app  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    spec = load_so101_spec()
    conv = spec.converter
    p = argparse.ArgumentParser(prog="script04_convert_so101_to_usd.py", description=__doc__.splitlines()[0])
    p.add_argument("--urdf", type=Path, default=spec.urdf_path, help="input URDF")
    p.add_argument("--output", type=Path, default=spec.usd_path, help="output USD file (dir + asset name)")
    p.add_argument("--force", action="store_true", help="rebuild even if the USD is up to date")
    p.add_argument("--merge_fixed_joints", choices=["true", "false"],
                   default=str(conv.get("merge_fixed_joints", True)).lower())
    p.add_argument("--collider_type", choices=["convex_hull", "convex_decomposition"],
                   default=conv.get("collider_type", "convex_decomposition"))
    p.add_argument("--no_instanceable", action="store_true", help="disable make_instanceable")
    return p


def preflight(urdf: Path, spec) -> list[str]:
    """Isaac-free checks before starting the app. Returns problems (empty = OK)."""
    problems = []
    if not urdf.is_file():
        return [f"URDF not found: {urdf}  -> run: python scripts/script02_fetch_so101_source.py"]
    import re  # noqa: PLC0415

    for mesh in re.findall(r'filename="([^"]+)"', urdf.read_text()):
        if mesh.startswith("package://"):
            problems.append(f"package:// mesh path not supported here: {mesh}")
        elif not (urdf.parent / mesh).is_file():
            problems.append(f"missing mesh {urdf.parent / mesh}  -> run script02 again")
    if urdf.resolve() == spec.urdf_path.resolve():
        manifest = load_yaml(spec.source_manifest_path)
        want = next((f["sha256"] for f in manifest["files"] if f["local_path"].endswith(urdf.name)), None)
        have = hashlib.sha256(urdf.read_bytes()).hexdigest()
        if want and want != have:
            problems.append(f"{urdf.name} sha256 differs from source_manifest.yaml (file was modified?)")
    return problems


def main(argv: list[str] | None = None) -> int:
    spec = load_so101_spec()
    parser = build_parser()
    pre_args, _ = parser.parse_known_args(argv)
    problems = preflight(pre_args.urdf.expanduser(), spec)
    if problems:
        print("[convert] FAIL preflight:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1

    args, simulation_app = launch_app(parser, argv)

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: PLC0415

    from dume_isaac.tasks.agent_compat import compat_cfg  # noqa: PLC0415

    conv = spec.converter
    drive = conv.get("joint_drive") or {}
    output = args.output.expanduser().resolve()
    cfg = compat_cfg(
        UrdfConverterCfg,
        asset_path=str(args.urdf.expanduser().resolve()),
        usd_dir=str(output.parent),
        usd_file_name=output.name,
        force_usd_conversion=args.force,
        make_instanceable=not args.no_instanceable and bool(conv.get("make_instanceable", True)),
        fix_base=bool(conv.get("fix_base", True)),
        merge_fixed_joints=args.merge_fixed_joints == "true",
        self_collision=bool(conv.get("self_collision", False)),
        collider_type=args.collider_type,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type=drive.get("drive_type", "force"),
            target_type=drive.get("target_type", "position"),
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=float(drive.get("stiffness", 17.8)),
                damping=float(drive.get("damping", 0.6)),
            ),
        ),
    )
    print(f"[convert] URDF   : {cfg.asset_path}")
    print(f"[convert] output : {output}")
    print(f"[convert] fix_base={cfg.fix_base} merge_fixed_joints={cfg.merge_fixed_joints} "
          f"collider={args.collider_type} instanceable={cfg.make_instanceable}")
    code = 0
    try:
        converter = UrdfConverter(cfg)
        usd_path = Path(converter.usd_path)
        if not usd_path.is_file():
            raise RuntimeError(f"converter returned {usd_path} but the file does not exist")
        print(f"[convert] PASS  USD written: {usd_path}")
        if usd_path.resolve() != spec.usd_path.resolve():
            print(f"[convert] NOTE  configs/my_so101.yaml asset.usd_path is {spec.usd_path}; update it to use this USD.")
    except Exception as exc:
        code = 1
        print(f"[convert] FAIL  {type(exc).__name__}: {exc}", file=sys.stderr)
        print("  Checks: URDF importer extension enabled? meshes present (script02)? "
              "try --collider_type convex_hull or --force; see docs/school_runtime_checklist.md Stage 2.",
              file=sys.stderr)
    simulation_app.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
