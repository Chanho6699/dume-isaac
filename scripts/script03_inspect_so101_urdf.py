#!/usr/bin/env python3
"""Stage 1b: inspect an SO-101 URDF: robot name, joints, types, parent/child, limits.

Python stdlib only for parsing (no Isaac dependency). Read-only.

Usage:
    python scripts/script03_inspect_so101_urdf.py                  # pinned URDF + mapping check
    python scripts/script03_inspect_so101_urdf.py path/to/so101.urdf [--check-mapping]

--check-mapping compares the URDF with configs/my_so101.yaml (joint names, EE body,
EE offset vs. the fixed TCP joint). It is on by default when no path is given.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

USAGE_HINT = """\
Usage:
    python scripts/script03_inspect_so101_urdf.py [path/to/so101.urdf] [--check-mapping]

The official SO-101 URDF is not committed. Fetch the pinned upstream files first:
    python scripts/script02_fetch_so101_source.py
(source and commit: dume_isaac/assets/my_so101/source_manifest.yaml)
"""


def parse_urdf(path: Path) -> dict:
    root = ET.parse(path).getroot()
    if root.tag != "robot":
        raise ValueError(f"root element is <{root.tag}>, expected <robot>")

    joints = []
    for j in root.findall("joint"):
        parent = j.find("parent")
        child = j.find("child")
        limit = j.find("limit")
        axis = j.find("axis")
        mimic = j.find("mimic")
        origin = j.find("origin")
        joints.append({
            "xyz": origin.get("xyz") if origin is not None else None,
            "name": j.get("name", "?"),
            "type": j.get("type", "?"),
            "parent": parent.get("link") if parent is not None else None,
            "child": child.get("link") if child is not None else None,
            "axis": axis.get("xyz") if axis is not None else None,
            "lower": limit.get("lower") if limit is not None else None,
            "upper": limit.get("upper") if limit is not None else None,
            "effort": limit.get("effort") if limit is not None else None,
            "velocity": limit.get("velocity") if limit is not None else None,
            "mimic": mimic.get("joint") if mimic is not None else None,
        })
    return {
        "name": root.get("name", "?"),
        "links": [link.get("name", "?") for link in root.findall("link")],
        "joints": joints,
    }


def _fmt(v) -> str:
    return "-" if v is None else str(v)


def print_report(path: Path, robot: dict) -> None:
    joints = robot["joints"]
    movable = [j for j in joints if j["type"] != "fixed"]
    print(f"=== URDF: {path} ===")
    print(f"Robot name : {robot['name']}")
    print(f"Links      : {len(robot['links'])}")
    print(f"Joints     : {len(joints)} total, {len(movable)} non-fixed\n")

    cols = ("name", "type", "parent", "child", "axis", "lower", "upper", "effort", "velocity")
    rows = [[_fmt(j[c]) for c in cols] for j in joints]
    widths = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c) for i, c in enumerate(cols)]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(v.ljust(w) for v, w in zip(r, widths)))

    mimics = [j for j in joints if j["mimic"]]
    if mimics:
        print("\nMimic joints:")
        for j in mimics:
            print(f"  {j['name']} mimics {j['mimic']}")

    print("\nNon-fixed joint names (must match `joints` in configs/my_so101.yaml):")
    for j in movable:
        print(f"  - {j['name']}")
    print("\nNote: URDF limits are CAD/model values, not the calibrated limits of the real robot.")


def check_mapping(robot: dict, spec) -> list[str]:
    """Compare a parsed URDF with the canonical my_so101 mapping. Returns problems (empty = OK)."""
    problems = []
    by_name = {j["name"]: j for j in robot["joints"]}
    links = set(robot["links"])
    for logical in spec.joint_map:
        actual = spec.actual(logical)
        if actual not in by_name:
            problems.append(f"joint '{logical}' -> '{actual}' not in URDF")
        elif by_name[actual]["type"] == "fixed":
            problems.append(f"joint '{actual}' is fixed in the URDF")
    for label, body in (("frames.base", spec.base_body), ("frames.end_effector.body", spec.ee_body)):
        if body not in links:
            problems.append(f"{label} '{body}' is not a URDF link")
    # EE frame: either the preserved fixed TCP link (offset 0, fixed joints kept) or a parent
    # body plus the fixed TCP joint's origin (fixed joints merged).
    into = [j for j in robot["joints"] if j["type"] == "fixed" and j["child"] == spec.ee_body]
    out_of = [j for j in robot["joints"] if j["type"] == "fixed" and j["parent"] == spec.ee_body and j["xyz"]]
    if into:
        if spec.merge_fixed_joints:
            problems.append(f"EE body '{spec.ee_body}' is a fixed-joint child but asset.converter.merge_fixed_joints "
                            "is true: it would be merged away (use the fallback in configs/my_so101.yaml)")
        if any(abs(v) > 1e-9 for v in spec.ee_offset):
            problems.append(f"EE body '{spec.ee_body}' is the TCP frame itself; offset must be [0, 0, 0]")
    elif out_of:
        xyz = [float(v) for v in out_of[0]["xyz"].split()]
        if any(abs(a - b) > 1e-6 for a, b in zip(xyz, spec.ee_offset)):
            problems.append(f"frames.end_effector.offset {list(spec.ee_offset)} != URDF {out_of[0]['name']} xyz {xyz}")
    limits = {j["name"]: j for j in robot["joints"] if j["lower"] is not None}
    for logical, q in spec.home.items():
        j = limits.get(spec.actual(logical))
        if j and not float(j["lower"]) <= q <= float(j["upper"]):
            problems.append(f"home_joint_pos.{logical}={q} outside URDF limits [{j['lower']}, {j['upper']}]")
    for label, q in (("gripper.open", spec.gripper_open), ("gripper.closed", spec.gripper_closed)):
        j = limits.get(spec.gripper_joint_name)
        if j and not float(j["lower"]) <= q <= float(j["upper"]):
            problems.append(f"{label}={q} outside URDF gripper limits [{j['lower']}, {j['upper']}]")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect joints/limits of an SO-101 URDF (stdlib only).")
    parser.add_argument("urdf", nargs="?", help="path to the URDF file (default: pinned my_so101 URDF)")
    parser.add_argument("--check-mapping", action="store_true", help="compare with configs/my_so101.yaml")
    args = parser.parse_args(argv)

    spec = None
    if not args.urdf or args.check_mapping:
        from dume_isaac.assets.my_so101.spec import load_so101_spec  # noqa: PLC0415

        spec = load_so101_spec()
    path = Path(args.urdf).expanduser() if args.urdf else spec.urdf_path
    if not path.is_file():
        print(f"Error: URDF file not found: {path}\n", file=sys.stderr)
        print(USAGE_HINT, file=sys.stderr)
        return 2
    try:
        robot = parse_urdf(path)
    except (ET.ParseError, ValueError) as exc:
        print(f"Error: could not parse URDF {path}: {exc}", file=sys.stderr)
        return 1
    print_report(path, robot)
    if spec is None:
        return 0
    problems = check_mapping(robot, spec)
    print("\n=== Mapping check vs configs/my_so101.yaml ===")
    if problems:
        for msg in problems:
            print(f"  FAIL {msg}")
        print("Fix the `joints` / `frames` / home / gripper values in configs/my_so101.yaml.")
        return 1
    print(f"  PASS joints {list(spec.all_joint_names)}, EE body '{spec.ee_body}', offset {list(spec.ee_offset)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
