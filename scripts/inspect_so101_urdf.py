#!/usr/bin/env python3
"""Inspect an SO-101 URDF: robot name, joints, types, parent/child, limits.

Python stdlib only (no Isaac dependency). Read-only.

Usage:
    python scripts/inspect_so101_urdf.py path/to/so101.urdf
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

USAGE_HINT = """\
Usage:
    python scripts/inspect_so101_urdf.py <path/to/so101.urdf>

No URDF is committed to this repo. Get the standard SO-101 URDF from the official
TheRobotStudio SO-ARM100 repository yourself, record its source URL and commit in
dume_isaac/assets/brainus_so101/README.md, then pass its path here.
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
        joints.append({
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

    print("\nNon-fixed joint names (copy to configs/brainus_so101.yaml after review):")
    for j in movable:
        print(f"  - {j['name']}")
    print("\nNote: URDF limits are CAD/model values, not the calibrated limits of the real robot.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect joints/limits of an SO-101 URDF (stdlib only).")
    parser.add_argument("urdf", nargs="?", help="path to the URDF file")
    args = parser.parse_args(argv)

    if not args.urdf:
        print(USAGE_HINT)
        return 2
    path = Path(args.urdf).expanduser()
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
