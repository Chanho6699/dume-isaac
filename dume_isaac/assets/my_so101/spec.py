"""Canonical SO-101 names/parameters, loaded from configs/my_so101.yaml.

Pure Python (no Isaac imports). Every task, script and robot config resolves joint and
frame names through this module, so a naming difference in the converted USD is fixed
in exactly one place: the `joints` / `frames` sections of configs/my_so101.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dume_isaac.config_io import load_yaml
from dume_isaac.paths import CONFIG_DIR, repo_path

LOGICAL_ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
LOGICAL_GRIPPER_JOINT = "gripper"
LOGICAL_JOINTS = LOGICAL_ARM_JOINTS + (LOGICAL_GRIPPER_JOINT,)


@dataclass(frozen=True)
class ActuatorGroup:
    name: str
    joints: tuple[str, ...]  # logical names
    stiffness: float
    damping: float
    armature: float | None = None
    effort_limit_sim: float | None = None
    velocity_limit_sim: float | None = None


@dataclass(frozen=True)
class So101Spec:
    joint_map: dict[str, str]          # logical -> actual
    arm_joints: tuple[str, ...]        # logical, action order
    gripper_joint: str                 # logical
    base_body: str
    ee_body: str
    ee_offset: tuple[float, float, float]
    home: dict[str, float]             # logical -> rad
    gripper_open: float
    gripper_closed: float
    actuator_groups: tuple[ActuatorGroup, ...]
    urdf_path: Path
    usd_path: Path
    source_manifest_path: Path
    converter: dict[str, Any] = field(default_factory=dict)
    smoke_test: dict[str, Any] = field(default_factory=dict)
    joint_motion_test: dict[str, Any] = field(default_factory=dict)

    def actual(self, logical: str) -> str:
        try:
            return self.joint_map[logical]
        except KeyError as exc:
            raise KeyError(f"unknown logical joint '{logical}'; known: {sorted(self.joint_map)}") from exc

    @property
    def arm_joint_names(self) -> tuple[str, ...]:
        return tuple(self.actual(j) for j in self.arm_joints)

    @property
    def gripper_joint_name(self) -> str:
        return self.actual(self.gripper_joint)

    @property
    def all_joint_names(self) -> tuple[str, ...]:
        return self.arm_joint_names + (self.gripper_joint_name,)

    def home_joint_pos_actual(self) -> dict[str, float]:
        return {self.actual(k): float(v) for k, v in self.home.items()}

    @property
    def merge_fixed_joints(self) -> bool:
        return bool(self.converter.get("merge_fixed_joints", False))


def urdf_fixed_joint_into(urdf_path: str | Path, child_link: str) -> tuple[str, str, tuple[float, float, float]] | None:
    """(joint name, parent link, origin xyz) of the fixed joint whose child is `child_link`, else None."""
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    for j in ET.parse(urdf_path).getroot().findall("joint"):
        child, parent = j.find("child"), j.find("parent")
        if j.get("type") == "fixed" and child is not None and child.get("link") == child_link:
            origin = j.find("origin")
            xyz = tuple(float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split())
            return j.get("name", "?"), parent.get("link") if parent is not None else "?", xyz  # type: ignore[return-value]
    return None


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"configs/my_so101.yaml: {msg}")


def load_so101_spec(path: str | Path | None = None) -> So101Spec:
    data = load_yaml(path or CONFIG_DIR / "my_so101.yaml")

    joint_map = {str(k): str(v) for k, v in (data.get("joints") or {}).items()}
    _require(set(joint_map) == set(LOGICAL_JOINTS),
             f"`joints` must map exactly {list(LOGICAL_JOINTS)}, got {sorted(joint_map)}")
    _require(len(set(joint_map.values())) == len(joint_map), "`joints` maps two logical joints to one name")

    arm = tuple(data.get("arm_joints") or ())
    _require(sorted(arm) == sorted(LOGICAL_ARM_JOINTS), f"`arm_joints` must be {list(LOGICAL_ARM_JOINTS)}")
    gripper = data.get("gripper_joint")
    _require(gripper == LOGICAL_GRIPPER_JOINT, "`gripper_joint` must be 'gripper'")

    frames = data.get("frames") or {}
    ee = frames.get("end_effector") or {}
    offset = tuple(float(v) for v in (ee.get("offset") or ()))
    _require(len(offset) == 3, "`frames.end_effector.offset` must be [x, y, z]")
    _require(bool(ee.get("body")) and bool(frames.get("base")), "`frames.base` and `frames.end_effector.body` are required")

    home = {str(k): float(v) for k, v in (data.get("home_joint_pos") or {}).items()}
    _require(set(home) == set(LOGICAL_JOINTS), "`home_joint_pos` must list all six logical joints")

    grip = data.get("gripper") or {}
    _require("open" in grip and "closed" in grip, "`gripper.open` and `gripper.closed` are required")
    _require(float(grip["open"]) != float(grip["closed"]), "`gripper.open` must differ from `gripper.closed`")

    groups = []
    covered: list[str] = []
    for name, g in (data.get("actuators") or {}).items():
        if not isinstance(g, dict):
            continue  # e.g. `status: ...`
        joints = tuple(g.get("joints") or ())
        _require(all(j in LOGICAL_JOINTS for j in joints), f"actuator group '{name}' has unknown joints {joints}")
        covered += joints
        groups.append(ActuatorGroup(
            name=name, joints=joints, stiffness=float(g["stiffness"]), damping=float(g["damping"]),
            armature=g.get("armature"), effort_limit_sim=g.get("effort_limit_sim"),
            velocity_limit_sim=g.get("velocity_limit_sim"),
        ))
    _require(sorted(covered) == sorted(LOGICAL_JOINTS), "actuator groups must cover every joint exactly once")

    asset = data.get("asset") or {}
    for key in ("urdf_path", "usd_path", "source_manifest"):
        _require(bool(asset.get(key)), f"`asset.{key}` is required")

    return So101Spec(
        joint_map=joint_map,
        arm_joints=arm,
        gripper_joint=gripper,
        base_body=str(frames["base"]),
        ee_body=str(ee["body"]),
        ee_offset=offset,  # type: ignore[arg-type]
        home=home,
        gripper_open=float(grip["open"]),
        gripper_closed=float(grip["closed"]),
        actuator_groups=tuple(groups),
        urdf_path=repo_path(asset["urdf_path"]),
        usd_path=repo_path(asset["usd_path"]),
        source_manifest_path=repo_path(asset["source_manifest"]),
        converter=dict(asset.get("converter") or {}),
        smoke_test=dict(data.get("smoke_test") or {}),
        joint_motion_test=dict(data.get("joint_motion_test") or {}),
    )
