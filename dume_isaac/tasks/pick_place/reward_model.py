"""Pick-place reward math, written once for both torch (runtime) and plain floats (home tests).

Pure Python: no torch import. Runtime mdp terms call these with `TorchOps`
(mdp/state.py); tests call them with `FloatOps` to compute whole-episode returns of
scripted policies (hover, carry, place, ...) and check the anti-hover property on the
accumulated return, not only on per-step weights.

State fields (robot base frame):
    ee_cube_dist     TCP-cube distance
    height           cube center height above its resting height on the table
    target_xy_dist   cube-target xy distance
    in_target        cube center inside the target footprint (minus margin)
    held             grasp proxy: cube off the table AND close to the TCP
    gripper_closing  gripper target is on the closed side
    speed            cube linear speed

Stage structure (anti reward-hacking):
    reach_object  1 - tanh(d / std)                         always
    grasp         [d < grasp_radius] * closing              always
    lift          held * (in_target ? 1 : clip(h / lift_height))
                  -> over the target, lowering the cube does NOT lose lift credit
    carry(_fine)  held * (1 - tanh(xy / std))
    descend       held * in_target * (1 - tanh(h / std))   increases while lowering
    place         placed (resting in target, slow, released, was lifted)
Along the intended path the per-step reward never drops at a stage boundary, and every
"held" state pays strictly less per step than the placed state (tests/test_reward_audit.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

STAGES = ("reach_object", "grasp", "lift", "carry", "carry_fine", "descend")


class FloatOps:
    """Scalar backend for home tests."""

    tanh = staticmethod(math.tanh)

    @staticmethod
    def f(b: Any) -> float:
        return float(b)

    @staticmethod
    def clip(x: float, lo: float | None, hi: float | None) -> float:
        if lo is not None:
            x = max(x, lo)
        if hi is not None:
            x = min(x, hi)
        return x

    @staticmethod
    def and_(a: Any, b: Any) -> bool:
        return bool(a) and bool(b)

    @staticmethod
    def not_(a: Any) -> bool:
        return not bool(a)

    @staticmethod
    def abs(x: float) -> float:
        return abs(x)

    @staticmethod
    def where(c: Any, a: Any, b: Any) -> Any:
        return a if c else b


@dataclass
class StageState:
    ee_cube_dist: Any
    height: Any
    target_xy_dist: Any
    in_target: Any
    held: Any
    gripper_closing: Any
    speed: Any


def stage_rewards(s: StageState, rew: dict, thr: dict, ops=FloatOps) -> dict[str, Any]:
    """Unweighted stage reward values. `rew` = rewards section, `thr` = thresholds of pick_place.yaml."""
    held = ops.f(s.held)
    over = ops.f(ops.and_(s.held, s.in_target))
    away = held - over
    lift_frac = ops.clip(s.height / thr["lift_height"], 0.0, 1.0)
    return {
        "reach_object": 1.0 - ops.tanh(s.ee_cube_dist / rew["reach_object"]["std"]),
        "grasp": ops.f(ops.and_(s.ee_cube_dist < thr["grasp_radius"], s.gripper_closing)),
        "lift": over + away * lift_frac,
        "carry": held * (1.0 - ops.tanh(s.target_xy_dist / rew["carry"]["std"])),
        "carry_fine": held * (1.0 - ops.tanh(s.target_xy_dist / rew["carry_fine"]["std"])),
        "descend": over * (1.0 - ops.tanh(ops.clip(s.height, 0.0, None) / rew["descend"]["std"])),
    }


def placed_now(s: StageState, thr: dict, ops=FloatOps) -> Any:
    """Cube resting on the table inside the target, slow, and (optionally) released."""
    ok = ops.and_(s.in_target, ops.abs(s.height) < thr["place_height_tol"])
    ok = ops.and_(ok, s.speed < thr["place_max_speed"])
    if thr["require_gripper_open"]:
        ok = ops.and_(ok, ops.not_(s.gripper_closing))
    return ok


def place_step(was_lifted: Any, hold_count: Any, s: StageState, thr: dict, ops=FloatOps):
    """One step of the episode-level place tracker.

    Returns (was_lifted, hold_count, placed, success). `placed` pays the place reward;
    `success` (placed for place_hold_steps consecutive steps) is the success metric.
    """
    was_lifted = ops.and_(s.held, s.height > thr["lift_height"]) | was_lifted
    placed = placed_now(s, thr, ops)
    if thr["require_was_lifted"]:
        placed = ops.and_(placed, was_lifted)
    hold_count = ops.where(placed, hold_count + 1, hold_count * 0)
    success = hold_count >= int(thr["place_hold_steps"])
    return was_lifted, hold_count, placed, success


def weighted_step_reward(s: StageState, placed: Any, rew: dict, thr: dict, ops=FloatOps) -> Any:
    """Sum of the positive staged terms with their weights (penalties excluded)."""
    values = stage_rewards(s, rew, thr, ops)
    total = sum(rew[k]["weight"] * v for k, v in values.items())
    return total + rew["place"]["weight"] * ops.f(placed)
