"""A. Home reward audit for pick-place: episode returns of scripted policies.

Uses the SAME reward math as the Isaac runtime (dume_isaac/tasks/pick_place/reward_model.py,
evaluated with FloatOps instead of TorchOps) and the real weights/thresholds/episode length
from configs/pick_place.yaml. Checks accumulated returns (undiscounted and with the PPO
gamma), not just per-step weights:

- hovering / carrying / holding for the whole remaining episode never beats placing
- placing still wins when started late (bounded number of remaining steps)
- the intended path (reach -> grasp -> lift -> carry -> descend) never loses per-step reward
- pushing the cube onto the target without lifting earns no place reward
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from dume_isaac.envs.tabletop.layout import load_tabletop_layout  # noqa: E402
from dume_isaac.tasks.pick_place.reward_model import StageState, place_step, weighted_step_reward  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = yaml.safe_load((ROOT / "configs/pick_place.yaml").read_text())
REW, THR = P["rewards"], P["thresholds"]
T = round(P["episode_length_s"] / (P["sim"]["dt"] * P["sim"]["decimation"]))
GAMMA = float(re.search(r"gamma=([0-9.]+)",
                        (ROOT / "dume_isaac/tasks/pick_place/agents/rsl_rl_ppo_cfg.py").read_text()).group(1))
LAYOUT = load_tabletop_layout()
HALF = min(LAYOUT.target_size) / 2 - THR["place_xy_margin"]  # in_target if xy offset <= HALF (one axis)

LIFT_H = THR["lift_height"] + 0.01      # carry height: above full-lift height
LOW_H = THR["held_min_height"] + 0.002  # lowest still-held height
GRIP_D = 0.005                          # TCP-cube distance while grasped
START_XY = 0.12                         # cube-target distance at reset
RELEASE_SPEEDS = (0.30, 0.15, 0.08, 0.03, 0.0)  # cube settling after opening the gripper


def state(d, h, xy, closing, speed=0.0) -> StageState:
    held = h > THR["held_min_height"] and d < THR["hold_radius"]
    return StageState(ee_cube_dist=d, height=h, target_xy_dist=xy, in_target=xy <= HALF,
                      held=held, gripper_closing=closing, speed=speed)


def lerp(a, b, n):
    return [a + (b - a) * (i + 1) / n for i in range(n)]


def approach_to_over_target() -> list[StageState]:
    """Intended path up to holding the cube low over the target center (gripper closed)."""
    traj = [state(d, 0.0, START_XY, False) for d in lerp(0.15, GRIP_D, 60)]        # reach
    traj += [state(GRIP_D, 0.0, START_XY, True)] * 10                              # close
    traj += [state(GRIP_D, h, START_XY, True) for h in lerp(0.0, LIFT_H, 20)]       # lift
    traj += [state(GRIP_D, LIFT_H, xy, True) for xy in lerp(START_XY, 0.0, 40)]     # carry
    traj += [state(GRIP_D, h, 0.0, True) for h in lerp(LIFT_H, LOW_H, 20)]          # descend
    return traj


def release_and_rest(n: int) -> list[StageState]:
    traj = [state(GRIP_D, 0.0, 0.0, False, v) for v in RELEASE_SPEEDS]
    return (traj + [state(GRIP_D, 0.0, 0.0, False)] * n)[:n]


def episode_return(traj: list[StageState], gamma: float = 1.0) -> tuple[float, bool]:
    assert len(traj) == T
    was_lifted, hold, ret, success = False, 0, 0.0, False
    for t, s in enumerate(traj):
        was_lifted, hold, placed, ok = place_step(was_lifted, hold, s, THR)
        success |= ok
        ret += gamma ** t * weighted_step_reward(s, placed, REW, THR)
    return ret, success


def hold_rest(s: StageState, prefix: list[StageState]) -> list[StageState]:
    return prefix + [s] * (T - len(prefix))


BASE = approach_to_over_target()
T0 = len(BASE)
POLICIES = {
    "hover_low_over_target": hold_rest(state(GRIP_D, LOW_H, 0.0, True), BASE),
    "hover_high_over_target": hold_rest(state(GRIP_D, LIFT_H, 0.0, True), BASE[:-20]),
    "carry_forever_off_target": hold_rest(state(GRIP_D, LIFT_H, START_XY, True), BASE[:90]),
    "set_down_gripper_closed": hold_rest(state(GRIP_D, 0.0, 0.0, True), BASE),
    "push_without_lift": (BASE[:70] + [state(GRIP_D, 0.0, xy, False) for xy in lerp(START_XY, 0.0, 60)]
                          + [state(GRIP_D, 0.0, 0.0, False)] * (T - 130)),
}
PLACE = BASE + release_and_rest(T - T0)


def test_episode_length_matches_config():
    assert T == 400 and T0 < T // 2


@pytest.mark.parametrize("gamma", [1.0, GAMMA])
@pytest.mark.parametrize("name", list(POLICIES))
def test_place_return_beats_non_place_policies(name, gamma):
    place, place_ok = episode_return(PLACE, gamma)
    other, other_ok = episode_return(POLICIES[name], gamma)
    assert place_ok and not other_ok
    assert place > other, f"{name}: return {other:.1f} >= place {place:.1f} (gamma {gamma})"


def test_place_margin_is_large_undiscounted():
    place, _ = episode_return(PLACE)
    best_other = max(episode_return(p)[0] for p in POLICIES.values())
    assert place > 1.25 * best_other, (place, best_other)


def test_push_without_lift_never_pays_place():
    was_lifted, hold = False, 0
    for s in POLICIES["push_without_lift"]:
        was_lifted, hold, placed, _ = place_step(was_lifted, hold, s, THR)
        assert not placed


@pytest.mark.parametrize("gamma", [1.0, GAMMA])
def test_late_placement_still_beats_hovering(gamma):
    """Hover low over the target until step t, then release. Placing must win whenever
    at least MAX_REMAINING steps (0.4 s) are left."""
    max_remaining = 20
    hover = POLICIES["hover_low_over_target"]
    hover_ret, _ = episode_return(hover, gamma)
    for t in range(T0, T - max_remaining + 1):
        traj = hover[:t] + release_and_rest(T - t)
        ret, ok = episode_return(traj, gamma)
        assert ok and ret > hover_ret, f"release at {t} ({T - t} steps left): {ret:.1f} <= hover {hover_ret:.1f}"


def test_per_step_held_sup_below_placed_min():
    """No held state (any height/distance/gripper) pays as much per step as any placed state."""
    held_sup = max(
        weighted_step_reward(state(d, h, xy, c), False, REW, THR)
        for d in (0.0, 0.005, 0.01, 0.02, 0.03, THR["hold_radius"] - 1e-4)
        for h in (THR["held_min_height"] + 1e-4, 0.015, 0.02, 0.03, 0.04, 0.06, 0.1)
        for xy in (0.0, 0.01, 0.02, HALF, HALF + 0.005, 0.05, 0.1, 0.2)
        for c in (True, False)
    )
    placed_min = min(weighted_step_reward(state(d, 0.0, xy, False), True, REW, THR)
                     for d in (0.0, 0.03, 0.1, 0.5) for xy in (0.0, HALF))
    assert held_sup < placed_min, (held_sup, placed_min)


def test_intended_path_reward_never_drops_before_release():
    rewards = [weighted_step_reward(s, False, REW, THR) for s in BASE]
    drops = [(t, rewards[t - 1], rewards[t]) for t in range(1, len(rewards)) if rewards[t] < rewards[t - 1] - 1e-9]
    assert not drops, drops[:5]
    placed_reward = weighted_step_reward(state(GRIP_D, 0.0, 0.0, False), True, REW, THR)
    assert placed_reward > max(rewards)
