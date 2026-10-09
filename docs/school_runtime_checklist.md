# School runtime checklist (Isaac Sim 5.1.0 / Isaac Lab v2.3.2)

이 문서만 보고 위에서부터 하나씩 진행한다. Stage가 PASS하면
[roadmap.md](roadmap.md)의 "Runtime validated"를 YES로 바꾼다.

준비 (한 번):

```bash
cd ~/Projects/dume-isaac && git pull
export ISAACLAB=<ISAACLAB_PATH>/isaaclab.sh     # 예: ~/IsaacLab/isaaclab.sh
```

Isaac이 필요한 스크립트는 모두 `$ISAACLAB -p scripts/...`로 실행한다. 공통 옵션:
`--headless`(GUI 없이), `--device cuda:0`.

---

## Stage 0 — environment check
- **Command:** `$ISAACLAB -p scripts/script01_check_environment.py --try-import`
- **Expected:** Isaac Sim/Isaac Lab `detected`, torch CUDA available, GPU RTX 5060 Laptop, `python match`
- **PASS condition:** 위 4개 모두 충족, import 에러 없음
- **If failed:** Isaac Lab interpreter로 실행했는지 확인 (`$ISAACLAB -p`). 결과를 그대로 기록
- **Next:** 출력 전체를 [environment.md](environment.md)에 붙여넣고 pending 항목 채우기

## Stage 1 — official source + URDF inspect
- **Command:**
  ```bash
  python scripts/script02_fetch_so101_source.py
  python scripts/script03_inspect_so101_urdf.py
  ```
- **Expected:** 15 files verified/fetched, joint 표 6 revolute + 1 fixed, mapping check PASS
- **PASS condition:** 두 스크립트 모두 exit 0 (`PASS` 출력)
- **If failed:** 네트워크(raw.githubusercontent.com) 확인. sha256 mismatch면 manifest와 upstream commit 확인 (임의 파일로 대체 금지)
- **Next:** Stage 2

## Stage 2 — URDF → USD
- **Command:** `$ISAACLAB -p scripts/script04_convert_so101_to_usd.py --headless`
- **Expected:** `[convert] PASS  USD written: .../dume_isaac/assets/my_so101/usd/my_so101.usd`
- **PASS condition:** exit 0, USD 파일 존재
- **If failed:** 에러 메시지의 Checks 확인. 오래 걸리면 convex decomposition 때문 (수 분 가능).
  재시도: `--force`, 실패 지속 시 `--collider_type convex_hull`로 우선 진행하고 기록.
  변환은 `merge_fixed_joints=false`로 `gripper_frame_link`(TCP, upstream에서 massless)를 보존한다.
  이 link 때문에 실패하면 fallback: `configs/my_so101.yaml`에서 `merge_fixed_joints: true` +
  `frames.end_effector` 를 주석의 `gripper_link` + offset 으로 바꾸고 `--force` 재변환
- **Next:** Stage 3

## Stage 3 — robot spawn (smoke test)
- **Command:** `$ISAACLAB -p scripts/script05_smoke_test_so101.py` (GUI) / `--headless`
- **Expected:** bodies/joints 출력 (`gripper_frame_link` 포함), 6 joints, results 전부 PASS
  (`EE frame`: |gripper_frame_link - gripper_link| ≈ 0.0985 m)
- **PASS condition:** `SMOKE TEST: PASS` + GUI에서 로봇이 바닥 위에 똑바로 서 있고 튀지 않음
- **If failed:**
  - mapping FAIL → 출력된 USD joint/body 이름으로 `configs/my_so101.yaml`의 `joints`/`frames`만 수정
  - hold home WARN/FAIL → actuator stiffness 부족 가능, `actuators.*.stiffness` 조정 후 기록
  - base drift FAIL → 변환 시 `fix_base` 확인, Stage 2 `--force`
  - EE frame FAIL 또는 `gripper_frame_link` 없음 / zero mass·inertia 경고 → Stage 2의 fallback 적용
- **Next:** Stage 4

## Stage 4 — joint motion
- **Command:** `$ISAACLAB -p scripts/script06_test_joint_motion.py` (GUI 권장)
- **Expected:** 각 joint 한 개씩 +0.25 rad 움직였다 복귀, gripper open/close, 표에 PASS
- **PASS condition:** `JOINT MOTION: PASS` (WARN은 기록 후 진행 가능) + GUI에서 "gripper open"이 실제로 벌어짐
- **If failed:** 특정 joint FAIL → 해당 actuator group stiffness/effort 확인.
  open/close가 반대로 보이면 `configs/my_so101.yaml`의 `gripper.open`/`closed` 교체
- **Next:** Stage 5

## Stage 5 — tabletop scene
- **Command:** `$ISAACLAB -p scripts/script07_run_tabletop_scene.py --num_envs 4`
- **Expected:** layout 값이 provisional로 표시, 로봇이 table 위, 파란 cube와 초록 target pad가 로봇 앞
  (target pad는 visual marker일 뿐 물리 객체가 아님: cube가 통과하거나 위에 놓여도 정상)
- **PASS condition:** `TABLETOP SCENE: PASS` + 시각 확인
- **If failed:** cube 관통/낙하 → table/cube 크기, 로봇이 table에 묻히거나 뜸 → `provisional_defaults.robot_base.position` z 조정
- **Next:** Stage 6

## Stage 6 — Reach random env sanity
- **Command:** `$ISAACLAB -p scripts/script08_random_policy.py --task reach --headless`
- **Expected:** obs shape 출력, reward 유한, episodes ended > 0 (timeout)
- **PASS condition:** `RANDOM POLICY GATE (reach): PASS`
- **If failed:** NaN → 해당 obs/reward term 확인. 생성 단계 에러 → 에러 메시지의 term 이름/파라미터 확인 (v2.3.2 API 차이 가능)
- **Next:** Stage 7

## Stage 7 — Reach PPO train / play
- **Command:**
  ```bash
  # 1) first PPO sanity run (pipeline only)
  $ISAACLAB -p scripts/script09_train_reach_ppo.py --headless --num_envs 64 --max_iterations 50 --run_name sanity64
  $ISAACLAB -p scripts/script10_play_reach_ppo.py --num_envs 16 --load_run sanity64
  # 2) after 1) passes: scale up step by step (see "Scaling num_envs" below), then the full run
  $ISAACLAB -p scripts/script09_train_reach_ppo.py --headless --num_envs <largest stable> --run_name full
  ```
- **Expected:** 1) 50 iteration 완료, `logs/rsl_rl/so101_reach/<run>_sanity64/model_*.pt` 생성, play 실행됨.
  full run: mean reward 상승, play에서 TCP가 목표 marker 쪽으로 이동
- **PASS condition:** sanity run이 에러 없이 끝나고 checkpoint로 play 가능 (수렴은 full run에서 판단)
- **If failed:** sanity run 실패 → env/agent cfg 에러 메시지 확인 (v2.3.2 API 차이). OOM → scale-up 중단 지점 기록.
  수렴 안 함 → reward 로그 확인, `configs/reach.yaml` 조정
- **Next:** Stage 8

## Stage 8 — Pick-place random env sanity
- **Command:** `$ISAACLAB -p scripts/script08_random_policy.py --task pick_place --headless`
- **Expected:** termination terms에 time_out / object_out_of_bounds, cube max speed < 10 m/s, place success ~0
- **PASS condition:** `RANDOM POLICY GATE (pick_place): PASS`
- **If failed:** cube 폭주/NaN → cube mass/friction, sim dt 확인. reset 이상 → `reset_object_and_target` 범위 확인
- **Next:** Stage 9

## Stage 9 — Pick-place PPO train / play
- **Command:**
  ```bash
  # 1) first PPO sanity run (pipeline only)
  $ISAACLAB -p scripts/script11_train_pick_place_ppo.py --headless --num_envs 64 --max_iterations 50 --difficulty easy --run_name sanity64
  $ISAACLAB -p scripts/script12_play_pick_place_ppo.py --num_envs 16 --difficulty easy --load_run sanity64
  # 2) after 1) passes: scale up step by step (see "Scaling num_envs" below), then the full run
  $ISAACLAB -p scripts/script11_train_pick_place_ppo.py --headless --num_envs <largest stable> --difficulty easy --run_name full
  $ISAACLAB -p scripts/script12_play_pick_place_ppo.py --num_envs 16 --difficulty easy
  ```
- **Expected:** reward 단계별 상승 (reach_object → grasp → lift → carry → place), `Episode_Metric/place_success` 상승
- **PASS condition:** play에서 place success가 0보다 의미 있게 큼 (목표치는 첫 결과를 보고 정함)
- **If failed:** grasp 안 됨 → gripper open/closed 값, cube friction/size, collider(convex decomposition) 확인.
  공중 hover → `place` weight가 held 합보다 큰지 확인. 학습 tensorboard: `logs/rsl_rl/so101_pick_place`
- **Next:** easy 성공 → `--difficulty medium` → `wide`, `--resume`으로 이어서 학습 가능

---

## Scaling num_envs (after the first 64-env sanity run passes)

Baseline cfg defaults stay 2048 (reach) / 1024 (pick-place), but are reached only step by step.
Each step: short run, watch `nvidia-smi` in another terminal, record, then go to the next size.

```bash
# reach: 64 -> 256 -> 512 -> 1024 -> 2048
$ISAACLAB -p scripts/script09_train_reach_ppo.py --headless --num_envs 256 --max_iterations 20 --run_name scale256
# pick_place: 64 -> 256 -> 512 -> 1024
$ISAACLAB -p scripts/script11_train_pick_place_ppo.py --headless --num_envs 256 --max_iterations 20 --difficulty easy --run_name scale256
```

- Go up only if the step finished without OOM / PhysX buffer errors and GPU memory stays below ~85%.
- Stop at the first failing size; the full run uses the last size that passed.
- PhysX "gpu ... capacity" errors at large sizes: record the message (buffer sizes live in
  `dume_isaac/envs/tabletop/env_cfg.py`).

| Task | num_envs | iter time / FPS | GPU mem | Result |
|---|---|---|---|---|
| reach | 64 | | | |
| reach | 256 | | | |
| reach | 512 | | | |
| reach | 1024 | | | |
| reach | 2048 | | | |
| pick_place | 64 | | | |
| pick_place | 256 | | | |
| pick_place | 512 | | | |
| pick_place | 1024 | | | |

---

## 실패 시 기록 양식

| Stage | 날짜 | 결과 | 에러/증상 | 수정한 것 |
|---|---|---|---|---|
| | | | | |
