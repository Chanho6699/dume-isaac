# dume-isaac

Project-Dum-E를 위한 **Isaac Sim / Isaac Lab 전용 시뮬레이션·학습 실험 repo**.

| Repo | 역할 |
|---|---|
| `physical-ai-dume` | 실제 로봇 시스템 (real robot, R3D/wrist 카메라, SAM2 VisualCue, 실제 제어) |
| `dume-isaac` (이 repo) | 시뮬레이션 / 학습 실험 / 강화 학습 / 환경 다양화 / 조건별 실험 |

두 repo는 코드 의존성 없이 분리되어 있다. 관측/행동의 의미 대응은
[docs/real_sim_contract.md](docs/real_sim_contract.md) 문서 계약으로만 맞춘다.

**Compatibility target (확정):** Isaac Sim 5.1.0 / Isaac Lab v2.3.2 / Python 3.11 / RTX 5060 Laptop.
Isaac Sim/Lab은 pip dependency가 아니며 학교에 설치된 Isaac Lab을 쓴다.

## 목표

1. SO-101 tabletop simulation
2. pick-and-place PPO teacher (full-state)
3. simulated ActorObservation
4. vision policy / distillation
5. synthetic demonstrations
6. sim-to-real evaluation

## 현재 상태

"Implemented" = 코드 있음 + 집 static/unit test 통과. "Runtime validated" = 학교 Isaac에서 해당 Stage PASS.

| 단계 | Implemented locally | Runtime validated on Isaac 5.1.0 |
|---|---|---|
| Stage 0 environment check | YES | NO |
| Stage 1 official source fetch + URDF inspect | YES | N/A (Isaac 불필요) — 집에서 실행 PASS |
| Stage 2 URDF → USD | YES | NO |
| Stage 3 robot spawn smoke test | YES | NO |
| Stage 4 joint motion test | YES | NO |
| Stage 5 tabletop scene | YES | NO |
| Stage 6 Reach random-policy gate | YES | NO |
| Stage 7 Reach PPO train/play | YES | NO |
| Stage 8 Pick-place random-policy gate | YES | NO |
| Stage 9 Pick-place PPO train/play | YES | NO |
| Vision ActorObservation / distillation / SmolVLA | NO | NO |

로드맵: [docs/roadmap.md](docs/roadmap.md)

## SO-101 asset: `my_so101`

- Isaac Lab v2.3.2에는 `SO101_CFG`가 없으므로 이 repo가 `MY_SO101_CFG`를 직접 관리한다.
- **Source 방식: pinned fetch.** 공식 `TheRobotStudio/SO-ARM100`의 commit 하나에 고정하고
  파일별 sha256을 [source_manifest.yaml](dume_isaac/assets/my_so101/source_manifest.yaml)에 기록.
  `script02`가 그 파일만 받아 검증한다 (upstream과 byte-identical, git에는 manifest만 커밋).
- joint/frame 이름은 [configs/my_so101.yaml](configs/my_so101.yaml) 한 곳에서만 매핑한다.
- actuator/home/gripper 값은 **provisional simulation value**이며 Brain Us 실물 검증값이 아니다.

## Scripts (실행 순서)

| # | Script | Stage | 내용 | Isaac |
|---|---|---|---|---|
| 01 | `script01_check_environment.py` | 0 | Isaac Sim/Lab, Python, GPU 등 실행 환경 확인 | 선택 |
| 02 | `script02_fetch_so101_source.py` | 1 | 공식 SO-101 URDF/mesh를 pinned commit에서 다운로드·검증 | no |
| 03 | `script03_inspect_so101_urdf.py` | 1 | URDF의 joint/frame/limit과 `my_so101` 매핑 검사 | no |
| 04 | `script04_convert_so101_to_usd.py` | 2 | SO-101 URDF를 Isaac용 USD로 변환 | yes |
| 05 | `script05_smoke_test_so101.py` | 3 | SO-101 spawn 및 articulation/frame/물리 기본 상태 확인 | yes |
| 06 | `script06_test_joint_motion.py` | 4 | 5개 arm joint + gripper를 하나씩 움직여 동작 검증 | yes |
| 07 | `script07_run_tabletop_scene.py` | 5 | table + SO-101 + cube + target 작업환경 실행 | yes |
| 08 | `script08_random_policy.py --task reach` / `--task pick_place` | 6 / 8 | PPO 전 reset/action/reward/termination 환경 sanity test | yes |
| 09 | `script09_train_reach_ppo.py` | 7 | 목표 위치까지 EE를 이동시키는 Reach PPO 학습 | yes |
| 10 | `script10_play_reach_ppo.py` | 7 | 학습된 Reach PPO checkpoint 실행·확인 | yes |
| 11 | `script11_train_pick_place_ppo.py` | 9 | cube를 target에 놓는 Pick-and-Place PPO teacher 학습 | yes |
| 12 | `script12_play_pick_place_ppo.py` | 9 | 학습된 Pick-and-Place PPO checkpoint 실행·평가 | yes |

학교 실행 절차와 단계별 PASS 기준: **[docs/school_runtime_checklist.md](docs/school_runtime_checklist.md)**

```bash
cd ~/Projects/dume-isaac && git pull
export ISAACLAB=<ISAACLAB_PATH>/isaaclab.sh

# Stage 0: 학교 Isaac 환경 확인
$ISAACLAB -p scripts/script01_check_environment.py --try-import

# Stage 1: 공식 SO-101 source 준비 및 URDF 검증
python scripts/script02_fetch_so101_source.py
python scripts/script03_inspect_so101_urdf.py

# Stage 2: URDF → USD 변환
$ISAACLAB -p scripts/script04_convert_so101_to_usd.py --headless

# Stage 3~9: robot → joints → tabletop → RL sanity → PPO
# 자세한 명령과 PASS 기준은 docs/school_runtime_checklist.md 참고
```

## 테스트 종류

| 종류 | 어디서 | 명령 | 범위 |
|---|---|---|---|
| A. Home static/unit | 집 (Isaac 없음) | `python3 -m pytest -q` | 문법, Isaac-free import, YAML/manifest/mapping, parser, checkpoint 해석, 문서 필수 항목, Isaac 없을 때 graceful exit |
| B. Isaac runtime | 학교만 | `scripts/script04`–`script12` | 실제 USD 변환, 물리, env 생성, PPO |

A가 통과해도 B는 검증된 것이 아니다. Isaac 코드를 mock으로 PASS시키지 않는다.

## 구조

```
configs/            my_so101.yaml (매핑/actuator), tabletop.yaml (실측 null + provisional),
                    reach.yaml, pick_place.yaml (reward/threshold/difficulty)
dume_isaac/
  assets/my_so101/  spec.py (pure), robot_cfg.py (MY_SO101_CFG), source_manifest.yaml, urdf/, usd/
  envs/tabletop/    layout.py (pure), scene_cfg.py, env_cfg.py, frames.py
  tasks/            registry.py (pure), reach/, pick_place/ (env cfg, mdp/, agents/rsl_rl_ppo_cfg.py)
  runtime/          app.py (AppLauncher 경계), rsl_rl_runner.py, checkpoints.py, single_robot.py
  actor/            (Phase 4, 비어 있음)
scripts/            script01 … script12
docs/               environment, roadmap, real_robot_spec, real_sim_contract, school_runtime_checklist
tests/              home static/unit tests
```

Isaac에 의존하는 모듈은 패키지 `__init__`에서 import하지 않는다. 스크립트가
`dume_isaac.runtime.app.launch_app()`으로 앱을 띄운 뒤에만 import된다.
