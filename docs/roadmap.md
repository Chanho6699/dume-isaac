# Roadmap

"Implemented" = 코드가 repo에 있고 집 static/unit test 통과.
"Runtime validated" = 학교 Isaac Sim 5.1.0 / Isaac Lab v2.3.2에서 해당 Stage PASS.
학교에서 PASS한 뒤에만 Runtime validated를 YES로 바꾼다.

| Phase | 내용 | Implemented | Runtime validated (Isaac 5.1.0) |
|---|---|---|---|
| 0 | 학교 환경 진단 | YES | NO |
| 1 | 공식 SO-101 source, URDF → USD, spawn, joint motion | YES | NO |
| 2 | table + cube + target scene | YES | NO |
| 3a | Reach sanity PPO (pipeline 검증) | YES | NO |
| 3b | pick-place full-state PPO teacher | YES | NO |
| 4 | simulated ActorObservation | NO | NO |
| 5 | state-to-vision distillation / vision policy | NO | NO |
| 6 | synthetic demonstrations / optional SmolVLA | NO | NO |
| 7 | domain randomization / sim-to-real | NO | NO |

## Phase 0: 학교 환경 진단
- Stage 0 ([school_runtime_checklist.md](school_runtime_checklist.md)), 결과를 [environment.md](environment.md)에 기록

## Phase 1: SO-101 asset
- 공식 TheRobotStudio source를 commit + sha256으로 pin (`source_manifest.yaml`) → Stage 1
- `script04`로 USD 변환 → Stage 2
- spawn smoke test, joint motion test → Stage 3–4
- 완료 조건: 6 joint가 명령을 따르고, NaN/폭주/base drift 없음

## Phase 2: tabletop scene
- `configs/tabletop.yaml`: 실측 전에는 provisional default 사용, 실측값을 채우면 자동 반영 → Stage 5
- 완료 조건: cube가 table 위에 안정적으로 놓이고 reset 반복 가능

## Phase 3: PPO teacher
- Reach sanity: random-policy gate → PPO train/play → Stage 6–7
- Pick-place: random-policy gate → PPO train/play (easy → medium → wide) → Stage 8–9
- 완료 조건: Reach 수렴, pick-place `Episode_Metric/place_success` 상승

## Phase 4: simulated ActorObservation
- [real_sim_contract.md](real_sim_contract.md)를 따르는 sim 관측 (RGB, depth, pointmap, GT mask, wrist RGB, joint state, goal)

## Phase 5: state-to-vision distillation / vision policy
- Phase 3 teacher → ActorObservation student

## Phase 6: synthetic demonstrations / optional SmolVLA experiments

## Phase 7: domain randomization / sim-to-real
- 실물 검증값([real_robot_spec.md](real_robot_spec.md)) 반영, `physical-ai-dume`에서 실물 평가
