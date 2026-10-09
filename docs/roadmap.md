# Roadmap

각 phase는 이전 phase의 완료 조건을 만족한 뒤 시작한다. Isaac 버전에 묶이는 코드는
Phase 0에서 버전이 확정된 뒤에만 작성한다.

## Phase 0: 학교 환경 진단
- `scripts/check_environment.py` 실행, 결과를 [environment.md](environment.md)에 기록
- Isaac Sim / Isaac Lab / Python / PyTorch / CUDA / driver 정확한 버전 확정
- Isaac Lab 공식 tutorial(빈 scene 실행 등) 1개가 도는지 확인
- **완료 조건**: environment.md 표가 실제 값으로 채워짐

## Phase 1: 공식/표준 SO-101 asset 준비 및 joint smoke test
- 설치된 Isaac Lab에 SO-101 config가 있는지 확인 (있다고 가정하지 않음)
- 없으면 표준 TheRobotStudio SO-101 URDF를 공식 repo에서 받아 USD로 import (출처/커밋 기록)
- `scripts/inspect_so101_urdf.py`로 joint 이름/limit 확인 → `configs/brainus_so101.yaml` 채우기
- smoke test: 로봇 spawn, 각 joint를 limit 내에서 하나씩 sweep, gripper open/close
- **완료 조건**: 6개 actuated joint가 의도한 방향으로 움직이고 폭주/관통 없음

## Phase 2: table + cube + bin scene
- 실측 후 `configs/tabletop.yaml` 채우기 (table, base pose, workspace, cube, bin, camera)
- fixed-base tabletop scene 구성, 외부 카메라 + wrist 카메라 배치
- **완료 조건**: scene reset 반복 시 물체/로봇이 안정적으로 초기화

## Phase 3: full-state PPO teacher
- privileged full state (joint state, object/target pose) 기반 pick-and-place
- reward / termination / reset 설계, 병렬 env 학습
- **완료 조건**: 고정 seed 평가에서 목표 성공률 달성 (목표치는 이 단계에서 정함)

## Phase 4: simulated ActorObservation
- [real_sim_contract.md](real_sim_contract.md)를 따르는 sim 관측 생성
  (RGB, depth, camera-centric pointmap, GT segmentation, wrist RGB, joint state, task goal)
- **완료 조건**: 규약(shape/dtype/단위/좌표계) 검사 테스트 통과

## Phase 5: state-to-vision distillation / vision policy
- Phase 3 teacher → ActorObservation 기반 student distillation
- **완료 조건**: student가 sim에서 teacher 대비 허용 범위 내 성능

## Phase 6: synthetic demonstrations / optional SmolVLA experiments
- teacher/student rollout으로 synthetic demonstration dataset 생성
- (선택) SmolVLA 등 VLA 학습 실험
- **완료 조건**: real 데이터와 동일 포맷의 synthetic dataset

## Phase 7: domain randomization / sim-to-real
- 조명, 텍스처, 카메라 pose, 물체 크기/질량, actuator gain randomization
- real robot spec 검증값([real_robot_spec.md](real_robot_spec.md)) 반영
- `physical-ai-dume`에서 실물 평가
- **완료 조건**: 실물 성공률 측정 및 sim 성능과의 gap 기록
