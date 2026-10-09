# brainus_so101 asset

Brain Us SO-101 follower의 시뮬레이션 asset 자리.

- [robot_spec.yaml](robot_spec.yaml): 확인된 실물 사실만 기록. 모르는 기하/동역학 값은 넣지 않는다.
- 시뮬레이션 설정 placeholder: [configs/brainus_so101.yaml](../../../configs/brainus_so101.yaml)

## Asset 방식 결정 (Phase 1, 학교에서)

1. 설치된 Isaac Lab에 SO-101 config(`SO101_CFG` 등)가 있는지 확인
   (`scripts/check_environment.py`의 SO-101 asset search). **있다고 가정하지 않는다.**
2. 없으면 표준 TheRobotStudio SO-101 URDF를 공식 repo에서 직접 받아 USD로 import.
   - 출처 URL + commit hash를 여기에 기록
   - `scripts/inspect_so101_urdf.py`로 joint 이름/limit 확인
   - 변환 중간 산출물은 `_generated/` 아래 (gitignore됨)
   - 작고 검증된 최종 USD만 명시적으로 커밋

## 기록 (학교에서 채우기)

| 항목 | 값 |
|---|---|
| 방식 (builtin cfg / URDF import) | TODO |
| URDF 출처 URL | TODO |
| URDF commit | TODO |
| USD 경로 | TODO |
| 변환 도구 / 버전 | TODO |
