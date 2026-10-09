# Real robot spec — Brain Us SO-101 follower

기계 판독용 사본: [robot_spec.yaml](../dume_isaac/assets/my_so101/robot_spec.yaml) (repo 내 canonical 이름: `my_so101`)

## 확인됨

| 항목 | 값 |
|---|---|
| 제품 | Brain Us SO-101 follower |
| 기반 설계 | 표준 TheRobotStudio SO-101 |
| 관절 | arm 5 + gripper 1 (총 6 actuated) |
| Follower servo | Feetech ST-3215-C018, 12V, 1:345 |

## 나중에 실물 확인 / 측정

| 항목 | 상태 | 측정 방법 / 메모 |
|---|---|---|
| calibrated joint min/max | TODO | `physical-ai-dume` calibration 결과에서 각 joint 값 기록 |
| real zero ↔ sim zero 대응 | TODO | 같은 자세에서 real 값과 sim joint 각도 비교, offset/부호 기록 |
| gripper 0~100 ↔ jaw opening | TODO | 여러 명령값에서 jaw 간격(mm) 측정 → sim gripper joint 값 매핑 |
| 실제 servo model label | TODO | 각 servo 라벨 사진 확인 (모든 joint가 C018인지) |
| 링크 구조가 표준 CAD와 같은지 | TODO | 주요 링크 길이 실측 후 표준 URDF와 비교 |
| response speed / overshoot | TODO | step 명령에 대한 joint 응답 기록 → sim actuator gain 튜닝 근거 |

검증 완료 시 `robot_spec.yaml`의 `verification.*` 플래그를 `true`로 바꾸고 근거를 여기 남긴다.

## 시뮬레이션에서 쓰는 값과의 관계

시뮬레이션의 inertia, friction, stiffness, damping, joint zero, joint limit, backlash는
공식 TheRobotStudio URDF/MJCF 값 또는 **provisional simulation value**이며, 위 표의
Brain Us 실물 측정이 끝나기 전까지 실물 검증값이 아니다
([configs/my_so101.yaml](../configs/my_so101.yaml) 참조).
