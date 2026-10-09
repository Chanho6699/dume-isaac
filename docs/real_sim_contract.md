# Real ↔ Sim observation contract

`physical-ai-dume`(real)와 `dume-isaac`(sim) 사이의 **의미 대응 계약**.
현재는 문서 계약만 존재한다. `physical-ai-dume` 코드를 복사하거나 dependency로 연결하지 않는다.
양쪽 구현이 이 문서를 각각 따르도록 한다.

## 관측 대응

| # | Real (`physical-ai-dume`) | Sim (`dume-isaac`) | 비고 |
|---|---|---|---|
| 1 | R3D RGB | external camera RGB | 같은 해상도 / intrinsics 목표 |
| 2 | R3D depth | simulated depth | meters |
| 3 | camera-centric pointmap | 같은 의미의 camera-centric pointmap | 아래 규약 |
| 4 | object SAM2 VisualCue | GT object segmentation | sim은 GT mask를 같은 형식으로 |
| 5 | target SAM2 VisualCue | GT target segmentation | 〃 |
| 6 | Wrist RGB | simulated wrist RGB | |
| 7 | robot state | simulated joint state | joint 순서/단위/zero는 calibration 후 확정 |
| 8 | language / task goal | 같은 language / task goal | 동일 문자열 체계 |

## Pointmap 규약

| 속성 | 값 |
|---|---|
| shape | `H x W x 3` |
| dtype | `float32` |
| 단위 | meters |
| 원점 | camera optical origin |
| X | right |
| Y | down |
| Z | forward |

즉 OpenCV camera optical frame. 유효하지 않은 depth 픽셀의 표현(NaN vs 0)은 TODO — real 쪽 구현과 맞춰 확정.

주의: Isaac Sim/USD 카메라의 기본 frame 규약은 OpenCV optical frame과 다를 수 있으므로,
sim pointmap 생성 시 반드시 위 규약으로 변환하고 테스트로 검증한다.

## 미확정 (TODO)

- 이미지 해상도, intrinsics
- segmentation 표현 (binary mask / instance id / VisualCue 포맷 세부)
- joint state 순서, 단위(rad vs servo tick), gripper 표현
- 타임스탬프 / 프레임 동기화 규약
- action 공간 정의
