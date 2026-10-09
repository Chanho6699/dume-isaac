# my_so101 asset

Canonical name for the SO-101 in this repo: **my_so101** (`MY_SO101_CFG`).

| 항목 | 값 |
|---|---|
| 실물 | Brain Us SO-101 follower (manufacturer: Brain Us) |
| 기반 설계 | TheRobotStudio SO-101 |
| 시뮬레이션 geometry | 공식 TheRobotStudio `Simulation/SO101/so101_new_calib.urdf` |
| Isaac Lab 기본 제공 cfg | 없음 (v2.3.2에 `SO101_CFG` 없음) → 이 repo에서 직접 관리 |

## 파일

| 파일 | 역할 | Isaac 필요 |
|---|---|---|
| [robot_spec.yaml](robot_spec.yaml) | 확인된 실물 사실만 | no |
| [source_manifest.yaml](source_manifest.yaml) | upstream repo / commit / 파일별 sha256 | no |
| [spec.py](spec.py) | `configs/my_so101.yaml` 로더: 논리 joint/frame 이름 → 실제 이름 (단일 매핑) | no |
| [robot_cfg.py](robot_cfg.py) | `make_my_so101_cfg()` / `MY_SO101_CFG` (Isaac Lab v2.3.2 `ArticulationCfg`) | yes |
| `urdf/` | upstream 파일 (fetch, git-ignored) | no |
| `usd/` | 변환된 USD (script04, git-ignored) | – |

## Source 방식 (선택: pinned fetch)

공식 repo `TheRobotStudio/SO-ARM100`의 한 commit에 고정하고 sha256으로 검증하는
fetch 스크립트를 쓴다. 파일은 upstream과 byte-identical이며 repo에는 manifest만 커밋한다.

```bash
python scripts/script02_fetch_so101_source.py     # Stage 1a
python scripts/script03_inspect_so101_urdf.py     # Stage 1b (mapping check 포함)
```

다른 출처(블로그, fork)의 URDF는 쓰지 않는다. upstream을 갱신하려면 manifest의 commit과
sha256을 함께 바꾼다.

## 이름 매핑을 바꿔야 할 때

USD의 joint/body 이름이 다르면 **`configs/my_so101.yaml`의 `joints` / `frames`만** 수정한다.
Reach / pick-place / 테스트 스크립트는 모두 `spec.py`를 통해 이름을 읽는다.

## 값의 상태

- URDF geometry / inertia: upstream CAD 값 (Brain Us 실물 검증 아님)
- actuator stiffness/damping/armature/effort: upstream MuJoCo STS3215 기본값을 쓴 **provisional simulation value**
- home pose, gripper open/closed: **provisional simulation value**
- joint zero / limits / gripper mapping / backlash / friction: Brain Us 실물 미검증
