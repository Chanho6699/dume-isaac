# dume-isaac

Project-Dum-E를 위한 **Isaac Sim / Isaac Lab 전용 시뮬레이션·학습 실험 repo**.

| Repo | 역할 |
|---|---|
| `physical-ai-dume` | 실제 로봇 시스템 (real robot, R3D/wrist 카메라, SAM2 VisualCue, 실제 제어) |
| `dume-isaac` (이 repo) | 시뮬레이션 / 학습 실험 |

두 repo는 코드 의존성 없이 분리되어 있다. 관측/행동의 의미 대응은
[docs/real_sim_contract.md](docs/real_sim_contract.md) 문서 계약으로만 맞춘다.

## 목표

1. SO-101 tabletop simulation
2. pick-and-place PPO teacher (full-state)
3. simulated ActorObservation
4. vision policy / distillation
5. synthetic demonstrations
6. sim-to-real evaluation

자세한 단계는 [docs/roadmap.md](docs/roadmap.md).

## 현재 상태

Foundation only. Isaac Sim scene, Isaac Lab EnvCfg, PPO 코드는 **아직 없다.**
학교 노트북은 Isaac Sim 5.1.x / Python 3.11로 *추정*되지만 Isaac Lab 버전은 미확인이므로,
버전에 묶이는 코드는 환경 확인 후 작성한다.

## 학교에서 첫 실행 순서

```bash
# 1. clone / pull
git clone <repo-url> ~/Projects/dume-isaac   # 또는: cd ~/Projects/dume-isaac && git pull

# 2. 환경 진단 (Isaac Lab을 실행하는 바로 그 Python으로 실행할 것)
python scripts/check_environment.py
#   Isaac Lab 소스 설치라면 예: ./isaaclab.sh -p ~/Projects/dume-isaac/scripts/check_environment.py
#   패키지를 실제 import까지 시도하려면: --try-import

# 3. 정확한 버전 기록 → docs/environment.md 에 결과 붙여넣기
#    OS / GPU / driver / Python / Isaac Sim / Isaac Lab / PyTorch / CUDA / 설치 경로

# 4. SO101_CFG 존재 여부 확인
#    check_environment.py 의 "SO-101 asset search" 항목 확인 (isaaclab_assets 소스를 텍스트 검색만 함)

# 5. 없으면: 표준 TheRobotStudio SO-101 URDF → USD import
#    (공식 SO-ARM100 repo에서 직접 받아 출처/커밋을 기록. 이 repo에 임의로 커밋하지 않음)
python scripts/inspect_so101_urdf.py <path/to/so101.urdf>
#    → Isaac Sim URDF importer 또는 Isaac Lab convert_urdf.py로 USD 변환

# 6. joint smoke test  (Phase 1, 환경 확인 후 작성)
# 7. tabletop scene    (Phase 2, configs/tabletop.yaml 실측 후 작성)
```

## 집에서 (Isaac 없이) 검증

```bash
python3 scripts/check_environment.py
python3 scripts/inspect_so101_urdf.py           # 사용법 출력
python3 -m pytest -q
```

## 구조

```
dume_isaac/          Python 패키지 (assets / envs / tasks / actor — 현재 placeholder)
configs/             시뮬레이션 설정 (미측정 값은 null)
scripts/             진단 / URDF 검사 도구 (stdlib 전용)
docs/                환경 기록, 로드맵, 실물 스펙, real↔sim 계약
tests/               Isaac 없이 도는 구조 테스트
```
