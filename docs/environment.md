# 실행 환경 기록

## 확정된 학교 runtime (compatibility target)

| 항목 | 값 | 상태 |
|---|---|---|
| Isaac Sim | 5.1.0 | 확정 |
| Isaac Lab | v2.3.2 | 확정 |
| Python | 3.11 | 확정 |
| GPU | RTX 5060 Laptop | 확정 |

코드/API는 Isaac Lab v2.3.2 기준이다. Isaac Lab 3.x / develop, Isaac Sim 6.x API는 쓰지 않는다.
Isaac Lab v2.3.2에는 기본 `SO101_CFG`가 없으므로 이 repo의 `MY_SO101_CFG`를 쓴다.
Isaac Sim / Isaac Lab은 pip dependency가 아니며, 학교에 설치된 Isaac Lab 환경을 그대로 쓴다.

## 학교에서 추가 확인 (pending)

| 항목 | 값 | 확인 방법 |
|---|---|---|
| OS | pending | `cat /etc/os-release` 또는 Windows 설정 |
| NVIDIA driver | pending | `nvidia-smi` |
| CUDA runtime | pending | `script01_check_environment.py` (torch CUDA) / `nvidia-smi` |
| PyTorch | pending | `script01_check_environment.py` |
| Isaac Lab path | pending | Isaac Lab clone 위치 (`isaaclab.sh`가 있는 폴더) |
| Python executable | pending | `script01_check_environment.py`의 Executable |
| Isaac launch command | pending | 예: `~/IsaacLab/isaaclab.sh -p` |

## 실행 명령

```bash
cd ~/Projects/dume-isaac
<ISAACLAB_PATH>/isaaclab.sh -p scripts/script01_check_environment.py --try-import
```

## script01_check_environment.py 출력 (Isaac Lab interpreter, --try-import)

```text
(여기에 붙여넣기)
```

## 집 환경 (참고, Isaac 없음)

| 항목 | 값 |
|---|---|
| OS | Ubuntu 24.04 (WSL2) |
| Python | 3.12 |
| GPU | RTX 3050 |
| Isaac Sim / Isaac Lab / PyTorch | 없음 |

집에서는 **A. static/unit test**만 가능하다 (`python3 -m pytest -q`).
Isaac runtime 검증(**B**)은 학교에서 [school_runtime_checklist.md](school_runtime_checklist.md) 순서로 한다.

## 메모

- 설치 시 경고/에러:
- 권한 제약 (sudo 불가 등):
- 디스크 여유 공간:
