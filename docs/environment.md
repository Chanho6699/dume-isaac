# 학교 환경 기록

학교 노트북에서 **실제로 확인한 값만** 기록한다. 추정치(예: "Isaac Sim 5.1.x / Python 3.11")는
확인 전까지 아래 표에 넣지 않는다.

기록 날짜: TODO
머신 이름 / 자산 번호: TODO

| 항목 | 값 | 확인 방법 |
|---|---|---|
| OS | TODO | `cat /etc/os-release` 또는 Windows 설정 |
| GPU | TODO | `nvidia-smi` |
| NVIDIA driver | TODO | `nvidia-smi` |
| Python | TODO | `check_environment.py` (Isaac Lab을 실행하는 interpreter 기준) |
| Isaac Sim | TODO | `check_environment.py`, Isaac Sim 설치 폴더의 `VERSION` |
| Isaac Lab | TODO | `check_environment.py`, Isaac Lab repo의 `git describe --tags` |
| PyTorch | TODO | `check_environment.py` |
| CUDA (torch) | TODO | `check_environment.py` |
| Isaac Sim 설치 경로 | TODO | |
| Isaac Lab 설치 경로 | TODO | |
| 가상환경 (conda / venv / isaaclab.sh -p) | TODO | |
| 실행 방식 (예: `./isaaclab.sh -p`) | TODO | |
| SO-101 builtin cfg 존재 여부 | TODO | `check_environment.py`의 SO-101 asset search |

## 실행 명령

```bash
cd ~/Projects/dume-isaac
python scripts/check_environment.py
# Isaac Lab interpreter로:
#   <ISAACLAB_PATH>/isaaclab.sh -p scripts/check_environment.py
# import까지 시도:
#   ... scripts/check_environment.py --try-import
```

## check_environment.py 출력 (기본 interpreter)

```text
(여기에 붙여넣기)
```

## check_environment.py 출력 (Isaac Lab interpreter, --try-import)

```text
(여기에 붙여넣기)
```

## 메모

- 설치 시 경고/에러:
- 권한 제약 (sudo 불가 등):
- 디스크 여유 공간:
