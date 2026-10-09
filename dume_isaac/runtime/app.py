"""Isaac app launch boundary.

Everything above `launch_app()` is pure Python. Isaac Lab requires the app to be
launched (AppLauncher) before importing any other isaaclab / omni module, so scripts call
`launch_app()` first and import Isaac modules afterwards.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys

EXIT_NO_ISAAC = 3
TARGET_ISAAC_SIM = "5.1.0"
TARGET_ISAAC_LAB = "2.3.2"


def isaaclab_available() -> bool:
    try:
        return importlib.util.find_spec("isaaclab") is not None
    except Exception:
        return False


def missing_isaac_message(script: str) -> str:
    return (
        f"[{script}] Isaac Lab is not importable from this Python ({sys.executable}).\n"
        f"  This step needs the school Isaac runtime (Isaac Sim {TARGET_ISAAC_SIM} / Isaac Lab v{TARGET_ISAAC_LAB}).\n"
        "  Run it with the Isaac Lab interpreter, e.g.:\n"
        f"    <ISAACLAB_PATH>/isaaclab.sh -p scripts/{script} [args]\n"
        "  At home only static/unit tests are possible: python3 -m pytest -q"
    )


def launch_app(parser: argparse.ArgumentParser, argv: list[str] | None = None):
    """Parse args, start Isaac Sim through AppLauncher, return (args, simulation_app).

    Without Isaac Lab: `--help` still works (script args only), anything else prints an
    actionable message and exits with EXIT_NO_ISAAC instead of a traceback.
    """
    if not isaaclab_available():
        parser.parse_known_args(argv)  # handles --help / bad script args
        print(missing_isaac_message(parser.prog), file=sys.stderr)
        sys.exit(EXIT_NO_ISAAC)

    from isaaclab.app import AppLauncher  # noqa: PLC0415

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    app_launcher = AppLauncher(args)
    return args, app_launcher.app
