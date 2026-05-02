"""
Entry points wired up in pyproject.toml [project.scripts].

  run-demo  →  python run_demo.py   (full APA end-to-end demo)
  compare   →  python compare.py    (APA vs GEPA vs MIPRO benchmark)
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent


def run_demo() -> None:
    sys.path.insert(0, str(_ROOT))
    runpy.run_path(str(_ROOT / "run_demo.py"), run_name="__main__")


def run_compare() -> None:
    sys.path.insert(0, str(_ROOT))
    runpy.run_path(str(_ROOT / "compare.py"), run_name="__main__")
