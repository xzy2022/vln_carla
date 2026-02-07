from __future__ import annotations

import pathlib
import sys

ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]
SRC: pathlib.Path = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
