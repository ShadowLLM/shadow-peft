from __future__ import annotations

import sys
from pathlib import Path

# Ensure `src/` is importable in tests without requiring an editable install.
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


