"""Make `pytest` work from a clean checkout without `pip install -e .` first."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
