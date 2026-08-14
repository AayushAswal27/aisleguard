"""
pytest configuration.

Adds the repository root to sys.path so tests can import the `src` package
(e.g. `from src.risk.ttc import time_to_collision`) regardless of where pytest
is invoked from.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))