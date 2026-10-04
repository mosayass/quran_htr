"""
tests/smoke_test.py
Test runner executing smoke tests for CRNN + CTC pipeline.
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import smoke_test

if __name__ == "__main__":
    smoke_test.main()
