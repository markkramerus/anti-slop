"""
Shared pytest fixtures.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path for all test imports
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
