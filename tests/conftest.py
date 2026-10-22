"""Shared test setup. None of these tests call Gemini or load the embedding
model -- anything that would is replaced with a fake, so the suite runs
offline, for free, in a few seconds.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pytest


@pytest.fixture(autouse=True)
def no_real_library(monkeypatch):
    """docker compose sets LIBRARY_DIR; tests must never touch the real
    document library. Tests that need one point it at a temp folder.
    """
    monkeypatch.delenv("LIBRARY_DIR", raising=False)
