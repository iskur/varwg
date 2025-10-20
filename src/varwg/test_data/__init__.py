"""Test fixtures for VG package tests."""
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent

def get_fixture_path(filename):
    """Get path to a test fixture file."""
    return FIXTURE_DIR / filename
