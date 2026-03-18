"""Shared test fixtures for the Riemann project."""
import os
import sqlite3
import tempfile
from pathlib import Path

import mpmath
import pytest

from riemann.config import ODLYZKO_DIR, DEFAULT_DPS
from riemann.engine.precision import precision_scope


@pytest.fixture
def high_precision():
    """Context manager fixture for 100-digit precision tests."""
    with precision_scope(100):
        yield


@pytest.fixture
def default_precision():
    """Context manager fixture for default (50-digit) precision tests."""
    with precision_scope(DEFAULT_DPS):
        yield


@pytest.fixture
def odlyzko_zeros() -> list[mpmath.mpf]:
    """Load first 100 Odlyzko zeros (imaginary parts) at high precision.

    Returns list of mpf values. Real part is always 0.5.
    """
    zeros_file = ODLYZKO_DIR / "zeros_100.txt"
    assert zeros_file.exists(), f"Odlyzko zeros file not found at {zeros_file}"

    zeros = []
    with open(zeros_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                zeros.append(mpmath.mpf(line))
    assert len(zeros) >= 10, f"Expected at least 10 zeros, got {len(zeros)}"
    return zeros


@pytest.fixture
def temp_db():
    """Temporary SQLite database for testing workbench operations."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    try:
        yield db_path
    finally:
        os.unlink(db_path)


@pytest.fixture
def first_zero_t() -> mpmath.mpf:
    """Imaginary part of the first non-trivial zero: 14.134725..."""
    with mpmath.workdps(60):
        return mpmath.zetazero(1).imag
