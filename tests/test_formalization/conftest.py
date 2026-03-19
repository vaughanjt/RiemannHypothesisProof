"""Shared fixtures for formalization pipeline tests."""
import subprocess
import tempfile
from pathlib import Path

import pytest

from riemann.config import PROJECT_ROOT


@pytest.fixture
def lean_project_dir() -> Path:
    """Return path to the Lean 4 project directory."""
    return PROJECT_ROOT / "lean_proofs"


@pytest.fixture
def sample_lean_error_output() -> str:
    """Multi-line Lean compiler output with errors and sorry warnings."""
    return (
        "RiemannProofs/Bad.lean:10:4: error: unknown identifier 'foo'\n"
        "RiemannProofs/Bad.lean:15:0: warning: declaration uses 'sorry'\n"
        "RiemannProofs/Bad.lean:20:0: warning: declaration uses 'sorry'\n"
    )


@pytest.fixture
def sample_lean_clean_output() -> str:
    """Clean build output with no errors or warnings."""
    return "Build completed successfully.\n"


@pytest.fixture
def sample_lean_source_with_sorry() -> str:
    """Lean source text containing sorry tokens."""
    return (
        "theorem t1 : True := by\n"
        "  sorry\n"
        "theorem t2 : False := by\n"
        "  sorry\n"
        "  sorry\n"
    )


@pytest.fixture
def sample_lean_source_clean() -> str:
    """Lean source with comments mentioning sorry but no actual sorry tokens."""
    return (
        "-- sorry is not a real proof\n"
        "/- this block comment mentions sorry -/\n"
        'def msg := "sorry not sorry"\n'
        "theorem t : True := True.intro\n"
    )


@pytest.fixture
def wsl_available() -> bool:
    """Check if WSL is available on this machine."""
    try:
        result = subprocess.run(
            ["wsl", "-e", "echo", "ok"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0 and "ok" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
