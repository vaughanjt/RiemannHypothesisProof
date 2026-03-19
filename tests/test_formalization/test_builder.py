"""Tests for WSL2 subprocess build runner."""
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from riemann.formalization.builder import (
    LakeBuildResult,
    windows_to_wsl_path,
    wsl_to_windows_path,
    run_lake_build,
    LEAN_PROJECT_DIR,
)


class TestLakeBuildResult:
    """Tests for the LakeBuildResult dataclass."""

    def test_creation_defaults(self):
        """LakeBuildResult can be created with default field values."""
        result = LakeBuildResult(
            success=True, returncode=0, output="Build completed."
        )
        assert result.success is True
        assert result.returncode == 0
        assert result.output == "Build completed."
        assert result.messages == []
        assert result.sorry_count == 0
        assert result.error_count == 0
        assert result.warning_count == 0
        assert result.duration_ms == 0.0

    def test_creation_with_all_fields(self):
        """LakeBuildResult stores all fields correctly."""
        from riemann.formalization.parser import LeanMessage
        msg = LeanMessage(file="a.lean", line=1, col=0, severity="error", message="bad")
        result = LakeBuildResult(
            success=False, returncode=1, output="error output",
            messages=[msg], sorry_count=2, error_count=1,
            warning_count=3, duration_ms=123.4,
        )
        assert result.success is False
        assert len(result.messages) == 1
        assert result.sorry_count == 2
        assert result.duration_ms == 123.4


class TestPathConversion:
    """Tests for Windows/WSL path conversion (mocked)."""

    @patch("riemann.formalization.builder.subprocess.run")
    def test_windows_to_wsl_path(self, mock_run):
        """windows_to_wsl_path calls wslpath and returns stripped output."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="/mnt/c/Users/test/project\n",
            stderr="",
        )
        result = windows_to_wsl_path("C:\\Users\\test\\project")
        assert result == "/mnt/c/Users/test/project"
        mock_run.assert_called_once()

    @patch("riemann.formalization.builder.subprocess.run")
    def test_wsl_to_windows_path(self, mock_run):
        """wsl_to_windows_path calls wslpath -w and returns stripped output."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="C:\\Users\\test\\project\n",
            stderr="",
        )
        result = wsl_to_windows_path("/mnt/c/Users/test/project")
        assert result == "C:\\Users\\test\\project"
        mock_run.assert_called_once()

    @patch("riemann.formalization.builder.subprocess.run")
    def test_windows_to_wsl_path_error(self, mock_run):
        """windows_to_wsl_path raises RuntimeError on failure."""
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="wslpath: error"
        )
        with pytest.raises(RuntimeError, match="wslpath failed"):
            windows_to_wsl_path("C:\\bad\\path")


class TestLeanProjectDir:
    """Test that LEAN_PROJECT_DIR is correctly configured."""

    def test_lean_project_dir_is_path(self):
        """LEAN_PROJECT_DIR is a Path ending with lean_proofs."""
        assert isinstance(LEAN_PROJECT_DIR, Path)
        assert LEAN_PROJECT_DIR.name == "lean_proofs"


@pytest.mark.integration
class TestRealLakeBuild:
    """Integration tests that require WSL2 and a working Lean 4 installation."""

    def test_real_lake_build(self, wsl_available, lean_project_dir):
        """Run actual lake build against the lean_proofs/ project."""
        if not wsl_available:
            pytest.skip("WSL not available")
        if not lean_project_dir.exists():
            pytest.skip("lean_proofs/ directory does not exist")

        result = run_lake_build(project_dir=lean_project_dir, timeout_seconds=120)
        assert isinstance(result, LakeBuildResult)
        assert result.success is True
        assert result.returncode == 0
        assert result.error_count == 0
        assert result.sorry_count == 0
        assert result.duration_ms > 0
