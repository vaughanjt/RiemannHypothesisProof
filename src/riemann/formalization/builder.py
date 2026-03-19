"""WSL2 subprocess build runner for Lean 4 projects."""
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from riemann.config import PROJECT_ROOT
from riemann.formalization.parser import LeanMessage, parse_lean_output

LEAN_PROJECT_DIR = PROJECT_ROOT / "lean_proofs"


@dataclass
class LakeBuildResult:
    """Result of a lake build invocation."""

    success: bool
    returncode: int
    output: str  # Combined stdout+stderr
    messages: list[LeanMessage] = field(default_factory=list)
    sorry_count: int = 0
    error_count: int = 0
    warning_count: int = 0
    duration_ms: float = 0.0


def windows_to_wsl_path(windows_path: str | Path) -> str:
    """Convert Windows path to WSL mount path using wslpath."""
    result = subprocess.run(
        ["wsl", "-e", "wslpath", "-u", str(windows_path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"wslpath failed: {result.stderr}")
    return result.stdout.strip()


def wsl_to_windows_path(wsl_path: str) -> str:
    """Convert WSL path to Windows path using wslpath."""
    result = subprocess.run(
        ["wsl", "-e", "wslpath", "-w", wsl_path],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"wslpath failed: {result.stderr}")
    return result.stdout.strip()


def run_lake_build(
    project_dir: Path | None = None,
    timeout_seconds: int = 300,
    fsync_delay: float = 0.1,
) -> LakeBuildResult:
    """Run lake build inside WSL2 and return structured result.

    Args:
        project_dir: Path to Lean project on Windows filesystem.
                     Defaults to LEAN_PROJECT_DIR.
        timeout_seconds: Max seconds before killing the build.
        fsync_delay: Seconds to wait after recent file writes
                     (mitigates 9P filesystem cache delay).

    Returns:
        LakeBuildResult with parsed messages, sorry count, etc.
    """
    project_dir = project_dir or LEAN_PROJECT_DIR
    wsl_path = windows_to_wsl_path(project_dir)

    # Small delay to let 9P filesystem flush (Pitfall 5 from RESEARCH.md)
    time.sleep(fsync_delay)

    cmd = f'source "$HOME/.elan/env" && cd "{wsl_path}" && lake build 2>&1'
    start = time.perf_counter()
    result = subprocess.run(
        ["wsl", "-e", "bash", "-c", cmd],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    output = result.stdout + result.stderr
    messages, sorry_count = parse_lean_output(output)
    error_count = sum(1 for m in messages if m.severity == "error")
    warning_count = sum(1 for m in messages if m.severity == "warning")

    return LakeBuildResult(
        success=result.returncode == 0,
        returncode=result.returncode,
        output=output,
        messages=messages,
        sorry_count=sorry_count,
        error_count=error_count,
        warning_count=warning_count,
        duration_ms=elapsed_ms,
    )
