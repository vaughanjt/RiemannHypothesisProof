"""SPC anomaly detection for zero distributions.

Scans sliding windows of zeros, compares local statistics against GUE
predictions, and flags deviations exceeding configurable sigma thresholds.
Warning and critical anomalies are auto-logged to the research workbench
as observations (evidence level 0).

Function-based API. Returns Anomaly dataclasses and conjecture IDs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from riemann.analysis.spacing import normalized_spacings
from riemann.types import ZetaZero
from riemann.workbench.conjecture import create_conjecture


# GUE reference constants (from random matrix theory)
GUE_MEAN_SPACING = 1.0
GUE_SPACING_VARIANCE = 1.0 - 2.0 / (np.pi ** 2)  # ~0.2732

DEFAULT_SIGMA_THRESHOLDS = {"info": 2.0, "warning": 3.0, "critical": 4.0}


@dataclass
class Anomaly:
    """A detected anomaly in zero distribution statistics."""
    zero_range: tuple[int, int]
    statistic: str
    observed_value: float
    expected_value: float
    sigma_deviation: float
    severity: str
    description: str


def _assign_severity(sigma: float, thresholds: dict[str, float]) -> str | None:
    """Assign severity based on sigma deviation and thresholds."""
    # Check from highest to lowest
    for level in ("critical", "warning", "info"):
        if level in thresholds and sigma >= thresholds[level]:
            return level
    return None


def detect_anomalies(
    zeros: list[ZetaZero],
    window_size: int = 50,
    stride: int = 25,
    sigma_thresholds: dict[str, float] | None = None,
) -> list[Anomaly]:
    """Scan zero windows for deviations from GUE predictions.

    Checks mean spacing and spacing variance in each window.

    Args:
        zeros: List of ZetaZero objects.
        window_size: Number of zeros per window.
        stride: Step between windows.
        sigma_thresholds: Custom thresholds {severity: sigma}. Defaults to 2/3/4.

    Returns:
        List of Anomaly objects sorted by sigma_deviation (highest first).
    """
    if sigma_thresholds is None:
        sigma_thresholds = DEFAULT_SIGMA_THRESHOLDS.copy()

    min_threshold = min(sigma_thresholds.values())
    anomalies = []

    n = len(zeros)
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        window_zeros = zeros[start:end]

        spacings = normalized_spacings(window_zeros)
        if len(spacings) < 2:
            continue

        # Statistic 1: mean spacing deviation
        observed_mean = float(np.mean(spacings))
        se_mean = np.sqrt(GUE_SPACING_VARIANCE / len(spacings))
        if se_mean > 0:
            z_mean = abs(observed_mean - GUE_MEAN_SPACING) / se_mean
            severity = _assign_severity(z_mean, sigma_thresholds)
            if severity is not None and z_mean >= min_threshold:
                anomalies.append(Anomaly(
                    zero_range=(start, end),
                    statistic="mean_spacing",
                    observed_value=observed_mean,
                    expected_value=GUE_MEAN_SPACING,
                    sigma_deviation=z_mean,
                    severity=severity,
                    description=(
                        f"Mean spacing {observed_mean:.4f} deviates {z_mean:.1f}σ "
                        f"from GUE expected {GUE_MEAN_SPACING:.4f} "
                        f"in zeros [{start}, {end})"
                    ),
                ))

        # Statistic 2: spacing variance deviation
        observed_var = float(np.var(spacings, ddof=1)) if len(spacings) > 1 else 0.0
        # Standard error of variance: sqrt(2/(n-1)) * sigma^2 (chi-squared-based)
        se_var = np.sqrt(2.0 / (len(spacings) - 1)) * GUE_SPACING_VARIANCE if len(spacings) > 1 else 1.0
        if se_var > 0:
            z_var = abs(observed_var - GUE_SPACING_VARIANCE) / se_var
            severity = _assign_severity(z_var, sigma_thresholds)
            if severity is not None and z_var >= min_threshold:
                anomalies.append(Anomaly(
                    zero_range=(start, end),
                    statistic="spacing_variance",
                    observed_value=observed_var,
                    expected_value=GUE_SPACING_VARIANCE,
                    sigma_deviation=z_var,
                    severity=severity,
                    description=(
                        f"Spacing variance {observed_var:.4f} deviates {z_var:.1f}σ "
                        f"from GUE expected {GUE_SPACING_VARIANCE:.4f} "
                        f"in zeros [{start}, {end})"
                    ),
                ))

    anomalies.sort(key=lambda a: a.sigma_deviation, reverse=True)
    return anomalies


def log_anomalies_to_workbench(
    anomalies: list[Anomaly],
    db_path: str | Path | None = None,
) -> list[str]:
    """Log warning and critical anomalies as workbench observations.

    Creates a conjecture record for each anomaly with severity >= warning.
    Evidence level is set to 0 (OBSERVATION).

    Args:
        anomalies: List of Anomaly objects.
        db_path: Optional path to workbench database.

    Returns:
        List of conjecture UUIDs created.
    """
    ids = []
    for a in anomalies:
        if a.severity not in ("warning", "critical"):
            continue

        tags = f"anomaly,{a.severity},{a.statistic}"
        cid = create_conjecture(
            statement=(
                f"Anomalous {a.statistic} in zeros {a.zero_range[0]}-{a.zero_range[1]}: "
                f"{a.sigma_deviation:.1f}-sigma from GUE"
            ),
            description=a.description,
            evidence_level=0,
            status="speculative",
            tags=tags,
            db_path=db_path,
        )
        ids.append(cid)

    return ids
