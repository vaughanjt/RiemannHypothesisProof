"""Stress-test framework for distinguishing genuine patterns from artifacts.

COMP-04: When a pattern is observed, the stress_test function re-runs the
computation at escalating precisions and/or expanded parameter ranges to verify
the pattern persists. If it vanishes at higher precision, it was a numerical artifact.

This is the #1 defense against Pitfall 2 (confusing evidence with proof).
"""
import time
from dataclasses import dataclass, field
from typing import Callable

import mpmath

from riemann.config import DEFAULT_DPS
from riemann.engine.precision import validated_computation
from riemann.types import ComputationResult


@dataclass
class StressTestResult:
    """Result of a stress test across multiple precision levels."""
    pattern_description: str
    consistent: bool
    results: list[ComputationResult]
    dps_levels: list[int]
    max_deviation: float
    predicate_results: list[bool] | None = None
    total_time_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


def stress_test(
    func: Callable,
    *args,
    dps_levels: list[int] | None = None,
    predicate: Callable | None = None,
    pattern_description: str = "",
    tolerance_digits: int | None = None,
) -> StressTestResult:
    """Run a computation at escalating precision levels to verify consistency.

    This is the core defense against numerical artifacts. If a pattern holds
    at 50 digits but vanishes at 200 digits, it was an artifact.

    Args:
        func: Computation function to stress-test. Called with (*args) under
            mpmath.workdps context.
        *args: Arguments to pass to func.
        dps_levels: List of precision levels to test. Default: [50, 100, 200].
        predicate: Optional function that takes a ComputationResult and returns
            bool. If provided, consistency requires predicate returning True
            at all levels.
        pattern_description: Human-readable description of the pattern being tested.
        tolerance_digits: Digits that must agree between consecutive levels.
            Default: min(level) - 10.

    Returns:
        StressTestResult with consistency assessment and per-level results.

    Raises:
        PrecisionError: If validated_computation detects P-vs-2P disagreement
            at any level (propagated from validated_computation).
    """
    if dps_levels is None:
        dps_levels = [DEFAULT_DPS, DEFAULT_DPS * 2, DEFAULT_DPS * 4]

    if tolerance_digits is None:
        tolerance_digits = min(dps_levels) - 10

    start_time = time.perf_counter()
    results = []
    predicate_results = [] if predicate else None

    for dps in dps_levels:
        result = validated_computation(
            func, *args,
            dps=dps,
            validate=True,
            algorithm=f"stress_test@{dps}dps",
        )
        results.append(result)

        if predicate:
            predicate_results.append(predicate(result))

    total_time = (time.perf_counter() - start_time) * 1000

    # Check consistency: consecutive results must agree
    max_deviation = 0.0
    consistent = True

    for i in range(1, len(results)):
        prev_val = results[i - 1].value
        curr_val = results[i].value

        if isinstance(prev_val, (mpmath.mpf, mpmath.mpc)):
            with mpmath.workdps(dps_levels[i] + 10):
                if abs(prev_val) == 0 and abs(curr_val) == 0:
                    deviation = 0.0
                else:
                    diff = abs(prev_val - curr_val)
                    scale = max(abs(prev_val), abs(curr_val), mpmath.mpf(1))
                    deviation = float(diff / scale)

                threshold = float(mpmath.power(10, -tolerance_digits))
                if deviation > threshold:
                    consistent = False
                max_deviation = max(max_deviation, deviation)

    # If predicate provided, all must be True for consistency
    if predicate_results and not all(predicate_results):
        consistent = False

    return StressTestResult(
        pattern_description=pattern_description,
        consistent=consistent,
        results=results,
        dps_levels=dps_levels,
        max_deviation=max_deviation,
        predicate_results=predicate_results,
        total_time_ms=total_time,
        metadata={
            "tolerance_digits": tolerance_digits,
            "num_levels": len(dps_levels),
        },
    )
