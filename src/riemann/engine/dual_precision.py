"""Dual-precision computation backend: mpmath + python-flint (per D-09).

Every computation runs in both mpmath (exploratory) and python-flint (ball
arithmetic). Disagreement beyond a threshold triggers flagging or raises
DualPrecisionError for catastrophic mismatches.

Usage:
    from riemann.engine.dual_precision import dual_compute, DualPrecisionError

    result = dual_compute(
        func_mpmath=lambda: mpmath.exp(-1),
        func_flint=lambda prec: arb(-1).exp(),
        dps=50,
        label="exp(-1)",
    )
    # result.agreement_digits > 15  (typically ~50 for well-conditioned)
    # result.flagged is False
"""
from __future__ import annotations

import math
import re

import mpmath
from flint import arb, ctx as flint_ctx  # noqa: F401 -- ensures flint is importable

from riemann.types import DualResult


class DualPrecisionError(Exception):
    """Raised when mpmath/flint catastrophically disagree (agreement < dps - 20)."""
    pass


def dps_to_prec(dps: int) -> int:
    """Convert decimal digits of precision to flint bit precision.

    Uses log2(10) ~ 3.32193 plus 20 guard bits.
    """
    return int(dps * 3.32193) + 20


def _arb_to_mpmath(fl_val, dps: int):
    """Convert a flint arb (or acb) to an mpmath number at full precision.

    Extracts the midpoint decimal string from arb.str(n_digits) and parses
    into mpmath.mpf. This avoids float64 truncation.
    """
    # Get string representation with enough decimal digits
    n_digits = dps + 5
    raw = fl_val.str(n_digits)
    # Parse: "[midpoint +/- error]" or just "midpoint"
    m = re.match(r'\[([-+]?[\d.]+(?:e[+-]?\d+)?)', raw)
    if m:
        return mpmath.mpf(m.group(1))
    # Fallback: try direct conversion
    return mpmath.mpf(str(float(fl_val)))


def dual_compute(
    func_mpmath,
    func_flint,
    *,
    dps: int = 50,
    label: str = "",
    threshold: float | None = None,
) -> DualResult:
    """Run a computation in both mpmath and python-flint, compare results.

    Args:
        func_mpmath: Callable taking no args, executed inside mpmath.workdps(dps+5).
                     Returns an mpmath number (mpf or mpc).
        func_flint:  Callable taking one arg `prec` (bit precision).
                     Returns a flint arb or acb.
        dps:         Decimal digits of precision (default 50).
        label:       Human-readable label for error messages.
        threshold:   Minimum agreement_digits before flagging. Default: dps - 10.
                     If agreement_digits < dps - 20, raises DualPrecisionError.

    Returns:
        DualResult with both values and agreement metric.

    Raises:
        DualPrecisionError: If agreement is catastrophically low (< dps - 20).
    """
    if threshold is None:
        threshold = dps - 10

    # Compute in mpmath at dps + 5 guard digits
    with mpmath.workdps(dps + 5):
        mp_val = func_mpmath()

    # Compute in flint at corresponding bit precision
    prec = dps_to_prec(dps)
    old_flint_prec = flint_ctx.prec
    flint_ctx.prec = prec
    try:
        fl_val = func_flint(prec)
    finally:
        flint_ctx.prec = old_flint_prec

    # Compare at full precision using mpmath (not float64 which caps at ~15 digits)
    with mpmath.workdps(dps + 5):
        # Convert flint result to mpmath via decimal string at full precision
        fl_as_mp = _arb_to_mpmath(fl_val, dps)

        mp_abs = abs(mpmath.mpf(mp_val))
        diff = abs(fl_as_mp - mpmath.mpf(mp_val))

        # Compute agreement in decimal digits
        if mp_abs != 0:
            relative = float(diff / mp_abs) if diff > 0 else 1e-300
            agreement = -math.log10(relative + 1e-300)
        else:
            agreement = -math.log10(float(diff) + 1e-300)

    # Check for catastrophic disagreement
    catastrophic_threshold = dps - 20
    if agreement < catastrophic_threshold:
        raise DualPrecisionError(
            f"Catastrophic disagreement in {label}: {agreement:.1f} digits "
            f"(need at least {catastrophic_threshold})"
        )

    # Check for flagged disagreement
    flagged = agreement < threshold

    return DualResult(
        mpmath_value=mp_val,
        flint_value=fl_val,
        agreement_digits=agreement,
        label=label,
        flagged=flagged,
    )
