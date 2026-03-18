"""Precision management: context managers and always-validate pattern.

CRITICAL: Never use bare mpmath.mp.dps = N. Always use precision_scope()
or mpmath.workdps(). Global state causes precision leak between computations.
"""
from contextlib import contextmanager
import time
import mpmath

from riemann.config import DEFAULT_DPS
from riemann.types import ComputationResult, PrecisionError


@contextmanager
def precision_scope(dps: int):
    """Set mpmath precision with guard digits for a computation block.

    Adds 5 guard digits beyond requested precision.
    Exception-safe: always restores original precision on exit.
    """
    with mpmath.workdps(dps + 5):
        yield dps


def validated_computation(
    func,
    *args,
    dps: int | None = None,
    validate: bool = True,
    tolerance: int | None = None,
    algorithm: str = "",
) -> ComputationResult:
    """Run func at dps and 2*dps, compare results to catch precision collapse.

    Args:
        func: Callable that performs computation using current mpmath precision.
        *args: Arguments to pass to func.
        dps: Decimal digits of precision. Default: config.DEFAULT_DPS (50).
        validate: If True, run at both P and 2P. If False, run once.
        tolerance: Number of digits that must agree. Default: dps - 5.
        algorithm: Name of algorithm for metadata.

    Returns:
        ComputationResult with the higher-precision result (or single result if validate=False).

    Raises:
        PrecisionError: If P and 2P results disagree within tolerance digits.
    """
    if dps is None:
        dps = DEFAULT_DPS
    if tolerance is None:
        tolerance = dps - 5

    start_time = time.perf_counter()

    if not validate:
        with mpmath.workdps(dps + 5):
            result = func(*args)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return ComputationResult(
            value=result,
            precision_digits=dps,
            validated=False,
            validation_precision=None,
            algorithm=algorithm,
            computation_time_ms=elapsed_ms,
        )

    # Compute at target precision
    with mpmath.workdps(dps + 5):
        result_p = func(*args)

    # Compute at double precision
    with mpmath.workdps(2 * dps + 5):
        result_2p = func(*args)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Compare: first 'tolerance' digits must agree
    if not _digits_agree(result_p, result_2p, tolerance):
        raise PrecisionError(
            f"Results disagree within {tolerance} digits at dps={dps}. "
            f"P result: {mpmath.nstr(result_p, 15)}, "
            f"2P result: {mpmath.nstr(result_2p, 15)}"
        )

    return ComputationResult(
        value=result_2p,
        precision_digits=dps,
        validated=True,
        validation_precision=2 * dps,
        algorithm=algorithm,
        computation_time_ms=elapsed_ms,
    )


def _digits_agree(a, b, digits: int) -> bool:
    """Check if two mpmath values agree to the specified number of digits."""
    if isinstance(a, mpmath.mpc) or isinstance(b, mpmath.mpc):
        a = mpmath.mpc(a)
        b = mpmath.mpc(b)
        # Both real and imaginary parts must agree
        if abs(a) == 0 and abs(b) == 0:
            return True
        diff = abs(a - b)
        scale = max(abs(a), abs(b), mpmath.mpf(1))
        relative_error = diff / scale
        threshold = mpmath.power(10, -digits)
        return relative_error < threshold
    else:
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        if a == 0 and b == 0:
            return True
        diff = abs(a - b)
        scale = max(abs(a), abs(b), mpmath.mpf(1))
        relative_error = diff / scale
        threshold = mpmath.power(10, -digits)
        return relative_error < threshold
