"""Riemann zeta function evaluation with always-validate pattern.

NEVER use scipy.special.zeta or float64 for critical strip evaluation.
All evaluation goes through mpmath with precision validation.
"""
import mpmath

from riemann.config import DEFAULT_DPS
from riemann.engine.precision import validated_computation
from riemann.types import ComputationResult


def zeta_eval(
    s,
    *,
    dps: int | None = None,
    validate: bool = True,
) -> ComputationResult:
    """Evaluate the Riemann zeta function at s with precision validation.

    Args:
        s: Complex or real point (mpmath.mpc, mpmath.mpf, or numeric).
        dps: Decimal digits of precision. Default: config.DEFAULT_DPS (50).
        validate: If True (default), run always-validate P-vs-2P pattern.

    Returns:
        ComputationResult with zeta(s) value, precision metadata, validation status.
    """
    if dps is None:
        dps = DEFAULT_DPS

    s_mp = mpmath.mpc(s) if not isinstance(s, (mpmath.mpf, mpmath.mpc)) else s

    return validated_computation(
        lambda: mpmath.zeta(s_mp),
        dps=dps,
        validate=validate,
        algorithm="mpmath.zeta",
    )


def zeta_on_critical_line(
    t,
    *,
    dps: int | None = None,
    validate: bool = True,
) -> ComputationResult:
    """Evaluate zeta(1/2 + it) on the critical line.

    Uses the standard zeta evaluation at s = 1/2 + it.
    For the magnitude |zeta(1/2+it)|, use Hardy's Z-function (siegelz)
    in the lfunctions module instead -- it is more numerically stable.

    Args:
        t: Real value (imaginary part on critical line).
        dps: Decimal digits of precision.
        validate: Always-validate flag.

    Returns:
        ComputationResult with zeta(0.5 + it).
    """
    if dps is None:
        dps = DEFAULT_DPS

    t_mp = mpmath.mpf(t) if not isinstance(t, mpmath.mpf) else t
    half = mpmath.mpf('0.5')  # Never hardcode 0.5 as float

    return validated_computation(
        lambda: mpmath.zeta(mpmath.mpc(half, t_mp)),
        dps=dps,
        validate=validate,
        algorithm="mpmath.zeta(0.5+it)",
    )
