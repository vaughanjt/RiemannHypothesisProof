"""Related function evaluation: Hardy Z, Dirichlet L, xi, Selberg zeta.

Uses mpmath built-ins where available (siegelz, dirichlet).
Xi function is hand-built from zeta + gamma (no mpmath built-in).
Selberg zeta is a stub for Phase 1 -- full implementation in Phase 2/3.

NEVER hand-roll: siegelz, siegeltheta, dirichlet. Use mpmath.
"""
import mpmath

from riemann.config import DEFAULT_DPS
from riemann.engine.precision import validated_computation
from riemann.types import ComputationResult


def hardy_z(
    t,
    *,
    dps: int | None = None,
    validate: bool = True,
) -> ComputationResult:
    """Evaluate Hardy's Z-function Z(t).

    Z(t) = exp(i*theta(t)) * zeta(1/2 + it) where theta is Riemann-Siegel theta.
    Z(t) is real-valued for real t, and |Z(t)| = |zeta(1/2 + it)|.
    Sign changes of Z(t) correspond to zeros of zeta on the critical line.

    Uses mpmath.siegelz(t) -- never hand-roll.
    """
    if dps is None:
        dps = DEFAULT_DPS

    # Store as string so t is reconstructed at working precision inside
    # validated_computation (avoids precision truncation from caller's scope).
    t_str = str(t) if not isinstance(t, (mpmath.mpf, mpmath.mpc)) else mpmath.nstr(t, 50)

    return validated_computation(
        lambda: mpmath.siegelz(mpmath.mpf(t_str)),
        dps=dps,
        validate=validate,
        algorithm="mpmath.siegelz",
    )


def dirichlet_l(
    s,
    chi: list,
    *,
    dps: int | None = None,
    validate: bool = True,
) -> ComputationResult:
    """Evaluate Dirichlet L-function L(s, chi).

    chi is a list representing the Dirichlet character values.
    chi = [1] gives the Riemann zeta function.
    chi = [0, 1, 0, -1] is a non-principal character mod 4.

    Uses mpmath.dirichlet(s, chi) -- never hand-roll.

    Args:
        s: Complex or real point.
        chi: List of character values (length = modulus).
        dps: Precision digits.
        validate: Always-validate flag.
    """
    if dps is None:
        dps = DEFAULT_DPS

    # Store as strings so s is reconstructed at working precision inside
    # validated_computation (avoids precision truncation from caller's scope).
    if isinstance(s, (mpmath.mpf, mpmath.mpc)):
        s_re = mpmath.nstr(mpmath.re(s), 50)
        s_im = mpmath.nstr(mpmath.im(s), 50)
    elif hasattr(s, 'real'):
        s_re, s_im = str(s.real), str(s.imag)
    else:
        s_re, s_im = str(s), "0"

    return validated_computation(
        lambda: mpmath.dirichlet(mpmath.mpc(s_re, s_im), chi),
        dps=dps,
        validate=validate,
        algorithm=f"mpmath.dirichlet(chi_mod{len(chi)})",
    )


def xi_function(
    s,
    *,
    dps: int | None = None,
    validate: bool = True,
) -> ComputationResult:
    """Evaluate the Riemann xi function.

    xi(s) = (1/2) * s * (s-1) * pi^(-s/2) * gamma(s/2) * zeta(s)

    The xi function is entire and satisfies xi(s) = xi(1-s).
    Its zeros are exactly the non-trivial zeros of zeta.

    NOT a built-in mpmath function -- hand-built from components.

    Args:
        s: Complex or real point.
        dps: Precision digits. Extra guard digits added internally.
        validate: Always-validate flag.
    """
    if dps is None:
        dps = DEFAULT_DPS

    # Store as strings so s is reconstructed at working precision inside
    # validated_computation (avoids precision truncation from caller's scope).
    if isinstance(s, (mpmath.mpf, mpmath.mpc)):
        s_re = mpmath.nstr(mpmath.re(s), 50)
        s_im = mpmath.nstr(mpmath.im(s), 50)
    elif hasattr(s, 'real'):
        s_re, s_im = str(s.real), str(s.imag)
    else:
        s_re, s_im = str(s), "0"

    def _compute_xi():
        s_local = mpmath.mpc(s_re, s_im)
        half = mpmath.mpf(1) / 2
        return (half * s_local * (s_local - 1)
                * mpmath.power(mpmath.pi, -s_local / 2)
                * mpmath.gamma(s_local / 2)
                * mpmath.zeta(s_local))

    return validated_computation(
        _compute_xi,
        dps=dps,
        validate=validate,
        tolerance=dps - 10,  # xi involves gamma, needs more guard digits
        algorithm="xi = s(s-1)/2 * pi^(-s/2) * gamma(s/2) * zeta(s)",
    )


def selberg_zeta_stub(spectral_data=None, **kwargs):
    """Stub for Selberg zeta function.

    The Selberg zeta function is defined for a discrete group acting on
    the hyperbolic plane and requires spectral data (lengths of closed
    geodesics). Full implementation deferred to Phase 2/3.

    Raises:
        NotImplementedError: Always. This is a placeholder.
    """
    raise NotImplementedError(
        "Selberg zeta function is not yet implemented. "
        "Requires spectral data (geodesic lengths) for a discrete group. "
        "Planned for Phase 2/3. See COMP-03 partial delivery note in RESEARCH.md."
    )
