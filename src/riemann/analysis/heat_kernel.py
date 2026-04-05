"""Heat kernel trace on SL(2,Z)\\H -- Phase 5 feasibility gate.

Computes Tr(e^{-t*Delta}) on the modular surface with three spectral
contributions: the constant term from the volume, the discrete Maass cusp
form sum, and the continuous Eisenstein spectrum integral.

The heat kernel trace connects to the Connes barrier interpretation:
if K(t) > 0 for all t > 0, this provides structural positivity information
about the spectral decomposition on the modular surface.

Mathematical reference:
    K(t) = K_constant(t) + K_discrete(t) + K_continuous(t)

    K_constant(t) = 1/(12*t)                  (vol(SL(2,Z)\\H) = pi/3)
    K_discrete(t) = sum_{j} exp(-(1/4 + r_j^2)*t)   (Maass cusp forms)
    K_continuous(t) = (1/(4*pi)) * int_0^inf exp(-(1/4+r^2)*t) * psi(r) dr

    psi(r) = -(phi'/phi)(1/2+ir) = scattering phase

Function-based API. All inputs string-serialized before mpmath operations.
"""
from __future__ import annotations

import json
import math

import mpmath

from riemann.config import DATA_DIR
from riemann.types import ConvergenceDiagnostic

try:
    from flint import arb, ctx as flint_ctx
    from riemann.engine.dual_precision import dual_compute, dps_to_prec
    _HAS_FLINT = True
except ImportError:
    _HAS_FLINT = False

# Module-level cache for Maass spectral parameters
_CACHED_PARAMS: list[float] | None = None


def load_maass_spectral_params(data_file=None) -> list[float]:
    """Load Maass cusp form spectral parameters from JSON.

    Returns list of r values (spectral parameters) sorted ascending.
    Cached after first call to avoid re-reading disk.

    Args:
        data_file: Path to JSON file. Default: DATA_DIR / "maass_forms.json".

    Returns:
        List of float r values (eigenvalue lambda_j = 1/4 + r_j^2).
    """
    global _CACHED_PARAMS
    if _CACHED_PARAMS is not None and data_file is None:
        return _CACHED_PARAMS

    if data_file is None:
        data_file = DATA_DIR / "maass_forms.json"

    with open(data_file) as f:
        data = json.load(f)

    params = [entry["r"] for entry in data["spectral_parameters"]]
    if data_file is None or data_file == DATA_DIR / "maass_forms.json":
        _CACHED_PARAMS = params
    return params


def maass_spectral_sum(t, *, n_terms=None, dps=50):
    """Compute the discrete Maass cusp form spectral sum.

    K_discrete(t) = sum_{j=1}^{N} exp(-(1/4 + r_j^2) * t)

    Auto-selects N terms based on precision: includes all terms where
    exp(-lambda_j * t) > 10^{-dps} (i.e., lambda_j < dps*ln(10)/t).

    Args:
        t: Heat kernel time parameter (positive real).
        n_terms: Number of terms to include. None = auto-select.
        dps: Decimal digits of precision.

    Returns:
        Tuple of (mpf sum value, ConvergenceDiagnostic).
    """
    spectral_params = load_maass_spectral_params()
    n_available = len(spectral_params)

    # Auto-select n_terms based on precision threshold
    if n_terms is None:
        lambda_threshold = dps * math.log(10) / float(t)
        n_terms = 0
        for r in spectral_params:
            lam = 0.25 + r * r
            if lam < lambda_threshold:
                n_terms += 1
            else:
                break
        n_terms = max(10, min(n_terms, n_available))
    else:
        n_terms = min(n_terms, n_available)

    with mpmath.workdps(dps + 5):
        t_mp = mpmath.mpf(str(t))
        total = mpmath.mpf('0')
        last_term = mpmath.mpf('0')
        second_to_last = mpmath.mpf('0')

        for j in range(n_terms):
            r_j = mpmath.mpf(str(spectral_params[j]))
            lam_j = mpmath.mpf('0.25') + r_j ** 2
            term = mpmath.exp(-lam_j * t_mp)
            second_to_last = last_term
            last_term = term
            total += term

        # Tail bound from Weyl's law: exp(-lambda_N * t) / (12 * t)
        if n_terms < n_available:
            r_N = mpmath.mpf(str(spectral_params[n_terms]))
        else:
            r_N = mpmath.mpf(str(spectral_params[-1]))
        lambda_N = mpmath.mpf('0.25') + r_N ** 2
        tail_bound = mpmath.exp(-lambda_N * t_mp) / (12 * t_mp)

        # Convergence rate
        if n_terms >= 2 and second_to_last != 0:
            conv_rate = float(abs(last_term / second_to_last))
        else:
            conv_rate = 0.0

        last_mag = float(abs(last_term))

    diag = ConvergenceDiagnostic(
        n_terms_used=n_terms,
        n_terms_available=n_available,
        last_term_magnitude=last_mag,
        tail_bound=float(tail_bound),
        convergence_rate=conv_rate,
    )

    return (total, diag)


def scattering_phase(r, *, dps=50):
    """Compute the scattering phase -(phi'/phi)(1/2 + ir) for SL(2,Z).

    psi(r) = -digamma(ir) + digamma(1/2+ir)
             - 2*(zeta'/zeta)(2ir) + 2*(zeta'/zeta)(1+2ir)

    Args:
        r: Real spectral parameter. Must be non-zero (pole at r=0).
        dps: Decimal digits of precision.

    Returns:
        mpf real value of the scattering phase.

    Raises:
        ValueError: If r is zero (pole in digamma(ir=0)).
    """
    r_float = float(r)
    if abs(r_float) < 1e-15:
        raise ValueError("scattering_phase has a pole at r=0")

    with mpmath.workdps(dps + 10):
        r_mp = mpmath.mpf(str(r))
        ir = r_mp * mpmath.mpc(0, 1)  # i*r
        s = mpmath.mpc('0.5', str(r))  # 1/2 + ir

        # Term 1: -digamma(ir)
        term1 = -mpmath.digamma(ir)

        # Term 2: +digamma(1/2 + ir)
        term2 = mpmath.digamma(s)

        # Term 3: -2 * (zeta'/zeta)(2ir)
        z_2ir = 2 * ir
        zeta_deriv_2ir = mpmath.zeta(z_2ir, derivative=True)
        zeta_2ir = mpmath.zeta(z_2ir)
        term3 = -2 * zeta_deriv_2ir / zeta_2ir

        # Term 4: +2 * (zeta'/zeta)(1 + 2ir)
        z_1_2ir = 1 + 2 * ir
        zeta_deriv_1_2ir = mpmath.zeta(z_1_2ir, derivative=True)
        zeta_1_2ir = mpmath.zeta(z_1_2ir)
        term4 = 2 * zeta_deriv_1_2ir / zeta_1_2ir

        result = term1 + term2 + term3 + term4

    return mpmath.re(result)


def eisenstein_continuous_integral(t, *, dps=50, cutoff_R=200):
    """Compute the continuous Eisenstein spectrum contribution.

    K_continuous(t) = (1/(4*pi)) * int_{eps}^{R} exp(-(1/4+r^2)*t) * psi(r) dr

    Uses mpmath.quad for adaptive numerical integration. The scattering phase
    formula is inlined for quadrature efficiency (avoids per-point function call
    overhead from the standalone scattering_phase function).

    Args:
        t: Heat kernel time parameter (positive real).
        dps: Decimal digits of precision.
        cutoff_R: Upper integration limit (replaces infinity).

    Returns:
        mpf real value of the continuous spectrum integral.
    """
    epsilon = mpmath.mpf('0.01')  # Avoid r=0 pole in digamma(ir)

    with mpmath.workdps(dps + 10):
        t_mp = mpmath.mpf(str(t))
        R_mp = mpmath.mpf(str(cutoff_R))
        prefactor = 1 / (4 * mpmath.pi)

        def _zeta_prime_ratio(z):
            """Compute zeta'(z)/zeta(z) inline."""
            return mpmath.zeta(z, derivative=True) / mpmath.zeta(z)

        def integrand(r):
            """Integrand: exp(-(1/4+r^2)*t) * psi_scatt(r)."""
            ir = r * mpmath.mpc(0, 1)
            s_half = mpmath.mpf('0.5') + ir

            # Inline scattering phase for quadrature efficiency
            psi = (
                -mpmath.digamma(ir)
                + mpmath.digamma(s_half)
                - 2 * _zeta_prime_ratio(2 * ir)
                + 2 * _zeta_prime_ratio(1 + 2 * ir)
            )

            exponential = mpmath.exp(-(mpmath.mpf('0.25') + r ** 2) * t_mp)
            return exponential * mpmath.re(psi)

        # Adaptive quadrature over [epsilon, R]
        integral = mpmath.quad(integrand, [epsilon, R_mp])

        # Estimated tail beyond cutoff: exp(-(1/4+R^2)*t) * R / (4*pi)
        _tail_est = mpmath.exp(-(mpmath.mpf('0.25') + R_mp ** 2) * t_mp) * R_mp * prefactor

        result = prefactor * integral

    return mpmath.re(result)


def heat_kernel_trace(t, *, n_maass=None, dps=50, use_dual=True):
    """Compute the full heat kernel trace on SL(2,Z)\\H.

    K(t) = K_constant(t) + K_discrete(t) + K_continuous(t)

    Args:
        t: Heat kernel time parameter (positive real).
        n_maass: Number of Maass terms. None = auto-select.
        dps: Decimal digits of precision.
        use_dual: If True and python-flint available, dual-compute constant
                  and discrete terms (D-09). Eisenstein integral is mpmath-only
                  because python-flint arb lacks digamma/zeta special functions.

    Returns:
        Dict with keys: total, constant_term, discrete_sum, continuous_integral,
        convergence, dual_results, dps.
    """
    dual_results = {"constant": None, "discrete": None, "continuous": None}

    with mpmath.workdps(dps + 5):
        t_mp = mpmath.mpf(str(t))

        # --- Constant term: vol(SL(2,Z)\H) contribution ---
        constant_term = 1 / (12 * t_mp)

        # --- Discrete Maass cusp form sum ---
        discrete_val, convergence = maass_spectral_sum(t, n_terms=n_maass, dps=dps)

        # --- Continuous Eisenstein spectrum ---
        continuous_val = eisenstein_continuous_integral(t, dps=dps)

        # --- Total ---
        total = constant_term + discrete_val + continuous_val

    # Dual-precision cross-validation (D-09)
    if use_dual and _HAS_FLINT:
        try:
            # Constant term: simple enough for flint arb
            t_str = str(t)
            dual_constant = dual_compute(
                func_mpmath=lambda: mpmath.mpf('1') / (12 * mpmath.mpf(t_str)),
                func_flint=lambda prec: arb(1) / (arb(12) * arb(t_str)),
                dps=dps,
                label="heat_kernel_constant",
            )
            dual_results["constant"] = dual_constant
        except Exception:
            pass  # Non-fatal: dual is diagnostic, not blocking

        try:
            # Discrete sum: flint arb supports exp
            spectral_params = load_maass_spectral_params()
            n_used = convergence.n_terms_used
            params_for_flint = spectral_params[:n_used]

            def _flint_discrete(prec):
                t_arb = arb(str(t))
                total_arb = arb(0)
                for r_val in params_for_flint:
                    r_arb = arb(str(r_val))
                    lam = arb('0.25') + r_arb ** 2
                    total_arb += (-lam * t_arb).exp()
                return total_arb

            def _mpmath_discrete():
                val, _ = maass_spectral_sum(t, n_terms=n_used, dps=dps)
                return val

            dual_discrete = dual_compute(
                func_mpmath=_mpmath_discrete,
                func_flint=_flint_discrete,
                dps=dps,
                label="heat_kernel_discrete",
            )
            dual_results["discrete"] = dual_discrete
        except Exception:
            pass  # Non-fatal

        # Eisenstein integral: mpmath-only (flint arb lacks digamma/zeta)
        # dual_results["continuous"] stays None -- documented limitation

    return {
        "total": float(total),
        "constant_term": float(constant_term),
        "discrete_sum": float(discrete_val),
        "continuous_integral": float(continuous_val),
        "convergence": convergence,
        "dual_results": dual_results,
        "dps": dps,
    }
