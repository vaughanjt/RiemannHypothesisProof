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
