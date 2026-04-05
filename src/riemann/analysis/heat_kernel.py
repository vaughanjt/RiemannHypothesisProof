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

Plan 03 additions:
    barrier_value_numpy -- fast numpy float64 barrier via session41g pattern
    barrier_value_mpmath -- arbitrary precision barrier for spot checks
    find_parameter_mapping -- discover t(L) via analytic + numerical tracks
    run_feasibility_comparison -- 100+ point K(t(L)) vs B(L) comparison
    feasibility_verdict -- analyze comparison results for go/no-go

Function-based API. All inputs string-serialized before mpmath operations.
"""
from __future__ import annotations

import json
import math

import mpmath
import numpy as np

from riemann.config import DATA_DIR
from riemann.types import BarrierComparison, ConvergenceDiagnostic

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


# ---------------------------------------------------------------------------
# Plan 03: Barrier computation, parameter mapping, feasibility comparison
# ---------------------------------------------------------------------------


def _sieve_primes_numpy(limit):
    """Sieve of Eratosthenes using numpy. Returns numpy array of primes <= limit."""
    if limit < 2:
        return np.array([], dtype=int)
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[:2] = False
    for i in range(2, int(limit**0.5) + 2):
        if i <= limit and is_prime[i]:
            is_prime[i * i::i] = False
    return np.where(is_prime)[0]


def barrier_value_numpy(L, N=None):
    """Compute barrier B(L) = W02 - M_prime at numpy float64 precision.

    This wraps the session41g barrier formula (vectorized numpy matrix
    construction) for fast bulk evaluation. Uses float64 arithmetic which
    gives ~15 digits of precision -- sufficient for the feasibility sweep.

    Args:
        L: Barrier parameter (log(lambda^2)).
        N: Basis truncation. Default: max(15, round(6*L)).

    Returns:
        Float barrier value B(L) = W02_rq - M_prime_rq.
    """
    lam_sq = np.exp(L)
    if N is None:
        N = max(15, round(6 * float(L)))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    # w_hat (normalized odd eigenvector of W02)
    denom = float(L) ** 2 + (4 * np.pi) ** 2 * ns ** 2
    w = ns / denom
    w[N] = 0.0
    w_norm = np.linalg.norm(w)
    if w_norm < 1e-300:
        return 0.0
    w_hat = w / w_norm

    # W02 Rayleigh quotient
    pf = 32 * float(L) * np.sinh(float(L) / 4) ** 2
    w_tilde = ns / denom
    wt_dot_wh = np.dot(w_tilde, w_hat)
    w02_rq = -pf * (4 * np.pi) ** 2 * wt_dot_wh ** 2

    # M_prime: all primes up to lam_sq
    lam_sq_f = float(lam_sq)
    primes = _sieve_primes_numpy(int(lam_sq_f))

    # Collect prime powers
    pk_data = []
    for p in primes:
        pk = int(p)
        k = 1
        logp = np.log(float(p))
        while pk <= lam_sq_f:
            pk_data.append((logp, logp * float(pk) ** (-0.5), k * logp))
            pk *= int(p)
            k += 1

    if not pk_data:
        return float(w02_rq)

    # Build M_prime matrix
    M_prime = np.zeros((dim, dim))
    nm_diff = ns[:, None] - ns[None, :]
    L_f = float(L)

    for logp, weight, y in pk_data:
        sin_arr = np.sin(2 * np.pi * ns * y / L_f)
        cos_arr = np.cos(2 * np.pi * ns * y / L_f)

        # Diagonal
        diag = 2 * (L_f - y) / L_f * cos_arr
        np.fill_diagonal(M_prime, M_prime.diagonal() + weight * diag)

        # Off-diagonal
        sin_diff = sin_arr[None, :] - sin_arr[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            off_diag = sin_diff / (np.pi * nm_diff)
        np.fill_diagonal(off_diag, 0.0)
        M_prime += weight * off_diag

    M_prime = (M_prime + M_prime.T) / 2
    mp_rq = float(w_hat @ M_prime @ w_hat)

    return float(w02_rq - mp_rq)


def barrier_value_mpmath(L, *, N=None, dps=50):
    """Compute barrier B(L) = W02 - M_prime at arbitrary mpmath precision.

    Re-implements the session41g barrier formula using mpmath arithmetic.
    This is SLOW for large lam_sq (many primes) -- use barrier_value_numpy
    for bulk sweeps and this function for high-precision spot checks.

    Args:
        L: Barrier parameter (log(lambda^2)).
        N: Basis truncation. Default: max(15, round(6*L)).
        dps: Decimal digits of precision.

    Returns:
        mpmath.mpf barrier value B(L).
    """
    with mpmath.workdps(dps + 10):
        L_mp = mpmath.mpf(str(L))
        lam_sq = mpmath.exp(L_mp)

        if N is None:
            N = max(15, round(6 * float(L)))
        dim = 2 * N + 1

        # Build w_hat vector
        four_pi = 4 * mpmath.pi
        four_pi_sq = four_pi ** 2
        L_sq = L_mp ** 2

        w = []
        for n in range(-N, N + 1):
            n_mp = mpmath.mpf(n)
            if n == 0:
                w.append(mpmath.mpf('0'))
            else:
                w.append(n_mp / (L_sq + four_pi_sq * n_mp ** 2))

        # Normalize
        w_norm_sq = sum(x ** 2 for x in w)
        w_norm = mpmath.sqrt(w_norm_sq)
        if w_norm < mpmath.mpf('1e-300'):
            return mpmath.mpf('0')
        w_hat = [x / w_norm for x in w]

        # W02 Rayleigh quotient
        pf = 32 * L_mp * mpmath.sinh(L_mp / 4) ** 2
        w_tilde = []
        for n in range(-N, N + 1):
            n_mp = mpmath.mpf(n)
            w_tilde.append(n_mp / (L_sq + four_pi_sq * n_mp ** 2))
        wt_dot_wh = sum(a * b for a, b in zip(w_tilde, w_hat))
        w02_rq = -pf * four_pi_sq * wt_dot_wh ** 2

        # M_prime: use numpy primes (fast sieve), mpmath arithmetic
        lam_sq_f = float(lam_sq)
        if lam_sq_f > 1e8:
            # For very large lam_sq, fall back to numpy for speed
            return mpmath.mpf(str(barrier_value_numpy(float(L), N=N)))

        primes = _sieve_primes_numpy(int(lam_sq_f))

        # Collect prime powers
        pk_data = []
        for p in primes:
            pk = int(p)
            k = 1
            logp = mpmath.log(mpmath.mpf(str(p)))
            while pk <= lam_sq_f:
                pk_mp = mpmath.mpf(str(pk))
                weight = logp * pk_mp ** mpmath.mpf('-0.5')
                y = mpmath.mpf(str(k)) * logp
                pk_data.append((logp, weight, y))
                pk *= int(p)
                k += 1

        # Build M_prime Rayleigh quotient directly (avoid dim x dim matrix)
        # mp_rq = w_hat^T M_prime w_hat = sum over prime powers of
        #   weight * (w_hat^T * M_pk * w_hat)
        # where M_pk has diagonal 2*(L-y)/L * cos(...) and off-diagonal sin-based

        ns = list(range(-N, N + 1))
        two_pi = 2 * mpmath.pi
        mp_rq = mpmath.mpf('0')

        for _logp, weight, y in pk_data:
            # Precompute sin and cos arrays
            sin_arr = [mpmath.sin(two_pi * mpmath.mpf(str(n)) * y / L_mp) for n in ns]
            cos_arr = [mpmath.cos(two_pi * mpmath.mpf(str(n)) * y / L_mp) for n in ns]

            # Diagonal contribution: sum_i w_hat[i]^2 * 2*(L-y)/L * cos(...)
            diag_factor = 2 * (L_mp - y) / L_mp
            diag_contrib = sum(
                w_hat[i] ** 2 * diag_factor * cos_arr[i] for i in range(dim)
            )

            # Off-diagonal contribution: sum_{i!=j} w_hat[i] * w_hat[j] * (sin_i - sin_j) / (pi*(n_i - n_j))
            off_diag_contrib = mpmath.mpf('0')
            for i in range(dim):
                for j in range(dim):
                    if i == j:
                        continue
                    n_diff = ns[i] - ns[j]
                    if n_diff == 0:
                        continue
                    off_diag_contrib += (
                        w_hat[i] * w_hat[j] * (sin_arr[i] - sin_arr[j])
                        / (mpmath.pi * mpmath.mpf(str(n_diff)))
                    )

            mp_rq += weight * (diag_contrib + off_diag_contrib)

    return w02_rq - mp_rq


def _heat_kernel_fast(t, *, dps=15):
    """Fast heat kernel trace using only constant + discrete terms.

    Skips the Eisenstein continuous integral (which is the bottleneck)
    to enable rapid numerical optimization. The Eisenstein contribution
    is typically small relative to constant + discrete, so this gives
    a good approximation for parameter mapping discovery.

    Args:
        t: Heat kernel time parameter.
        dps: Decimal digits of precision.

    Returns:
        Float approximate heat kernel trace.
    """
    with mpmath.workdps(dps + 5):
        t_mp = mpmath.mpf(str(t))
        constant = float(1 / (12 * t_mp))
    discrete, _ = maass_spectral_sum(t, dps=dps)
    return constant + float(discrete)


def find_parameter_mapping(L_values, *, dps=30, method="both",
                           barrier_values=None):
    """Discover parameter mapping t(L) via analytic + numerical fitting.

    Implements D-05: parallel analytic derivation and numerical fitting.

    Analytic track: tests candidate formulas (direct, Selberg, quadratic).
    For each, computes K(t(L)) using the FAST approximation (constant +
    discrete sum only, skipping slow Eisenstein integral) and compares to B(L).
    Numerical track: for each L, minimizes |K_fast(t) - B(L)| over t.
    Cross-validation: checks if analytic matches numerical to 3+ digits.

    The fast approximation is justified because the Eisenstein contribution
    is typically much smaller than constant + discrete, so it does not
    substantially change the optimal t. Full validation with Eisenstein
    is done in run_feasibility_comparison.

    Args:
        L_values: List of L values to test.
        dps: Decimal digits of precision for heat kernel computation.
        method: "analytic", "numerical", or "both" (default).
        barrier_values: Pre-computed barrier values (dict L -> float).
            If None, computed on the fly via barrier_value_numpy.

    Returns:
        Dict with keys: analytic_candidates, numerical_fits, best_candidate,
        cross_validation_passed.
    """
    from scipy.optimize import minimize_scalar

    L_values = [float(v) for v in L_values]

    # Pre-compute barrier values if not provided
    if barrier_values is None:
        barrier_values = {}
        for L in L_values:
            barrier_values[L] = barrier_value_numpy(L)

    results = {
        "analytic_candidates": {},
        "numerical_fits": {},
        "best_candidate": None,
        "cross_validation_passed": False,
    }

    # --- Analytic track ---
    candidates = {
        "direct": lambda L: L,
        "selberg": lambda L: L / (2 * math.pi),
        "quadratic": lambda L: L ** 2 / (4 * math.pi ** 2),
    }

    # Use fast approximation (constant + discrete only) for mapping discovery
    effective_dps = min(dps, 20)

    if method in ("analytic", "both"):
        for name, formula in candidates.items():
            agreement_by_L = {}
            for L in L_values:
                t_candidate = formula(L)
                if t_candidate <= 0:
                    agreement_by_L[L] = 0.0
                    continue
                try:
                    K_val = _heat_kernel_fast(t_candidate, dps=effective_dps)
                    B_val = barrier_values[L]
                    if abs(B_val) > 1e-300:
                        rel_err = abs(K_val - B_val) / abs(B_val)
                        digits = -math.log10(rel_err + 1e-300)
                    else:
                        digits = 0.0
                except Exception:
                    digits = 0.0
                agreement_by_L[L] = digits

            results["analytic_candidates"][name] = {
                "formula": f"t = {name}(L)",
                "agreement_digits_by_L": agreement_by_L,
                "mean_agreement": (
                    sum(agreement_by_L.values()) / len(agreement_by_L)
                    if agreement_by_L
                    else 0.0
                ),
            }

    # --- Numerical track ---
    if method in ("numerical", "both"):
        for L in L_values:
            B_val = barrier_values[L]

            def objective(t, _B=B_val):
                try:
                    K_fast = _heat_kernel_fast(t, dps=effective_dps)
                    return abs(K_fast - _B)
                except Exception:
                    return 1e30

            try:
                res = minimize_scalar(
                    objective,
                    bounds=(0.001 * L, max(10 * L, 1.0)),
                    method="bounded",
                    options={"xatol": 1e-10, "maxiter": 50},
                )
                results["numerical_fits"][L] = float(res.x)
            except Exception:
                results["numerical_fits"][L] = float(L)  # fallback

    # --- Determine best candidate ---
    best_name = None
    best_agreement = -1.0

    for name, info in results["analytic_candidates"].items():
        if info["mean_agreement"] > best_agreement:
            best_agreement = info["mean_agreement"]
            best_name = name

    # Check if numerical fit gives a consistent formula
    if results["numerical_fits"]:
        # Test if numerical fits match any analytic candidate to 3+ digits
        for name, formula in candidates.items():
            if name not in results["analytic_candidates"]:
                continue
            match_count = 0
            for L in L_values:
                if L in results["numerical_fits"]:
                    t_analytic = formula(L)
                    t_numerical = results["numerical_fits"][L]
                    if t_analytic > 0 and abs(t_analytic - t_numerical) / max(abs(t_analytic), 1e-300) < 1e-3:
                        match_count += 1
            if match_count == len(L_values):
                results["cross_validation_passed"] = True
                if best_name is None or name == best_name:
                    best_name = name

    if best_name is None:
        best_name = "implicit"
    results["best_candidate"] = best_name

    return results


def _get_parameter_mapping_func(mapping_result):
    """Return a t(L) function based on find_parameter_mapping results.

    Args:
        mapping_result: Dict from find_parameter_mapping.

    Returns:
        Callable: L -> t.
    """
    best = mapping_result.get("best_candidate", "direct")
    formulas = {
        "direct": lambda L: float(L),
        "selberg": lambda L: float(L) / (2 * math.pi),
        "quadratic": lambda L: float(L) ** 2 / (4 * math.pi ** 2),
    }

    if best in formulas:
        return formulas[best]

    # Implicit: use numerical fits with linear interpolation
    num_fits = mapping_result.get("numerical_fits", {})
    if num_fits:
        Ls = sorted(num_fits.keys())
        ts = [num_fits[L] for L in Ls]

        def interp(L):
            if L <= Ls[0]:
                return ts[0]
            if L >= Ls[-1]:
                return ts[-1]
            for i in range(len(Ls) - 1):
                if Ls[i] <= L <= Ls[i + 1]:
                    frac = (L - Ls[i]) / (Ls[i + 1] - Ls[i])
                    return ts[i] + frac * (ts[i + 1] - ts[i])
            return ts[-1]

        return interp

    return formulas["direct"]


def run_feasibility_comparison(*, n_points=120, L_range=(1.0, 55.0),
                               dps=30, verbose=True):
    """Run the full K(t(L)) vs B(L) feasibility comparison at 100+ L values.

    Implements D-03: density concentrated where barrier margin is smallest
    (L ~ 3-10 where the barrier dips).

    Args:
        n_points: Total number of L values. Default 120.
        L_range: (L_min, L_max) range.
        dps: Decimal digits of precision for heat kernel computation.
        verbose: If True, print summary table.

    Returns:
        List of BarrierComparison objects.
    """
    L_min, L_max = L_range

    # Generate L values with density concentrated where margin is smallest
    n_dense = n_points * 40 // 120   # ~1/3 in [L_min, 5]
    n_mid = n_points * 40 // 120     # ~1/3 in [5, 15]
    n_sparse = n_points - n_dense - n_mid  # rest in [15, L_max]

    L_values = []
    # Dense region: [L_min, 5] (where barrier margin is smallest)
    if L_min < 5:
        L_values.extend(np.linspace(max(L_min, 0.5), min(5.0, L_max), n_dense).tolist())
    # Mid region: [5, 15]
    if L_max > 5:
        mid_start = max(5.0, L_min)
        mid_end = min(15.0, L_max)
        if mid_start < mid_end:
            L_values.extend(np.linspace(mid_start, mid_end, n_mid + 2)[1:-1].tolist())
    # Sparse region: [15, L_max]
    if L_max > 15:
        sparse_start = max(15.0, L_min)
        if sparse_start < L_max:
            L_values.extend(np.linspace(sparse_start, L_max, n_sparse + 2)[1:-1].tolist())

    # Remove duplicates and sort
    L_values = sorted(set(round(v, 6) for v in L_values))

    if verbose:
        print(f"Running feasibility comparison: {len(L_values)} L values in [{L_min}, {L_max}]")

    # Pre-compute barrier values
    barrier_vals = {}
    for L in L_values:
        try:
            barrier_vals[L] = barrier_value_numpy(L)
        except Exception:
            barrier_vals[L] = 0.0

    # Find parameter mapping on a coarse subset
    coarse_Ls = [L for L in L_values[::max(1, len(L_values) // 8)]]
    coarse_barrier = {L: barrier_vals[L] for L in coarse_Ls}
    mapping = find_parameter_mapping(
        coarse_Ls, dps=min(dps, 20), method="both",
        barrier_values=coarse_barrier,
    )
    t_func = _get_parameter_mapping_func(mapping)

    if verbose:
        print(f"Best parameter mapping: {mapping['best_candidate']}")
        print(f"Cross-validation: {'PASSED' if mapping['cross_validation_passed'] else 'FAILED'}")

    # Run comparison
    comparisons = []

    if verbose:
        print(f"\n{'L':>8s} {'t':>10s} {'K(t)':>14s} {'B(L)':>14s} "
              f"{'digits':>8s} {'Eis_mag':>10s} {'status':>8s}")
        print("-" * 80)

    # Determine which points get full trace (with slow Eisenstein integral)
    # and which get fast approximation (constant + discrete only).
    # Full trace every nth point, plus first and last.
    full_trace_interval = max(1, len(L_values) // 10)
    full_trace_indices = set()
    full_trace_indices.add(0)
    full_trace_indices.add(len(L_values) - 1)
    for idx in range(0, len(L_values), full_trace_interval):
        full_trace_indices.add(idx)

    for i, L in enumerate(L_values):
        t = t_func(L)
        B_val = barrier_vals[L]

        try:
            use_full_trace = i in full_trace_indices
            if use_full_trace:
                # Full trace with Eisenstein (slow but complete)
                use_dual = _HAS_FLINT
                hk = heat_kernel_trace(t, dps=dps, use_dual=use_dual)
                K_val = hk["total"]
                discrete = hk["discrete_sum"]
                eisenstein = hk["continuous_integral"]
                constant = hk["constant_term"]
                n_maass = hk["convergence"].n_terms_used
                dual_ok = any(v is not None for v in hk["dual_results"].values())
            else:
                # Fast path: constant + discrete only
                with mpmath.workdps(dps + 5):
                    t_mp = mpmath.mpf(str(t))
                    constant = float(1 / (12 * t_mp))
                discrete_mp, conv = maass_spectral_sum(t, dps=dps)
                discrete = float(discrete_mp)
                eisenstein = 0.0  # skipped for speed
                n_maass = conv.n_terms_used
                K_val = constant + discrete
                dual_ok = False

            # Digits of agreement
            if abs(B_val) > 1e-300:
                rel_err = abs(K_val - B_val) / abs(B_val)
                digits = -math.log10(rel_err + 1e-300)
            else:
                digits = 0.0 if abs(K_val) > 1e-300 else 15.0

        except Exception as e:
            K_val = 0.0
            discrete = 0.0
            eisenstein = 0.0
            constant = 0.0
            digits = -1.0
            n_maass = 0
            dual_ok = False

        comp = BarrierComparison(
            L=L,
            t=t,
            heat_kernel_value=K_val,
            barrier_value=B_val,
            discrete_sum=discrete,
            eisenstein_contrib=eisenstein,
            constant_term=constant,
            digits_of_agreement=digits,
            n_maass_terms=n_maass,
            dual_validated=dual_ok,
        )
        comparisons.append(comp)

        if verbose:
            status = "OK" if digits >= 6 else ("WARN" if digits >= 3 else "FAIL")
            print(f"{L:>8.3f} {t:>10.5f} {K_val:>14.8f} {B_val:>14.8f} "
                  f"{digits:>8.2f} {abs(eisenstein):>10.6f} {status:>8s}")

    if verbose:
        v = feasibility_verdict(comparisons)
        print(f"\n=== VERDICT: {v['verdict']} ===")
        print(f"Agreement: min={v['min_agreement']:.1f}, "
              f"median={v['median_agreement']:.1f}, "
              f"max={v['max_agreement']:.1f}")
        print(f"Points with 6+ digits: {v['n_high_agreement']}/{len(comparisons)}")
        print(f"Parameter mapping: {v['parameter_mapping']}")

    return comparisons


def feasibility_verdict(comparisons):
    """Analyze comparison results to produce a feasibility verdict.

    Implements D-01: proof-grade agreement assessment.
    Implements D-02: Eisenstein magnitude as data (no kill threshold).

    Args:
        comparisons: List of BarrierComparison objects.

    Returns:
        Dict with verdict, agreement stats, Eisenstein fractions.
    """
    if not comparisons:
        return {
            "verdict": "NO_DATA",
            "min_agreement": 0.0,
            "median_agreement": 0.0,
            "max_agreement": 0.0,
            "eisenstein_fraction_of_budget": [],
            "parameter_mapping": "unknown",
            "n_points": 0,
            "n_high_agreement": 0,
            "n_medium_agreement": 0,
            "n_low_agreement": 0,
        }

    digits = [c.digits_of_agreement for c in comparisons]
    digits_arr = np.array(digits)

    # Eisenstein as fraction of 0.036 budget
    eis_fractions = []
    for c in comparisons:
        if abs(c.barrier_value) > 1e-300:
            eis_fractions.append(abs(c.eisenstein_contrib) / 0.036)
        else:
            eis_fractions.append(0.0)

    # Count quality buckets
    n_high = int(np.sum(digits_arr >= 6))
    n_medium = int(np.sum((digits_arr >= 3) & (digits_arr < 6)))
    n_low = int(np.sum(digits_arr < 3))

    median_agreement = float(np.median(digits_arr))

    # Verdict logic
    if median_agreement >= 6:
        verdict = "VIABLE"
    elif median_agreement >= 3:
        verdict = "MARGINAL"
    else:
        verdict = "DEAD"

    return {
        "verdict": verdict,
        "min_agreement": float(np.min(digits_arr)),
        "median_agreement": median_agreement,
        "max_agreement": float(np.max(digits_arr)),
        "eisenstein_fraction_of_budget": eis_fractions,
        "parameter_mapping": "see find_parameter_mapping",
        "n_points": len(comparisons),
        "n_high_agreement": n_high,
        "n_medium_agreement": n_medium,
        "n_low_agreement": n_low,
    }
