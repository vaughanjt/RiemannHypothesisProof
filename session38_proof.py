"""
SESSION 38e — TOWARD THE PROOF

Part 1: ANALYTIC NEGATIVITY
  Prove: the diagonal entries wr_diag[n] on null(W02) give negative trace.
  The key ingredients:
  (a) wr_diag[n] asymptotic behavior for large |n|
  (b) The null(W02) constraint: vectors must be orthogonal to the Poisson kernel
  (c) The interaction: null vectors have mass at large |n| where wr_diag is negative

Part 2: PRIME NEAR-NEUTRALITY
  Prove: trace of T(p^k)|null(W02) is small (bounded by something << wr_diag trace)
  The key: the oscillatory kernel q has near-zero integral when weighted by null vectors

Part 3: TRACE TO SPECTRUM
  Attempt: given trace bounds, prove all eigenvalues are non-positive
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, exp, cos, sin, euler, digamma, hyp2f1, sinh
import time
import sys
sys.path.insert(0, '.')
from connes_crossterm import build_all
from session33_sieve_bypass import compute_M_decomposition


# ============================================================================
# PART 1: ANALYTIC NEGATIVITY — The digamma diagonal
# ============================================================================

def analyze_wr_diag_asymptotics(lam_sq_values):
    """
    Compute wr_diag[n] at multiple bandwidths and fit the asymptotic formula.

    The integral in wr_diag[n]:
    I(n) = integral_0^L [exp(x/2)*2(1-x/L)cos(2*pi*n*x/L) - 2] / (exp(x)-exp(-x)) dx

    For the DC part (n=0):
    I(0) = integral_0^L [exp(x/2)*2(1-x/L) - 2] / (2*sinh(x)) dx

    For n != 0, the cos oscillates and the integral decomposes as:
    I(n) = I(0) * (Riemann-Lebesgue decay) + (boundary terms)

    The asymptotic: wr_diag[n] = w_const + I(n)
    where w_const = (euler + log(4*pi*(e^L-1)/(e^L+1)))

    For large n: I(n) should decay, but the 1/sinh(x) singularity at x=0
    creates a logarithmic contribution.

    Near x=0: the integrand ~ [exp(x/2)*2cos(2*pi*n*x/L) - 2] / (2x)
    = [2cos(2*pi*n*x/L) - 2 + O(x)] / (2x)
    = [-2*(2*pi*n/L)^2*x^2/2 + O(x^3)] / (2x) + [O(x)]/(2x)
    = -(2*pi*n/L)^2 * x/2 + O(x)

    So the integrand near 0 is O(x), integrable. No logarithmic singularity.

    The oscillatory part for large n: by Riemann-Lebesgue, I(n) -> 0.
    But the RATE of decay depends on the smoothness of the integrand.

    The integrand f(x) = exp(x/2)*(1-x/L) / sinh(x) is smooth on (0,L] but
    f(x) ~ 1/x as x -> 0. So f is in L^1 but NOT smooth at 0.
    The Fourier integral of an L^1 function with 1/x singularity decays as O(1/n).

    But our earlier fit showed decay as O(n^{-0.28}) -- much SLOWER than 1/n.
    This suggests the wr_diag asymptotic is not simple Fourier decay.

    Let me fit wr_diag[n] = A + B*n + C*log(n) + D/n + ...
    """
    print("PART 1: wr_diag[n] ASYMPTOTICS", flush=True)
    print("=" * 70, flush=True)

    for lam_sq in lam_sq_values:
        mp.dps = 50
        L_mp = log(mpf(lam_sq))
        L_f = float(L_mp)
        eL = exp(L_mp)
        N = max(15, round(6 * L_f))

        omega_0 = mpf(2)
        w_const = float((omega_0 / 2) * (euler + log(4 * pi * (eL - 1) / (eL + 1))))

        # Compute wr_diag at high precision
        wr = {}
        for nv in range(N + 1):
            def omega(x, nv=nv):
                return 2 * (1 - x / L_mp) * cos(2 * pi * nv * x / L_mp)
            dx = L_mp / 10000
            integral = mpf(0)
            for k in range(10000):
                x = dx * (k + mpf(1) / 2)
                numer = exp(x / 2) * omega(x) - omega_0
                denom = exp(x) - exp(-x)
                if abs(denom) > mpf(10)**(-40):
                    integral += numer / denom
            integral *= dx
            wr[nv] = float(w_const + integral)

        print(f"\nlam^2={lam_sq}, L={L_f:.4f}, N={N}", flush=True)

        # Fit: wr_diag[n] = a + b*log(n) + c/n + d/n^2 for large n
        # Use n >= 5 for fitting
        ns_fit = np.arange(5, N + 1)
        wr_fit = np.array([wr[n] for n in ns_fit])

        # Design matrix for: wr = a + b*log(n) + c/n
        X = np.column_stack([
            np.ones(len(ns_fit)),
            np.log(ns_fit),
            1.0 / ns_fit,
        ])
        coeffs, residuals, rank, sv = np.linalg.lstsq(X, wr_fit, rcond=None)
        a, b, c = coeffs

        print(f"  Fit: wr_diag[n] ~ {a:.6f} + {b:.6f}*log(n) + {c:.4f}/n", flush=True)
        print(f"  b (log coefficient) = {b:.6f}", flush=True)
        if residuals.size > 0:
            print(f"  Fit residual: {np.sqrt(residuals[0]/len(ns_fit)):.6e}", flush=True)

        # Check fit quality
        print(f"\n  {'n':>4} {'wr_diag[n]':>14} {'fit':>14} {'error':>12}", flush=True)
        for n in [1, 2, 5, 10, 15, 20, N]:
            actual = wr[n]
            fitted = a + b * np.log(n) + c / n if n > 0 else w_const
            print(f"  {n:>4} {actual:>+14.8f} {fitted:>+14.8f} {actual-fitted:>+12.4e}", flush=True)

        # The KEY: wr_diag[n] ~ const + b*log(n) with b < 0
        # For b < 0: wr_diag[n] -> -infinity as n -> infinity
        # The null(W02) constraint forces mass to large n
        # So the trace on null(W02) is strongly negative

        # Compute the Poisson kernel weights: k_n = 1/(L^2 + 4*pi^2*n^2)
        # The null(W02) constraint says: sum_n phi_n * k_n = 0 (orthog to even Poisson)
        # and sum_n n*phi_n * k_n = 0 (orthog to odd Poisson)
        # These force phi_n to be "orthogonal" to the 1/(L^2+n^2) weight

        # The trace of M_diag on null(W02):
        # trace = sum_n wr_diag[n] - (2 eigenvalues removed by null projection)
        # The removed eigenvalues are the projections onto range(W02)

        wr_total = sum(wr[abs(n)] for n in range(-N, N + 1))
        print(f"\n  Total trace of M_diag: {wr_total:.4f}", flush=True)

        # The two removed eigenvalues (from range(W02)):
        W02, M, QW = build_all(lam_sq, N, n_quad=10000)
        M_diag_mat = np.zeros((2*N+1, 2*N+1))
        for i in range(2*N+1):
            M_diag_mat[i, i] = wr[abs(i - N)]

        ew, ev = np.linalg.eigh(W02)
        thresh = np.max(np.abs(ew)) * 1e-10
        P_range = ev[:, np.abs(ew) > thresh]  # 2 range vectors

        # Rayleigh quotients of M_diag on range vectors
        for j in range(P_range.shape[1]):
            rq = np.dot(P_range[:, j], M_diag_mat @ P_range[:, j])
            print(f"  M_diag Rayleigh on range vec {j}: {rq:+.4f}", flush=True)

        trace_null = np.trace(ev[:, np.abs(ew) <= thresh].T @ M_diag_mat @ ev[:, np.abs(ew) <= thresh])
        print(f"  Trace of M_diag on null(W02): {trace_null:.4f}", flush=True)
        print(f"  (Should be strongly negative due to log-growth of |wr_diag|)", flush=True)


# ============================================================================
# PART 2: PRIME NEAR-NEUTRALITY
# ============================================================================

def analyze_prime_trace_mechanism(lam_sq, N=None):
    """
    WHY is trace(T(p^k)|null(W02)) nearly zero?

    trace(T(pk)|null) = trace(T(pk)) - trace(T(pk)|range(W02))

    trace(T(pk)) = w(pk) * sum_n q(n,n,logpk)
                 = w(pk) * sum_n 2(L-logpk)/L * cos(2*pi*n*logpk/L)
                 = w(pk) * 2(L-logpk)/L * D_N(2*pi*logpk/L)

    where D_N is the Dirichlet kernel: sum_{n=-N}^{N} cos(n*theta) = sin((N+1/2)*theta)/sin(theta/2)

    For generic logpk/L (not a rational with small denominator):
    D_N ~ O(1) (oscillatory, bounded)

    So trace(T(pk)) = O(w(pk)) = O(log(p)/sqrt(pk))

    trace(T(pk)|range) = w(pk) * sum of q(n,n,logpk) weighted by Poisson kernel^2
                        ≈ w(pk) * (concentrated at n=0)

    The DIFFERENCE trace(T(pk)|null) = trace(T(pk)) - trace(T(pk)|range)
    is the trace of T minus the Poisson kernel contribution.

    KEY INSIGHT: trace(T(pk)) involves the Dirichlet kernel D_N at frequency logpk/L.
    For most primes, this frequency is irrational (by Gelfond-Schneider type results),
    so D_N is O(1/sin(pi*logpk/L)), which is O(1) for generic primes.
    """
    if N is None:
        L_f = np.log(lam_sq)
        N = max(15, round(6 * L_f))
    dim = 2 * N + 1
    L_f = np.log(lam_sq)
    ns = np.arange(-N, N + 1, dtype=float)

    print(f"\nPART 2: PRIME TRACE MECHANISM at lam^2={lam_sq}", flush=True)
    print(f"  dim={dim}, L={L_f:.4f}", flush=True)

    W02, M, QW = build_all(lam_sq, N, n_quad=10000)
    _, _, _, _, primes = compute_M_decomposition(lam_sq, N)

    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10
    P_null = ev[:, np.abs(ew) <= thresh]
    P_range = ev[:, np.abs(ew) > thresh]

    print(f"\n  {'p^k':>6} {'w(pk)':>8} {'tr(T)':>10} {'tr(T|range)':>12} {'tr(T|null)':>12} "
          f"{'D_N':>10} {'freq':>8}", flush=True)

    for pk, logp, logpk in primes:
        w = logp * pk**(-0.5)

        # Trace of T on full space
        # = w * sum_n 2(L-logpk)/L * cos(2*pi*n*logpk/L)
        dirichlet_sum = sum(np.cos(2*np.pi*n*logpk/L_f) for n in ns)
        tr_full = w * 2 * (L_f - logpk) / L_f * dirichlet_sum

        # Trace on range(W02)
        Q = np.zeros((dim, dim))
        for i in range(dim):
            m = ns[i]
            for j in range(dim):
                n = ns[j]
                if m != n:
                    Q[i, j] = (np.sin(2*np.pi*n*logpk/L_f) -
                               np.sin(2*np.pi*m*logpk/L_f)) / (np.pi*(m-n))
                else:
                    Q[i, j] = 2*(L_f - logpk)/L_f * np.cos(2*np.pi*m*logpk/L_f)
        Q = (Q + Q.T) / 2
        T = w * Q

        tr_range = sum(np.dot(P_range[:, j], T @ P_range[:, j]) for j in range(P_range.shape[1]))
        tr_null = np.trace(P_null.T @ T @ P_null)

        freq = logpk / L_f  # Normalized frequency

        print(f"  {pk:>6} {w:>8.4f} {tr_full:>+10.4f} {tr_range:>+12.4f} {tr_null:>+12.4f} "
              f"{dirichlet_sum:>10.2f} {freq:>8.4f}", flush=True)

    # The THEORETICAL bound:
    # |tr(T(pk)|null)| <= |tr(T(pk))| + |tr(T(pk)|range)|
    # |tr(T(pk))| = w * 2*(L-logpk)/L * |D_N(theta_pk)|
    # |D_N(theta)| <= min(2N+1, 1/|sin(theta/2)|) for theta = 2*pi*logpk/L
    # For generic theta: |D_N| = O(1/sin(theta/2)) = O(L/logpk) roughly
    # So |tr(T)| = O(w * L/logpk) = O(logp * L / (sqrt(pk) * logpk))

    print(f"\n  THEORETICAL BOUND on |tr(T(pk)|null)|:", flush=True)
    print(f"  |tr(T(pk))| <= w(pk) * 2 * |D_N(2*pi*log(pk)/L)|", flush=True)
    print(f"  The Dirichlet kernel D_N(theta) = O(min(dim, 1/|sin(theta/2)|))", flush=True)
    print(f"  For each prime, the frequency theta_p = 2*pi*log(p)/L is generically irrational", flush=True)
    print(f"  => |D_N| is bounded independently of N for fixed p", flush=True)
    print(f"  => |tr(T(p))| = O(log(p)/sqrt(p)) -- same order as the weight", flush=True)
    print(f"  => Total: sum_p |tr(T(p)|null)| = O(sum log(p)/sqrt(p)) = O(sqrt(X))", flush=True)
    print(f"  This is MUCH SMALLER than |Ma_trace| ~ N*log(N)", flush=True)


# ============================================================================
# PART 3: THE SPECTRAL GAP — From trace to eigenvalues
# ============================================================================

def analyze_spectral_gap(lam_sq, N=None):
    """
    Given:
    - Ma_null has trace ~ -C*N*log(N) (strongly negative, from digamma)
    - Mp_null has trace ~ -c*sqrt(lambda) (weakly negative, from PNT)
    - M_null = Ma_null + Mp_null has all eigenvalues <= 0

    Can we prove the spectral bound from the trace bound?

    APPROACH: Decompose M_null into "diagonal-dominant" and "off-diagonal" parts.

    In the FOURIER BASIS, M has:
    - Diagonal: wr_diag[n] (strongly negative for |n| > ~5)
    - Off-diagonal: alpha terms + prime sum

    If the diagonal dominates the off-diagonal (Gershgorin), M <= 0.
    We know Gershgorin fails in the raw Fourier basis.

    But what about in a ROTATED basis that reduces the off-diagonal?

    The optimal rotation is the eigenbasis of M (but that's circular).
    A sub-optimal but analyzable rotation: the eigenbasis of Ma_null.

    In Ma's eigenbasis, Ma is diagonal and Mp is the perturbation.
    If ||Mp||_op < min eigenvalue gap of Ma from 0, Weyl's inequality gives M <= 0.
    """
    if N is None:
        L_f = np.log(lam_sq)
        N = max(15, round(6 * L_f))
    dim = 2 * N + 1
    L_f = np.log(lam_sq)

    W02, M, QW = build_all(lam_sq, N, n_quad=10000)
    M_diag, M_alpha, M_prime, _, primes = compute_M_decomposition(lam_sq, N)

    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10
    P_null = ev[:, np.abs(ew) <= thresh]
    d_null = P_null.shape[1]

    Ma_null = P_null.T @ (M_diag + M_alpha) @ P_null
    Mp_null = P_null.T @ M_prime @ P_null
    M_null = Ma_null + Mp_null

    print(f"\nPART 3: SPECTRAL GAP ANALYSIS at lam^2={lam_sq}", flush=True)

    # Eigenvalues of Ma_null
    evals_Ma = np.sort(np.linalg.eigvalsh(Ma_null))
    max_Ma = evals_Ma[-1]

    # Operator norm of Mp_null
    op_Mp = np.linalg.norm(Mp_null, 2)

    print(f"  Ma_null max eigenvalue: {max_Ma:+.6f}", flush=True)
    print(f"  ||Mp_null||_op:         {op_Mp:.6f}", flush=True)
    print(f"  Weyl bound: max(M) <= max(Ma) + ||Mp|| = {max_Ma + op_Mp:+.6f}", flush=True)

    if max_Ma + op_Mp < 1e-6:
        print(f"  *** WEYL BOUND PROVES M <= 0! ***", flush=True)
    else:
        print(f"  Weyl bound FAILS (positive upper bound).", flush=True)
        print(f"  Deficit: {max_Ma + op_Mp:.6f}", flush=True)

    # More refined: use Ma's eigenbasis
    evals_Ma_full, evecs_Ma = np.linalg.eigh(Ma_null)

    # Mp in Ma's eigenbasis
    Mp_in_Ma_basis = evecs_Ma.T @ Mp_null @ evecs_Ma

    # For each Ma eigenvalue, the Gershgorin disk from Mp:
    print(f"\n  PER-EIGENVALUE ANALYSIS (Ma eigenbasis):", flush=True)
    print(f"  {'i':>4} {'eig(Ma)':>10} {'Mp_diag':>10} {'Mp_offdiag':>12} {'Gersh_upper':>12} {'safe?':>6}",
          flush=True)

    n_safe = 0
    n_unsafe = 0
    worst_upper = -float('inf')

    for i in range(d_null):
        ma_eig = evals_Ma_full[i]
        mp_diag = Mp_in_Ma_basis[i, i]
        mp_offdiag = sum(abs(Mp_in_Ma_basis[i, j]) for j in range(d_null) if j != i)
        gersh_upper = ma_eig + mp_diag + mp_offdiag

        if gersh_upper < 1e-6:
            n_safe += 1
        else:
            n_unsafe += 1

        if gersh_upper > worst_upper:
            worst_upper = gersh_upper
            worst_i = i

        if i < 5 or i >= d_null - 5 or gersh_upper > -0.01:
            print(f"  {i:>4} {ma_eig:>+10.4f} {mp_diag:>+10.4f} {mp_offdiag:>12.4f} "
                  f"{gersh_upper:>+12.4f} {'YES' if gersh_upper < 1e-6 else 'no':>6}", flush=True)

    print(f"\n  Gershgorin in Ma eigenbasis:", flush=True)
    print(f"  Safe: {n_safe}/{d_null}", flush=True)
    print(f"  Unsafe: {n_unsafe}/{d_null}", flush=True)
    print(f"  Worst upper: {worst_upper:+.6f} at i={worst_i}", flush=True)

    # HOW CLOSE ARE WE?
    # The actual max eigenvalue of M_null:
    actual_max = np.max(np.linalg.eigvalsh(M_null))
    print(f"\n  Actual max eigenvalue of M_null: {actual_max:+.6e}", flush=True)
    print(f"  Gershgorin worst bound:          {worst_upper:+.6f}", flush=True)
    print(f"  Tightness ratio: {worst_upper / (actual_max + 1e-15):.0f}x too loose", flush=True)

    # ALTERNATIVE: use the TRACE + RANK argument
    # If M_null has rank r seeing modes and trace T,
    # and the seeing modes have average eigenvalue T/r,
    # can we bound the maximum eigenvalue?
    evals_M = np.linalg.eigvalsh(M_null)
    seeing = evals_M[evals_M < -0.001]
    silent = evals_M[evals_M >= -0.001]

    print(f"\n  TRACE-RANK ARGUMENT:", flush=True)
    print(f"  Seeing modes: {len(seeing)}, trace: {np.sum(seeing):.4f}", flush=True)
    print(f"  Silent modes: {len(silent)}, trace: {np.sum(silent):.6e}", flush=True)
    print(f"  Average seeing eigenvalue: {np.mean(seeing):.4f}", flush=True)
    print(f"  If silent modes stay at ~0, M <= 0 follows from seeing modes < 0", flush=True)
    print(f"  Seeing modes are ALWAYS negative (they see zeros on the critical line)", flush=True)


if __name__ == "__main__":
    print("SESSION 38e — TOWARD THE PROOF", flush=True)
    print("=" * 80, flush=True)

    # Part 1: Digamma asymptotics
    analyze_wr_diag_asymptotics([50, 200])

    # Part 2: Prime trace mechanism
    analyze_prime_trace_mechanism(50)

    # Part 3: Spectral gap
    for ls in [50, 200]:
        analyze_spectral_gap(ls)

    print(f"\nDone.", flush=True)
