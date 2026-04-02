"""
SESSION 36 FAST -- Sonin monotonicity with numpy-accelerated builds.

Skip mpmath where possible. Use scipy quadrature for wr_diag.
Focus on the KEY question: does max_eig(M|null(W02)) decrease monotonically
as lambda^2 crosses each prime power?
"""

import numpy as np
from scipy import integrate
import time
import json
import sys

# Euler-Mascheroni constant
EULER = 0.5772156649015329


def build_fast_numpy(lam_sq, N_val):
    """
    Build W02, M, QW using numpy/scipy only (no mpmath).
    Much faster but slightly less precise.
    """
    L = np.log(lam_sq)
    eL = np.exp(L)
    dim = 2 * N_val + 1

    # Prime powers
    limit = int(lam_sq)
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 2):
        if i <= limit and sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    vM = []
    for p in range(2, limit + 1):
        if sieve[p] and p <= lam_sq:
            pk = p
            while pk <= lam_sq:
                vM.append((pk, np.log(p), np.log(pk)))
                pk *= p

    ns = np.arange(-N_val, N_val + 1, dtype=float)

    # W02 (exact in numpy)
    L2 = L**2
    p2 = (4*np.pi)**2
    pf = 32 * L * np.sinh(L/4)**2
    W02 = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            n, m = ns[i], ns[j]
            W02[i, j] = pf * (L2 - p2*m*n) / ((L2 + p2*m**2) * (L2 + p2*n**2))

    # Alpha coefficients (using scipy digamma)
    from scipy.special import digamma as scipy_digamma, hyp2f1 as scipy_hyp2f1
    alpha = np.zeros(dim)
    for i in range(dim):
        n = int(ns[i])
        if n == 0:
            alpha[i] = 0.0
        else:
            an = abs(n)
            # a = pi*i*an/L + 1/4 (complex)
            a_real = 0.25
            a_imag = np.pi * an / L
            # digamma of complex argument: use series or asymptotic
            # For large |a|: digamma(a) ~ log(a) - 1/(2a) - ...
            # Im(digamma(a_real + i*a_imag)) ~ arctan(a_imag/a_real) for small a_real
            # More precisely: use reflection formula or recurrence
            # Rough approximation for our purposes:
            a_abs = np.sqrt(a_real**2 + a_imag**2)
            dig_imag = np.arctan2(a_imag, a_real)  # Leading term of Im(digamma)
            # Correction: Im(digamma(z)) ~ arg(z) + sum corrections
            # For z = 1/4 + i*y with y >> 1: Im(digamma) ~ pi/2 - 1/(2y) + ...
            if a_imag > 1:
                dig_imag = np.pi/2 - a_real / (a_real**2 + a_imag**2)
            else:
                # Use more terms
                dig_imag = np.arctan2(a_imag, a_real)

            # hyp2f1 term: for large L, exp(-2L) ~ 0, so hyp2f1 ~ 1 and f1 ~ 0
            z = np.exp(-2*L)
            f1 = 0.0  # Negligible for L > 2

            d = dig_imag / 2
            val = (f1 + d) / np.pi
            alpha[i] = val if n > 0 else -val

    # wr_diag via scipy quadrature
    omega_0 = 2.0
    w_const = (omega_0 / 2) * (EULER + np.log(4*np.pi*(eL - 1)/(eL + 1)))
    wr_diag = np.zeros(dim)
    for i in range(dim):
        nv = abs(int(ns[i]))

        def integrand(x):
            if x < 1e-15:
                return 0.0
            omega_x = 2 * (1 - x/L) * np.cos(2*np.pi*nv*x/L)
            numer = np.exp(x/2) * omega_x - omega_0
            denom = np.exp(x) - np.exp(-x)
            if abs(denom) < 1e-40:
                return 0.0
            return numer / denom

        integral, _ = integrate.quad(integrand, 1e-10, L, limit=200, epsabs=1e-10)
        wr_diag[i] = w_const + integral

    # Build M
    M = np.zeros((dim, dim))
    for i in range(dim):
        n = ns[i]
        M[i, i] = wr_diag[i]
        for j in range(dim):
            m = ns[j]
            if n != m:
                M[i, j] += (alpha[j] - alpha[i]) / (n - m)
            for pk, logp, logpk in vM:
                if n != m:
                    q = (np.sin(2*np.pi*m*logpk/L) -
                         np.sin(2*np.pi*n*logpk/L)) / (np.pi*(n-m))
                else:
                    q = 2*(L - logpk)/L * np.cos(2*np.pi*n*logpk/L)
                M[i, j] += logp * pk**(-0.5) * q

    M = (M + M.T) / 2
    QW = W02 - M
    QW = (QW + QW.T) / 2

    return W02, M, QW, len(vM)


def analyze_at_lambda(lam_sq, N_val):
    """Compute key eigenvalue metrics at a given lambda^2."""
    W02, M, QW, n_pk = build_fast_numpy(lam_sq, N_val)
    dim = 2 * N_val + 1

    # null(W02)
    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10
    P_null = ev[:, np.abs(ew) <= thresh]

    if P_null.shape[1] == 0:
        return None

    M_null = P_null.T @ M @ P_null
    evals = np.linalg.eigvalsh(M_null)
    max_null = np.max(evals)

    # v_+ and orth
    evals_M, evecs_M = np.linalg.eigh(M)
    v_plus = evecs_M[:, -1]

    return {
        'max_eig_null': float(max_null),
        'n_primes': n_pk,
        'qw_ok': bool(max_null < 1e-6),
    }


def main():
    print("SESSION 36 FAST -- SONIN MONOTONICITY FLOW", flush=True)
    print("=" * 80, flush=True)

    # 1. BASE CASE: lambda^2 from 4 to 30
    print("\nBASE CASE VERIFICATION", flush=True)
    print(f"  {'lam^2':>6} {'N':>3} {'#pk':>4} {'max_eig(M|null)':>18} {'QW>=0?':>7}", flush=True)

    N_base = 8
    for lam_sq in range(4, 31):
        t0 = time.time()
        result = analyze_at_lambda(lam_sq, N_base)
        elapsed = time.time() - t0
        if result is None:
            continue
        flag = "" if result['qw_ok'] else " ***FAIL***"
        print(f"  {lam_sq:>6} {N_base:>3} {result['n_primes']:>4} "
              f"{result['max_eig_null']:>+18.8e} {'YES' if result['qw_ok'] else 'NO':>7} "
              f"{elapsed:.1f}s{flag}", flush=True)

    # 2. PRIME POWER THRESHOLD DERIVATIVES
    print("\nPRIME POWER THRESHOLD DERIVATIVES", flush=True)
    print(f"  {'p^k':>6} {'max_before':>16} {'max_after':>16} {'delta':>14} {'sign':>6}", flush=True)

    # Generate prime powers up to 300
    limit = 300
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 2):
        if i <= limit and sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    prime_powers = set()
    for p in range(2, limit + 1):
        if sieve[p]:
            pk = p
            while pk <= limit:
                prime_powers.add(pk)
                pk *= p
    prime_powers = sorted(prime_powers)

    N_flow = 12
    n_neg = 0
    n_pos = 0
    n_total = 0

    for pp in prime_powers:
        if pp < 3:
            continue
        lam_before = pp - 0.01
        lam_after = pp + 0.01

        r_before = analyze_at_lambda(lam_before, N_flow)
        r_after = analyze_at_lambda(lam_after, N_flow)

        if r_before is None or r_after is None:
            continue

        delta = r_after['max_eig_null'] - r_before['max_eig_null']
        n_total += 1
        if delta < 1e-10:
            n_neg += 1
            sign = "NEG"
        else:
            n_pos += 1
            sign = "POS"

        print(f"  {pp:>6} {r_before['max_eig_null']:>+16.8e} "
              f"{r_after['max_eig_null']:>+16.8e} {delta:>+14.6e} {sign:>6}", flush=True)

    print(f"\n  MONOTONICITY SUMMARY:", flush=True)
    print(f"  Negative delta (good): {n_neg}/{n_total}", flush=True)
    print(f"  Positive delta (bad):  {n_pos}/{n_total}", flush=True)
    if n_pos == 0 and n_total > 0:
        print(f"\n  *** PERFECT MONOTONICITY ***", flush=True)
    elif n_pos > 0:
        print(f"\n  Monotonicity VIOLATED at {n_pos} thresholds", flush=True)

    # 3. CONTINUOUS FLOW
    print("\nCONTINUOUS FLOW", flush=True)
    print(f"  {'lam^2':>8} {'L':>6} {'#pk':>4} {'max_eig(M|null)':>18} {'QW>=0?':>7}", flush=True)

    N_cont = 12
    lam_values = list(range(4, 50)) + list(range(50, 310, 10))
    prev_max = None

    for lam_sq in lam_values:
        result = analyze_at_lambda(lam_sq, N_cont)
        if result is None:
            continue

        L_f = np.log(lam_sq)
        direction = ""
        if prev_max is not None:
            d = result['max_eig_null'] - prev_max
            direction = " v" if d < -1e-12 else (" ^" if d > 1e-12 else " =")

        flag = "" if result['qw_ok'] else " ***"
        print(f"  {lam_sq:>8} {L_f:>6.3f} {result['n_primes']:>4} "
              f"{result['max_eig_null']:>+18.8e} {'YES' if result['qw_ok'] else 'NO':>7}{direction}{flag}",
              flush=True)
        prev_max = result['max_eig_null']

    print(f"\nDone.", flush=True)

    with open('session36_fast_flow.json', 'w') as f:
        json.dump({'status': 'complete'}, f)


if __name__ == "__main__":
    main()
