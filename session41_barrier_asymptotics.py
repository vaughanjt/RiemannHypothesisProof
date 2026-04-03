"""
SESSION 41 — BARRIER CONSTANT ASYMPTOTICS

Goal: Prove that the barrier QW_barrier = <w, QW, w> / ||w||^2 > c > 0
for all lambda^2, where w is the odd eigenvector of W02.

Strategy:
1. Compute all 4 barrier components at many lambda^2 values
2. Derive closed-form expressions for each component
3. Prove the cancellation analytically

The barrier decomposes as:
    QW_barrier = <w, W02-M, w> / ||w||^2
               = <w, W02, w> - <w, M_prime, w> - <w, M_diag, w> - <w, M_alpha, w>
    all divided by ||w||^2.

Key quantities (a = L/(4*pi), L = log(lam^2)):
    w[n] = n / (L^2 + (4*pi)^2 * n^2) = n / ((4*pi)^2 * (a^2 + n^2))

Usage:
    python session41_barrier_asymptotics.py
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, euler, exp, cos, sin, hyp2f1, digamma, sinh, coth
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from connes_crossterm import build_all
from session33_sieve_bypass import compute_M_decomposition


# ═══════════════════════════════════════════════════════════════
# PART 1: NUMERICAL BARRIER SWEEP
# ═══════════════════════════════════════════════════════════════

def compute_barrier_components(lam_sq, N=None, n_quad=10000):
    """
    Compute the 4 components of the barrier on the odd eigenvector w of W02.

    Returns dict with:
        w02_term:   <w, W02, w> / ||w||^2
        mprime_term: <w, M_prime, w> / ||w||^2
        mdiag_term:  <w, M_diag, w> / ||w||^2
        malpha_term: <w, M_alpha, w> / ||w||^2
        barrier:     w02_term - mprime_term - mdiag_term - malpha_term
    """
    L_f = np.log(lam_sq)
    if N is None:
        N = max(15, round(6 * L_f))
    dim = 2 * N + 1

    # Build all matrices
    W02, M, QW = build_all(lam_sq, N, n_quad=n_quad)
    M_diag, M_alpha, M_prime, M_full, vM = compute_M_decomposition(lam_sq, N, n_quad=n_quad)

    # Find the odd eigenvector of W02
    # W02 has rank 2: one even eigenvalue and one odd eigenvalue
    ew, ev = np.linalg.eigh(W02)
    # Non-zero eigenvalues
    nz_idx = np.where(np.abs(ew) > np.max(np.abs(ew)) * 1e-10)[0]

    center = N  # index of n=0
    ns = np.arange(-N, N + 1, dtype=float)

    # Identify even/odd eigenvectors
    w_vec = None
    u_vec = None
    w_eval = None
    u_eval = None

    for idx in nz_idx:
        v = ev[:, idx]
        # Check parity: even if v[center+k] ≈ v[center-k], odd if v[center+k] ≈ -v[center-k]
        even_err = sum(abs(v[center + k] - v[center - k]) for k in range(1, N + 1))
        odd_err = sum(abs(v[center + k] + v[center - k]) for k in range(1, N + 1))
        if odd_err < even_err:
            w_vec = v
            w_eval = ew[idx]
        else:
            u_vec = v
            u_eval = ew[idx]

    if w_vec is None:
        raise ValueError(f"Could not find odd eigenvector at lam_sq={lam_sq}")

    # Normalize
    w_norm_sq = np.dot(w_vec, w_vec)
    w_hat = w_vec / np.sqrt(w_norm_sq)

    # Compute Rayleigh quotients
    w02_term = w_hat @ W02 @ w_hat
    mprime_term = w_hat @ M_prime @ w_hat
    mdiag_term = w_hat @ M_diag @ w_hat
    malpha_term = w_hat @ M_alpha @ w_hat
    barrier = w_hat @ QW @ w_hat  # = w02 - mprime - mdiag - malpha

    # Also compute on u direction for comparison
    u_norm_sq = np.dot(u_vec, u_vec) if u_vec is not None else 1.0
    u_hat = u_vec / np.sqrt(u_norm_sq) if u_vec is not None else None

    u_barrier = u_hat @ QW @ u_hat if u_hat is not None else None

    # Full QW minimum eigenvalue
    eps_0 = np.linalg.eigvalsh(QW)[0]

    return {
        'lam_sq': lam_sq,
        'L': L_f,
        'N': N,
        'dim': dim,
        'w_eval': w_eval,
        'u_eval': u_eval,
        'w02_term': w02_term,
        'mprime_term': mprime_term,
        'mdiag_term': mdiag_term,
        'malpha_term': malpha_term,
        'barrier': barrier,
        'barrier_check': w02_term - mprime_term - mdiag_term - malpha_term,
        'u_barrier': u_barrier,
        'eps_0': eps_0,
        'w_norm_sq': w_norm_sq,
    }


# ═══════════════════════════════════════════════════════════════
# PART 2: ANALYTICAL FORMULAS
# ═══════════════════════════════════════════════════════════════

def analytical_w02_term(lam_sq):
    """
    Closed-form for <w_hat, W02, w_hat>.

    W02 = pf * (L^2 * |u><u| - (4pi)^2 * |w><w|)
    where u[n] = 1/(L^2 + (4pi)^2 n^2), w[n] = n/(L^2 + (4pi)^2 n^2).

    Since <u, w> = 0 (odd function):
    <w, W02, w> = -pf * (4pi)^2 * ||w||^4
    <w_hat, W02, w_hat> = -pf * (4pi)^2 * ||w||^2

    where ||w||^2 = sum_n n^2 / (L^2 + (4pi)^2 n^2)^2.

    Using a = L/(4pi):
    ||w||^2 = (4pi)^{-4} * sum_n n^2 / (a^2 + n^2)^2

    The infinite sum: sum_{n=-inf}^{inf} n^2/(a^2+n^2)^2 = pi*coth(pi*a)/(2a) - pi^2/(2*sinh^2(pi*a))
    """
    mp.dps = 50
    L = log(mpf(lam_sq))
    a = L / (4 * pi)
    pf = 32 * L * sinh(L / 4) ** 2

    # ||w||^2 via exact formula
    S2 = pi * coth(pi * a) / (2 * a) - pi**2 / (2 * sinh(pi * a)**2)
    w_norm_sq = S2 / (4 * pi)**4

    # <w_hat, W02, w_hat> = -pf * (4pi)^2 * ||w||^2
    result = -pf * (4 * pi)**2 * w_norm_sq

    return float(result), float(w_norm_sq)


def analytical_sums(lam_sq, N_terms=500):
    """
    Compute all key sums analytically using the digamma/coth representations.

    Returns dict with:
        S0 = sum n^0 / (a^2 + n^2)^1 = pi*coth(pi*a)/a   [Eisenstein]
        S1 = sum n^2 / (a^2 + n^2)^2                        [for ||w||^2]
        S2 = sum n^4 / (a^2 + n^2)^3                        [higher order]
    All sums from n = -inf to +inf.
    """
    mp.dps = 50
    L = log(mpf(lam_sq))
    a = L / (4 * pi)

    # S_0 = sum 1/(a^2 + n^2) = pi*coth(pi*a) / a
    S0 = pi * coth(pi * a) / a

    # sum 1/(a^2+n^2)^2 = d/d(a^2)[-S0] / ... let me use the formula:
    # sum 1/(a^2+n^2)^2 = (pi*coth(pi*a)/(2*a^3) + pi^2/(2*a^2*sinh^2(pi*a)))
    S0_2 = pi * coth(pi * a) / (2 * a**3) + pi**2 / (2 * a**2 * sinh(pi * a)**2)

    # S1 = sum n^2/(a^2+n^2)^2 = sum [1/(a^2+n^2) - a^2/(a^2+n^2)^2]
    #    = S0 - a^2 * S0_2
    S1 = S0 - a**2 * S0_2

    # Simplify S1:
    # = pi*coth(pi*a)/a - a^2*[pi*coth(pi*a)/(2*a^3) + pi^2/(2*a^2*sinh^2(pi*a))]
    # = pi*coth(pi*a)/a - pi*coth(pi*a)/(2*a) - pi^2/(2*sinh^2(pi*a))
    # = pi*coth(pi*a)/(2*a) - pi^2/(2*sinh^2(pi*a))
    S1_check = pi * coth(pi * a) / (2 * a) - pi**2 / (2 * sinh(pi * a)**2)

    return {
        'a': float(a),
        'S0': float(S0),
        'S0_2': float(S0_2),
        'S1': float(S1),
        'S1_check': float(S1_check),
    }


def analytical_mdiag_term(lam_sq, n_quad=10000):
    """
    Attempt closed-form for <w_hat, M_diag, w_hat>.

    M_diag[n,n] = wr_diag[n] where:
    wr_diag[n] = (omega_0/2)*(gamma + log(4*pi*(e^L-1)/(e^L+1))) + integral

    This is the Weil explicit formula diagonal, which for large L simplifies to:
    wr_diag[n] ≈ gamma + log(4*pi) + integral_term(n)

    <w_hat, M_diag, w_hat> = sum_n |w_hat[n]|^2 * wr_diag[n]
    """
    # This requires the numerical wr_diag values; analytical form is complex
    # For now, just return the numerical value for comparison
    mp.dps = 50
    L = log(mpf(lam_sq))
    L_f = float(L)
    eL = exp(L)
    N_val = max(15, round(6 * L_f))
    dim = 2 * N_val + 1

    # Compute wr_diag
    omega_0 = mpf(2)
    wr_diag = {}
    for nv in range(N_val + 1):
        def omega(x, nv=nv):
            return 2 * (1 - x / L) * cos(2 * pi * nv * x / L)
        w_const = (omega_0 / 2) * (euler + log(4 * pi * (eL - 1) / (eL + 1)))
        dx = L / n_quad
        integral = mpf(0)
        for k in range(n_quad):
            x = dx * (k + mpf(1) / 2)
            numer = exp(x / 2) * omega(x) - omega_0
            denom = exp(x) - exp(-x)
            if abs(denom) > mpf(10)**(-40):
                integral += numer / denom
        integral *= dx
        wr_diag[nv] = float(w_const + integral)
        wr_diag[-nv] = wr_diag[nv]

    # Build w vector and compute
    ns = np.arange(-N_val, N_val + 1, dtype=float)
    w_vec = ns / (L_f**2 + (4 * np.pi)**2 * ns**2)
    w_vec[N_val] = 0.0  # n=0 term
    w_norm = np.linalg.norm(w_vec)
    w_hat = w_vec / w_norm

    diag_vals = np.array([wr_diag[int(n)] for n in ns])
    mdiag_rayleigh = np.sum(w_hat**2 * diag_vals)

    # Analyze wr_diag as function of |n|
    # For large |n|, wr_diag[n] should approach a constant + O(1/n^2)
    return {
        'mdiag_rayleigh': mdiag_rayleigh,
        'wr_diag_0': wr_diag[0],
        'wr_diag_1': wr_diag[1],
        'wr_diag_5': wr_diag[5] if 5 <= N_val else None,
        'wr_diag_10': wr_diag[10] if 10 <= N_val else None,
    }


# ═══════════════════════════════════════════════════════════════
# PART 3: ASYMPTOTIC ANALYSIS
# ═══════════════════════════════════════════════════════════════

def fit_power_law(xs, ys, label=""):
    """Fit y = A * x^alpha + B to data."""
    # First try log-log fit for the growth
    logx = np.log(xs)
    logy = np.log(np.abs(ys))
    # Linear fit in log-log
    coeffs = np.polyfit(logx, logy, 1)
    alpha = coeffs[0]
    A = np.exp(coeffs[1]) * np.sign(ys[0])
    return alpha, A


# ═══════════════════════════════════════════════════════════════
# MAIN COMPUTATION
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print()
    print('#' * 70)
    print('  SESSION 41 — BARRIER CONSTANT ASYMPTOTICS')
    print('#' * 70)

    # ── Phase 1: Numerical sweep ──
    print('\n  PHASE 1: Numerical barrier sweep')
    print('  ' + '=' * 60)

    lam_sq_values = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    results = []

    for lam_sq in lam_sq_values:
        t0 = time.time()
        r = compute_barrier_components(lam_sq)
        dt = time.time() - t0
        results.append(r)
        print(f'\n  lam^2 = {lam_sq:>6d}  (L = {r["L"]:.3f}, N = {r["N"]}, dim = {r["dim"]}, {dt:.1f}s)')
        print(f'    W02 eigenvalues:  even = {r["u_eval"]:+.6f},  odd = {r["w_eval"]:+.6f}')
        print(f'    Barrier components:')
        print(f'      <w, W02, w>     = {r["w02_term"]:+.6f}')
        print(f'      <w, M_prime, w> = {r["mprime_term"]:+.6f}')
        print(f'      <w, M_diag, w>  = {r["mdiag_term"]:+.6f}')
        print(f'      <w, M_alpha, w> = {r["malpha_term"]:+.6f}')
        print(f'    Barrier = {r["barrier"]:+.8f}  (check: {r["barrier_check"]:+.8f})')
        print(f'    u-direction barrier = {r["u_barrier"]:+.8f}')
        print(f'    Full eps_0 = {r["eps_0"]:.6e}')

    # ── Summary table ──
    print('\n\n  BARRIER SUMMARY TABLE')
    print('  ' + '-' * 80)
    print(f'  {"lam^2":>7s} {"L":>6s} {"W02":>10s} {"M_prime":>10s} '
          f'{"M_diag":>10s} {"M_alpha":>10s} {"Barrier":>10s} {"u_bar":>10s}')
    print('  ' + '-' * 80)
    for r in results:
        print(f'  {r["lam_sq"]:>7d} {r["L"]:>6.2f} {r["w02_term"]:>+10.5f} '
              f'{r["mprime_term"]:>+10.5f} {r["mdiag_term"]:>+10.5f} '
              f'{r["malpha_term"]:>+10.5f} {r["barrier"]:>+10.6f} '
              f'{r["u_barrier"]:>+10.6f}')

    # ── Log-slopes (d log(component) / d log(lam^2)) ──
    print('\n\n  LOG-SLOPES (d log|component| / d log(lam^2))')
    print('  ' + '-' * 70)
    Ls = np.array([r['L'] for r in results])
    w02s = np.array([abs(r['w02_term']) for r in results])
    mps = np.array([abs(r['mprime_term']) for r in results])
    mds = np.array([abs(r['mdiag_term']) for r in results])
    mas = np.array([abs(r['malpha_term']) for r in results])
    barriers = np.array([r['barrier'] for r in results])

    for i in range(1, len(results)):
        dlogL = np.log(results[i]['lam_sq'] / results[i-1]['lam_sq'])
        s_w02 = np.log(w02s[i] / w02s[i-1]) / dlogL if w02s[i-1] > 0 else 0
        s_mp = np.log(mps[i] / mps[i-1]) / dlogL if mps[i-1] > 0 else 0
        s_md = np.log(mds[i] / mds[i-1]) / dlogL if mds[i-1] > 0 else 0
        s_ma = np.log(mas[i] / mas[i-1]) / dlogL if mas[i-1] > 0 else 0
        s_b = (barriers[i] - barriers[i-1]) / dlogL  # linear slope for barrier

        print(f'  [{results[i-1]["lam_sq"]:>5d} -> {results[i]["lam_sq"]:>5d}]  '
              f'W02: {s_w02:+.4f}  M_pr: {s_mp:+.4f}  M_dg: {s_md:+.4f}  '
              f'M_al: {s_ma:+.4f}  barrier_delta: {s_b:+.6f}')

    # ── Phase 2: Analytical comparison for W02 term ──
    print('\n\n  PHASE 2: Analytical W02 formula verification')
    print('  ' + '=' * 60)

    print(f'\n  {"lam^2":>7s} {"Numerical":>12s} {"Analytical":>12s} {"Rel Error":>12s} {"||w||^2 anal":>14s}')
    print('  ' + '-' * 60)
    for lam_sq in [50, 200, 1000, 5000]:
        # Find numerical result
        r = [x for x in results if x['lam_sq'] == lam_sq]
        if r:
            num = r[0]['w02_term']
        else:
            num = compute_barrier_components(lam_sq)['w02_term']

        anal, wnorm = analytical_w02_term(lam_sq)
        rel_err = abs(num - anal) / abs(anal) if abs(anal) > 1e-20 else 0
        print(f'  {lam_sq:>7d} {num:>+12.6f} {anal:>+12.6f} {rel_err:>12.2e} {wnorm:>14.6e}')

    # ── Phase 3: Component growth rates ──
    print('\n\n  PHASE 3: Growth rate analysis')
    print('  ' + '=' * 60)

    if len(results) >= 4:
        # Use last 4 points for power-law fit
        tail = results[-4:]
        lams = np.array([r['lam_sq'] for r in tail])

        for name, vals in [('|W02|', [abs(r['w02_term']) for r in tail]),
                           ('|M_prime|', [abs(r['mprime_term']) for r in tail]),
                           ('|M_diag|', [abs(r['mdiag_term']) for r in tail]),
                           ('|M_alpha|', [abs(r['malpha_term']) for r in tail])]:
            alpha, A = fit_power_law(lams, np.array(vals))
            print(f'  {name:12s} ~ {A:+.4f} * lam^{alpha:.4f}')

        # Barrier: should be ~constant, check
        bar_vals = [r['barrier'] for r in tail]
        print(f'\n  Barrier range: [{min(bar_vals):.6f}, {max(bar_vals):.6f}]')
        print(f'  Barrier mean:  {np.mean(bar_vals):.6f}')
        print(f'  Barrier std:   {np.std(bar_vals):.6f}')

    # ── Phase 4: Analytical sum verification ──
    print('\n\n  PHASE 4: Key sum verification (S1 = sum n^2/(a^2+n^2)^2)')
    print('  ' + '=' * 60)

    for lam_sq in [50, 200, 1000, 5000]:
        sums = analytical_sums(lam_sq)
        L_f = np.log(lam_sq)
        a = sums['a']

        # Numerical check: compute S1 by direct summation
        N_check = 500
        ns = np.arange(-N_check, N_check + 1, dtype=float)
        S1_num = np.sum(ns**2 / (a**2 + ns**2)**2)

        print(f'\n  lam^2 = {lam_sq}, a = L/(4pi) = {a:.4f}')
        print(f'    S1 analytical = {sums["S1"]:.10f}')
        print(f'    S1 check      = {sums["S1_check"]:.10f}')
        print(f'    S1 numerical  = {S1_num:.10f}')
        print(f'    S1 asymptotic = {np.pi / (2 * a):.10f}  [pi/(2a)]')
        print(f'    Rel error (anal vs num) = {abs(sums["S1"] - S1_num) / abs(S1_num):.2e}')

    # ── Phase 5: Barrier vs 1/(8*pi) ──
    print('\n\n  PHASE 5: Barrier vs candidate constants')
    print('  ' + '=' * 60)
    candidates = {
        '1/(8*pi)': 1 / (8 * np.pi),
        '1/(4*pi^2)': 1 / (4 * np.pi**2),
        '1/(2*pi)^2': 1 / (2 * np.pi)**2,
        'log(2)/(4*pi)': np.log(2) / (4 * np.pi),
        'gamma/(4*pi)': 0.5772156649 / (4 * np.pi),
    }
    print(f'  Candidate constants:')
    for name, val in candidates.items():
        print(f'    {name:20s} = {val:.8f}')

    if results:
        last_barrier = results[-1]['barrier']
        print(f'\n  Last computed barrier: {last_barrier:.8f}')
        for name, val in candidates.items():
            print(f'    {name:20s}: ratio = {last_barrier / val:.6f}')

    print('\n' + '#' * 70)
    print('  SESSION 41 COMPLETE')
    print('#' * 70)
