"""
SESSION 41c — ANALYTICAL DERIVATIONS FOR BARRIER COMPONENTS

Derives closed-form expressions for each barrier component using:
1. Poisson summation for Lorentzian sums
2. Digamma function representations
3. Exact vs truncated sum comparisons

Key quantity: a = L/(4*pi) where L = log(lam^2).
Note: a is SMALL (0.18 to 0.73 for lam^2 in [10, 10000]).
So we CANNOT use large-a asymptotics. Need exact formulas.

The normalized weight function:
    |w_hat[n]|^2 = n^2 / (a^2 + n^2)^2 / S1
    S1 = sum_{n=-inf}^{inf} n^2/(a^2+n^2)^2 = pi*coth(pi*a)/(2a) - pi^2/(2*sinh^2(pi*a))
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, euler, exp, cos, sin,
                    coth, sinh, cosh, psi, gamma as mpgamma,
                    fsum, nsum, inf)
import time


mp.dps = 50


# ═══════════════════════════════════════════════════════════════
# EXACT SUMS VIA POISSON/DIGAMMA
# ═══════════════════════════════════════════════════════════════

def S_power(a, p, q):
    """
    Compute sum_{n=-inf}^{inf} n^{2p} / (a^2 + n^2)^q  exactly.

    Uses the identity: sum 1/(a^2+n^2) = pi*coth(pi*a)/a
    and derivatives with respect to a^2 for higher q.

    Results for small cases:
    (p=0, q=1): pi*coth(pi*a)/a
    (p=0, q=2): pi*coth(pi*a)/(2*a^3) + pi^2/(2*a^2*sinh^2(pi*a))
    (p=1, q=2): pi*coth(pi*a)/(2*a) - pi^2/(2*sinh^2(pi*a))
    (p=1, q=3): computed from the above
    """
    a = mpf(a)
    if p == 0 and q == 1:
        return pi * coth(pi * a) / a
    elif p == 0 and q == 2:
        return pi * coth(pi * a) / (2 * a**3) + pi**2 / (2 * a**2 * sinh(pi * a)**2)
    elif p == 1 and q == 2:
        # S(1,2) = S(0,1) - a^2 * S(0,2)
        return pi * coth(pi * a) / (2 * a) - pi**2 / (2 * sinh(pi * a)**2)
    elif p == 0 and q == 3:
        # -d/d(a^2) S(0,2) = S(0,3)
        # Need d/da of S(0,2), then divide by 2a
        # S(0,2) = f(a) = pi*coth(pi*a)/(2*a^3) + pi^2/(2*a^2*sinh^2(pi*a))
        # Compute numerically via mpmath
        return nsum(lambda n: 1/(a**2 + n**2)**3, [-inf, inf])
    elif p == 1 and q == 3:
        return S_power(a, 0, 2) - a**2 * S_power(a, 0, 3)
    elif p == 2 and q == 3:
        return S_power(a, 0, 1) - 2*a**2 * S_power(a, 0, 2) + a**4 * S_power(a, 0, 3)
    else:
        # Fallback to numerical
        return nsum(lambda n: n**(2*p) / (a**2 + n**2)**q, [-inf, inf])


def fourier_lorentzian_sum(a, t, p=1, q=2):
    """
    Compute sum_{n=-inf}^{inf} n^{2p} * cos(2*pi*n*t) / (a^2 + n^2)^q.

    For p=1, q=2:
    sum n^2 cos(2*pi*n*t) / (a^2+n^2)^2
    = sum cos(...)/(a^2+n^2) - a^2 * sum cos(...)/(a^2+n^2)^2

    Base case sum cos(2*pi*n*t)/(a^2+n^2):
    Using Poisson: = (pi/a) * sum_k exp(-2*pi*a*|t-k|)
    For 0 <= t <= 1: = (pi/a) * [exp(-2*pi*a*t) + exp(-2*pi*a*(1-t))] / (1 - exp(-2*pi*a))
                        + corrections for |k| >= 1

    Actually the exact formula is:
    sum_{n=-inf}^inf cos(2*pi*n*t) / (a^2+n^2)
    = (pi/a) * cosh(2*pi*a*(1/2 - |t mod 1|)) / sinh(pi*a)   for 0 < |t| < 1
    """
    a = mpf(a)
    t = mpf(t)

    if p == 0 and q == 1:
        # Exact: (pi/a) * cosh(2*pi*a*(1/2 - t)) / sinh(pi*a)  for 0 < t < 1
        if t <= 0 or t >= 1:
            return nsum(lambda n: cos(2*pi*n*t) / (a**2 + n**2), [-inf, inf])
        return (pi / a) * cosh(2*pi*a*(mpf(1)/2 - t)) / sinh(pi*a)

    elif p == 0 and q == 2:
        # -d/d(a^2) of F(0,1)(t)
        # Use numerical differentiation or series formula
        return nsum(lambda n: cos(2*pi*n*t) / (a**2 + n**2)**2, [-inf, inf])

    elif p == 1 and q == 2:
        # F(1,2) = F(0,1) - a^2 * F(0,2)
        return fourier_lorentzian_sum(a, t, 0, 1) - a**2 * fourier_lorentzian_sum(a, t, 0, 2)

    else:
        return nsum(lambda n: n**(2*p) * cos(2*pi*n*t) / (a**2 + n**2)**q, [-inf, inf])


# ═══════════════════════════════════════════════════════════════
# ANALYTICAL M_PRIME ON w DIRECTION
# ═══════════════════════════════════════════════════════════════

def analytical_mprime_diagonal(lam_sq):
    """
    Compute the diagonal part of <w_hat, M_prime, w_hat> analytically.

    <w_hat, M_prime^diag, w_hat> = sum_{p^k <= lam^2} log(p)*p^{-k/2} * R(t)

    where t = log(p^k)/L and
    R(t) = 2(1-t) * G(t) / S1

    G(t) = sum n^2 cos(2*pi*n*t) / (a^2+n^2)^2  [Fourier-Lorentzian sum]
    S1 = G(0) = sum n^2 / (a^2+n^2)^2
    """
    L = log(mpf(lam_sq))
    a = L / (4 * pi)
    L_f = float(L)

    # S1 = ||w||^2 normalization
    S1 = S_power(a, 1, 2)

    # Enumerate prime powers
    limit = int(lam_sq)
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 2):
        if i <= limit and sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False

    total = mpf(0)
    terms = []

    for p in range(2, limit + 1):
        if not sieve[p]:
            continue
        pk = p
        k = 1
        while pk <= lam_sq:
            logp = log(mpf(p))
            logpk = k * logp
            t = logpk / L
            weight = logp * mpf(p)**(-mpf(k)/2)

            # G(t) = sum n^2 cos(2*pi*n*t) / (a^2+n^2)^2
            G_t = fourier_lorentzian_sum(a, float(t), p=1, q=2)

            # R(t) = 2(1-t) * G(t) / S1
            R_t = 2 * (1 - t) * G_t / S1

            contribution = weight * R_t
            total += contribution

            if p <= 7 or pk == p:
                terms.append((int(p), k, float(t), float(weight),
                              float(G_t/S1), float(R_t), float(contribution)))

            pk *= p
            k += 1

    return float(total), float(S1), terms


# ═══════════════════════════════════════════════════════════════
# ANALYTICAL W02 ON w DIRECTION
# ═══════════════════════════════════════════════════════════════

def analytical_w02(lam_sq):
    """
    Exact formula: <w_hat, W02, w_hat> = -pf * (4*pi)^2 * ||w_tilde||^2

    where pf = 32*L*sinh^2(L/4)
    and ||w_tilde||^2 = S1 / (4*pi)^4
    """
    L = log(mpf(lam_sq))
    a = L / (4 * pi)
    pf = 32 * L * sinh(L / 4)**2
    S1 = S_power(a, 1, 2)
    wnorm2 = S1 / (4 * pi)**4

    result = -pf * (4 * pi)**2 * wnorm2
    # Simplify: -32*L*sinh^2(L/4) * 16*pi^2 * S1/(256*pi^4)
    #         = -2*L*sinh^2(L/4) * S1 / pi^2
    result_check = -2 * L * sinh(L / 4)**2 * S1 / pi**2

    return float(result), float(result_check), float(S1)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print()
    print('#' * 70)
    print('  SESSION 41c — ANALYTICAL DERIVATIONS')
    print('#' * 70)

    # ── Part 1: S1 exact values ──
    print('\n  PART 1: Key sum S1 = sum n^2/(a^2+n^2)^2')
    print('  ' + '=' * 60)

    for lam_sq in [10, 50, 200, 1000, 5000, 10000]:
        L = float(log(mpf(lam_sq)))
        a = L / (4 * np.pi)
        S1 = float(S_power(mpf(a), 1, 2))
        S1_asymp = np.pi / (2 * a)
        ratio = S1 / S1_asymp
        print(f'  lam^2={lam_sq:>6d}  a={a:.4f}  S1={S1:.8f}  '
              f'pi/(2a)={S1_asymp:.8f}  ratio={ratio:.6f}')

    # ── Part 2: W02 analytical vs numerical ──
    print('\n\n  PART 2: W02 analytical formula')
    print('  ' + '=' * 60)

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from connes_crossterm import build_all

    for lam_sq in [50, 200, 1000, 5000]:
        L_f = np.log(lam_sq)
        N = max(15, round(6 * L_f))

        # Numerical
        W02, M, QW = build_all(lam_sq, N, n_quad=8000)
        ns = np.arange(-N, N + 1, dtype=float)
        w_vec = ns / (L_f**2 + (4*np.pi)**2 * ns**2)
        w_vec[N] = 0.0
        w_hat = w_vec / np.linalg.norm(w_vec)
        num = w_hat @ W02 @ w_hat

        # Analytical (infinite sum)
        anal, anal_check, S1 = analytical_w02(lam_sq)

        # Truncated analytical (sum from -N to N)
        a = L_f / (4 * np.pi)
        S1_trunc = sum(n**2 / (a**2 + n**2)**2 for n in range(-N, N+1))
        wnorm2_trunc = S1_trunc / (4 * np.pi)**4
        pf = 32 * L_f * np.sinh(L_f / 4)**2
        anal_trunc = -pf * (4 * np.pi)**2 * wnorm2_trunc

        print(f'\n  lam^2={lam_sq}  (a={a:.4f}, N={N})')
        print(f'    Numerical:           {num:+.8f}')
        print(f'    Analytical (inf):    {anal:+.8f}  (err={abs(num-anal)/abs(num):.2e})')
        print(f'    Analytical (trunc):  {anal_trunc:+.8f}  (err={abs(num-anal_trunc)/abs(num):.2e})')

    # ── Part 3: M_prime analytical ──
    print('\n\n  PART 3: M_prime diagonal contribution (exact Fourier-Lorentzian)')
    print('  ' + '=' * 60)

    for lam_sq in [50, 200]:
        print(f'\n  lam^2 = {lam_sq}')
        t0 = time.time()
        total, S1, terms = analytical_mprime_diagonal(lam_sq)
        dt = time.time() - t0

        # Numerical comparison
        from session33_sieve_bypass import compute_M_decomposition
        L_f = np.log(lam_sq)
        N = max(15, round(6 * L_f))
        M_diag, M_alpha, M_prime, M_full, vM = compute_M_decomposition(lam_sq, N, n_quad=8000)
        ns = np.arange(-N, N + 1, dtype=float)
        w_vec = ns / (L_f**2 + (4*np.pi)**2 * ns**2)
        w_vec[N] = 0.0
        w_hat = w_vec / np.linalg.norm(w_vec)
        mprime_num = w_hat @ M_prime @ w_hat

        # M_prime = diagonal + off-diagonal
        M_prime_diag = np.diag(np.diag(M_prime))
        M_prime_off = M_prime - M_prime_diag
        mprime_diag_num = w_hat @ M_prime_diag @ w_hat
        mprime_off_num = w_hat @ M_prime_off @ w_hat

        print(f'    S1 = {S1:.8f}')
        print(f'    Analytical (diag only): {total:+.8f}  ({dt:.1f}s)')
        print(f'    Numerical (full):       {mprime_num:+.8f}')
        print(f'    Numerical (diag only):  {mprime_diag_num:+.8f}')
        print(f'    Numerical (off-diag):   {mprime_off_num:+.8f}')
        print(f'    Off-diag fraction:      {abs(mprime_off_num)/abs(mprime_num)*100:.2f}%')

        print(f'\n    Prime contributions (p<=7):')
        print(f'    {"p":>3s} {"k":>2s} {"t=logpk/L":>10s} {"weight":>10s} '
              f'{"G(t)/S1":>10s} {"R(t)":>10s} {"contrib":>10s}')
        print('    ' + '-' * 62)
        for p, k, t, w, gs, rt, c in terms[:15]:
            print(f'    {p:>3d} {k:>2d} {t:>10.6f} {w:>10.6f} '
                  f'{gs:>10.6f} {rt:>10.6f} {c:>10.6f}')

    # ── Part 4: Filter function G(t)/S1 ──
    print('\n\n  PART 4: Spectral filter G(t)/S1 vs t')
    print('  ' + '=' * 60)
    print('  This is the "window function" that weights each prime\'s contribution')

    for lam_sq in [200, 1000, 10000]:
        L = log(mpf(lam_sq))
        a = L / (4 * pi)
        S1 = S_power(a, 1, 2)

        print(f'\n  lam^2={lam_sq}, a={float(a):.4f}')
        print(f'  {"t":>6s} {"G(t)/S1":>12s} {"2(1-t)*G/S1":>12s} {"exp(-t*L)":>12s}')
        print('  ' + '-' * 48)

        for t_val in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            if t_val == 0:
                Gt = S1
            elif t_val >= 1:
                Gt = mpf(0)
            else:
                Gt = fourier_lorentzian_sum(a, t_val, p=1, q=2)
            ratio = Gt / S1 if S1 != 0 else 0
            R_t = 2 * (1 - t_val) * ratio
            exp_decay = float(exp(-mpf(t_val) * L))
            print(f'  {t_val:>6.2f} {float(ratio):>12.8f} {float(R_t):>12.8f} {exp_decay:>12.8f}')

    # ── Part 5: The barrier as a function of exact sums ──
    print('\n\n  PART 5: Barrier decomposition in exact sums')
    print('  ' + '=' * 60)
    print('  barrier = |M_prime| - |W02| - M_diag - M_alpha')
    print('  = (excess of M_prime over W02 in magnitude) - (analytic terms)')

    for lam_sq in [200, 1000]:
        L = log(mpf(lam_sq))
        a = L / (4 * pi)
        S1 = S_power(a, 1, 2)

        # W02 exact
        pf = 32 * L * sinh(L / 4)**2
        w02_exact = -pf * (4 * pi)**2 * S1 / (4 * pi)**4
        # = -2*L*sinh^2(L/4)*S1/pi^2

        print(f'\n  lam^2={lam_sq}')
        print(f'    a = {float(a):.6f}')
        print(f'    S1 = {float(S1):.8f}')
        print(f'    pf = {float(pf):.4f}')
        print(f'    W02 exact = {float(w02_exact):.8f}')
        print(f'    2*L*sinh^2(L/4)*S1/pi^2 = {float(2*L*sinh(L/4)**2*S1/pi**2):.8f}')

    print('\n' + '#' * 70)
    print('  SESSION 41c COMPLETE')
    print('#' * 70)
