"""
SESSION 69 -- LP EMERGENCE: HOW DO PRIMES CREATE THE LP PROPERTY?

F alone fails d>=3 Turan. xi = F*zeta passes all tested orders.
The LP property is ARITHMETIC -- created by the primes.

Key question: how many primes are needed?
Add primes one at a time and track when d=3,4 Jensen polynomials
become hyperbolic. This reveals the emergence mechanism.
"""

import sys
import numpy as np
import mpmath
from mpmath import mp, mpf

mp.dps = 50

sys.path.insert(0, '.')
from session41g_uncapped_barrier import sieve_primes


def compute_taylor_coeffs(lam_sq, K=12, use_all_primes=True, prime_subset=None):
    """Compute Taylor coefficients of xi(1/2+it)/xi(1/2) in t^{2k}.

    If prime_subset is given, use only those primes (plus archimedean).
    """
    # Build xi_partial = F * zeta_partial where zeta_partial uses given primes
    L = float(np.log(lam_sq))

    if use_all_primes:
        def func(ss):
            return mpf('0.5') * ss * (ss-1) * mpmath.power(mpmath.pi, -ss/2) * \
                   mpmath.gamma(ss/2) * mpmath.zeta(ss)
    else:
        # F * partial Euler product
        def func(ss):
            F = mpf('0.5') * ss * (ss-1) * mpmath.power(mpmath.pi, -ss/2) * \
                mpmath.gamma(ss/2)
            zeta_partial = mpf(1)
            for p in prime_subset:
                zeta_partial *= 1 / (1 - mpmath.power(mpf(int(p)), -ss))
            return F * zeta_partial

    s = mpf('0.5')
    f_val = func(s)

    Z = []
    for k in range(1, K + 1):
        deriv = mpmath.diff(func, s, n=2*k)
        Z.append(deriv / f_val)

    c = [mpf(1)]
    for k in range(1, K + 1):
        c.append(Z[k-1] * (-1)**k / mpmath.factorial(2*k))

    return [float(x) for x in c]


def check_jensen(c, d, k):
    """Check if Jensen polynomial J_{d,k}(x) = Sum C(d,j)*c[k+j]*x^j has all real zeros."""
    from math import comb
    coeffs = [comb(d, j) * c[k+j] for j in range(d+1)]
    roots = np.roots(coeffs[::-1])
    return all(abs(r.imag) < 1e-6 * max(1, abs(r.real)) for r in roots)


def d3_turan(c, k):
    """Compute d=3 higher-order Turan inequality at shift k."""
    T2_k = c[k]**2 - c[k-1]*c[k+1]
    T2_k1 = c[k+1]**2 - c[k]*c[k+2]
    cross = c[k]*c[k+1] - c[k-1]*c[k+2]
    return 4*T2_k*T2_k1 - cross**2


def run():
    print()
    print('#' * 76)
    print('  SESSION 69 -- LP EMERGENCE')
    print('#' * 76)

    # ==================================================================
    # PART 1: F ALONE vs FULL XI at d=2,3,4,5,6,7,8
    # ==================================================================
    print('\n  === PART 1: JENSEN POLYNOMIAL HYPERBOLICITY ===')
    print('  Check degrees d=2..8 at shifts k=0..5 for F and xi.\n')

    K = 14

    # F only (no primes)
    mp.dps = 60
    def F_func(ss):
        return mpf('0.5') * ss * (ss-1) * mpmath.power(mpmath.pi, -ss/2) * \
               mpmath.gamma(ss/2)

    def xi_func(ss):
        return F_func(ss) * mpmath.zeta(ss)

    s = mpf('0.5')
    F_val = F_func(s)
    xi_val = xi_func(s)

    F_Z = []
    xi_Z = []
    for k in range(1, K+1):
        F_Z.append(mpmath.diff(F_func, s, n=2*k) / F_val)
        xi_Z.append(mpmath.diff(xi_func, s, n=2*k) / xi_val)

    c_F = [1.0]
    c_xi = [1.0]
    for k in range(1, K+1):
        fac = float(mpmath.factorial(2*k))
        c_F.append(float(F_Z[k-1]) * (-1)**k / fac)
        c_xi.append(float(xi_Z[k-1]) * (-1)**k / fac)

    mp.dps = 50

    print(f'  {"d":>3} {"k":>3} {"J(xi) real?":>12} {"J(F) real?":>12}')
    print('  ' + '-' * 34)

    for d in range(2, 9):
        for k in range(min(6, K - d)):
            xi_hyp = check_jensen(c_xi, d, k)
            F_hyp = check_jensen(c_F, d, k)
            if d <= 4 or k == 0:
                print(f'  {d:>3d} {k:>3d} {"YES" if xi_hyp else "NO":>12} '
                      f'{"YES" if F_hyp else "NO":>12}')
    sys.stdout.flush()

    # ==================================================================
    # PART 2: ADD PRIMES ONE AT A TIME -- WHEN DOES d=3 PASS?
    # ==================================================================
    print(f'\n  === PART 2: PRIME-BY-PRIME LP EMERGENCE ===')
    print(f'  Add primes cumulatively. Track d=3 Turan at k=1.\n')

    all_primes = list(sieve_primes(200))

    print(f'  {"#primes":>8} {"last p":>8} {"d3_T(k=1)":>14} {"d3>0?":>6}')
    print('  ' + '-' * 40)

    # F only (0 primes)
    d3_F = d3_turan(c_F, 1)
    print(f'  {0:>8d} {"(F)":>8} {d3_F:>+14.6e} {"YES" if d3_F > 0 else "NO":>6}')

    for n_primes in [1, 2, 3, 5, 10, 20, 50, len(all_primes)]:
        subset = all_primes[:n_primes]
        c_partial = compute_taylor_coeffs(200, K=K, use_all_primes=False,
                                           prime_subset=subset)
        d3_val = d3_turan(c_partial, 1)
        last_p = subset[-1] if subset else 0
        print(f'  {n_primes:>8d} {int(last_p):>8d} {d3_val:>+14.6e} '
              f'{"YES" if d3_val > 0 else "NO":>6}')

    # Full xi for comparison
    d3_xi = d3_turan(c_xi, 1)
    print(f'  {"(full)":>8} {"xi":>8} {d3_xi:>+14.6e} {"YES" if d3_xi > 0 else "NO":>6}')
    sys.stdout.flush()

    # ==================================================================
    # PART 3: THE ZETA BOOST ANATOMY
    # ==================================================================
    print(f'\n  === PART 3: WHAT ZETA CHANGES IN THE COEFFICIENTS ===')
    print(f'  Compare c_k(F) vs c_k(xi) and the correction.\n')

    print(f'  {"k":>3} {"c_k(F)":>16} {"c_k(xi)":>16} {"correction":>16} {"ratio xi/F":>12}')
    print('  ' + '-' * 66)
    for k in range(K+1):
        corr = c_xi[k] - c_F[k]
        ratio = c_xi[k] / c_F[k] if abs(c_F[k]) > 1e-30 else float('inf')
        print(f'  {k:>3d} {c_F[k]:>+16.8e} {c_xi[k]:>+16.8e} {corr:>+16.8e} {ratio:>12.6f}')
    sys.stdout.flush()

    # ==================================================================
    # PART 4: THE EULER PRODUCT STRUCTURE
    # ==================================================================
    print(f'\n  === PART 4: INDIVIDUAL PRIME CONTRIBUTIONS TO d=3 ===')
    print(f'  Which primes contribute most to making d=3 pass?\n')

    # Compute d3 with each prime REMOVED
    c_full = compute_taylor_coeffs(200, K=K, use_all_primes=True)
    d3_full = d3_turan(c_full, 1)

    print(f'  Full d3(k=1) = {d3_full:+.6e}')
    print()
    print(f'  {"removed p":>10} {"d3 without p":>14} {"shift":>14} {"still >0?":>10}')
    print('  ' + '-' * 52)

    for p in all_primes[:20]:
        subset = [q for q in all_primes if q != p]
        c_partial = compute_taylor_coeffs(200, K=K, use_all_primes=False,
                                           prime_subset=subset)
        d3_val = d3_turan(c_partial, 1)
        shift = d3_val - d3_full
        ok = d3_val > 0
        print(f'  {int(p):>10d} {d3_val:>+14.6e} {shift:>+14.6e} '
              f'{"YES" if ok else "**NO**":>10}')
    sys.stdout.flush()

    # ==================================================================
    # VERDICT
    # ==================================================================
    print()
    print('=' * 76)
    print('  SESSION 69 VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
