"""
SESSION 70 -- CUMULANT DECAY AND JENSEN HYPERBOLICITY AT ALL SCALES

Proof path: Euler product Gaussianization forces xi near-Gaussian.
Gaussian IS LP (Hermite polynomials are hyperbolic).
RH = corrections (higher cumulants) preserve LP at ALL (d, n).

Key computation:
  1. Compute cumulants kappa_{2k} from Taylor coeffs of xi
  2. Map Jensen polynomial hyperbolicity across full (d, n) plane
  3. Find the tightest point -- where does hyperbolicity barely hold?
  4. Connect to cumulant decay rate: is it fast enough?

The Jensen polynomial J_{d,n}(x) = Sum_{j=0}^d C(d,j) c_{n+j} x^j
is hyperbolic (all real zeros) iff xi is in LP class.

GORZ proved: hyperbolic for all d >= 1 at sufficiently large n.
Gap: small n, large d.
"""

import sys
import numpy as np
import mpmath
from mpmath import mp, mpf

mp.dps = 50


def xi_func(s):
    return mpf('0.5') * s * (s - 1) * mpmath.power(mpmath.pi, -s / 2) * \
           mpmath.gamma(s / 2) * mpmath.zeta(s)


def compute_taylor_coeffs(K=15):
    """Compute K+1 Taylor coefficients c_0, ..., c_K of xi(1/2+it)/xi(1/2).

    Uses Richardson extrapolation for speed (mpmath.diff with method='quad').
    """
    s = mpf('0.5')
    xi_val = xi_func(s)

    c = [mpf(1)]
    for k in range(1, K + 1):
        deriv = mpmath.diff(xi_func, s, n=2 * k)
        c_k = deriv / xi_val * mpf(-1)**k / mpmath.factorial(2 * k)
        c.append(c_k)
        if k % 5 == 0:
            print(f'    ... computed c_{k}')
            sys.stdout.flush()

    return c


def coeffs_to_cumulants(c, K_cum):
    """Convert Taylor coefficients to cumulants via log-generating function.

    If phi(t) = Sum c_k t^{2k} is the characteristic function,
    log phi(t) = Sum kappa_{2k} t^{2k} / (2k)!

    So the cumulants come from log(Sum c_k w^k) where w = t^2.
    Let a_k = c_k. Then log(Sum a_k w^k) = Sum b_k w^k where
    b_k are the log-series coefficients.

    Relation: b_1 = a_1, b_n = a_n - (1/n) Sum_{j=1}^{n-1} j * b_j * a_{n-j}
    Then kappa_{2k} = (2k)! * b_k.
    """
    a = [float(c[k]) for k in range(K_cum + 1)]

    # log-series coefficients b_k (k = 1, ..., K_cum)
    b = [0.0] * (K_cum + 1)
    b[1] = a[1]
    for n in range(2, K_cum + 1):
        s = sum(j * b[j] * a[n - j] for j in range(1, n))
        b[n] = a[n] - s / n

    # Cumulants: kappa_{2k} = (2k)! * b_k
    kappa = {}
    for k in range(1, K_cum + 1):
        fac = 1
        for j in range(1, 2 * k + 1):
            fac *= j
        kappa[2 * k] = b[k] * fac

    return kappa, b


def check_jensen_roots(c, d, n):
    """Check if Jensen polynomial J_{d,n}(x) has all real zeros.

    J_{d,n}(x) = Sum_{j=0}^d C(d,j) c_{n+j} x^j

    Returns (is_hyperbolic, min_imag_ratio, roots).
    """
    from math import comb

    if n + d >= len(c):
        return None, None, None

    coeffs = [comb(d, j) * float(c[n + j]) for j in range(d + 1)]

    if abs(coeffs[-1]) < 1e-30:
        return None, None, None

    roots = np.roots(coeffs[::-1])

    # Hyperbolicity: all roots real
    imag_ratios = [abs(r.imag) / max(1e-30, abs(r.real)) for r in roots]
    max_imag_ratio = max(imag_ratios) if imag_ratios else 0

    is_hyp = max_imag_ratio < 1e-6

    return is_hyp, max_imag_ratio, roots


def jensen_discriminant_margin(c, d, n):
    """For d=2: explicit discriminant gives a margin measure.

    J_{2,n}(x) = c_n + 2*c_{n+1}*x + c_{n+2}*x^2
    Discriminant = 4*c_{n+1}^2 - 4*c_n*c_{n+2} = 4*(c_{n+1}^2 - c_n*c_{n+2})
    Hyperbolic iff discriminant >= 0, i.e., Turan inequality holds.
    """
    cn = float(c[n])
    cn1 = float(c[n + 1])
    cn2 = float(c[n + 2])

    turan = cn1**2 - cn * cn2
    # Normalized margin
    if abs(cn * cn2) > 1e-30:
        ratio = cn1**2 / (cn * cn2)
    else:
        ratio = float('inf')

    return turan, ratio


def run():
    print()
    print('#' * 76)
    print('  SESSION 70 -- CUMULANT DECAY AND JENSEN HYPERBOLICITY')
    print('#' * 76)

    # ==================================================================
    # STEP 1: Compute Taylor coefficients to high order
    # ==================================================================
    print('\n  === STEP 1: TAYLOR COEFFICIENTS OF xi(1/2+it)/xi(1/2) ===\n')

    K = 15
    print(f'  Computing c_0 through c_{K} at {mp.dps}-digit precision...')
    sys.stdout.flush()

    c = compute_taylor_coeffs(K)

    print(f'  {"k":>3} {"c_k":>22} {"|c_k|":>14}')
    print('  ' + '-' * 42)
    for k in range(K + 1):
        ck = float(c[k])
        print(f'  {k:>3d} {ck:>+22.14e} {abs(ck):>14.6e}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 2: Cumulants from Taylor coefficients
    # ==================================================================
    print(f'\n  === STEP 2: CUMULANTS kappa_{{2k}} ===\n')

    K_cum = min(K, 12)
    kappa, b = coeffs_to_cumulants(c, K_cum)

    print(f'  {"2k":>4} {"kappa_{{2k}}":>22} {"b_k (log coeff)":>22} {"kappa/kappa_2^k":>16}')
    print('  ' + '-' * 68)
    kappa_2 = kappa.get(2, 1.0)
    for k in range(1, K_cum + 1):
        kap = kappa[2 * k]
        norm = kap / kappa_2**k if abs(kappa_2) > 1e-30 else 0
        print(f'  {2*k:>4d} {kap:>+22.14e} {b[k]:>+22.14e} {norm:>+16.8e}')
    sys.stdout.flush()

    # Gaussian test: if kappa_{2k} = 0 for k >= 2, it's Gaussian
    print(f'\n  Gaussian test: kappa_2 = {kappa[2]:+.10e}')
    print(f'  Deviations from Gaussian (should be 0 for pure Gaussian):')
    for k in range(2, min(K_cum + 1, 8)):
        print(f'    kappa_{2*k:>2d} / kappa_2^{k} = {kappa[2*k]/kappa_2**k:+.10e}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 3: Map Jensen polynomial hyperbolicity across (d, n) plane
    # ==================================================================
    print(f'\n  === STEP 3: JENSEN HYPERBOLICITY MAP (d, n) ===\n')

    max_d = min(12, K)
    max_n = K

    print(f'  J_{{d,n}}(x) hyperbolic? (Y=yes, N=no, .=insufficient data)')
    print()
    header = '  n\\d  ' + ''.join(f'{d:>4d}' for d in range(2, max_d + 1))
    print(header)
    print('  ' + '-' * (6 + 4 * (max_d - 1)))

    hyp_map = {}
    for n in range(0, max_n - 1):
        row = f'  {n:>3d}  '
        for d in range(2, max_d + 1):
            if n + d > K:
                row += '   .'
            else:
                is_hyp, margin, _ = check_jensen_roots(c, d, n)
                if is_hyp is None:
                    row += '   .'
                else:
                    hyp_map[(d, n)] = (is_hyp, margin)
                    row += '   Y' if is_hyp else '  *N'
        print(row)
    sys.stdout.flush()

    # ==================================================================
    # STEP 4: Detailed margin analysis for d=2 (Turan) and d=3
    # ==================================================================
    print(f'\n  === STEP 4: MARGIN ANALYSIS ===\n')

    # d=2: Turan ratios
    print(f'  d=2 Turan ratios R_k = c_k^2 / (c_{{k-1}} * c_{{k+1}}):')
    print(f'  {"k":>3} {"R_k":>14} {"R_k - 1":>14} {"k*(R_k-1)":>14}')
    print('  ' + '-' * 48)
    for k in range(1, K):
        ck = float(c[k])
        ckm = float(c[k - 1])
        ckp = float(c[k + 1])
        if abs(ckm * ckp) > 1e-50:
            R = ck**2 / (ckm * ckp)
            print(f'  {k:>3d} {R:>14.8f} {R-1:>+14.8e} {k*(R-1):>14.8f}')
    sys.stdout.flush()

    # d=3: higher-order Turan
    print(f'\n  d=3 higher-order Turan T3_k = 4*(c_k^2 - c_{{k-1}}c_{{k+1}})*(c_{{k+1}}^2 - c_k c_{{k+2}}) - (c_k c_{{k+1}} - c_{{k-1}} c_{{k+2}})^2:')
    print(f'  {"k":>3} {"T3_k":>22} {">0?":>5}')
    print('  ' + '-' * 34)
    for k in range(1, K - 1):
        ck = float(c[k])
        ckm = float(c[k - 1])
        ckp = float(c[k + 1])
        ckpp = float(c[k + 2])
        t2_k = ck**2 - ckm * ckp
        t2_k1 = ckp**2 - ck * ckpp
        cross = ck * ckp - ckm * ckpp
        t3 = 4 * t2_k * t2_k1 - cross**2
        print(f'  {k:>3d} {t3:>+22.10e} {"YES" if t3 > 0 else "**NO**":>5}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 5: The closest-to-failure point in (d, n) space
    # ==================================================================
    print(f'\n  === STEP 5: CLOSEST TO FAILURE ===\n')

    # For each Jensen polynomial, compute how close the roots are to
    # becoming non-real (ratio of imag to real part)
    print(f'  {"d":>3} {"n":>3} {"max |Im/Re|":>14} {"status":>8}')
    print('  ' + '-' * 32)

    worst_margin = 0
    worst_d = worst_n = 0
    for d in range(2, max_d + 1):
        for n in range(0, max_n - d):
            is_hyp, margin, roots = check_jensen_roots(c, d, n)
            if is_hyp is not None:
                if is_hyp and margin > worst_margin:
                    worst_margin = margin
                    worst_d = d
                    worst_n = n
                if not is_hyp or margin > 1e-8:
                    print(f'  {d:>3d} {n:>3d} {margin:>14.8e} '
                          f'{"HYPER" if is_hyp else "**FAIL**":>8}')
    sys.stdout.flush()

    # Find the tightest passing case
    print(f'\n  Tightest passing Jensen polynomials (smallest max|Im/Re| among passing):')
    tight_cases = []
    for d in range(2, max_d + 1):
        for n in range(0, max_n - d):
            is_hyp, margin, roots = check_jensen_roots(c, d, n)
            if is_hyp is not None and is_hyp:
                # Compute a better margin: for a hyperbolic polynomial,
                # the margin is how far the roots are from coalescing
                # (discriminant-like measure)
                real_roots = sorted([r.real for r in roots])
                if len(real_roots) >= 2:
                    min_gap = min(abs(real_roots[i+1] - real_roots[i])
                                  for i in range(len(real_roots) - 1))
                    spread = max(abs(r) for r in real_roots) if real_roots else 1
                    rel_gap = min_gap / max(spread, 1e-30)
                    tight_cases.append((rel_gap, d, n, min_gap))

    tight_cases.sort()
    print(f'  {"d":>3} {"n":>3} {"rel_gap":>14} {"min_gap":>14}')
    print('  ' + '-' * 38)
    for rel_gap, d, n, min_gap in tight_cases[:10]:
        print(f'  {d:>3d} {n:>3d} {rel_gap:>14.8e} {min_gap:>14.8e}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 6: Cumulant expansion of Jensen polynomials
    # ==================================================================
    print(f'\n  === STEP 6: CUMULANT INTERPRETATION ===\n')

    # For a Gaussian (kappa_{2k}=0, k>=2), the Taylor coeffs are
    # c_k = kappa_2^k / (2k)! * (2k)! / k! = kappa_2^k / k!  ... no.
    # Actually: phi(t) = exp(kappa_2 * t^2 / 2) for Gaussian.
    # So c_k = (kappa_2/2)^k / k! = sigma^{2k} / (2^k * k!)
    # where sigma^2 = kappa_2.

    sigma_sq = kappa[2]
    print(f'  sigma^2 = kappa_2 = {sigma_sq:+.10e}')
    print()
    print(f'  Gaussian approx: c_k^{{Gauss}} = (kappa_2/2)^k / k!')
    print(f'  {"k":>3} {"c_k(xi)":>18} {"c_k(Gauss)":>18} {"ratio":>12} {"delta":>14}')
    print('  ' + '-' * 68)

    for k in range(K + 1):
        ck_xi = float(c[k])
        # Gaussian: exp(sigma^2 * w / 2) => coeff of w^k is (sigma^2/2)^k / k!
        ck_gauss = (sigma_sq / 2)**k
        fac = 1
        for j in range(1, k + 1):
            fac *= j
        ck_gauss /= fac
        ratio = ck_xi / ck_gauss if abs(ck_gauss) > 1e-50 else float('inf')
        delta = (ck_xi - ck_gauss) / abs(ck_xi) if abs(ck_xi) > 1e-50 else 0
        print(f'  {k:>3d} {ck_xi:>+18.10e} {ck_gauss:>+18.10e} {ratio:>12.6f} {delta:>+14.6e}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 7: Normalized cumulants and decay rate
    # ==================================================================
    print(f'\n  === STEP 7: NORMALIZED CUMULANT DECAY ===\n')

    # The key question: how fast do the normalized cumulants
    # kappa_{2k} / kappa_2^k decay?
    # For LP: need fast enough decay that Jensen polys stay hyperbolic.

    print(f'  Normalized cumulants kappa_{{2k}} / kappa_2^k:')
    print(f'  {"k":>3} {"kappa_{{2k}}/kappa_2^k":>22} {"log|.|":>12} {"predicted slope":>16}')
    print('  ' + '-' * 56)

    log_norms = []
    for k in range(1, K_cum + 1):
        norm = kappa[2 * k] / kappa_2**k
        log_norm = np.log10(abs(norm)) if abs(norm) > 1e-50 else -50
        log_norms.append((k, log_norm))
        print(f'  {k:>3d} {norm:>+22.14e} {log_norm:>12.4f}')

    # Fit decay rate: log|kappa_{2k}/kappa_2^k| ~ alpha * k + beta
    ks = np.array([x[0] for x in log_norms[1:]])  # skip k=1 (kappa_2/kappa_2 = 1)
    logs = np.array([x[1] for x in log_norms[1:]])
    valid = np.isfinite(logs)
    if np.sum(valid) >= 2:
        fit = np.polyfit(ks[valid], logs[valid], 1)
        print(f'\n  Decay fit: log10|kappa_{{2k}}/kappa_2^k| ~ {fit[0]:.4f} * k + {fit[1]:.4f}')
        print(f'  Exponential decay rate: |kappa_{{2k}}/kappa_2^k| ~ {10**fit[1]:.4e} * {10**fit[0]:.6f}^k')
        print(f'  Per-step decay factor: {10**fit[0]:.6f}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 8: Archimedean (F) vs full (xi) cumulants
    # ==================================================================
    print(f'\n  === STEP 8: F vs XI CUMULANTS ===\n')

    def F_func(s):
        return mpf('0.5') * s * (s - 1) * mpmath.power(mpmath.pi, -s / 2) * \
               mpmath.gamma(s / 2)

    s = mpf('0.5')
    F_val = F_func(s)

    c_F = [mpf(1)]
    for k in range(1, K + 1):
        deriv = mpmath.diff(F_func, s, n=2 * k)
        c_F.append(deriv / F_val * mpf(-1)**k / mpmath.factorial(2 * k))
        if k % 5 == 0:
            print(f'    ... computed F c_{k}')
            sys.stdout.flush()

    kappa_F, b_F = coeffs_to_cumulants(c_F, K_cum)
    kappa_xi = kappa

    print(f'  {"2k":>4} {"kappa(F)":>22} {"kappa(xi)":>22} {"correction":>22}')
    print('  ' + '-' * 74)
    for k in range(1, K_cum + 1):
        kF = kappa_F[2 * k]
        kX = kappa_xi[2 * k]
        corr = kX - kF
        print(f'  {2*k:>4d} {kF:>+22.14e} {kX:>+22.14e} {corr:>+22.14e}')
    sys.stdout.flush()

    # ==================================================================
    # VERDICT
    # ==================================================================
    print()
    print('=' * 76)
    print('  SESSION 70 VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
