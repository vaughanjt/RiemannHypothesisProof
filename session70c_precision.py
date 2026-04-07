"""
SESSION 70c -- HIGH-PRECISION JENSEN FRONTIER VIA HADAMARD PRODUCT

Compute Taylor coefficients from zeros (trivially fast, arbitrary order).

xi(1/2+it)/xi(1/2) = prod_k (1 - t^2/gamma_k^2)

log(phi(w)) = Sum_k log(1 - w/gamma_k^2) = -Sum_k Sum_{m>=1} w^m / (m * gamma_k^{2m})

So b_m = -Sum_k 1/(m * gamma_k^{2m})

Then c_k = coefficients of exp(Sum b_m w^m) via exponential recursion.

NOTE: This uses the zeros, so it's "cheating" for proof purposes.
But it maps the Jensen frontier precisely to guide the proof strategy.
"""

import sys
import time
import numpy as np
import mpmath
from mpmath import mp, mpf

mp.dps = 50


def compute_zeros(N_zeros):
    """Compute first N_zeros zeta zeros."""
    print(f'  Computing {N_zeros} zeta zeros...')
    sys.stdout.flush()
    gammas = []
    for k in range(1, N_zeros + 1):
        g = float(mpmath.im(mpmath.zetazero(k)))
        gammas.append(g)
        if k % 100 == 0:
            print(f'    {k}/{N_zeros} zeros computed')
            sys.stdout.flush()
    return np.array(gammas)


def log_coeffs_from_zeros(gammas, K):
    """Compute log-series coefficients b_m = -Sum_k 1/(m * gamma_k^{2m})."""
    b = np.zeros(K + 1)
    g2 = gammas**2
    for m in range(1, K + 1):
        b[m] = -np.sum(1.0 / (m * g2**m))
    return b


def exp_series(b, K):
    """Given log-series b_m, compute exp-series c_k.

    If log(phi) = Sum b_m w^m, then phi = Sum c_k w^k.
    Recursion: c_0 = exp(b_0) = 1 (since b_0 = 0),
    c_n = (1/n) * Sum_{j=1}^{n} j * b_j * c_{n-j}
    """
    c = np.zeros(K + 1)
    c[0] = 1.0
    for n in range(1, K + 1):
        c[n] = sum(j * b[j] * c[n - j] for j in range(1, n + 1)) / n
    return c


def check_jensen(c, d, n):
    from math import comb
    if n + d >= len(c):
        return None, None

    coeffs = [comb(d, j) * c[n + j] for j in range(d + 1)]
    if abs(coeffs[-1]) < 1e-200:
        return None, None

    roots = np.roots(coeffs[::-1])
    imag_ratios = [abs(r.imag) / max(1e-30, abs(r)) for r in roots]
    max_ir = max(imag_ratios)

    if max_ir < 1e-6:
        real_roots = sorted(r.real for r in roots)
        if len(real_roots) >= 2:
            gaps = [real_roots[i + 1] - real_roots[i] for i in range(len(real_roots) - 1)]
            min_gap = min(gaps)
            spread = real_roots[-1] - real_roots[0]
            rel_gap = min_gap / max(abs(spread), 1e-30)
            return True, rel_gap
        return True, 1.0
    return False, -max_ir


def validate_coefficients(c, K_validate=12):
    """Cross-validate against mpmath.diff for first few coefficients."""
    print(f'  Cross-validating c_0..c_{K_validate} against mpmath.diff...')
    sys.stdout.flush()

    def xi_func(s):
        return mpf('0.5') * s * (s - 1) * mpmath.power(mpmath.pi, -s / 2) * \
               mpmath.gamma(s / 2) * mpmath.zeta(s)

    s = mpf('0.5')
    xi_val = xi_func(s)

    max_err = 0
    for k in range(1, K_validate + 1):
        deriv = mpmath.diff(xi_func, s, n=2 * k)
        c_exact = float(deriv / xi_val * mpf(-1)**k / mpmath.factorial(2 * k))
        rel_err = abs(c[k] - c_exact) / max(abs(c_exact), 1e-50)
        max_err = max(max_err, rel_err)
        if k <= 5 or rel_err > 1e-8:
            print(f'    c_{k:>2d}: zeros={c[k]:>+18.10e}, diff={c_exact:>+18.10e}, '
                  f'rel_err={rel_err:.2e}')
        sys.stdout.flush()

    print(f'  Max relative error over c_1..c_{K_validate}: {max_err:.2e}')
    return max_err


def run():
    print()
    print('#' * 76)
    print('  SESSION 70c -- JENSEN FRONTIER VIA HADAMARD PRODUCT')
    print('#' * 76)

    # ==================================================================
    # STEP 1: Compute Taylor coefficients from zeros
    # ==================================================================
    print(f'\n  === STEP 1: TAYLOR COEFFICIENTS FROM ZEROS ===\n')

    N_zeros = 500
    K = 50  # go to c_50 (can check d up to 50)

    t0 = time.time()
    gammas = compute_zeros(N_zeros)
    print(f'  Zeros computed in {time.time()-t0:.1f}s')

    b = log_coeffs_from_zeros(gammas, K)
    c = exp_series(b, K)

    # Validate
    max_err = validate_coefficients(c, K_validate=12)

    if max_err > 1e-4:
        print(f'  WARNING: validation error {max_err:.2e} > 1e-4.')
        print(f'  Need more zeros for convergence.')

    # Show coefficients
    print(f'\n  Taylor coefficients c_k (from {N_zeros} zeros, K={K}):')
    for k in range(min(K + 1, 35)):
        print(f'    c_{k:>2d} = {c[k]:>+18.10e}')
    sys.stdout.flush()

    # Check convergence: add more zeros and see if coefficients change
    print(f'\n  Convergence check: sensitivity to N_zeros...')
    for N_test in [100, 200, 500]:
        g_test = gammas[:N_test]
        b_test = log_coeffs_from_zeros(g_test, 30)
        c_test = exp_series(b_test, 30)
        diff_at_15 = abs(c_test[15] - c[15]) / max(abs(c[15]), 1e-50)
        diff_at_20 = abs(c_test[20] - c[20]) / max(abs(c[20]), 1e-50)
        diff_at_25 = abs(c_test[25] - c[25]) / max(abs(c[25]), 1e-50)
        print(f'    N={N_test:>4d}: |delta c_15|/|c_15| = {diff_at_15:.2e}, '
              f'|delta c_20| = {diff_at_20:.2e}, |delta c_25| = {diff_at_25:.2e}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 2: Jensen hyperbolicity frontier to d=40
    # ==================================================================
    print(f'\n  === STEP 2: JENSEN HYPERBOLICITY FRONTIER ===\n')

    # n=0 frontier
    print(f'  n=0 frontier:')
    print(f'  {"d":>3} {"hyper?":>7} {"margin":>14}')
    print('  ' + '-' * 28)
    margins_n0 = []
    first_fail_n0 = None
    for d in range(2, min(K, 45) + 1):
        h, m = check_jensen(c, d, 0)
        if h is None:
            break
        status = '   YES' if h else '**NO**'
        print(f'  {d:>3d} {status} {m:>+14.8e}')
        if h:
            margins_n0.append((d, m))
        elif first_fail_n0 is None:
            first_fail_n0 = d
    sys.stdout.flush()

    # n=1 frontier
    print(f'\n  n=1 frontier:')
    margins_n1 = []
    first_fail_n1 = None
    for d in range(2, min(K - 1, 40) + 1):
        h, m = check_jensen(c, d, 1)
        if h is None:
            break
        if h:
            margins_n1.append((d, m))
            if d <= 20 or d % 5 == 0:
                print(f'  d={d:>2d}: YES, margin={m:.6e}')
        else:
            if first_fail_n1 is None:
                first_fail_n1 = d
            print(f'  d={d:>2d}: **NO**, fail={-m:.6e}')
    sys.stdout.flush()

    # Full map up to d=25
    print(f'\n  Hyperbolicity map:')
    d_max = min(K - 1, 25)
    n_max = min(K - 2, 12)
    header = '  n\\d  ' + ''.join(f'{d:>4d}' for d in range(2, d_max + 1))
    print(header)
    print('  ' + '-' * (6 + 4 * (d_max - 1)))

    frontier = {}  # d -> max n that passes
    for n in range(0, n_max + 1):
        row = f'  {n:>3d}  '
        for d in range(2, d_max + 1):
            h, m = check_jensen(c, d, n)
            if h is None:
                row += '   .'
            elif h:
                row += '   Y'
                frontier[d] = max(frontier.get(d, -1), n)
            else:
                row += '  *N'
        print(row)
    sys.stdout.flush()

    # ==================================================================
    # STEP 3: Frontier characterization
    # ==================================================================
    print(f'\n  === STEP 3: FRONTIER d + n ~ const? ===\n')

    print(f'  Frontier: max n passing for each d:')
    print(f'  {"d":>3} {"max_n":>6} {"d+max_n":>8}')
    print('  ' + '-' * 20)
    for d in sorted(frontier.keys()):
        print(f'  {d:>3d} {frontier[d]:>6d} {d + frontier[d]:>8d}')

    if len(frontier) >= 3:
        ds = np.array(sorted(frontier.keys()))
        ns = np.array([frontier[d] for d in ds])
        # Is d + n ~ const?
        sums = ds + ns
        print(f'\n  Sum d+n: min={min(sums)}, max={max(sums)}, mean={np.mean(sums):.1f}')
        print(f'  Frontier shape: d + n ~ {np.mean(sums):.1f} ± {np.std(sums):.1f}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 4: Margin decay at n=0
    # ==================================================================
    print(f'\n  === STEP 4: MARGIN DECAY AT n=0 ===\n')

    if len(margins_n0) >= 3:
        ds = np.array([x[0] for x in margins_n0])
        ms = np.array([x[1] for x in margins_n0])

        # Filter out any non-positive margins
        valid = ms > 0
        ds = ds[valid]
        ms = ms[valid]

        if len(ds) >= 3:
            log_ms = np.log(ms)

            # Exponential fit
            fit = np.polyfit(ds, log_ms, 1)
            print(f'  Exponential fit: margin ~ {np.exp(fit[1]):.4f} * exp({fit[0]:.6f} * d)')
            print(f'  Decay per degree: {np.exp(fit[0]):.6f} ({np.exp(fit[0])*100:.1f}%)')
            for d_target in [20, 30, 50, 100]:
                pred = np.exp(fit[0] * d_target + fit[1])
                print(f'    Predicted margin at d={d_target}: {pred:.2e}')

            # Power-law fit: margin ~ d^alpha
            log_ds = np.log(ds)
            fit_p = np.polyfit(log_ds, log_ms, 1)
            print(f'\n  Power-law fit: margin ~ {np.exp(fit_p[1]):.4f} * d^{fit_p[0]:.4f}')

    sys.stdout.flush()

    # ==================================================================
    # STEP 5: Turan ratios extended to high k
    # ==================================================================
    print(f'\n  === STEP 5: TURAN RATIOS TO k={K-1} ===\n')

    print(f'  {"k":>3} {"R_k":>14} {"R_k-1":>14} {"k*(R_k-1)":>14}')
    print('  ' + '-' * 48)
    kr_data = []
    for k in range(1, K):
        if abs(c[k - 1] * c[k + 1]) > 1e-200:
            R = c[k]**2 / (c[k - 1] * c[k + 1])
            kr = k * (R - 1)
            kr_data.append((k, R, kr))
            if k <= 25 or k % 5 == 0:
                print(f'  {k:>3d} {R:>14.8f} {R-1:>+14.8e} {kr:>14.8f}')

    if len(kr_data) >= 5:
        ks = np.array([x[0] for x in kr_data])
        krs = np.array([x[2] for x in kr_data])
        fit = np.polyfit(1.0 / ks, krs, 1)
        print(f'\n  k*(R_k-1) ~ {fit[1]:.6f} + {fit[0]:.4f}/k')
        print(f'  Limit as k->inf: {fit[1]:.6f}')
        print(f'  GORZ prediction: 0.5 (Hermite/GUE)')
        print(f'  Ratio: {fit[1]/0.5:.4f}x')
    sys.stdout.flush()

    # ==================================================================
    # STEP 6: Log-series structure
    # ==================================================================
    print(f'\n  === STEP 6: LOG-SERIES COEFFICIENTS ===\n')

    print(f'  {"k":>3} {"b_k":>22} {"b_k/b_1^k":>18}')
    print('  ' + '-' * 46)
    b1 = b[1]
    for k in range(1, min(K + 1, 30)):
        bk = b[k]
        norm = bk / b1**k if abs(b1**k) > 1e-50 else 0
        print(f'  {k:>3d} {bk:>+22.14e} {norm:>+18.8e}')

    # Decay fit
    log_b_data = [(k, np.log10(abs(b[k]))) for k in range(1, min(K + 1, 30))
                  if abs(b[k]) > 1e-50]
    if len(log_b_data) >= 3:
        ks_b = np.array([x[0] for x in log_b_data])
        logs_b = np.array([x[1] for x in log_b_data])
        fit_b = np.polyfit(ks_b, logs_b, 1)
        print(f'\n  b_k decay: |b_k| ~ {10**fit_b[1]:.4e} * {10**fit_b[0]:.6f}^k')
    sys.stdout.flush()

    # ==================================================================
    # VERDICT
    # ==================================================================
    print()
    print('=' * 76)
    print('  SESSION 70c VERDICT')
    print('=' * 76)
    print()

    if first_fail_n0 is not None:
        print(f'  Jensen polynomials HYPERBOLIC for d=2..{first_fail_n0-1} at n=0.')
        print(f'  FIRST FAILURE at d={first_fail_n0}, n=0.')
    else:
        print(f'  All Jensen polynomials HYPERBOLIC up to d={margins_n0[-1][0]}.')

    print()
    if len(margins_n0) >= 3:
        ds = np.array([x[0] for x in margins_n0])
        ms = np.array([x[1] for x in margins_n0])
        valid = ms > 0
        if np.sum(valid) >= 3:
            fit = np.polyfit(ds[valid], np.log(ms[valid]), 1)
            print(f'  Margin decay: exp({fit[0]:.4f} * d) — '
                  f'{np.exp(fit[0])*100:.1f}% per degree')


if __name__ == '__main__':
    run()
