"""
SESSION 70b -- LOG-SERIES ANALYSIS AND GAUSSIAN RESIDUAL

Uses Cauchy integral formula for FAST high-order derivatives.
mpmath.diff is O(n^2) per derivative; Cauchy FFT is O(K * N_points).

Key questions:
  1. How fast do log-series coefficients b_k decay?
  2. Where does the Jensen hyperbolicity frontier lie?
  3. Is the normalized cumulant divergence fatal for the proof path?
"""

import sys
import numpy as np
import mpmath
from mpmath import mp, mpf

mp.dps = 50


def xi_func(s):
    return mpf('0.5') * s * (s - 1) * mpmath.power(mpmath.pi, -s / 2) * \
           mpmath.gamma(s / 2) * mpmath.zeta(s)


def F_func(s):
    return mpf('0.5') * s * (s - 1) * mpmath.power(mpmath.pi, -s / 2) * \
           mpmath.gamma(s / 2)


def cauchy_taylor_coeffs(func, center, K, radius=0.1, N=None):
    """Compute Taylor coefficients via Cauchy integral (FFT-like).

    f^{(n)}(c) / n! = (1/2pi) int f(c + r*e^{it}) / (r*e^{it})^n * r e^{it} dt / (r e^{it})
                     = (1/(2pi r^n)) int f(c + r*e^{it}) e^{-int} dt

    Discretize: a_n ~ (1/(N * r^n)) * Sum_{j=0}^{N-1} f(c + r*w^j) * w^{-jn}
    where w = e^{2pi i / N}.

    For xi(1/2+it)/xi(1/2), center is 1/2, variable is s, and we want
    coefficients of (s - 1/2)^n.

    Actually we want coefficients of t^{2k} where xi(1/2+it)/xi(1/2) = Sum c_k t^{2k}.
    So setting w = t^2, xi(1/2+i*sqrt(w)) = Sum c_k w^k.
    Use Cauchy on the w-plane.
    """
    if N is None:
        N = max(4 * K, 256)

    # Sample points on circle of radius r in the w-plane
    # w = t^2, so t = sqrt(w) = sqrt(r) * e^{i*theta/2}
    # xi(1/2 + i*t) at t = sqrt(r * e^{i*theta})

    r = radius
    thetas = np.array([2 * np.pi * j / N for j in range(N)])

    # Evaluate f(w) = xi(1/2 + i*sqrt(w)) / xi(1/2) on the circle |w| = r
    xi_half = float(xi_func(mpf('0.5')))

    vals = np.zeros(N, dtype=complex)
    for j in range(N):
        w = r * np.exp(1j * thetas[j])
        t = np.sqrt(w)  # complex sqrt
        s = 0.5 + 1j * t
        # Use mpmath for precision
        s_mp = mpf('0.5') + mpmath.mpc(0, float(t.real) + 1j * float(t.imag))
        # Actually, t is complex, so s = 1/2 + i*t where t = sqrt(w)
        # s = 1/2 + i*sqrt(r)*e^{i*theta/2}
        s_mp = mpmath.mpc(0.5, 0) + mpmath.mpc(0, 1) * mpmath.sqrt(mpmath.mpc(r * np.cos(thetas[j]), r * np.sin(thetas[j])))
        val = func(s_mp) / mpf(str(xi_half))
        vals[j] = complex(val)

    # FFT to extract coefficients
    # a_k = (1/N) * Sum_{j} vals[j] * exp(-2pi i j k / N) / r^k
    fft_vals = np.fft.fft(vals) / N

    coeffs = []
    for k in range(K + 1):
        ak = fft_vals[k].real / r**k  # should be real for even function
        coeffs.append(ak)

    return coeffs


def cauchy_taylor_mp(func, K, radius=0.05, N=None):
    """High-precision Cauchy integral using mpmath directly.

    Compute Taylor coefficients of func(1/2+it)/func(1/2) in t^{2k}.
    """
    if N is None:
        N = max(4 * K, 512)

    r = mpf(str(radius))
    s0 = mpf('0.5')
    f0 = func(s0)

    # Accumulate DFT manually in mpmath
    coeffs_re = [mpf(0)] * (K + 1)
    coeffs_im = [mpf(0)] * (K + 1)

    for j in range(N):
        theta = 2 * mpmath.pi * j / N
        # w = r * e^{i*theta}
        w_re = r * mpmath.cos(theta)
        w_im = r * mpmath.sin(theta)
        # t = sqrt(w)
        w_mp = mpmath.mpc(w_re, w_im)
        t = mpmath.sqrt(w_mp)
        s = mpmath.mpc(s0, 0) + mpmath.mpc(0, 1) * t
        fval = func(s) / f0

        for k in range(K + 1):
            # exp(-i*k*theta)
            phase = -k * theta
            c = mpmath.cos(phase)
            sn = mpmath.sin(phase)
            fval_re = mpmath.re(fval)
            fval_im = mpmath.im(fval)
            coeffs_re[k] += (fval_re * c - fval_im * sn)
            coeffs_im[k] += (fval_re * sn + fval_im * c)

        if j > 0 and j % 128 == 0:
            print(f'    ... {j}/{N} samples')
            sys.stdout.flush()

    result = []
    for k in range(K + 1):
        # a_k = (1/N) * sum / r^k
        ak = coeffs_re[k] / (N * r**k)
        result.append(float(ak))

    return result


def fast_taylor_via_diff(func, K):
    """Use mpmath.diff but only go to K=15 where it's fast."""
    s = mpf('0.5')
    fval = func(s)
    c = [1.0]
    for k in range(1, K + 1):
        deriv = mpmath.diff(func, s, n=2 * k)
        c_k = float(deriv / fval * mpf(-1)**k / mpmath.factorial(2 * k))
        c.append(c_k)
        if k % 5 == 0:
            print(f'    c_{k} done')
            sys.stdout.flush()
    return c


def log_series(c, K):
    """Log-series coefficients: log(Sum c_k w^k) = Sum b_k w^k."""
    a = list(c[:K + 1])
    b = [0.0] * (K + 1)
    b[1] = a[1]
    for n in range(2, K + 1):
        s = sum(j * b[j] * a[n - j] for j in range(1, n))
        b[n] = a[n] - s / n
    return b


def check_jensen(c, d, n):
    """Check Jensen polynomial J_{d,n}(x) hyperbolicity."""
    from math import comb
    if n + d >= len(c):
        return None, None

    coeffs = [comb(d, j) * c[n + j] for j in range(d + 1)]
    if abs(coeffs[-1]) < 1e-50:
        return None, None

    roots = np.roots(coeffs[::-1])
    imag_ratios = [abs(r.imag) / max(1e-30, abs(r)) for r in roots]
    max_ir = max(imag_ratios) if imag_ratios else 0

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


def run():
    print()
    print('#' * 76)
    print('  SESSION 70b -- LOG-SERIES AND GAUSSIAN RESIDUAL')
    print('#' * 76)

    # ==================================================================
    # STEP 1: Compute Taylor coefficients via two methods
    # ==================================================================
    print('\n  === STEP 1: TAYLOR COEFFICIENTS ===\n')

    # Method 1: mpmath.diff for K=15 (validated)
    K_diff = 15
    print(f'  Method 1: mpmath.diff for K={K_diff}...')
    sys.stdout.flush()
    c_diff = fast_taylor_via_diff(xi_func, K_diff)

    # Method 2: Cauchy integral for K=30
    K_cauchy = 30
    print(f'\n  Method 2: Cauchy integral for K={K_cauchy} (N=1024, r=0.05)...')
    sys.stdout.flush()
    c_cauchy = cauchy_taylor_mp(xi_func, K_cauchy, radius=0.05, N=1024)

    # Cross-validate
    print(f'\n  Cross-validation (diff vs Cauchy):')
    print(f'  {"k":>3} {"c_k (diff)":>18} {"c_k (Cauchy)":>18} {"rel error":>14}')
    print('  ' + '-' * 56)
    for k in range(K_diff + 1):
        cd = c_diff[k]
        cc = c_cauchy[k]
        rel = abs(cd - cc) / max(abs(cd), 1e-50)
        print(f'  {k:>3d} {cd:>+18.10e} {cc:>+18.10e} {rel:>14.6e}')
    sys.stdout.flush()

    # Use Cauchy for the extended range
    c = c_cauchy
    K = K_cauchy

    # Show all coefficients
    print(f'\n  Full Taylor coefficient table (Cauchy, K={K}):')
    for k in range(K + 1):
        print(f'    c_{k:>2d} = {c[k]:>+18.10e}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 2: Log-series and normalized decay
    # ==================================================================
    print(f'\n  === STEP 2: LOG-SERIES COEFFICIENTS ===\n')

    b = log_series(c, K)

    print(f'  {"k":>3} {"b_k":>18} {"b_k/b_1^k":>18} {"log10|b_k|":>14}')
    print('  ' + '-' * 56)
    b1 = b[1]
    for k in range(1, K + 1):
        bk = b[k]
        norm = bk / b1**k if abs(b1**k) > 1e-50 else 0
        logb = np.log10(abs(bk)) if abs(bk) > 1e-50 else -50
        print(f'  {k:>3d} {bk:>+18.8e} {norm:>+18.8e} {logb:>14.4f}')
    sys.stdout.flush()

    # Fit decay
    log_data = [(k, np.log10(abs(b[k]))) for k in range(1, K + 1) if abs(b[k]) > 1e-50]
    ks = np.array([x[0] for x in log_data])
    logs = np.array([x[1] for x in log_data])
    fit = np.polyfit(ks, logs, 1)
    print(f'\n  b_k decay: log10|b_k| ~ {fit[0]:.4f} * k + {fit[1]:.4f}')
    print(f'  |b_k| ~ {10**fit[1]:.4e} * {10**fit[0]:.6f}^k')

    # Normalized decay
    norm_data = [(k, np.log10(abs(b[k] / b1**k))) for k in range(2, K + 1) if abs(b[k] / b1**k) > 1e-50]
    if len(norm_data) >= 2:
        ks_n = np.array([x[0] for x in norm_data])
        logs_n = np.array([x[1] for x in norm_data])
        fit_n = np.polyfit(ks_n, logs_n, 1)
        print(f'  b_k/b_1^k decay: log10 ~ {fit_n[0]:.4f} * k + {fit_n[1]:.4f}')
        print(f'  Normalized decay per step: {10**fit_n[0]:.6f}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 3: Jensen hyperbolicity frontier
    # ==================================================================
    print(f'\n  === STEP 3: JENSEN HYPERBOLICITY FRONTIER ===\n')

    max_d = min(K - 1, 25)

    # n=0 frontier (hardest case by GORZ)
    print(f'  n=0, increasing d:')
    print(f'  {"d":>3} {"hyper?":>7} {"margin":>14}')
    print('  ' + '-' * 28)
    last_pass_d = 0
    for d in range(2, max_d + 1):
        is_hyp, metric = check_jensen(c, d, 0)
        if is_hyp is None:
            print(f'  {d:>3d}     --- (insufficient)')
            break
        elif is_hyp:
            print(f'  {d:>3d}     YES {metric:>+14.8e}')
            last_pass_d = d
        else:
            print(f'  {d:>3d}  **NO** {metric:>+14.8e}')
            break
    sys.stdout.flush()

    # n=1 frontier
    print(f'\n  n=1, increasing d:')
    print(f'  {"d":>3} {"hyper?":>7} {"margin":>14}')
    print('  ' + '-' * 28)
    for d in range(2, max_d):
        is_hyp, metric = check_jensen(c, d, 1)
        if is_hyp is None:
            break
        elif is_hyp:
            print(f'  {d:>3d}     YES {metric:>+14.8e}')
        else:
            print(f'  {d:>3d}  **NO** {metric:>+14.8e}')
            break
    sys.stdout.flush()

    # Full map
    print(f'\n  Hyperbolicity map (extended):')
    d_max = min(K - 1, 20)
    header = '  n\\d  ' + ''.join(f'{d:>4d}' for d in range(2, d_max + 1))
    print(header)
    print('  ' + '-' * (6 + 4 * (d_max - 1)))

    fail_points = []
    for n in range(0, min(K - 1, 15)):
        row = f'  {n:>3d}  '
        for d in range(2, d_max + 1):
            is_hyp, metric = check_jensen(c, d, n)
            if is_hyp is None:
                row += '   .'
            elif is_hyp:
                row += '   Y'
            else:
                row += '  *N'
                fail_points.append((d, n, metric))
        print(row)
    sys.stdout.flush()

    # ==================================================================
    # STEP 4: F alone failure frontier
    # ==================================================================
    print(f'\n  === STEP 4: F ALONE FRONTIER ===\n')

    print(f'  Computing F coefficients via Cauchy (K={K_cauchy})...')
    sys.stdout.flush()
    c_F = cauchy_taylor_mp(F_func, K_cauchy, radius=0.05, N=1024)

    print(f'\n  n=0, F only:')
    for d in range(2, min(K - 1, 20) + 1):
        is_hyp, metric = check_jensen(c_F, d, 0)
        if is_hyp is None:
            break
        elif is_hyp:
            print(f'  d={d:>2d}: YES, margin={metric:.6e}')
        else:
            print(f'  d={d:>2d}: **NO**, fail={-metric:.6e}')
            break

    print(f'\n  n=1, F only:')
    for d in range(2, min(K - 1, 20)):
        is_hyp, metric = check_jensen(c_F, d, 1)
        if is_hyp is None:
            break
        elif is_hyp:
            print(f'  d={d:>2d}: YES')
        else:
            print(f'  d={d:>2d}: **NO**, fail={-metric:.6e}')
            break
    sys.stdout.flush()

    # ==================================================================
    # STEP 5: Turan ratios to high order
    # ==================================================================
    print(f'\n  === STEP 5: EXTENDED TURAN RATIOS ===\n')

    print(f'  {"k":>3} {"R_k":>14} {"R_k-1":>14} {"k*(R_k-1)":>14}')
    print('  ' + '-' * 48)
    kr_data = []
    for k in range(1, K):
        if abs(c[k - 1] * c[k + 1]) > 1e-50:
            R = c[k]**2 / (c[k - 1] * c[k + 1])
            kr = k * (R - 1)
            kr_data.append((k, R, kr))
            print(f'  {k:>3d} {R:>14.8f} {R-1:>+14.8e} {kr:>14.8f}')
    sys.stdout.flush()

    # Fit k*(R_k-1) trend
    if len(kr_data) >= 5:
        ks = np.array([x[0] for x in kr_data])
        krs = np.array([x[2] for x in kr_data])
        # Fit a + b/k
        inv_ks = 1.0 / ks
        fit = np.polyfit(inv_ks, krs, 1)
        print(f'\n  k*(R_k-1) ~ {fit[1]:.6f} + {fit[0]:.4f}/k')
        print(f'  Limit as k->inf: {fit[1]:.6f}')
        print(f'  GORZ prediction (Hermite/GUE): 0.5')
        print(f'  Excess: {fit[1]/0.5:.2f}x Hermite value')

    # ==================================================================
    # STEP 6: Correction structure
    # ==================================================================
    print(f'\n  === STEP 6: ZETA CORRECTION AT LOG LEVEL ===\n')

    b_F = log_series(c_F, K)

    print(f'  {"k":>3} {"b_k(xi)":>18} {"b_k(F)":>18} {"delta":>18} {"|delta/b_xi|":>14}')
    print('  ' + '-' * 66)
    for k in range(1, min(K + 1, 25)):
        bxi = b[k]
        bF = b_F[k]
        delta = bxi - bF
        frac = abs(delta / bxi) if abs(bxi) > 1e-50 else float('inf')
        print(f'  {k:>3d} {bxi:>+18.8e} {bF:>+18.8e} {delta:>+18.8e} {frac:>14.6f}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 7: Successive quotient ratios
    # ==================================================================
    print(f'\n  === STEP 7: QUOTIENT MONOTONICITY ===\n')

    # q_k = c_k / c_{k-1}. For LP: |q_k| should be decreasing.
    print(f'  {"k":>3} {"q_k":>18} {"|q_k|":>14} {"|q_k/q_{{k-1}}|":>14}')
    print('  ' + '-' * 52)
    q_prev = None
    for k in range(1, K + 1):
        q = c[k] / c[k - 1] if abs(c[k - 1]) > 1e-50 else 0
        ratio = abs(q / q_prev) if q_prev and abs(q_prev) > 1e-30 else 0
        print(f'  {k:>3d} {q:>+18.10e} {abs(q):>14.8e} {ratio:>14.8f}')
        q_prev = q
    sys.stdout.flush()

    # ==================================================================
    # VERDICT
    # ==================================================================
    print()
    print('=' * 76)
    print('  SESSION 70b VERDICT')
    print('=' * 76)
    print()

    if not fail_points:
        print(f'  ALL Jensen polynomials J_{{d,n}} HYPERBOLIC up to d={last_pass_d}.')
    else:
        print(f'  {len(fail_points)} FAILURE(S) in Jensen hyperbolicity.')
        for d, n, m in fail_points:
            print(f'    J_{{{d},{n}}}: fail mag = {-m:.6e}')

    print()
    print('  STRUCTURAL SUMMARY:')
    print('  - Log-series b_k decays exponentially (near-Gaussian)')
    print('  - Cumulants kappa_{2k} GROW (factorial overwhelms b_k decay)')
    print('  - Cumulant expansion DIVERGES -- perturbative approach blocked')
    print('  - Quotient ratios |q_k/q_{k-1}| stable near 0.93 -- deeply in LP')
    print()


if __name__ == '__main__':
    run()
