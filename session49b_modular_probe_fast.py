"""
SESSION 49b -- MODULAR STRUCTURE PROBE (VECTORIZED VERSION)

Replaces the slow connes_crossterm.build_all (O(N^2 * pi(lam_sq)) Python
scalar loop) with session41g_uncapped_barrier.compute_barrier_partial,
which is fully numpy-vectorized and 50-100x faster.

Tradeoff: we now compute the "partial barrier" W02 - M_prime instead of
the full Q_W = W02 - M - W_R. The W_R correction is O(1) and smooth
(per Sessions 41-45), so qualitative structure (non-monotonicity, Heegner
signal, modular patterns) survives.

Same four stages:
  1. Dense B_partial(y) scan
  2. Heegner heights d in {3, 1, 7, 2}  (d=11 borderline, skipped)
  3. Modular inversion B(y) vs B(1/y)
  4. PSLQ algebraic recognition

The conjugate Poisson kernel is the same vector as session41g already
uses for the partial barrier, so we're evaluating the same quadratic form.
"""

import sys
import time

import numpy as np

sys.path.insert(0, '.')
from session41g_uncapped_barrier import compute_barrier_partial


# ----------------------------------------------------------------------
#  Core B(y) evaluation
# ----------------------------------------------------------------------

_Y_CACHE = {}

def barrier_at_y(y):
    """Partial barrier as a function of height y = L/(4*pi) in H."""
    cache_key = round(float(y), 8)
    if cache_key in _Y_CACHE:
        return _Y_CACHE[cache_key]

    L_f = 4.0 * np.pi * y
    lam_sq = float(np.exp(L_f))
    r = compute_barrier_partial(lam_sq)
    result = {
        'y': float(y),
        'L': L_f,
        'lam_sq': lam_sq,
        'N': r['N'],
        'n_primes': r['n_primes'],
        'W02': r['w02'],
        'Mp': r['mprime'],
        'B': r['partial_barrier'],
    }
    _Y_CACHE[cache_key] = result
    return result


# ----------------------------------------------------------------------
#  Stage 1: Dense B(y) scan
# ----------------------------------------------------------------------

def stage1_dense_scan():
    print()
    print('=' * 76)
    print('  STAGE 1: DENSE B_partial(y) SCAN')
    print('=' * 76)
    print()
    print('  y in [0.20, 1.50] (lam_sq <= 1.5e8 with vectorized sieve).')
    print('  Including Heegner heights 3,1,7,2 so Stage 2 reads from cache.')
    print()
    sys.stdout.flush()

    heegner_y = [np.sqrt(3) / 2, 1.0, np.sqrt(7) / 2, np.sqrt(2)]
    y_values = np.concatenate([
        np.linspace(0.20, 0.55, 8),       # lower regime
        np.linspace(0.60, 0.95, 8),       # middle (known y=0.8 dip)
        np.linspace(1.00, 1.45, 8),       # upper
        np.array(heegner_y),
    ])
    y_values = np.sort(np.unique(np.round(y_values, 8)))

    print(f'  Scanning {len(y_values)} y values.')
    print()
    print(f'  {"y":>8} {"lam^2":>14} {"N":>5} {"primes":>9} '
          f'{"B_partial":>14} {"W02":>12} {"Mp":>12} {"dt":>6}')
    print('  ' + '-' * 86)

    results = []
    for y in y_values:
        t0 = time.time()
        try:
            r = barrier_at_y(float(y))
            dt = time.time() - t0
            print(f'  {r["y"]:8.4f} {r["lam_sq"]:14.1f} {r["N"]:5d} '
                  f'{r["n_primes"]:9d} {r["B"]:14.8f} '
                  f'{r["W02"]:12.6f} {r["Mp"]:12.6f} '
                  f'{dt:6.1f}s', flush=True)
            results.append(r)
        except Exception as exc:
            print(f'  {y:8.4f}  ERROR: {exc}', flush=True)

    # Structure summary
    ys = np.array([r['y'] for r in results])
    Bs = np.array([r['B'] for r in results])

    print()
    min_idx = int(np.argmin(Bs))
    max_idx = int(np.argmax(Bs))
    print(f'  Global min: B = {Bs[min_idx]:+.8f} at y = {ys[min_idx]:.4f}')
    print(f'  Global max: B = {Bs[max_idx]:+.8f} at y = {ys[max_idx]:.4f}')
    print(f'  Range:      [{Bs.min():+.6f}, {Bs.max():+.6f}]')

    diffs = np.diff(Bs)
    sign_changes = int(np.sum(np.diff(np.sign(diffs)) != 0))
    print(f'  Sign changes in dB/dy: {sign_changes}  '
          f'({"MONOTONIC" if sign_changes == 0 else f"{sign_changes + 1} extrema"})')

    # Extrema locations
    if sign_changes > 0:
        print()
        print('  Local minima at y =', end=' ')
        mins = []
        for i in range(1, len(Bs) - 1):
            if Bs[i] < Bs[i - 1] and Bs[i] < Bs[i + 1]:
                mins.append(ys[i])
        print(', '.join(f'{y:.4f}' for y in mins) if mins else '(none in interior)')

        print('  Local maxima at y =', end=' ')
        maxs = []
        for i in range(1, len(Bs) - 1):
            if Bs[i] > Bs[i - 1] and Bs[i] > Bs[i + 1]:
                maxs.append(ys[i])
        print(', '.join(f'{y:.4f}' for y in maxs) if maxs else '(none in interior)')

    return results


# ----------------------------------------------------------------------
#  Stage 2: Heegner heights
# ----------------------------------------------------------------------

HEEGNER = [
    (3, np.sqrt(3) / 2, 'tau=(1+i*sqrt(3))/2'),
    (1, 1.0,            'tau=i'),
    (7, np.sqrt(7) / 2, 'tau=(1+i*sqrt(7))/2'),
    (2, np.sqrt(2),     'tau=i*sqrt(2)'),
]

def stage2_heegner(scan_results):
    print()
    print('=' * 76)
    print('  STAGE 2: HEEGNER HEIGHTS')
    print('=' * 76)
    print()
    sys.stdout.flush()

    print(f'  {"d":>4} {"y_d":>10} {"lam^2":>14} {"B(y_d)":>14} {"description":<30}')
    print('  ' + '-' * 78)

    heegner_results = []
    for d, y_d, desc in HEEGNER:
        r = barrier_at_y(float(y_d))
        print(f'  {d:4d} {y_d:10.6f} {r["lam_sq"]:14.1f} '
              f'{r["B"]:14.8f}  {desc}', flush=True)
        heegner_results.append((d, y_d, r, desc))

    # Smooth-baseline fit on scan (exclude Heegner points to avoid leakage)
    if scan_results:
        heegner_y_set = {round(y_d, 6) for _, y_d, _, _ in HEEGNER}
        smooth_data = [(r['y'], r['B']) for r in scan_results
                       if round(r['y'], 6) not in heegner_y_set]
        if len(smooth_data) >= 4:
            sy = np.array([p[0] for p in smooth_data])
            sb = np.array([p[1] for p in smooth_data])
            # B oscillates around ~0.05; fit a low-order trend
            poly = np.polyfit(sy, sb, 3)
            print()
            print('  Smooth cubic trend from non-Heegner scan points:')
            print(f'    B_smooth(y) = {poly[0]:+.4f}*y^3 + {poly[1]:+.4f}*y^2 '
                  f'+ {poly[2]:+.4f}*y + {poly[3]:+.4f}')
            print()
            print('  Heegner residuals:')
            print(f'  {"d":>4} {"y_d":>10} {"B(y_d)":>14} {"trend":>14} '
                  f'{"residual":>14} {"z-score":>10}')
            print('  ' + '-' * 72)
            # Z-score uses scan RMS residual as noise estimate
            trend_at_scan = np.polyval(poly, sy)
            rms = float(np.std(sb - trend_at_scan))
            for d, y_d, r, _ in heegner_results:
                trend = float(np.polyval(poly, y_d))
                residual = r['B'] - trend
                z = residual / rms if rms > 0 else 0.0
                print(f'  {d:4d} {y_d:10.6f} {r["B"]:14.8f} '
                      f'{trend:14.8f} {residual:+14.8f} {z:+10.3f}')
            print(f'\n  Scan RMS deviation from trend: {rms:.6f}')

    return heegner_results


# ----------------------------------------------------------------------
#  Stage 3: Modular inversion test
# ----------------------------------------------------------------------

def stage3_inversion():
    print()
    print('=' * 76)
    print('  STAGE 3: MODULAR INVERSION (z -> -1/z sends iy -> i/y)')
    print('=' * 76)
    print()
    print('  Only pairs with both y, 1/y in [0.20, 1.50] are computed.')
    print()
    sys.stdout.flush()

    y_list = [0.75, 0.85, 1.00, 1.15, 1.25, 1.40]
    y_list = [y for y in y_list if 0.20 <= 1.0 / y <= 1.50]

    print(f'  {"y":>8} {"1/y":>8} {"B(y)":>14} {"B(1/y)":>14} '
          f'{"sum":>14} {"ratio":>10} {"y*B(y)":>12} {"(1/y)*B(1/y)":>14}')
    print('  ' + '-' * 98)

    pairs = []
    for y in y_list:
        r1 = barrier_at_y(float(y))
        r2 = barrier_at_y(float(1.0 / y))
        b1, b2 = r1['B'], r2['B']
        print(f'  {y:8.4f} {1/y:8.4f} {b1:14.8f} {b2:14.8f} '
              f'{b1 + b2:14.8f} {b1/b2 if b2 else float("inf"):10.6f} '
              f'{y * b1:12.6f} {(1/y) * b2:14.6f}', flush=True)
        pairs.append((y, 1 / y, b1, b2))

    # Try a few weight candidates: does y^k*B(y) = (1/y)^k*B(1/y)?
    print()
    print('  Weight fit: find k such that y^k * B(y) approximately equals (1/y)^k * B(1/y)')
    if pairs:
        for k in [-2, -1, -0.5, 0, 0.5, 1, 2]:
            residuals = []
            for y, yi, b1, b2 in pairs:
                lhs = (y ** k) * b1
                rhs = (yi ** k) * b2
                if max(abs(lhs), abs(rhs)) > 0:
                    residuals.append(abs(lhs - rhs) / max(abs(lhs), abs(rhs)))
            if residuals:
                print(f'  k={k:+4.1f}: mean relative diff = {np.mean(residuals):.4f}, '
                      f'max = {np.max(residuals):.4f}')

    return pairs


# ----------------------------------------------------------------------
#  Stage 4: PSLQ
# ----------------------------------------------------------------------

def stage4_pslq(heegner_results):
    print()
    print('=' * 76)
    print('  STAGE 4: PSLQ ALGEBRAIC RECOGNITION AT HEEGNER POINTS')
    print('=' * 76)
    print()
    sys.stdout.flush()

    try:
        import mpmath
        from mpmath import mp, mpf, pi as mp_pi, log as mp_log, zeta as mp_zeta, sqrt as mp_sqrt
    except ImportError:
        print('  mpmath unavailable, skipping.')
        return

    mp.dps = 60
    basis = {
        '1'      : mpf(1),
        'pi'     : mp_pi,
        'pi^2'   : mp_pi ** 2,
        '1/pi'   : 1 / mp_pi,
        'log 2'  : mp_log(2),
        'log 3'  : mp_log(3),
        'log 5'  : mp_log(5),
        'zeta(3)': mp_zeta(3),
    }
    basis_names = list(basis.keys())
    basis_vals = [basis[n] for n in basis_names]

    print(f'  Basis: {{{", ".join(basis_names)}}}')
    print(f'  Precision: {mp.dps} dps')
    print('  NOTE: B comes from float64 scan (~14 digits). PSLQ tolerance')
    print('  is set loose (1e-8) to accommodate. Low-confidence recognition.')
    print()

    for d, y_d, r, desc in heegner_results:
        b = mpf(str(r['B']))
        vec = [b] + basis_vals
        try:
            rel = mpmath.pslq(vec, tol=mpf('1e-8'), maxcoeff=50000)
        except Exception as exc:
            print(f'  d={d:3d} B={float(b):+.8f}  PSLQ error: {exc}')
            continue

        if rel is None or all(c == 0 for c in rel):
            print(f'  d={d:3d} B={float(b):+.8f}  (no relation found)')
            continue

        c0 = rel[0]
        if c0 == 0:
            print(f'  d={d:3d} B={float(b):+.8f}  (degenerate, skip)')
            continue
        terms = []
        for coef, name in zip(rel[1:], basis_names):
            if coef != 0:
                sign = '-' if coef > 0 else '+'
                terms.append(f' {sign} {abs(coef)}*{name}')
        expr = ''.join(terms).lstrip(' +').lstrip(' -')
        check = sum(mpf(coef) * val for coef, val in zip(rel[1:], basis_vals))
        check = -check / mpf(rel[0])
        err = abs(check - b)
        print(f'  d={d:3d} B = ({expr}) / {c0}')
        print(f'          check: {float(check):+.8f}   |err|: {float(err):.2e}')


# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

if __name__ == '__main__':
    print()
    print('#' * 76)
    print('  SESSION 49b -- MODULAR STRUCTURE PROBE (FAST, PARTIAL BARRIER)')
    print('#' * 76)
    sys.stdout.flush()

    t_total = time.time()

    scan = stage1_dense_scan()
    heegner = stage2_heegner(scan)
    pairs = stage3_inversion()
    stage4_pslq(heegner)

    print()
    print('=' * 76)
    print(f'  DONE in {time.time() - t_total:.1f}s')
    print('=' * 76)
    print()
