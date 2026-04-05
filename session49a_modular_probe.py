"""
SESSION 49a -- MODULAR STRUCTURE PROBE OF B(y)

Follow-up to Session 48's conjugate Poisson discovery. The barrier
  B(L) = <Q_y, Q_W Q_y>   with y = L/(4*pi)
is a function on SL(2,Z)\H. Conjecture 3 in structural_analysis_draft.tex
asks whether B(y) admits an expansion in classical automorphic objects.

This is a first probe of two live angles:

  (b) B(y) as function on the fundamental domain      (Conjecture 3)
  (c) Ramanujan-style evaluation at Heegner heights   (roadmap Phase 7)

Four tests, in order of escalating expense:

  1. Dense B(y) scan on y in [0.15, 1.66] (25 points).
     Refines the y=0.8 dip discovered in Session 48, maps shape.

  2. Heegner heights y_d = Im(tau_d) for d in {3, 1, 7}
     (class-number-one values with lam_sq = exp(4*pi*y) tractable
     for naive prime sieving; d=2,11,19,43,67,163 require lam_sq
     > 4e7 which the current build_all can't reach cheaply).

  3. Modular inversion test: compare B(y) against B(1/y).
     The SL(2,Z) generator z -> -1/z sends iy -> i/y. If B has any
     SL(2,Z) equivariance (even approximate), this should be visible.

  4. PSLQ algebraic recognition at Heegner points against a basis
     {1, pi, pi^2, log 2, log 3, zeta(3)} at 50-dps precision.

KILL CRITERIA (honest):
  If B(y) is smooth and featureless, Heegner values are unremarkable,
  B(y) vs B(1/y) has no structure, and PSLQ finds nothing — this angle
  joins heat kernel / RKHS / Rankin-Selberg in the dead pile.

  If any ONE of the four tests lights up cleanly, that's a lead.
"""

import sys
import time

import numpy as np

sys.path.insert(0, '.')
from connes_crossterm import build_all

# connes_crossterm sets mp.dps = 50 at import time. Outputs of build_all are
# cast to float64 before we touch them, so ~25 dps is ample and ~2x faster
# in the alpha / digamma loop that dominates per-point cost.
import mpmath
mpmath.mp.dps = 25


# ----------------------------------------------------------------------
#  Core barrier evaluation (float64 bulk, mpmath for precision re-runs)
# ----------------------------------------------------------------------

_Y_CACHE = {}

def barrier_at_y(y, N=None, dps=None):
    """
    Compute B(y) = <Q_y_hat, Q_W Q_y_hat> where Q_y is the conjugate
    Poisson kernel at height y and Q_W is the Connes cross-term quadratic
    form at lam_sq = exp(4*pi*y).

    y       : height in upper half-plane (> 0).
    N       : half-width of matrix (dim = 2N+1). None -> auto.
    dps     : if not None, returns mpmath-precision B (slower).
    """
    cache_key = round(float(y), 8)
    if cache_key in _Y_CACHE:
        return _Y_CACHE[cache_key]

    L_f = 4.0 * np.pi * y
    lam_sq = np.exp(L_f)
    if N is None:
        # Need enough Fourier modes so Q_y is well-resolved.
        # For y small, Q_y(n) = n/(y^2+n^2) decays as 1/n; need ~ 6*L modes.
        N = max(20, int(round(6 * L_f)))

    W02, M, QW = build_all(lam_sq, N)

    ns = np.arange(-N, N + 1, dtype=float)
    kernel = ns / (y * y + ns * ns)     # conjugate Poisson, odd, 0 at n=0
    kernel[N] = 0.0
    norm = np.linalg.norm(kernel)
    if norm > 0:
        kernel_hat = kernel / norm
    else:
        kernel_hat = kernel

    w02 = float(kernel_hat @ W02 @ kernel_hat)
    m = float(kernel_hat @ M @ kernel_hat)
    b = float(kernel_hat @ QW @ kernel_hat)

    result = {
        'y': y, 'L': L_f, 'lam_sq': lam_sq, 'N': N,
        'W02': w02, 'M': m, 'B': b,
    }
    _Y_CACHE[cache_key] = result
    return result


# ----------------------------------------------------------------------
#  Stage 1: Dense B(y) scan
# ----------------------------------------------------------------------

def stage1_dense_scan():
    print()
    print('=' * 74)
    print('  STAGE 1: DENSE B(y) SCAN')
    print('=' * 74)
    print()
    print('  Tractable range y in [0.15, 1.35] (lam_sq <= 1.3e7).')
    print('  Looking for: extrema, monotonicity, fundamental-domain features.')
    print()
    sys.stdout.flush()

    # Denser near known dip at y=0.8, sparser elsewhere.
    # y > 1.35 requires lam_sq > 1.3e7 which makes prime sieving costly.
    # We include the 3 in-range Heegner heights (sqrt(3)/2, 1, sqrt(7)/2)
    # directly in the scan so Stage 2 reads them from cache.
    heegner_in_scan = [np.sqrt(3) / 2, 1.0, np.sqrt(7) / 2]
    y_values = np.concatenate([
        np.linspace(0.20, 0.55, 8),
        np.linspace(0.62, 0.95, 6),
        np.array([1.15, 1.30]),
        np.array(heegner_in_scan),
    ])
    y_values = np.sort(np.unique(np.round(y_values, 8)))

    print(f'  {"y":>8} {"lam^2":>12} {"N":>5} {"B(y)":>14} '
          f'{"W02":>12} {"M":>12} {"dt":>6}')
    print('  ' + '-' * 72)

    results = []
    for y in y_values:
        t0 = time.time()
        try:
            r = barrier_at_y(float(y))
            dt = time.time() - t0
            print(f'  {r["y"]:8.4f} {r["lam_sq"]:12.1f} {r["N"]:5d} '
                  f'{r["B"]:14.8f} {r["W02"]:12.6f} {r["M"]:12.6f} '
                  f'{dt:6.1f}s', flush=True)
            results.append(r)
        except Exception as exc:
            dt = time.time() - t0
            print(f'  {y:8.4f}  ERROR ({dt:.1f}s): {exc}', flush=True)

    # Extrema and monotonicity
    ys = np.array([r['y'] for r in results])
    Bs = np.array([r['B'] for r in results])

    print()
    min_idx = int(np.argmin(Bs))
    max_idx = int(np.argmax(Bs))
    print(f'  Global min: B = {Bs[min_idx]:.8f} at y = {ys[min_idx]:.4f}')
    print(f'  Global max: B = {Bs[max_idx]:.8f} at y = {ys[max_idx]:.4f}')

    diffs = np.diff(Bs)
    sign_changes = int(np.sum(np.diff(np.sign(diffs)) != 0))
    print(f'  Sign changes in dB/dy: {sign_changes}')
    if sign_changes == 0:
        print('  Verdict: B(y) MONOTONIC on scanned range.')
    else:
        print(f'  Verdict: B(y) NON-MONOTONIC ({sign_changes + 1} regions).')

    return results


# ----------------------------------------------------------------------
#  Stage 2: Heegner heights
# ----------------------------------------------------------------------

# Class-number-1 discriminants. For d with d mod 4 == 3, the fundamental
# domain representative is tau_d = (1 + i*sqrt(d))/2, so height y_d = sqrt(d)/2.
# For d mod 4 in {1, 2}, the representative is tau_d = i*sqrt(d), height sqrt(d).
# Class number 1 discriminants: 1, 2, 3, 7, 11, 19, 43, 67, 163.
# Cap y_d <= 1.35 -> only d = 3, 1, 7 fit in the tractable range.
# d = 2 (y=1.414), 11 (1.658), 19+ (2.18+) require lam_sq > 4e7 and are
# deferred pending a sieve-free construction.

HEEGNER = [
    # (d, y = Im(tau_d), tau formula)
    (3, np.sqrt(3) / 2, 'tau = (1+i*sqrt(3))/2, d=3'),
    (1, 1.0,            'tau = i, d=1'),
    (7, np.sqrt(7) / 2, 'tau = (1+i*sqrt(7))/2, d=7'),
]

def stage2_heegner(scan_results):
    print()
    print('=' * 74)
    print('  STAGE 2: HEEGNER HEIGHTS')
    print('=' * 74)
    print()
    print('  B(y) evaluated at class-number-1 CM heights (tractable subset).')
    print('  Looking for: anomalous values, clean ratios, pattern vs neighbors.')
    print()

    print(f'  {"d":>4} {"y_d":>10} {"lam^2":>14} {"N":>5} '
          f'{"B(y_d)":>14} {"tau":>30}')
    print('  ' + '-' * 82)

    heegner_results = []
    for d, y_d, desc in HEEGNER:
        t0 = time.time()
        try:
            r = barrier_at_y(float(y_d))
            dt = time.time() - t0
            print(f'  {d:4d} {y_d:10.6f} {r["lam_sq"]:14.1f} {r["N"]:5d} '
                  f'{r["B"]:14.8f}  {desc}  ({dt:.1f}s)', flush=True)
            heegner_results.append((d, y_d, r, desc))
        except Exception as exc:
            print(f'  {d:4d} {y_d:10.6f}  ERROR: {exc}', flush=True)

    # Compare Heegner values against a smooth baseline fit from scan
    # using log-log linear regression on the scan tail.
    if scan_results:
        scan_ys = np.array([r['y'] for r in scan_results])
        scan_Bs = np.array([r['B'] for r in scan_results])
        mask = scan_ys > 0.3  # avoid boundary effects
        if np.sum(mask) >= 4:
            poly = np.polyfit(scan_ys[mask], np.log(np.abs(scan_Bs[mask])), 2)
            print()
            print('  Smooth baseline (quadratic log-fit from Stage 1):')
            print(f'    log|B(y)| ~ {poly[0]:+.4f}*y^2 + {poly[1]:+.4f}*y + '
                  f'{poly[2]:+.4f}')
            print()
            print('  Heegner residuals vs smooth baseline:')
            print(f'  {"d":>4} {"y_d":>10} {"B(y_d)":>14} '
                  f'{"smooth":>14} {"residual":>14} {"ratio":>8}')
            print('  ' + '-' * 72)
            for d, y_d, r, _ in heegner_results:
                smooth = np.exp(np.polyval(poly, y_d))
                residual = r['B'] - smooth
                ratio = r['B'] / smooth if smooth != 0 else float('inf')
                print(f'  {d:4d} {y_d:10.6f} {r["B"]:14.8f} '
                      f'{smooth:14.8f} {residual:+14.8f} {ratio:8.4f}')

    return heegner_results


# ----------------------------------------------------------------------
#  Stage 3: Modular inversion test
# ----------------------------------------------------------------------

def stage3_inversion():
    print()
    print('=' * 74)
    print('  STAGE 3: MODULAR INVERSION TEST')
    print('=' * 74)
    print()
    print('  SL(2,Z) generator S: z -> -1/z sends iy -> i/y.')
    print('  If B is SL(2,Z)-equivariant under any simple representation,')
    print('  B(y) and B(1/y) should be related by a clean transformation.')
    print('  Both y and 1/y must be in [0.15, 1.35] -> y in [0.74, 1.35].')
    print()
    sys.stdout.flush()

    # Tight inversion probe: only 3 pairs, pick y values so 1/y is also tractable
    y_pairs = [0.80, 1.00, 1.20]

    print(f'  {"y":>8} {"1/y":>8} {"B(y)":>14} {"B(1/y)":>14} '
          f'{"B(y)+B(1/y)":>14} {"B(y)*B(1/y)":>14} {"B(y)/B(1/y)":>12}')
    print('  ' + '-' * 92)

    pairs = []
    for y in y_pairs:
        if 1.0 / y > 1.35 or 1.0 / y < 0.15:
            continue
        try:
            r1 = barrier_at_y(float(y))
            r2 = barrier_at_y(float(1.0 / y))
            b1, b2 = r1['B'], r2['B']
            print(f'  {y:8.4f} {1/y:8.4f} {b1:14.8f} {b2:14.8f} '
                  f'{b1+b2:14.8f} {b1*b2:14.8f} '
                  f'{b1/b2 if b2 else float("inf"):12.6f}', flush=True)
            pairs.append((y, 1/y, b1, b2))
        except Exception as exc:
            print(f'  {y:8.4f}  ERROR: {exc}', flush=True)

    # Check simple invariants
    if pairs:
        print()
        print('  Simple-invariant checks:')
        print(f'  {"y":>8} {"y*B(y)-y^-1*B(1/y)":>22} {"y^s*B(y)-y^{-s}*B(1/y), s=0.5":>34}')
        print('  ' + '-' * 70)
        for y, yi, b1, b2 in pairs:
            inv1 = y * b1 - yi * b2
            inv2 = (y ** 0.5) * b1 - (yi ** 0.5) * b2
            print(f'  {y:8.4f} {inv1:22.8f} {inv2:34.8f}')

    return pairs


# ----------------------------------------------------------------------
#  Stage 4: PSLQ algebraic recognition
# ----------------------------------------------------------------------

def stage4_pslq(heegner_results):
    print()
    print('=' * 74)
    print('  STAGE 4: PSLQ ALGEBRAIC RECOGNITION AT HEEGNER POINTS')
    print('=' * 74)
    print()

    try:
        import mpmath
        from mpmath import mp, mpf, pi as mp_pi, log as mp_log, zeta as mp_zeta
    except ImportError:
        print('  mpmath unavailable, skipping.')
        return

    mp.dps = 60
    basis = {
        '1'       : mpf(1),
        'pi'      : mp_pi,
        'pi^2'    : mp_pi ** 2,
        'pi^-1'   : 1 / mp_pi,
        'log 2'   : mp_log(2),
        'log 3'   : mp_log(3),
        'zeta(3)' : mp_zeta(3),
    }
    basis_names = list(basis.keys())
    basis_vals = [basis[n] for n in basis_names]

    print(f'  Basis: {{{", ".join(basis_names)}}}')
    print(f'  Precision: {mp.dps} dps')
    print('  Note: B(y_d) values come from float64 scan (~15 dps). PSLQ needs')
    print('  higher precision to be trustworthy -- this is a low-confidence probe.')
    print()

    for d, y_d, r, desc in heegner_results:
        b = mpf(str(r['B']))  # float64 precision only
        vec = [b] + basis_vals
        try:
            # Relatively loose tolerance because b is only ~1e-15 accurate
            rel = mpmath.pslq(vec, tol=mpf('1e-10'), maxcoeff=10000)
        except Exception as exc:
            rel = None
            print(f'  d={d:3d} y={y_d:.6f} B={float(b):.8f}  PSLQ error: {exc}')
            continue

        if rel is None or all(c == 0 for c in rel):
            print(f'  d={d:3d} y={y_d:.6f} B={float(b):.8f}  (no relation found)')
        else:
            c0 = rel[0]
            if c0 == 0:
                print(f'  d={d:3d} y={y_d:.6f} B={float(b):.8f}  (degenerate relation)')
                continue
            terms = []
            for coef, name in zip(rel[1:], basis_names):
                if coef != 0:
                    terms.append(f'{-coef}*{name}')
            expr = ' + '.join(terms) if terms else '0'
            print(f'  d={d:3d} y={y_d:.6f} B ~ ({expr}) / {c0}')
            # Verify
            check = mpf(0)
            for coef, val in zip(rel[1:], basis_vals):
                check += mpf(coef) * val
            check = -check / mpf(rel[0])
            err = abs(check - b)
            print(f'           check: {float(check):.8f}  err: {float(err):.2e}')


# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

if __name__ == '__main__':
    print()
    print('#' * 74)
    print('  SESSION 49a -- MODULAR STRUCTURE PROBE OF B(y)')
    print('#' * 74)

    t_total = time.time()

    scan = stage1_dense_scan()
    heegner = stage2_heegner(scan)
    pairs = stage3_inversion()
    stage4_pslq(heegner)

    print()
    print('=' * 74)
    print(f'  DONE in {time.time() - t_total:.1f}s')
    print('=' * 74)
    print()
