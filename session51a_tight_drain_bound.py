"""
SESSION 51a -- TIGHT DRAIN BOUND

Thread 1 of Session 51, running in parallel with 51b.

Goal: bound |drain(L)| tightly enough that we can compare to margin(L)
across all L (not just a finite computed range).

Context from Session 46f:
  margin(L) -> 0.269 as L -> inf (analytic, monotonically increasing)
  |drain(L)| -> 0.240 as L -> inf (finite prime sum, oscillates)
  asymptotic gap -> 0.029 (small but positive in computed range)
  minimum numerical gap was never found negative in [lam^2 = 50, 10000]

Session 46f tried:
  (A) Triangle inequality: too loose
  (B) Equidistribution (RMS): too loose
  (C) Numerical max: works but only over finite L
  (E) Per-prime worst-case: too loose

This session takes a sharper empirical approach: compute |drain(L)| at
very fine L spacing, look for the genuine maximum over the whole range,
and check whether it tracks a simple functional form (e.g., first-prime
dominated). The goal is NOT to prove anything here -- it's to identify
the structure of the drain's L-dependence that a proof attempt would
need to exploit.

Specifically:
  1. Compute drain(L) densely across L in [1.5, 6.5] at dL = 0.005 (1000 pts)
  2. Decompose per-prime contributions at each L
  3. Find arg max_L |drain(L)| and identify which primes are "active" there
  4. Check: does max |drain| match sum of WORST absolute per-prime contributions?
     If so, there's phase locking; if not, there's cancellation.
  5. Fit: does |drain(L)| ~ C + D/L or similar simple form?
"""

import sys
import time

import numpy as np

sys.path.insert(0, '.')
from session41g_uncapped_barrier import sieve_primes, compute_barrier_partial
from session42j_margin_vs_drain import mprime_pnt_integral, mdiag_malpha
from session45n_pi_predicts_primes import w02_only


def drain_and_margin_at_L(L_val, N=None):
    """Compute (margin, drain) at a single L value."""
    lam_sq = int(round(np.exp(L_val)))
    if lam_sq < 2:
        lam_sq = 2
    if N is None:
        N = max(15, round(6 * L_val))

    w02 = w02_only(lam_sq, N)
    r = compute_barrier_partial(lam_sq, N)
    actual_mp = r['mprime']
    pnt_mp = mprime_pnt_integral(lam_sq, N)
    md, ma = mdiag_malpha(lam_sq, N)

    margin = w02 - pnt_mp - md - ma
    drain = actual_mp - pnt_mp
    return margin, drain, lam_sq, N


def run():
    print()
    print('#' * 76)
    print('  SESSION 51a -- TIGHT DRAIN BOUND PROBE')
    print('#' * 76)
    print()
    print('  Computing (margin, drain) densely across L in [1.5, 6.5]')
    print('  dL = 0.02 -> 250 points (kept modest for runtime; mdiag_malpha is slow)')
    sys.stdout.flush()

    L_values = np.arange(1.5, 6.5 + 0.01, 0.02)
    print(f'\n  {len(L_values)} L values, estimated ~{len(L_values) * 15 / 60:.0f} min')
    print()
    print(f'  {"L":>7} {"lam^2":>10} {"margin":>12} {"|drain|":>12} '
          f'{"gap":>12} {"min_gap":>12}')
    print('  ' + '-' * 72)
    sys.stdout.flush()

    margins = []
    drains = []
    min_gap = float('inf')
    min_gap_L = None

    t_total = time.time()
    for i, L in enumerate(L_values):
        t0 = time.time()
        margin, drain, lam_sq, N = drain_and_margin_at_L(float(L))
        margins.append(margin)
        drains.append(drain)
        gap = margin - abs(drain)
        if gap < min_gap:
            min_gap = gap
            min_gap_L = float(L)
        dt = time.time() - t0
        if i % 10 == 0 or i == len(L_values) - 1:
            print(f'  {L:7.3f} {lam_sq:10d} {margin:+12.6f} {abs(drain):12.6f} '
                  f'{gap:+12.6f} {min_gap:+12.6f}  ({dt:.1f}s)', flush=True)

    margins = np.array(margins)
    drains = np.array(drains)
    abs_drains = np.abs(drains)

    print(f'\n  Scan done in {time.time() - t_total:.1f}s')
    print()
    print(f'  margin: min={margins.min():+.6f} max={margins.max():+.6f}')
    print(f'  |drain|: min={abs_drains.min():.6f} max={abs_drains.max():.6f}')
    print(f'  minimum gap: {min_gap:+.6f} at L = {min_gap_L:.3f}')
    print(f'  max arg: L = {L_values[int(np.argmax(abs_drains))]:.3f} '
          f'|drain| = {abs_drains.max():.6f}')

    # Structure: is the maximum |drain| explained by a few primes at one L?
    max_idx = int(np.argmax(abs_drains))
    L_max = float(L_values[max_idx])
    print()
    print(f'  AT THE WORST L ({L_max:.3f}):')
    print(f'    lam^2 = {int(round(np.exp(L_max)))}')
    print(f'    |drain| = {abs_drains[max_idx]:.6f}')
    print(f'    margin = {margins[max_idx]:+.6f}')
    print(f'    gap = {margins[max_idx] - abs_drains[max_idx]:+.6f}')

    # Ratio of |drain| to margin across the scan
    ratio = abs_drains / margins
    print()
    print(f'  |drain| / margin statistics over scan:')
    print(f'    mean    = {ratio.mean():.4f}')
    print(f'    max     = {ratio.max():.4f}')
    print(f'    std dev = {ratio.std():.4f}')

    # Fit: |drain(L)| = a + b/L + c cos(omega L + phi)?
    # Start with simple: linear trend + dominant oscillation
    print()
    print('  Linear trend fit: |drain|(L) ~ a + b*L')
    A = np.column_stack([np.ones_like(L_values), L_values])
    coeffs, *_ = np.linalg.lstsq(A, abs_drains, rcond=None)
    lin_fit = A @ coeffs
    lin_res = abs_drains - lin_fit
    print(f'    a = {coeffs[0]:+.6f}, b = {coeffs[1]:+.6f}')
    print(f'    residual RMS = {lin_res.std():.6f}')
    print(f'    residual max = {np.max(np.abs(lin_res)):.6f}')

    # Save data for later analysis
    np.savez('session51a_scan.npz',
             L_values=L_values, margins=margins, drains=drains)
    print()
    print('  Data saved to session51a_scan.npz')

    # Verdict
    print()
    print('=' * 76)
    print('  VERDICT (empirical, not a proof)')
    print('=' * 76)
    if min_gap > 0.01:
        print(f'  Minimum gap over scan: {min_gap:.4f} (> 0.01)')
        print('  Margin exceeds |drain| with comfortable margin over the scanned L.')
        print('  Asymptotic behavior: margin -> 0.269, |drain| -> 0.240, gap -> 0.029.')
        print('  To upgrade to a PROOF, need: (a) bound for L > L_max via PNT error,')
        print('  (b) explicit-constant verification for L in [1.5, L_max].')
    elif min_gap > 0:
        print(f'  Minimum gap: {min_gap:+.6f} (> 0 but tight)')
        print('  Margin barely exceeds |drain|. A proof attempt needs tight bounds.')
    else:
        print(f'  Minimum gap: {min_gap:+.6f} (NEGATIVE)')
        print('  |drain| > margin at some L in the scan -- the margin-drain approach')
        print('  as currently formulated fails.')


if __name__ == '__main__':
    run()
