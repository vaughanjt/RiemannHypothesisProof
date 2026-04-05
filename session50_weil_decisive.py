"""
SESSION 50 -- DECISIVE WEIL-AMPLITUDE RESIDUAL TEST

Open thread from Session 49c: constant-amplitude cosine fit at known
zero frequencies + log primes explained only 94% of B_full. The 6%
residual was suspected to be amplitude mis-specification (the true
explicit-formula amplitudes are L-dependent, not constants).

This session uses a SUPERSET basis: polynomial-in-L modulation of each
cosine. That is, for each zero gamma_n, we include columns

    L^k * cos(gamma_n * L)   and   L^k * sin(gamma_n * L)

for k = 0, 1, ..., d_zero. Any smooth-in-L amplitude is captured by a
sufficiently large d_zero over a finite L interval. Same for primes.

Sweep over (K_zeros, K_primes, d_zero, d_prime) and watch R^2.

Decisive criteria:
  - If R^2 converges to ~1 (say, > 0.9999) as basis grows
        -> naive modular angle is DEAD with full confidence.
           B_full = explicit-formula content (zeros + primes with smooth
           L-modulation), no hidden structure beyond.
  - If R^2 plateaus below 0.999 even with large d_zero and K
        -> genuine content at frequencies NOT in the known zero/prime set
           -> LEAD worth deeper investigation.
  - Intermediate (0.99 < R^2 < 0.9999): ambiguous, report and discuss.

Reuses build_all_fast from session49c_weil_residual.py.
"""

import sys
import time

import numpy as np

sys.path.insert(0, '.')
from session49c_weil_residual import (
    barrier_full_at_L,
    load_zero_ordinates,
    first_n_log_primes,
    autocorr,
    esprit_frequencies,
)


# --------------------------------------------------------------------------
#  Basis construction
# --------------------------------------------------------------------------

def build_basis(L_values, gammas, log_primes, poly_deg, d_zero, d_prime):
    """
    Columns:
      1, L, ..., L^poly_deg                          (poly_deg+1 cols)
      for each g in gammas:
        L^k * cos(g L), L^k * sin(g L)  for k = 0..d_zero  (2*(d_zero+1) cols)
      for each lp in log_primes:
        L^k * cos(lp L), L^k * sin(lp L) for k = 0..d_prime (2*(d_prime+1) cols)
    """
    cols = [L_values ** k for k in range(poly_deg + 1)]
    for g in gammas:
        c = np.cos(g * L_values)
        s = np.sin(g * L_values)
        for k in range(d_zero + 1):
            Lk = L_values ** k if k > 0 else np.ones_like(L_values)
            cols.append(Lk * c)
            cols.append(Lk * s)
    for lp in log_primes:
        c = np.cos(lp * L_values)
        s = np.sin(lp * L_values)
        for k in range(d_prime + 1):
            Lk = L_values ** k if k > 0 else np.ones_like(L_values)
            cols.append(Lk * c)
            cols.append(Lk * s)
    return np.column_stack(cols)


def fit_and_residual(L_values, B_values, gammas, log_primes,
                     poly_deg, d_zero, d_prime):
    X = build_basis(L_values, gammas, log_primes, poly_deg, d_zero, d_prime)
    coeffs, *_ = np.linalg.lstsq(X, B_values, rcond=None)
    fit = X @ coeffs
    residual = B_values - fit
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((B_values - B_values.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rms = float(np.sqrt(np.mean(residual ** 2)))
    return X.shape[1], coeffs, residual, r2, rms


# --------------------------------------------------------------------------
#  Main
# --------------------------------------------------------------------------

def run():
    print()
    print('#' * 76)
    print('  SESSION 50 -- DECISIVE WEIL-AMPLITUDE RESIDUAL TEST')
    print('#' * 76)

    # ---- Compute B_full on uniform L grid ----
    dL = 0.02
    L_min, L_max = 1.0, 6.5
    L_values = np.arange(L_min, L_max + dL / 2, dL)
    n_pts = len(L_values)

    print(f'\n  Computing B_full at {n_pts} points, L in [{L_min}, {L_max}]')
    sys.stdout.flush()
    t0 = time.time()
    B = np.array([barrier_full_at_L(float(L)) for L in L_values])
    print(f'  done in {time.time() - t0:.1f}s')
    print(f'  B range: [{B.min():+.6f}, {B.max():+.6f}]   span: {B.max()-B.min():.6f}')

    B_span = float(B.max() - B.min())

    # ---- Load zeros and primes ----
    # Use 50 zeros -- Session 50 showed ESPRIT finding gamma_40, gamma_41, gamma_43
    # in the residual when only 30 were fitted.
    gammas_all = load_zero_ordinates(60)
    log_primes_all = first_n_log_primes(30)
    print(f'\n  First 5 gammas: {[round(g,3) for g in gammas_all[:5]]}')
    print(f'  First 5 log primes: {[round(lp,3) for lp in log_primes_all[:5]]}')

    # ---- Sweep basis configurations ----
    print()
    print('=' * 76)
    print('  BASIS SWEEP -- L-polynomial modulation of each cosine')
    print('=' * 76)
    print()
    print(f'  {"K_zeros":>8} {"d_zero":>7} {"K_prim":>7} {"d_prim":>7} '
          f'{"cols":>6} {"R^2":>16} {"RMS_res":>14} {"RMS/span":>12}')
    print('  ' + '-' * 92)
    sys.stdout.flush()

    results = {}
    configs = [
        # (K_zeros, d_zero, K_primes, d_prime)
        # Baseline trend: constant amplitude, low K
        (20, 0, 20, 0),
        (30, 0, 30, 0),
        # Add L modulation, keep K moderate
        (20, 1, 20, 1),
        (30, 1, 30, 1),
        (30, 2, 30, 1),
        # Push K higher — the key test: does R^2 -> 1 as K grows?
        (40, 1, 30, 1),
        (40, 2, 30, 1),
        (50, 1, 30, 1),
        (50, 2, 30, 1),
        (60, 1, 30, 1),
        (60, 2, 30, 1),
        # Aggressive: large K + aggressive modulation
        (60, 3, 30, 2),
    ]

    poly_deg_trend = 4  # smooth trend polynomial (same for all configs)
    for K_z, d_z, K_p, d_p in configs:
        gammas = gammas_all[:K_z]
        logps = log_primes_all[:K_p]
        try:
            n_cols, _, res, r2, rms = fit_and_residual(
                L_values, B, gammas, logps, poly_deg_trend, d_z, d_p)
            print(f'  {K_z:>8d} {d_z:>7d} {K_p:>7d} {d_p:>7d} '
                  f'{n_cols:>6d} {r2:>16.12f} {rms:>14.3e} '
                  f'{rms / B_span:>12.3e}', flush=True)
            results[(K_z, d_z, K_p, d_p)] = (n_cols, res, r2, rms)
        except Exception as exc:
            print(f'  {K_z:>8d} {d_z:>7d} {K_p:>7d} {d_p:>7d}  FAILED: {exc}')

    # ---- Best fit analysis ----
    best_key, (best_cols, best_res, best_r2, best_rms) = max(
        results.items(), key=lambda kv: kv[1][2])
    K_z, d_z, K_p, d_p = best_key
    print()
    print('=' * 76)
    print(f'  BEST FIT: K_z={K_z}, d_z={d_z}, K_p={K_p}, d_p={d_p}, cols={best_cols}')
    print('=' * 76)
    print(f'  R^2                 = {best_r2:.12f}')
    print(f'  RMS residual        = {best_rms:.3e}')
    print(f'  Residual range      = [{best_res.min():+.3e}, {best_res.max():+.3e}]')
    print(f'  RMS / B_span        = {best_rms / B_span:.3e}')
    print(f'  1 - R^2             = {1 - best_r2:.3e}  '
          f'(fraction of variance unexplained)')

    # Autocorrelation
    acf = autocorr(best_res, max_lag=50)
    print()
    print('  Residual autocorrelation:')
    for lag in (0, 1, 2, 5, 10, 20, 49):
        print(f'    lag {lag:3d}: {acf[lag]:+.4f}')
    acf_tail = float(np.sum(acf[1:] ** 2))
    print(f'  sum(acf[1:]^2)       = {acf_tail:.4f}  '
          f'(noise -> small, structure -> large)')

    # ESPRIT on residual
    print()
    print('  ESPRIT on residual (looking for unexplained frequencies):')
    for n_comp in (5, 10):
        try:
            freqs, _ = esprit_frequencies(best_res, n_comp, dL)
            pos = np.sort(freqs[freqs > 0.5])
            if len(pos) == 0:
                print(f'    n_comp={n_comp}: no positive frequencies')
                continue
            print(f'    n_comp={n_comp}: {len(pos)} positive frequencies')
            for f in pos[:8]:
                near_g = float(min(gammas_all, key=lambda g: abs(g - f)))
                err_g = abs(f - near_g)
                near_p = float(min(log_primes_all, key=lambda lp: abs(lp - f)))
                err_p = abs(f - near_p)
                tag = 'unassigned'
                if err_g < 1.0:
                    tag = f'near gamma~{near_g:.2f}'
                elif err_p < 0.3:
                    tag = f'near log p~{near_p:.2f}'
                print(f'      f={f:8.4f}  {tag}  (err_g={err_g:.3f}, err_p={err_p:.3f})')
        except Exception as exc:
            print(f'    n_comp={n_comp}: FAILED ({exc})')

    # ---- Verdict ----
    print()
    print('=' * 76)
    print('  VERDICT')
    print('=' * 76)
    unexplained = 1 - best_r2
    if unexplained < 1e-4:
        verdict = (
            "DEAD CLEAN: R^2 > 0.9999, basically machine-precision fit.\n"
            "  B_full is zero + prime oscillations with smooth L-modulation.\n"
            "  There is NO residual structure beyond the explicit formula.\n"
            "  The naive modular angle is decisively ruled out."
        )
    elif unexplained < 1e-3:
        verdict = (
            "DEAD (with small caveat): R^2 in (0.999, 0.9999). Most of\n"
            "  the variance is captured by zeros + primes with smooth\n"
            "  modulation. The small remaining residual is plausibly\n"
            "  finite-basis truncation. Modular angle effectively dead."
        )
    elif unexplained < 0.01:
        verdict = (
            "LEANING DEAD: R^2 in (0.99, 0.999). Significant improvement\n"
            "  over constant-amplitude baseline (Session 49c: 0.940), but\n"
            "  residual is still above numerical precision. Decide based on\n"
            "  ESPRIT output and acf structure: if truly structureless, dead;\n"
            "  if autocorrelated, warrants another look."
        )
    else:
        verdict = (
            f"NOT DEAD: R^2 = {best_r2:.4f}, {100*unexplained:.1f}% unexplained.\n"
            "  Even with L-polynomial-modulated cosines at known frequencies,\n"
            "  a significant chunk of B_full is NOT captured by known\n"
            "  zero/prime content. This is a real LEAD. Examine ESPRIT output\n"
            "  for the frequency location of the residual."
        )
    print('  ' + verdict.replace('\n', '\n  '))
    print()


if __name__ == '__main__':
    run()
