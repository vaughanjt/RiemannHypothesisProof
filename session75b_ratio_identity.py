"""
SESSION 75b -- IS THE COUPLING RATIO RELATED TO A KNOWN CONSTANT?

The coupling ratio at Schur step 0 satisfies:
  1 - ratio ~ 0.000245 * L^{-4.54}

Is the exponent -4.54 exactly -9/2? Is 0.000245 a recognizable constant?
Is L^{-4.54} really the right model, or is it (something)^{-L}?

More importantly: is the MARGIN (= |a_1| - coupling) itself a recognizable
quantity? It's ~5e-7 at lam^2=1000 and shrinks with lambda.

Probes:
  1. Precision test: is the exponent exactly -9/2 or -4.5 or something else?
  2. Is the margin related to lambda_max(M_odd)? (they're both ~10^{-7})
  3. Is the margin related to the Schur margin from Session 62?
  4. Does the MARGIN (not the ratio) have a cleaner scaling law?
  5. Test alternative models: exponential, log-correction, etc.
  6. Connection to the displacement identity?
"""

import sys
import numpy as np

sys.path.insert(0, '.')
from session49c_weil_residual import build_all_fast


def odd_block(M, N):
    dim = 2 * N + 1
    P = np.zeros((dim, N))
    for n in range(1, N + 1):
        P[N + n, n - 1] = 1.0 / np.sqrt(2)
        P[N - n, n - 1] = -1.0 / np.sqrt(2)
    return P.T @ M @ P


def run():
    print()
    print('#' * 76)
    print('  SESSION 75b -- IS THE EXPONENT -9/2?')
    print('#' * 76)

    # ==================================================================
    # STEP 1: Dense lambda scan of the margin
    # ==================================================================
    print(f'\n  === STEP 1: DENSE MARGIN SCAN ===\n')

    data = []
    print(f'  {"lam^2":>8} {"L":>8} {"|a_1|":>12} {"coupling":>12} '
          f'{"margin":>14} {"lam_max(Mo)":>14}')
    print('  ' + '-' * 72)

    for lam_sq in [15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 500,
                    750, 1000, 1500, 2000, 3000, 5000, 7500, 10000,
                    15000, 20000, 30000, 50000]:
        try:
            L = np.log(lam_sq)
            N = max(15, round(6 * L))
            _, M, _ = build_all_fast(lam_sq, N)
            Mo = odd_block(M, N)

            a1 = Mo[0, 0]
            c = Mo[0, 1:]
            B = Mo[1:, 1:]
            Binv_c = np.linalg.solve(B, c)
            coupling = -float(c @ Binv_c)
            margin = abs(a1) - coupling

            lam_max = np.linalg.eigvalsh(Mo)[-1]

            data.append((lam_sq, L, abs(a1), coupling, margin, lam_max))
            print(f'  {lam_sq:>8d} {L:>8.4f} {abs(a1):>12.6f} {coupling:>12.6f} '
                  f'{margin:>+14.6e} {lam_max:>+14.6e}')
        except Exception as e:
            print(f'  {lam_sq:>8d} ERROR: {e}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 2: Is margin proportional to lambda_max(M_odd)?
    # ==================================================================
    print(f'\n  === STEP 2: MARGIN vs LAMBDA_MAX(M_ODD) ===\n')

    margins = np.array([d[4] for d in data])
    lam_maxs = np.array([d[5] for d in data])
    Ls = np.array([d[1] for d in data])

    ratios_ml = margins / np.abs(lam_maxs)
    print(f'  margin / |lambda_max(M_odd)|:')
    print(f'  {"lam^2":>8} {"margin":>14} {"|lam_max|":>14} {"ratio":>12}')
    print('  ' + '-' * 52)
    for i, d in enumerate(data):
        print(f'  {d[0]:>8d} {margins[i]:>14.6e} {abs(lam_maxs[i]):>14.6e} '
              f'{ratios_ml[i]:>12.6f}')

    print(f'\n  Ratio range: [{ratios_ml.min():.4f}, {ratios_ml.max():.4f}]')
    print(f'  Mean: {ratios_ml.mean():.4f}, Std: {ratios_ml.std():.4f}')
    print(f'  Constant? {ratios_ml.std() / ratios_ml.mean() < 0.1}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 3: Fit margin to various models
    # ==================================================================
    print(f'\n  === STEP 3: MARGIN SCALING MODELS ===\n')

    valid = margins > 0
    Ls_v = Ls[valid]
    margins_v = margins[valid]
    log_L = np.log(Ls_v)
    log_m = np.log(margins_v)

    # Model A: margin ~ a * L^b (power law)
    fit_A = np.polyfit(log_L, log_m, 1)
    resid_A = np.std(log_m - np.polyval(fit_A, log_L))
    print(f'  Model A (power law): margin ~ {np.exp(fit_A[1]):.6e} * L^{fit_A[0]:.4f}')
    print(f'    Residual std: {resid_A:.6f}')

    # Model B: margin ~ a * L^b * (log L)^c — hard to fit, try b fixed at -4.5
    pred_B = -4.5 * log_L
    fit_B_intercept = np.mean(log_m - pred_B)
    resid_B = np.std(log_m - pred_B - fit_B_intercept)
    print(f'  Model B (L^{{-4.5}}): margin ~ {np.exp(fit_B_intercept):.6e} * L^{{-4.5}}')
    print(f'    Residual std: {resid_B:.6f}')

    # Model C: margin ~ a * exp(-b * L) (exponential)
    fit_C = np.polyfit(Ls_v, log_m, 1)
    resid_C = np.std(log_m - np.polyval(fit_C, Ls_v))
    print(f'  Model C (exponential): margin ~ {np.exp(fit_C[1]):.6e} * exp({fit_C[0]:.4f} * L)')
    print(f'    Residual std: {resid_C:.6f}')

    # Model D: margin ~ a / L^b with b from data
    # Already done as Model A
    # Model E: margin ~ a * L^b * exp(-c*L)
    # Fit: log(margin) = log(a) + b*log(L) - c*L
    A_E = np.column_stack([np.ones_like(Ls_v), log_L, Ls_v])
    fit_E, _, _, _ = np.linalg.lstsq(A_E, log_m, rcond=None)
    resid_E = np.std(log_m - A_E @ fit_E)
    print(f'  Model E (power*exp): margin ~ {np.exp(fit_E[0]):.6e} * L^{fit_E[1]:.4f} * exp({fit_E[2]:.6f}*L)')
    print(f'    Residual std: {resid_E:.6f}')

    print(f'\n  Best fit: {"A" if resid_A == min(resid_A, resid_B, resid_C, resid_E) else "B" if resid_B == min(resid_A, resid_B, resid_C, resid_E) else "C" if resid_C == min(resid_A, resid_B, resid_C, resid_E) else "E"}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 4: Is the exponent exactly -9/2?
    # ==================================================================
    print(f'\n  === STEP 4: TESTING EXACT EXPONENTS ===\n')

    for exp_name, exp_val in [('-4', -4), ('-9/2', -4.5), ('-4.54', -4.54),
                                ('-14/3', -14/3), ('-5', -5), ('-pi', -np.pi),
                                ('-sqrt(2)*pi', -np.sqrt(2)*np.pi)]:
        pred = exp_val * log_L
        intercept = np.mean(log_m - pred)
        resid = np.std(log_m - pred - intercept)
        coeff = np.exp(intercept)
        print(f'  L^{{{exp_name:>10s}}} = L^{exp_val:>8.4f}: '
              f'coeff = {coeff:.6e}, residual = {resid:.6f}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 5: What IS the margin?
    # ==================================================================
    print(f'\n  === STEP 5: THE MARGIN IN ABSOLUTE TERMS ===\n')

    # The margin = |a_1| - coupling = |a_1| * (1 - ratio)
    # = |a_1| * C * L^{-4.54}
    # Since |a_1| ~ L (from Step 1 data), margin ~ L * L^{-4.54} = L^{-3.54}

    # But let's check |a_1| scaling
    a1s = np.array([d[2] for d in data])
    fit_a1 = np.polyfit(log_L, np.log(a1s), 1)
    print(f'  |a_1| scaling: |a_1| ~ {np.exp(fit_a1[1]):.4f} * L^{fit_a1[0]:.4f}')

    # So margin ~ |a_1| * (1-ratio) ~ L^{fit_a1[0]} * L^{exponent}
    combined_exp = fit_a1[0] + fit_A[0]  # should give margin ~ L^combined
    print(f'  Combined: margin ~ L^{combined_exp:.4f}')
    print(f'  Direct fit: margin ~ L^{fit_A[0]:.4f}')
    print(f'  Consistency: {abs(combined_exp - fit_A[0]) < 0.2}')

    # Is the margin related to lambda_max via a simple constant?
    # From Step 2: ratio_ml is not constant. Let's check if margin/|lam_max| has a trend
    valid2 = np.abs(lam_maxs) > 1e-15
    fit_ml = np.polyfit(log_L[valid2], np.log(ratios_ml[valid2]), 1)
    print(f'\n  margin/|lam_max| trend: {np.exp(fit_ml[1]):.4f} * L^{fit_ml[0]:.4f}')
    print(f'  (if exponent ~ 0, they scale together)')
    sys.stdout.flush()

    # ==================================================================
    # STEP 6: Connection to known constants
    # ==================================================================
    print(f'\n  === STEP 6: MATCHING CONSTANTS ===\n')

    # The margin at L=1 (extrapolated): margin(L=1) ~ coeff from fit
    coeff_margin = np.exp(fit_A[1])
    print(f'  Extrapolated margin at L=1: {coeff_margin:.6e}')

    # Known constants for comparison
    candidates = [
        ('1/gamma_1^2', 1/14.134725**2),
        ('1/(4*pi^2)', 1/(4*np.pi**2)),
        ('1/(2*pi)', 1/(2*np.pi)),
        ('Z/2', 0.0462),
        ('Z', 0.0924),
        ('1/e^3', np.exp(-3)),
        ('gamma_E/100', 0.5772/100),
        ('2*pi*Z', 2*np.pi*0.0462),
    ]

    print(f'  Candidate matches for margin coefficient {coeff_margin:.6e}:')
    for name, val in candidates:
        ratio = coeff_margin / val
        print(f'    {name:>15s} = {val:.6e}, ratio = {ratio:.4f}')

    # The 1-ratio coefficient
    coeff_1mr = 0.000245
    print(f'\n  Candidate matches for (1-ratio) coefficient {coeff_1mr:.6e}:')
    for name, val in candidates:
        ratio = coeff_1mr / val
        print(f'    {name:>15s} = {val:.6e}, ratio = {ratio:.6f}')
    sys.stdout.flush()

    # ==================================================================
    # VERDICT
    # ==================================================================
    print()
    print('=' * 76)
    print('  SESSION 75b VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
