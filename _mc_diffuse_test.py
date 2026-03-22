"""Monte Carlo significance test for the diffuse 30% residual.

The three-component decomposition of zeta zero ACF excess:
  1. Oscillatory (~53%): prime-frequency cosines
  2. Short-range (~17%): enhanced level repulsion
  3. Diffuse (~30%): collectively detectable (chi2 z~3.6) but individually sub-threshold

This test determines whether chi2 z is expected from GUE finite-sample
noise + ACF lag correlation, or indicates genuine third-component structure.

Method: Generate 500 pure-GUE samples of ~10k spacings each, run the same
analysis pipeline, build null distribution of chi2 z-scores, compare to observed.
"""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from scipy.linalg import eigvalsh_tridiagonal
from sympy import primerange
from riemann.analysis.bost_connes_operator import spacing_autocorrelation, polynomial_unfold

# ============================================================
# CONSTANTS (matching _trace_convergence.py / _selberg_convergence.py)
# ============================================================
MAX_LAG = 400
T_BASE = 267653395647.0
LOG_T = np.log(T_BASE / (2 * np.pi))
GUE_N = 1200
N_BASELINE = 100
N_TRIALS = 500
MATS_PER_TRIAL = 11  # 11 * ~959 spacings ≈ 10.5k per trial
BEST_ALPHA = 0.758

# ============================================================
# FAST GUE EIGENVALUE GENERATION (Dumitriu-Edelman tridiagonal)
# ============================================================
def gue_eigs(n, rng):
    """GUE(n) eigenvalues via tridiagonal beta-ensemble (exact, O(n))."""
    d = rng.standard_normal(n)
    e = np.sqrt(rng.chisquare(2 * np.arange(n - 1, 0, -1)) / 2)
    return eigvalsh_tridiagonal(d, e) / np.sqrt(n)


def gue_spacings_batch(n_mats, mat_size, rng):
    """Concatenated normalized GUE spacings from multiple matrices."""
    all_sp = []
    for _ in range(n_mats):
        eigs = gue_eigs(mat_size, rng)
        sp = polynomial_unfold(eigs, trim_fraction=0.1)
        all_sp.append(sp)
    sp = np.concatenate(all_sp)
    sp /= np.mean(sp)
    return sp


def chi2_zscore(residual, se, dof):
    """Chi-squared z-score: (sum(z^2) - dof) / sqrt(2*dof)."""
    return (np.sum((residual / se) ** 2) - dof) / np.sqrt(2 * dof)


# ============================================================
# STEP 1: GUE BASELINE
# ============================================================
t_start = time.time()
print('Step 1: GUE baseline (100 matrices at N=1200)...')
rng_bl = np.random.default_rng(42)
bl_acfs = []
for _ in range(N_BASELINE):
    eigs = gue_eigs(GUE_N, rng_bl)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(sp) > MAX_LAG + 10:
        bl_acfs.append(spacing_autocorrelation(sp, MAX_LAG))
baseline = np.mean(bl_acfs, axis=0)[1:MAX_LAG + 1]
print(f'  {len(bl_acfs)} matrices, {time.time() - t_start:.1f}s')

# ============================================================
# STEP 2: REAL DATA (Odlyzko zeros at T~2.7e11)
# ============================================================
print('Step 2: Loading Odlyzko zeros3.txt...')
zeros = []
with open('data/odlyzko/zeros3.txt') as f:
    for line in f:
        try:
            zeros.append(float(line.strip()))
        except ValueError:
            pass
zeros = np.array(zeros)
density = LOG_T / (2 * np.pi)
sp_real = np.diff(zeros) * density
sp_real /= np.mean(sp_real)
N_real = len(sp_real)
se_real = 1.0 / np.sqrt(N_real)
acf_real = spacing_autocorrelation(sp_real, MAX_LAG)[1:MAX_LAG + 1]
excess_real = acf_real - baseline
print(f'  {N_real} spacings, se={se_real:.6f}')

# ============================================================
# STEP 3: DESIGN MATRICES
# ============================================================
k_arr = np.arange(1, MAX_LAG + 1, dtype=float)

# --- Model A: Free cos+sin, p<=200 m<=6, capped at 100 freqs + 6 short-range ---
# (reproduces _trace_convergence.py Phase 5/6 where chi2 z~3.6)
freqs = []
for p in primerange(2, 201):
    for m in range(1, 7):
        f = m * np.log(p) / LOG_T
        if f < 0.5:
            freqs.append(f)
uq = []
for f in freqs:
    if not any(abs(f - u) < 1e-6 for u in uq):
        uq.append(f)
if len(uq) > MAX_LAG // 4:
    uq = uq[:MAX_LAG // 4]

cols_A = []
for f in uq:
    cols_A.append(np.cos(2 * np.pi * k_arr * f))
    cols_A.append(np.sin(2 * np.pi * k_arr * f))
short_6 = [
    np.exp(-k_arr / 1.0), np.exp(-k_arr / 3.0), np.exp(-k_arr / 10.0),
    1.0 / k_arr, 1.0 / k_arr ** 2, (-1) ** k_arr / k_arr,
]
cols_A.extend(short_6)
X_A = np.column_stack(cols_A)
n_A = X_A.shape[1]
dof_A = MAX_LAG - n_A
M_A = np.eye(MAX_LAG) - X_A @ np.linalg.pinv(X_A)

# --- Model B: Selberg constrained, 5 params (from _selberg_convergence.py) ---
cos_sel = np.zeros(MAX_LAG)
sin_sel = np.zeros(MAX_LAG)
for p in primerange(2, 500):
    for m in range(1, 9):
        f = m * np.log(p) / LOG_T
        if f >= 0.5:
            break
        w = np.log(p) / p ** (BEST_ALPHA * m)
        cos_sel += w * np.cos(2 * np.pi * k_arr * f)
        sin_sel += w * np.sin(2 * np.pi * k_arr * f)
short_3 = [np.exp(-k_arr / 1.0), np.exp(-k_arr / 3.0), 1.0 / k_arr ** 2]
X_B = np.column_stack([cos_sel, sin_sel] + short_3)
n_B = X_B.shape[1]
dof_B = MAX_LAG - n_B
M_B = np.eye(MAX_LAG) - X_B @ np.linalg.pinv(X_B)

print(f'  Model A: {n_A} params ({len(uq)} freqs x2 + 6 short), dof={dof_A}')
print(f'  Model B: {n_B} params (Selberg cos+sin + 3 short), dof={dof_B}')

# Observed chi2 z-scores
z_obs_A = chi2_zscore(M_A @ excess_real, se_real, dof_A)
z_obs_B = chi2_zscore(M_B @ excess_real, se_real, dof_B)
print(f'  Observed chi2 z:  A = {z_obs_A:+.2f},  B = {z_obs_B:+.2f}')

# ============================================================
# STEP 4: MONTE CARLO (pure GUE null)
# ============================================================
print(f'\nStep 4: Monte Carlo ({N_TRIALS} trials, {MATS_PER_TRIAL} x GUE({GUE_N})/trial)...')
t_mc = time.time()
mc_z_A = np.zeros(N_TRIALS)
mc_z_B = np.zeros(N_TRIALS)
mc_n_sp = np.zeros(N_TRIALS, dtype=int)
rng_mc = np.random.default_rng(12345)

for trial in range(N_TRIALS):
    if (trial + 1) % 100 == 0:
        dt = time.time() - t_mc
        eta = dt / (trial + 1) * (N_TRIALS - trial - 1)
        print(f'  {trial + 1}/{N_TRIALS}  ({dt:.0f}s elapsed, ~{eta:.0f}s remaining)')

    sp = gue_spacings_batch(MATS_PER_TRIAL, GUE_N, rng_mc)
    N_sp = len(sp)
    se = 1.0 / np.sqrt(N_sp)
    acf = spacing_autocorrelation(sp, MAX_LAG)[1:MAX_LAG + 1]
    excess = acf - baseline

    mc_z_A[trial] = chi2_zscore(M_A @ excess, se, dof_A)
    mc_z_B[trial] = chi2_zscore(M_B @ excess, se, dof_B)
    mc_n_sp[trial] = N_sp

dt_mc = time.time() - t_mc
print(f'  Done: {dt_mc:.1f}s ({dt_mc / N_TRIALS * 1000:.0f}ms/trial)')

# ============================================================
# STEP 5: RESULTS
# ============================================================
print('\n' + '=' * 70)
print('MONTE CARLO RESULTS')
print('=' * 70)
print(f'Trials: {N_TRIALS}')
print(f'Spacings/trial: {int(np.mean(mc_n_sp))} (real data: {N_real})')


def report(name, z_obs, mc_z, dof, n_params):
    pval = np.mean(mc_z >= z_obs)
    mc_mean = np.mean(mc_z)
    mc_std = np.std(mc_z)
    print(f'\n--- {name} ---')
    print(f'  Parameters: {n_params}, DOF: {dof}')
    print(f'  Null distribution (pure GUE):')
    print(f'    Mean:     {mc_mean:+.2f}')
    print(f'    Std:      {mc_std:.2f}')
    print(f'    [5%, 95%]: [{np.percentile(mc_z, 5):+.2f}, {np.percentile(mc_z, 95):+.2f}]')
    print(f'    Min/Max:  [{np.min(mc_z):+.2f}, {np.max(mc_z):+.2f}]')
    print(f'  Observed:   z = {z_obs:+.2f}')
    print(f'  P-value:    {pval:.4f}')
    # Effective DOF (from MC variance inflation)
    if mc_std > 0:
        # Under chi2(dof), z has std=1. If MC std > 1, effective DOF < nominal
        dof_eff = dof / mc_std ** 2
        print(f'  Variance inflation: {mc_std ** 2:.2f}x  (eff. DOF ~ {dof_eff:.0f})')
    # Calibrated z (accounting for null shift/spread)
    z_cal = (z_obs - mc_mean) / mc_std
    print(f'  Calibrated z: {z_cal:+.2f} (observed vs MC null)')
    return pval, z_cal


pval_A, zcal_A = report('Model A: Free cos+sin (206 params)', z_obs_A, mc_z_A, dof_A, n_A)
pval_B, zcal_B = report('Model B: Selberg constrained (5 params)', z_obs_B, mc_z_B, dof_B, n_B)

# ============================================================
# VERDICT
# ============================================================
print('\n' + '=' * 70)
print('VERDICT: IS THE DIFFUSE 30% REAL?')
print('=' * 70)

# Primary test is Model A (the one that produced z~3.6 in original analysis)
if pval_A > 0.05:
    print(f'\n>>> NOISE: chi2 z = {z_obs_A:+.2f} is EXPECTED under GUE (p = {pval_A:.3f})')
    print(f'>>> The GUE null produces comparable z-scores routinely.')
    if np.std(mc_z_A) > 1.3:
        print(f'>>> Root cause: ACF lags are correlated (variance inflation = {np.std(mc_z_A)**2:.2f}x)')
        print(f'>>> The chi2 test assumed independence, inflating the z-score.')
    print()
    print('DECOMPOSITION SIMPLIFIES: 3 components --> 2')
    print('  1. Oscillatory: prime-frequency modulation (Selberg 1/p amplitude)')
    print('  2. Short-range: enhanced nearest-neighbor repulsion')
    print('  [diffuse: not a component — GUE sampling noise]')
    print()
    print('The non-GUE ACF is FULLY explained by:')
    print('  ACF_excess(k) = Sum_p A(p)*cos(2*pi*k*f_p + phi_p) + short-range(k)')
    print(f'  with A(p) ~ log(p)/p^0.76 (Selberg/pair-correlation hybrid)')
elif pval_A > 0.01:
    print(f'\n>>> MARGINAL: chi2 z = {z_obs_A:+.2f} (p = {pval_A:.3f})')
    print(f'>>> Cannot decisively classify the diffuse 30% as noise or signal.')
    print(f'>>> More zeros or better baseline estimation may resolve this.')
    print()
    print('Three-component decomposition: INCONCLUSIVE')
else:
    print(f'\n>>> REAL: chi2 z = {z_obs_A:+.2f} is ANOMALOUS (p = {pval_A:.4f})')
    print(f'>>> The diffuse 30% is NOT explained by GUE sampling noise.')
    print()
    print('THREE-COMPONENT DECOMPOSITION STANDS:')
    print('  1. Oscillatory (~53%): prime cosines with Selberg 1/p amplitude')
    print('  2. Short-range (~17%): enhanced level repulsion')
    print('  3. Diffuse (~30%): genuine sub-threshold collective structure')

# Model B cross-check
print(f'\nCross-check (Selberg 5-param model): '
      f'z={z_obs_B:+.2f}, calibrated={zcal_B:+.2f}, p={np.mean(mc_z_B >= z_obs_B):.4f}')
if (pval_A > 0.05) == (np.mean(mc_z_B >= z_obs_B) > 0.05):
    print('Both models AGREE on the verdict.')
else:
    print('Models DISAGREE — investigate further.')

print(f'\nTotal time: {time.time() - t_start:.1f}s')
