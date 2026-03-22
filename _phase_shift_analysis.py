"""Per-prime amplitude and phase extraction from the 206-param fit.

The MC test proved the 206-param model captures ALL non-GUE ACF structure.
This script extracts (C_p, phi_p) for each prime and tests whether the phases
follow a predictable law — which would enable a constrained model with far
fewer parameters.

If phi_p ~ c * log(p), the entire ACF excess becomes a 3-4 parameter model:
  ACF_excess(k) = sum_p C(p) * cos(2*pi*k*f_p + c*log(p)) + short-range(k)
"""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from scipy.linalg import eigvalsh_tridiagonal
from scipy.optimize import minimize
from sympy import primerange
from riemann.analysis.bost_connes_operator import spacing_autocorrelation, polynomial_unfold

MAX_LAG = 400
T_BASE = 267653395647.0
LOG_T = np.log(T_BASE / (2 * np.pi))
k_arr = np.arange(1, MAX_LAG + 1, dtype=float)

# ============================================================
# SETUP: Reproduce data + baseline + fit (same as MC test)
# ============================================================
t0 = time.time()
print('Setup: loading data and computing baseline...')

def gue_eigs(n, rng):
    d = rng.standard_normal(n)
    e = np.sqrt(rng.chisquare(2 * np.arange(n - 1, 0, -1)) / 2)
    return eigvalsh_tridiagonal(d, e) / np.sqrt(n)

rng_bl = np.random.default_rng(42)
bl_acfs = []
for _ in range(100):
    eigs = gue_eigs(1200, rng_bl)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(sp) > MAX_LAG + 10:
        bl_acfs.append(spacing_autocorrelation(sp, MAX_LAG))
baseline = np.mean(bl_acfs, axis=0)[1:MAX_LAG + 1]

zeros = []
with open('data/odlyzko/zeros3.txt') as f:
    for line in f:
        try: zeros.append(float(line.strip()))
        except ValueError: pass
zeros = np.array(zeros)
sp_real = np.diff(zeros) * (LOG_T / (2 * np.pi))
sp_real /= np.mean(sp_real)
N_real = len(sp_real)
se = 1.0 / np.sqrt(N_real)
acf_real = spacing_autocorrelation(sp_real, MAX_LAG)[1:MAX_LAG + 1]
excess = acf_real - baseline
ss_tot = np.sum(excess ** 2)
print(f'  {N_real} spacings, excess L2={np.sqrt(ss_tot):.4f}, {time.time()-t0:.1f}s')

# ============================================================
# STEP 1: Per-prime cos+sin fit (100 primes, first harmonic only)
# ============================================================
print('\nStep 1: Per-prime amplitude/phase extraction...')
primes = list(primerange(2, 600))[:100]
freqs = [np.log(p) / LOG_T for p in primes]

# Build design matrix: 2 columns per prime (cos, sin) + 3 short-range
cols = []
for f in freqs:
    cols.append(np.cos(2 * np.pi * k_arr * f))
    cols.append(np.sin(2 * np.pi * k_arr * f))
short_range = [np.exp(-k_arr / 1.0), np.exp(-k_arr / 3.0), 1.0 / k_arr ** 2]
cols.extend(short_range)
X = np.column_stack(cols)
amps, _, _, _ = np.linalg.lstsq(X, excess, rcond=None)
pred = X @ amps
ss_res = np.sum((excess - pred) ** 2)
R2 = 1 - ss_res / ss_tot
R2_adj = 1 - (1 - R2) * (MAX_LAG - 1) / (MAX_LAG - X.shape[1] - 1)
print(f'  100 primes cos+sin + 3 short: R2={R2:.4f}, R2_adj={R2_adj:.4f} ({X.shape[1]} params)')

# Extract per-prime amplitude and phase
print(f'\n{"p":>5} {"freq":>8} {"A_cos":>9} {"A_sin":>9} {"C_p":>9} {"phi_p":>8} {"phi/logp":>9}')
print('-' * 70)
amplitudes = []
phases = []
for i, p in enumerate(primes):
    a_cos = amps[2 * i]
    a_sin = amps[2 * i + 1]
    C_p = np.sqrt(a_cos ** 2 + a_sin ** 2)
    phi_p = np.arctan2(-a_sin, a_cos)  # phase in cos(wk + phi)
    amplitudes.append(C_p)
    phases.append(phi_p)
    if i < 30 or p in [97, 101, 127, 131, 137]:
        print(f'{p:>5} {freqs[i]:>8.5f} {a_cos:>+9.5f} {a_sin:>+9.5f} {C_p:>9.5f} {phi_p:>+8.3f} {phi_p/np.log(p):>+9.4f}')

amplitudes = np.array(amplitudes)
phases = np.array(phases)
log_primes = np.array([np.log(p) for p in primes])

# Short-range coefficients
print(f'\nShort-range: exp(-k/1)={amps[-3]:+.5f}, exp(-k/3)={amps[-2]:+.5f}, 1/k^2={amps[-1]:+.5f}')

# ============================================================
# STEP 2: Amplitude law verification
# ============================================================
print('\n' + '=' * 70)
print('AMPLITUDE LAW: C_p vs theoretical predictions')
print('=' * 70)

# Test: C_p ~ scale * log(p) / p^alpha
def amp_residual(params, primes, C_obs):
    scale, alpha = params
    C_pred = np.array([scale * np.log(p) / p ** alpha for p in primes])
    return np.sum((C_obs - C_pred) ** 2)

res = minimize(amp_residual, [0.01, 0.75], args=(primes, amplitudes), method='Nelder-Mead')
best_scale, best_alpha = res.x
C_pred = np.array([best_scale * np.log(p) / p ** best_alpha for p in primes])
R2_amp = 1 - np.sum((amplitudes - C_pred) ** 2) / np.sum((amplitudes - np.mean(amplitudes)) ** 2)

print(f'Best fit: C_p = {best_scale:.5f} * log(p) / p^{best_alpha:.3f}')
print(f'Amplitude R2: {R2_amp:.4f}')

# Compare specific laws
for label, alpha in [('Explicit 1/sqrt(p)', 0.5), ('Selberg 1/p', 1.0), ('Fitted', best_alpha), ('3/4 power', 0.75)]:
    # Fit just the scale
    C_model = np.array([np.log(p) / p ** alpha for p in primes])
    scale_opt = np.dot(C_model, amplitudes) / np.dot(C_model, C_model)
    C_scaled = scale_opt * C_model
    R2_law = 1 - np.sum((amplitudes - C_scaled) ** 2) / np.sum((amplitudes - np.mean(amplitudes)) ** 2)
    print(f'  {label:<25} alpha={alpha:.3f}  scale={scale_opt:.5f}  R2={R2_law:.4f}')

# ============================================================
# STEP 3: Phase structure analysis
# ============================================================
print('\n' + '=' * 70)
print('PHASE STRUCTURE: phi_p vs prime index / log(p)')
print('=' * 70)

# Test 1: phi_p = constant (no phase dependence)
mean_phase = np.mean(phases)
R2_const = 1 - np.sum((phases - mean_phase) ** 2) / np.sum((phases - np.mean(phases)) ** 2)
print(f'Mean phase: {mean_phase:+.4f} rad ({np.degrees(mean_phase):+.1f} deg)')
print(f'Phase std: {np.std(phases):.4f} rad ({np.degrees(np.std(phases)):.1f} deg)')

# Test 2: phi_p = c * log(p)
# Use weighted regression (weight by amplitude — high-amplitude primes matter more)
weights = amplitudes / np.sum(amplitudes)

def phase_model_residual(c, log_p, phi_obs, w):
    phi_pred = c * log_p
    # Wrap to [-pi, pi]
    diff = np.angle(np.exp(1j * (phi_obs - phi_pred)))
    return np.sum(w * diff ** 2)

res_phase = minimize(phase_model_residual, [0.0], args=(log_primes, phases, weights),
                     method='Nelder-Mead')
c_logp = res_phase.x[0]
phi_pred_logp = c_logp * log_primes
diff_logp = np.angle(np.exp(1j * (phases - phi_pred_logp)))
R2_logp = 1 - np.sum(weights * diff_logp ** 2) / np.sum(weights * (phases - np.average(phases, weights=weights)) ** 2)
print(f'\nphi_p = c * log(p):  c = {c_logp:+.5f}, weighted R2 = {R2_logp:.4f}')

# Test 3: phi_p = a + b * log(p)
def phase_model_affine(params, log_p, phi_obs, w):
    a, b = params
    phi_pred = a + b * log_p
    diff = np.angle(np.exp(1j * (phi_obs - phi_pred)))
    return np.sum(w * diff ** 2)

res_aff = minimize(phase_model_affine, [0.0, 0.0], args=(log_primes, phases, weights),
                   method='Nelder-Mead')
a_aff, b_aff = res_aff.x
phi_pred_aff = a_aff + b_aff * log_primes
diff_aff = np.angle(np.exp(1j * (phases - phi_pred_aff)))
R2_aff = 1 - np.sum(weights * diff_aff ** 2) / np.sum(weights * (phases - np.average(phases, weights=weights)) ** 2)
print(f'phi_p = a + b*log(p): a={a_aff:+.4f}, b={b_aff:+.5f}, weighted R2 = {R2_aff:.4f}')

# Test 4: phi_p = a + b * log(p) + c * log(p)^2
def phase_model_quad(params, log_p, phi_obs, w):
    a, b, c = params
    phi_pred = a + b * log_p + c * log_p ** 2
    diff = np.angle(np.exp(1j * (phi_obs - phi_pred)))
    return np.sum(w * diff ** 2)

res_quad = minimize(phase_model_quad, [0.0, 0.0, 0.0], args=(log_primes, phases, weights),
                    method='Nelder-Mead')
a_q, b_q, c_q = res_quad.x
phi_pred_quad = a_q + b_q * log_primes + c_q * log_primes ** 2
diff_quad = np.angle(np.exp(1j * (phases - phi_pred_quad)))
R2_quad = 1 - np.sum(weights * diff_quad ** 2) / np.sum(weights * (phases - np.average(phases, weights=weights)) ** 2)
print(f'phi_p = quadratic:    a={a_q:+.4f}, b={b_q:+.5f}, c={c_q:+.6f}, weighted R2 = {R2_quad:.4f}')

# Test 5: Are phases just noise? (amplitude-weighted circular dispersion)
# Under null (random phases), mean resultant length R -> 0
# Rayleigh test
z_vals = amplitudes * np.exp(1j * phases)
R_resultant = np.abs(np.sum(z_vals)) / np.sum(amplitudes)
n_eff = len(phases)
rayleigh_z = n_eff * R_resultant ** 2
print(f'\nRayleigh test for phase uniformity:')
print(f'  Mean resultant length: R = {R_resultant:.4f}')
print(f'  Rayleigh z = {rayleigh_z:.2f} (z > 3 => phases non-uniform)')

# Test 6: Connection to S(T) — the argument of zeta
# S(T) = N(T) - theta(T)/pi - 1 where theta is the Riemann-Siegel theta
# At T = T_BASE, we can estimate S(T) from the zero count
# N(T) ~ (T/2pi) * log(T/2pie) + 7/8 + S(T)
# From the data: the zeros start near the 10^10-th zero
# Actually, S(T) fluctuates O(log T) and we can compute theta(T)
import mpmath
mpmath.mp.dps = 30
T = mpmath.mpf(T_BASE)
theta_T = mpmath.siegeltheta(T)
# N(T) estimate from smooth part
N_smooth = float(theta_T / mpmath.pi + 1)  # smooth part of zero counting function
# The exact N(T) would tell us S(T), but we don't know the exact zero index
# Instead, test if the phase structure is consistent with S(T) modulation
S_T_estimate = 0  # We'll estimate from phase data itself
print(f'\nRiemann-Siegel theta(T) = {float(theta_T):.4f}')
print(f'theta(T)/pi = {float(theta_T/mpmath.pi):.4f}')

# The theoretical phase prediction from the explicit formula:
# The pair correlation contribution at height T has phase related to
# the imaginary part of log zeta(1/2 + iT) summed over nearby zeros
# For the ACF, the phase shift for prime p should be:
#   phi_p ~ pi * f_p * S(T) where f_p = log(p)/log(T/2pi)
# This is because S(T) represents the "phase offset" of the zero staircase
# Let's test: phi_p = pi * S * log(p) / log(T/2pi)
# If this works, we can estimate S from the data

# Estimate S from the phase data using weighted least squares
# phi_p ≈ c * log(p) where c ≈ pi * S / log(T/2pi)
S_from_phases = c_logp * LOG_T / np.pi
print(f'\nS(T) estimated from phases: {S_from_phases:+.4f}')
print(f'  (via phi_p = [pi*S/log(T/2pi)] * log(p), c = {c_logp:+.5f})')

# ============================================================
# STEP 4: Constrained phase models — how few params suffice?
# ============================================================
print('\n' + '=' * 70)
print('CONSTRAINED MODELS: amplitude law + phase law')
print('=' * 70)

def build_constrained_model(primes, alpha, phase_func, n_data, log_T):
    """Build model with amplitude law and phase function.
    Free parameters: amplitude_scale + phase params + 3 short-range = varies."""
    model_cos = np.zeros(n_data)
    model_sin = np.zeros(n_data)
    for p in primes:
        f = np.log(p) / log_T
        if f >= 0.5:
            continue
        amp = np.log(p) / p ** alpha
        phi = phase_func(p)
        # cos(wk + phi) = cos(phi)*cos(wk) - sin(phi)*sin(wk)
        model_cos += amp * np.cos(phi) * np.cos(2 * np.pi * k_arr * f)
        model_cos -= amp * np.sin(phi) * np.sin(2 * np.pi * k_arr * f)
    return model_cos

def fit_constrained(model_osc, excess, ss_tot, n_data):
    """Fit: scale * model_osc + 3 short-range terms."""
    X = np.column_stack([model_osc] + short_range)
    amps_fit, _, _, _ = np.linalg.lstsq(X, excess, rcond=None)
    pred = X @ amps_fit
    ss_res = np.sum((excess - pred) ** 2)
    R2 = 1 - ss_res / ss_tot
    n_params = X.shape[1]
    R2_adj = 1 - (1 - R2) * (n_data - 1) / (n_data - n_params - 1)
    residual = excess - pred
    chi2 = np.sum((residual / se) ** 2)
    dof = n_data - n_params
    chi2_z = (chi2 - dof) / np.sqrt(2 * dof)
    return R2, R2_adj, chi2_z, n_params

print(f'\n{"Model":<50} {"params":>6} {"R2":>7} {"R2_adj":>7} {"chi2_z":>7}')
print('-' * 82)

# Model 1: Pure cosine (phi=0), alpha=0.758
m = build_constrained_model(primes, 0.758, lambda p: 0, MAX_LAG, LOG_T)
r2, r2a, cz, np_ = fit_constrained(m, excess, ss_tot, MAX_LAG)
print(f'{"phi=0, alpha=0.758":<50} {np_:>6} {r2:>7.4f} {r2a:>7.4f} {cz:>+7.2f}')

# Model 2: phi = c*log(p), alpha=0.758 (2 + 3 = 5 DOF)
m = build_constrained_model(primes, 0.758, lambda p: c_logp * np.log(p), MAX_LAG, LOG_T)
r2, r2a, cz, np_ = fit_constrained(m, excess, ss_tot, MAX_LAG)
print(f'{"phi=c*log(p) [fitted c], alpha=0.758":<50} {np_:>6} {r2:>7.4f} {r2a:>7.4f} {cz:>+7.2f}')

# Model 3: phi = a+b*log(p), alpha=0.758 (still 4 DOF total after fitting a,b)
m = build_constrained_model(primes, 0.758, lambda p: a_aff + b_aff * np.log(p), MAX_LAG, LOG_T)
r2, r2a, cz, np_ = fit_constrained(m, excess, ss_tot, MAX_LAG)
print(f'{"phi=a+b*log(p), alpha=0.758":<50} {np_:>6} {r2:>7.4f} {r2a:>7.4f} {cz:>+7.2f}')

# Model 4: Jointly optimize alpha and phase coefficient
print('\n--- Joint optimization: alpha + phase_coeff + scale + 3 short-range ---')

def neg_R2_joint(params):
    alpha, c_phase = params
    m = build_constrained_model(primes, alpha, lambda p: c_phase * np.log(p), MAX_LAG, LOG_T)
    X = np.column_stack([m] + short_range)
    a, _, _, _ = np.linalg.lstsq(X, excess, rcond=None)
    return np.sum((excess - X @ a) ** 2) / ss_tot

res_joint = minimize(neg_R2_joint, [0.758, c_logp], method='Nelder-Mead',
                     options={'xatol': 1e-5, 'fatol': 1e-8})
opt_alpha, opt_c = res_joint.x
m = build_constrained_model(primes, opt_alpha, lambda p: opt_c * np.log(p), MAX_LAG, LOG_T)
r2, r2a, cz, np_ = fit_constrained(m, excess, ss_tot, MAX_LAG)
# Total free params: alpha (pre-fit) + c_phase (pre-fit) + scale + 3 short = 6
r2a_6 = 1 - (1 - r2) * (MAX_LAG - 1) / (MAX_LAG - 6 - 1)
print(f'  alpha={opt_alpha:.4f}, c_phase={opt_c:+.5f}')
print(f'  R2={r2:.4f}, R2_adj(6 DOF)={r2a_6:.4f}, chi2_z={cz:+.2f}')

# Model 5: alpha + affine phase (a + b*log(p))
def neg_R2_affine(params):
    alpha, a_ph, b_ph = params
    m = build_constrained_model(primes, alpha, lambda p: a_ph + b_ph * np.log(p), MAX_LAG, LOG_T)
    X = np.column_stack([m] + short_range)
    a, _, _, _ = np.linalg.lstsq(X, excess, rcond=None)
    return np.sum((excess - X @ a) ** 2) / ss_tot

res_aff2 = minimize(neg_R2_affine, [opt_alpha, 0, opt_c], method='Nelder-Mead',
                    options={'xatol': 1e-5, 'fatol': 1e-8})
opt2_alpha, opt2_a, opt2_b = res_aff2.x
m2 = build_constrained_model(primes, opt2_alpha,
                              lambda p: opt2_a + opt2_b * np.log(p), MAX_LAG, LOG_T)
r2_2, r2a_2, cz_2, np_2 = fit_constrained(m2, excess, ss_tot, MAX_LAG)
r2a_7 = 1 - (1 - r2_2) * (MAX_LAG - 1) / (MAX_LAG - 7 - 1)
print(f'\n  Affine phase: alpha={opt2_alpha:.4f}, a={opt2_a:+.4f}, b={opt2_b:+.5f}')
print(f'  R2={r2_2:.4f}, R2_adj(7 DOF)={r2a_7:.4f}, chi2_z={cz_2:+.2f}')

# ============================================================
# STEP 5: Cos + sin with predicted amplitude (Selberg constrained)
# Using separate cos_scale and sin_scale but constrained amplitudes
# ============================================================
print('\n' + '=' * 70)
print('HYBRID MODEL: Selberg amplitude + separate cos/sin scales')
print('=' * 70)

# The 5-param Selberg model failed because it used ONE scale each for the
# weighted cos and sin sums. But different primes have different phase offsets.
# What if we allow per-prime cos+sin but constrain the AMPLITUDE to follow
# the Selberg law? That's 2 DOF per prime (direction of the phasor) but
# the magnitude is constrained.

# Actually, the cleanest test: use the per-prime amplitudes from the fit
# and check if a constrained-amplitude model with FREE per-prime phases
# matches the free model.

# Per-prime phasor model: C_p * cos(2*pi*k*f_p + phi_p)
# = C_p * [cos(phi_p)*cos(2*pi*k*f_p) - sin(phi_p)*sin(2*pi*k*f_p)]
# If C_p is constrained but phi_p is free, each prime has 1 DOF (phi_p)
# Total: 100 + 1 (global scale) + 3 (short-range) = 104 params

# But we can also check: with the OBSERVED per-prime amplitudes frozen,
# what's the chi2_z? This tells us if the amplitude law matters at all.

# Actually, the simplest informative test: compare R2 across model families
# to find where phase constraint starts hurting

print(f'\n{"Model":<55} {"DOF":>5} {"R2":>7} {"R2_adj":>7}')
print('-' * 80)

# Free per-prime cos+sin (already computed above)
print(f'{"Free cos+sin (100 primes) + 3 short":<55} {203:>5} {R2:>7.4f} {R2_adj:>7.4f}')

# Selberg constrained (5 params)
cos_sel = np.zeros(MAX_LAG)
sin_sel = np.zeros(MAX_LAG)
for p in primes:
    f = np.log(p) / LOG_T
    if f >= 0.5: continue
    w = np.log(p) / p ** 0.758
    cos_sel += w * np.cos(2 * np.pi * k_arr * f)
    sin_sel += w * np.sin(2 * np.pi * k_arr * f)
X5 = np.column_stack([cos_sel, sin_sel] + short_range)
a5, _, _, _ = np.linalg.lstsq(X5, excess, rcond=None)
pred5 = X5 @ a5
R2_5 = 1 - np.sum((excess - pred5) ** 2) / ss_tot
R2a_5 = 1 - (1 - R2_5) * (MAX_LAG - 1) / (MAX_LAG - 6)
print(f'{"Selberg constrained (cos+sin scales) + 3 short":<55} {5:>5} {R2_5:>7.4f} {R2a_5:>7.4f}')

# Joint optimal (6 params)
print(f'{"Joint alpha+c*log(p) phase + scale + 3 short":<55} {6:>5} {r2:>7.4f} {r2a_6:>7.4f}')

# Affine phase (7 params)
print(f'{"Joint alpha + affine phase + scale + 3 short":<55} {7:>5} {r2_2:>7.4f} {r2a_7:>7.4f}')

# ============================================================
# STEP 6: Per-prime phase residuals — structure or noise?
# ============================================================
print('\n' + '=' * 70)
print('PHASE RESIDUALS: after removing c*log(p) trend')
print('=' * 70)

phase_resid = np.angle(np.exp(1j * (phases - opt_c * log_primes)))
# Weight by amplitude
w_amp = amplitudes / np.sum(amplitudes)
wrms = np.sqrt(np.sum(w_amp * phase_resid ** 2))
print(f'Weighted RMS phase residual: {wrms:.3f} rad ({np.degrees(wrms):.1f} deg)')

# Are the residuals structured or random?
# Test: autocorrelation of phase residuals (ordered by prime)
acf_phase = np.correlate(phase_resid - np.mean(phase_resid),
                          phase_resid - np.mean(phase_resid), 'full')
acf_phase = acf_phase[len(phase_resid)-1:] / acf_phase[len(phase_resid)-1]
print(f'Phase residual ACF (lag 1): {acf_phase[1]:+.3f}')
print(f'Phase residual ACF (lag 2): {acf_phase[2]:+.3f}')
print(f'Phase residual ACF (lag 3): {acf_phase[3]:+.3f}')
if abs(acf_phase[1]) < 2.0 / np.sqrt(len(phase_resid)):
    print('  -> Consistent with independent noise')
else:
    print('  -> Structured! Phase residuals are correlated')

# Show the amplitude-phase scatter for top 20 primes
print(f'\nTop 20 primes by amplitude:')
print(f'{"p":>5} {"C_p":>9} {"phi_p":>8} {"c*logp":>8} {"resid":>8}')
order = np.argsort(amplitudes)[::-1]
for idx in order[:20]:
    p = primes[idx]
    print(f'{p:>5} {amplitudes[idx]:>9.5f} {phases[idx]:>+8.3f} {opt_c*log_primes[idx]:>+8.3f} {phase_resid[idx]:>+8.3f}')

# ============================================================
# VERDICT
# ============================================================
print('\n' + '=' * 70)
print('VERDICT: PHASE SHIFT MODEL')
print('=' * 70)

gap_free = R2_adj  # free 203-param model
gap_constrained = r2a_6  # constrained 6-param model
gap_ratio = gap_constrained / gap_free if gap_free > 0 else 0

print(f'\nFree model (203 params): R2_adj = {gap_free:.4f}')
print(f'Constrained (6 params):  R2_adj = {r2a_6:.4f}')
print(f'Recovery ratio: {gap_ratio:.3f} ({gap_ratio*100:.1f}% of free model\'s explanatory power)')
print(f'\nPhase law: phi_p = {opt_c:+.5f} * log(p)')
print(f'Amplitude law: C_p ~ {best_scale:.5f} * log(p) / p^{opt_alpha:.3f}')
print(f'S(T) estimate: {opt_c * LOG_T / np.pi:+.4f}')

if gap_ratio > 0.85:
    print(f'\n>>> STRONG: 6 parameters capture {gap_ratio*100:.0f}% of the free model')
    print('>>> The per-prime phases are PREDICTABLE from log(p)')
    print('>>> The non-GUE ACF is a ~6-parameter function')
elif gap_ratio > 0.50:
    print(f'\n>>> PARTIAL: 6 parameters capture {gap_ratio*100:.0f}% of the free model')
    print('>>> The phase law captures the trend but misses per-prime variation')
    print('>>> The non-GUE ACF needs ~10-30 effective parameters')
else:
    print(f'\n>>> WEAK: 6 parameters capture only {gap_ratio*100:.0f}% of the free model')
    print('>>> Per-prime phases are essentially free parameters')
    print('>>> The non-GUE ACF has high intrinsic dimensionality')

print(f'\nTotal time: {time.time() - t0:.1f}s')
