"""Re-run trace formula convergence with Selberg 1/p amplitude law.

The Montgomery pair correlation form factor gives A(p) ~ log(p)/(pi*p),
not the explicit formula's log(p)/sqrt(p). Test whether this correct
amplitude law + cos+sin basis + short-range pushes convergence higher."""
import sys
sys.path.insert(0, 'src')
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from sympy import primerange
from riemann.analysis.bost_connes_operator import (
    spacing_autocorrelation, polynomial_unfold
)

max_lag = 400

# Load data
def load_zeros(path):
    values = []
    with open(path) as f:
        for line in f:
            try:
                values.append(float(line.strip()))
            except ValueError:
                continue
    return np.array(values)

res = load_zeros('data/odlyzko/zeros3.txt')
T_base = 267653395647.0
log_T = np.log(T_base / (2*np.pi))
density = log_T / (2*np.pi)
sp = np.diff(res) * density
sp = sp / np.mean(sp)
N = len(sp)
se = 1.0 / np.sqrt(N)
n_data = max_lag

acf = spacing_autocorrelation(sp, max_lag)

# GUE baseline
print('Computing GUE baseline...')
gue_N = 1200
rng = np.random.default_rng(42)
gue_acfs = []
for _ in range(100):
    A = rng.standard_normal((gue_N, gue_N)) + 1j * rng.standard_normal((gue_N, gue_N))
    H = (A + A.conj().T) / (2 * np.sqrt(2 * gue_N))
    eigs = np.linalg.eigvalsh(H)
    s = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(s) > max_lag + 10:
        gue_acfs.append(spacing_autocorrelation(s, max_lag))
gue_acf = np.mean(gue_acfs, axis=0)
excess = acf[1:max_lag+1] - gue_acf[1:max_lag+1]
ss_tot = np.sum(excess**2)

print(f'Data: {N} spacings, {n_data} lags, excess L2={np.sqrt(ss_tot):.4f}')

# ============================================================
# AMPLITUDE LAW COMPARISON (1-DOF models)
# ============================================================
print('\n' + '='*70)
print('1-DOF AMPLITUDE LAW COMPARISON')
print('='*70)

def build_predicted_model(primes, max_m, amp_func, n_data, log_T):
    """Build model with predicted amplitudes. Returns column vector."""
    model = np.zeros(n_data)
    for p in primes:
        for m in range(1, max_m + 1):
            freq = m * np.log(p) / log_T
            if freq >= 0.5:
                break
            amp = amp_func(p, m)
            for k in range(n_data):
                model[k] += amp * np.cos(2*np.pi*(k+1)*freq)
    return model

def fit_scale(model, excess, ss_tot):
    """Fit single scale parameter, return R2, R2_adj, scale."""
    # Optimal scale = <model, excess> / <model, model>
    dot = np.dot(model, excess)
    norm_sq = np.dot(model, model)
    if norm_sq < 1e-30:
        return 0, 0, 0
    scale = dot / norm_sq
    pred = scale * model
    ss_res = np.sum((excess - pred)**2)
    R2 = 1 - ss_res / ss_tot
    R2_adj = 1 - (1 - R2) * (n_data - 1) / (n_data - 2)
    return R2, R2_adj, scale

# Amplitude laws to test
laws = {
    'log(p)/sqrt(p)':     lambda p, m: np.log(p) / p**(m/2.0),
    'log(p)/p^m':         lambda p, m: np.log(p) / p**m,
    'log(p)/(pi*p^m)':    lambda p, m: np.log(p) / (np.pi * p**m),
    '1/sqrt(p)':          lambda p, m: 1.0 / p**(m/2.0),
    '1/p^m':              lambda p, m: 1.0 / p**m,
    'log(p)/p^(3/4*m)':   lambda p, m: np.log(p) / p**(0.75*m),
}

primes_all = list(primerange(2, 500))

print(f'\n{"Law":<22} {"max_p":>6} {"m<=":>4} {"R2":>8} {"R2_adj":>8} {"scale":>10}')
print('-'*65)

best_R2_adj = -999
best_law = None

for law_name, amp_func in laws.items():
    for max_p_val, max_m_val in [(30, 1), (100, 1), (200, 1), (100, 4), (200, 4)]:
        primes = [p for p in primes_all if p <= max_p_val]
        model = build_predicted_model(primes, max_m_val, amp_func, n_data, log_T)
        R2, R2_adj, scale = fit_scale(model, excess, ss_tot)
        tag = ''
        if R2_adj > best_R2_adj:
            best_R2_adj = R2_adj
            best_law = (law_name, max_p_val, max_m_val)
            tag = ' <-- BEST'
        if max_p_val == 100 and max_m_val == 1:
            print(f'{law_name:<22} {max_p_val:>6} {max_m_val:>4} {R2:>8.4f} {R2_adj:>8.4f} {scale:>10.6f}{tag}')

print(f'\nBest 1-DOF: {best_law[0]} (p<={best_law[1]}, m<={best_law[2]}), R2_adj={best_R2_adj:.4f}')

# ============================================================
# 2-DOF MODEL: scale + decay exponent
# ============================================================
print('\n' + '='*70)
print('2-DOF MODEL: A(p,m) = scale * log(p) / p^(alpha*m)')
print('='*70)

def parametric_model(params, primes, max_m, n_data, log_T):
    scale, alpha = params
    model = np.zeros(n_data)
    for p in primes:
        for m in range(1, max_m + 1):
            freq = m * np.log(p) / log_T
            if freq >= 0.5:
                break
            amp = scale * np.log(p) / p**(alpha * m)
            for k in range(n_data):
                model[k] += amp * np.cos(2*np.pi*(k+1)*freq)
    return model

primes_100 = [p for p in primes_all if p <= 100]

def neg_R2_parametric(params):
    model = parametric_model(params, primes_100, 4, n_data, log_T)
    ss_res = np.sum((excess - model)**2)
    return ss_res / ss_tot

from scipy.optimize import minimize as scipy_minimize
result = scipy_minimize(neg_R2_parametric, [0.01, 0.75], method='Nelder-Mead',
                        options={'xatol': 1e-6, 'fatol': 1e-8})
best_scale, best_alpha = result.x
R2_param = 1 - result.fun
R2_adj_param = 1 - (1 - R2_param) * (n_data - 1) / (n_data - 3)

print(f'Best fit: scale={best_scale:.6f}, alpha={best_alpha:.4f}')
print(f'R2={R2_param:.4f}, R2_adj={R2_adj_param:.4f}')
print(f'alpha=0.5 would be explicit formula (1/sqrt(p))')
print(f'alpha=1.0 would be Selberg/Montgomery (1/p)')
print(f'alpha={best_alpha:.4f} is the data-preferred decay')

# ============================================================
# FULL MODEL: Selberg cos+sin + short-range
# ============================================================
print('\n' + '='*70)
print('FULL MODEL: SELBERG 1/p COS+SIN + SHORT-RANGE')
print('='*70)

def make_cos(freq, n):
    return np.array([np.cos(2*np.pi*k*freq) for k in range(1, n+1)])

def make_sin(freq, n):
    return np.array([np.sin(2*np.pi*k*freq) for k in range(1, n+1)])

k_arr = np.arange(1, n_data + 1, dtype=float)
short_range = [
    np.exp(-k_arr / 1.0),
    np.exp(-k_arr / 3.0),
    1.0 / k_arr**2,
]

def fit_full_model(freqs, excess, extra_cols=None):
    """Cos+sin at each freq + optional extra columns."""
    cols = []
    for f in freqs:
        cols.append(make_cos(f, len(excess)))
        cols.append(make_sin(f, len(excess)))
    if extra_cols:
        cols.extend(extra_cols)
    X = np.column_stack(cols)
    amps, _, _, _ = np.linalg.lstsq(X, excess, rcond=None)
    pred = X @ amps
    ss_res = np.sum((excess - pred)**2)
    R2 = 1 - ss_res / ss_tot
    p = X.shape[1]
    R2_adj = 1 - (1 - R2) * (n_data - 1) / (n_data - p - 1) if n_data > p + 1 else R2
    return R2, R2_adj, pred, amps

def fit_constrained_sincos(primes, max_m, alpha, excess, extra_cols=None):
    """Cos+sin with CONSTRAINED amplitudes: A(p,m) = log(p)/p^(alpha*m).
    Only 2 free params per prime: overall cos_scale and sin_scale,
    sharing the amplitude profile across harmonics.
    Actually, let's do it differently: build the Selberg-predicted cos and sin
    vectors (weighted sum), then fit 2 scales."""
    cos_model = np.zeros(len(excess))
    sin_model = np.zeros(len(excess))
    for p in primes:
        for m in range(1, max_m + 1):
            freq = m * np.log(p) / log_T
            if freq >= 0.5:
                break
            weight = np.log(p) / p**(alpha * m)
            cos_model += weight * make_cos(freq, len(excess))
            sin_model += weight * make_sin(freq, len(excess))
    cols = [cos_model, sin_model]
    if extra_cols:
        cols.extend(extra_cols)
    X = np.column_stack(cols)
    amps, _, _, _ = np.linalg.lstsq(X, excess, rcond=None)
    pred = X @ amps
    ss_res = np.sum((excess - pred)**2)
    R2 = 1 - ss_res / ss_tot
    p = X.shape[1]
    R2_adj = 1 - (1 - R2) * (n_data - 1) / (n_data - p - 1) if n_data > p + 1 else R2
    return R2, R2_adj, pred, amps

print(f'\n{"Description":<55} {"params":>7} {"R2":>8} {"R2_adj":>8}')
print('-'*85)

# A. Previous best: free-fit cos+sin + short-range
for label, max_p, max_m in [('p<=100 m<=6', 101, 7), ('p<=200 m<=6', 201, 7)]:
    freqs = []
    for p in primerange(2, max_p):
        for m in range(1, max_m):
            f = m * np.log(p) / log_T
            if f < 0.5:
                freqs.append(f)
    uq = []
    for f in freqs:
        if not any(abs(f - u) < 1e-6 for u in uq):
            uq.append(f)
    if len(uq) > n_data // 4:
        uq = uq[:n_data // 4]
    R2, R2_adj, _, _ = fit_full_model(uq, excess, short_range)
    print(f'FREE {label} cos+sin+short{"":>20} {2*len(uq)+3:>7} {R2:>8.4f} {R2_adj:>8.4f}')

# B. Selberg-constrained: only 2+3=5 free params (cos_scale, sin_scale, 3 short-range)
for label, max_p, max_m, alpha in [
    ('Selberg a=1.0 p<=100 m<=4', 100, 4, 1.0),
    ('Selberg a=1.0 p<=200 m<=6', 200, 6, 1.0),
    ('Selberg a=1.0 p<=500 m<=8', 500, 8, 1.0),
    ('Explicit a=0.5 p<=100 m<=4', 100, 4, 0.5),
    ('Explicit a=0.5 p<=200 m<=6', 200, 6, 0.5),
    (f'Optimal a={best_alpha:.2f} p<=100 m<=4', 100, 4, best_alpha),
    (f'Optimal a={best_alpha:.2f} p<=200 m<=6', 200, 6, best_alpha),
    (f'Optimal a={best_alpha:.2f} p<=500 m<=8', 500, 8, best_alpha),
]:
    primes = [p for p in primes_all if p <= max_p]
    R2, R2_adj, pred, amps = fit_constrained_sincos(primes, max_m, alpha, excess, short_range)
    n_params = 2 + len(short_range)  # cos_scale + sin_scale + short-range terms
    print(f'{label:<55} {n_params:>7} {R2:>8.4f} {R2_adj:>8.4f}')

# C. Per-prime cos+sin (2 DOF per prime) + Selberg amplitude SHAPE constraint
# Each prime gets free (a_p, b_p) but constrained by: a_p^2 + b_p^2 ~ (log(p)/p^alpha)^2
# This is equivalent to free cos+sin but we report the amplitude profile
print('\n--- Per-prime amplitude profile (free cos+sin, 30 primes) ---')
primes_30 = list(primerange(2, 128))[:30]
freqs_30 = [np.log(p)/log_T for p in primes_30]
R2_30, R2_adj_30, pred_30, amps_30 = fit_full_model(freqs_30, excess, short_range)
print(f'30 primes cos+sin+short: R2={R2_30:.4f}, R2_adj={R2_adj_30:.4f} ({2*30+3} params)')

print(f'\n{"p":>5} {"cos_amp":>10} {"sin_amp":>10} {"|amp|":>10} {"Selberg":>10} {"Explicit":>10} {"ratio_S":>10}')
print('-'*70)
for i, p in enumerate(primes_30[:20]):
    a_cos = amps_30[2*i]
    a_sin = amps_30[2*i+1]
    total_amp = np.sqrt(a_cos**2 + a_sin**2)
    selberg_pred = np.log(p) / p
    explicit_pred = np.log(p) / np.sqrt(p)
    ratio = total_amp / selberg_pred if selberg_pred > 0 else 0
    print(f'{p:>5} {a_cos:>+10.6f} {a_sin:>+10.6f} {total_amp:>10.6f} {selberg_pred:>10.6f} {explicit_pred:>10.6f} {ratio:>10.4f}')

# ============================================================
# BEST CONSTRAINED MODEL: RESIDUAL ANALYSIS
# ============================================================
print('\n' + '='*70)
print('BEST CONSTRAINED MODEL: RESIDUAL ANALYSIS')
print('='*70)

# Use optimal alpha with largest prime range
primes_500 = [p for p in primes_all if p <= 500]
R2_best, R2_adj_best, pred_best, amps_best = fit_constrained_sincos(
    primes_500, 8, best_alpha, excess, short_range)
residual = excess - pred_best
residual_z = residual / se
n_sig = np.sum(np.abs(residual_z) > 2.5)

dof = n_data - (2 + len(short_range))
chi2_obs = np.sum(residual_z**2)
chi2_z = (chi2_obs - dof) / np.sqrt(2*dof)

print(f'Model: Selberg a={best_alpha:.4f}, p<=500, m<=8, 5 params')
print(f'R2={R2_best:.4f}, R2_adj={R2_adj_best:.4f}')
print(f'Residual max |z|: {np.max(np.abs(residual_z)):.2f}')
print(f'Significant lags (|z|>2.5): {n_sig}/{n_data} (expected: {n_data*0.012:.1f})')
print(f'Chi2 z-score: {chi2_z:+.2f}')
print(f'DOF remaining: {dof}')

# Top residuals
print(f'\nTop residuals:')
print(f'{"Lag":<5} {"Excess":>8} {"Model":>8} {"Residual":>8} {"z":>6}')
for k in np.argsort(np.abs(residual_z))[-10:][::-1]:
    lag = k + 1
    print(f'{lag:<5} {excess[k]:>+8.4f} {pred_best[k]:>+8.4f} {residual[k]:>+8.4f} {residual_z[k]:>+6.2f}')

# ============================================================
# VERDICT
# ============================================================
print('\n' + '='*70)
print('SELBERG CONVERGENCE VERDICT')
print('='*70)
print(f'\nOptimal decay exponent: alpha = {best_alpha:.4f}')
print(f'  (0.5 = explicit formula, 1.0 = Selberg/Montgomery)')
print(f'\n5-parameter Selberg model (cos+sin scales + 3 short-range):')
print(f'  R2 = {R2_best:.4f}')
print(f'  R2_adj = {R2_adj_best:.4f}')
print(f'  Chi2 z = {chi2_z:+.2f}')
print(f'  Significant residual lags: {n_sig}/{n_data}')
print()
if chi2_z < 2 and n_sig <= n_data * 0.02:
    print('>>> CONVERGED with 5 parameters!')
    print('>>> The Selberg amplitude law + phase shifts + short-range correction')
    print('>>> captures ALL statistically significant structure.')
elif R2_adj_best > 0.50:
    print('>>> SUBSTANTIAL: 5-param Selberg model captures >50% (adjusted)')
    print('>>> Comparable to free-fit models with 60-200 parameters')
else:
    print('>>> The constrained model underperforms free-fit')
    print('>>> The amplitude law needs refinement')
