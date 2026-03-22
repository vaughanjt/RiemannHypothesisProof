"""Theoretical derivation of alpha(T) from the Montgomery pair correlation formula.

The Montgomery pair correlation conjecture gives the form factor:
  F(tau) = |tau|  for |tau| <= 1  (asymptotically as T -> inf)

At FINITE T, the form factor is approximated by the partial Euler product:
  F(tau, T) = |tau| * Product_{p <= X} (1 - p^{-1+2*pi*i*tau/log(T/2pi)}) / (1 - p^{-1})
  where X ~ T^{1/2} is the prime cutoff

The ACF excess at lag k involves the Fourier coefficients of F(tau, T) - F_GUE(tau).
The contribution of prime p to the form factor at finite T gives:
  A(p, T) ~ log(p) * |1 - p^{-1} * p^{2*pi*i*k*log(p)/log(T/2pi)}| / (something)

The KEY INSIGHT: at finite T, the form factor is not simply |tau| but includes
a smooth cutoff that modifies the effective amplitude of each prime's contribution.

The effective alpha comes from HOW the form factor approaches |tau|:
  F(tau, T) ≈ |tau| * (1 - correction(tau, T))
  correction ~ sum_p 1/p * (something involving tau and p)

When we fit A(p) ~ log(p)/p^alpha, the alpha we measure reflects the
TRUNCATION of the prime sum, not a fundamental exponent.

This script derives alpha(T) from first principles and compares to data.
"""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from scipy.linalg import eigvalsh_tridiagonal
from scipy.optimize import minimize_scalar, minimize, curve_fit
from sympy import primerange
from riemann.analysis.bost_connes_operator import spacing_autocorrelation, polynomial_unfold

MAX_LAG = 400
k_arr = np.arange(1, MAX_LAG + 1, dtype=float)
primes_all = list(primerange(2, 500))

# ============================================================
# SETUP: load the high-T data
# ============================================================
t0 = time.time()
print('Loading data...')

def gue_eigs(n, rng):
    d = rng.standard_normal(n)
    e = np.sqrt(rng.chisquare(2 * np.arange(n - 1, 0, -1)) / 2)
    return eigvalsh_tridiagonal(d, e) / np.sqrt(n)

rng_bl = np.random.default_rng(42)
bl_acfs = []
for _ in range(100):
    eigs = gue_eigs(1200, rng_bl)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(sp) > 410:
        bl_acfs.append(spacing_autocorrelation(sp, 400))
baseline = np.mean(bl_acfs, axis=0)[1:MAX_LAG + 1]

zeros_high = []
with open('data/odlyzko/zeros3.txt') as f:
    for line in f:
        try: zeros_high.append(float(line.strip()))
        except ValueError: pass
zeros_high = np.array(zeros_high)
T_HIGH = 267653395647.0
LOG_T_HIGH = np.log(T_HIGH / (2 * np.pi))

density = LOG_T_HIGH / (2 * np.pi)
sp = np.diff(zeros_high) * density
sp /= np.mean(sp)
se = 1.0 / np.sqrt(len(sp))
acf = spacing_autocorrelation(sp, MAX_LAG)[1:MAX_LAG + 1]
excess = acf - baseline
ss_tot = np.sum(excess ** 2)
print(f'  {len(sp)} spacings at T~2.7e11, {time.time()-t0:.1f}s')

short_3 = [np.exp(-k_arr / 1.0), np.exp(-k_arr / 3.0), 1.0 / k_arr ** 2]

# ============================================================
# STEP 1: The Montgomery form factor at finite T
# ============================================================
print('\n' + '=' * 70)
print('STEP 1: MONTGOMERY FORM FACTOR AT FINITE T')
print('=' * 70)

# The pair correlation function R_2(x) has Fourier transform (form factor):
#   F(tau) = 1 - delta(tau)  for GUE (beta=2)
# Montgomery conjectured F(tau) -> |tau| for |tau| < 1.
#
# The EXPLICIT FORMULA gives, for the pair correlation of zeta zeros:
#   R_2(x) - 1 = -delta(x) + 1 - (sin(pi*x)/(pi*x))^2
#                 + 2 * Re[sum_{n=2}^infty Lambda(n)/n * n^{-2*pi*i*x/log(T/2pi)}]
#                   / log(T/2pi)
#
# where Lambda(n) = log(p) if n = p^m, else 0 (von Mangoldt function).
#
# The form factor at tau = k * delta_freq (where delta_freq = 1/log(T/2pi)):
#   F(tau, T) = |tau| + 2/logT * sum_{n>=2} Lambda(n)/n * cos(2*pi*tau*log(n))
#             ≈ |tau| + 2/logT * sum_p log(p)/p * cos(2*pi*tau*log(p))  [keeping m=1 only]
#
# But this is the ASYMPTOTIC form factor. At finite T, the explicit formula
# involves a sum truncated at n ~ T:
#   R_2(x) - 1 - R_2^GUE(x) ≈ 2/logT * sum_{p<=T^{1/2}} log(p)/p
#                                 * cos(2*pi*x*log(p)/logT) + ...
#
# When we fit A(p) ~ log(p)/p^alpha to the ACF excess, we are fitting:
#   ACF_excess(k) ≈ C/logT * sum_p log(p)/p * cos(2*pi*k*log(p)/logT)
#
# The "true" amplitude is A(p) = C * log(p) / (p * logT), i.e., alpha = 1.
# But the FINITE SUM and WINDOWING modify the effective alpha.

# Let's compute the PREDICTED ACF excess from the Montgomery formula directly:
def montgomery_prediction(log_T, max_lag, primes, include_higher_harmonics=False):
    """Predicted ACF excess from the Montgomery/Bogomolny-Keating formula.

    ACF_excess(k) ≈ (2/logT) * sum_p (log(p)/p) * cos(2*pi*k*log(p)/logT)
                   + higher harmonics if requested

    This is the ASYMPTOTIC prediction with alpha = 1 (Selberg).
    """
    k_arr = np.arange(1, max_lag + 1, dtype=float)
    model = np.zeros(max_lag)
    for p in primes:
        freq = np.log(p) / log_T
        if freq >= 0.5:
            continue
        # Asymptotic amplitude: log(p) / p  (from form factor)
        amp = np.log(p) / p
        model += amp * np.cos(2 * np.pi * k_arr * freq)
        if include_higher_harmonics:
            for m in range(2, 9):
                freq_m = m * np.log(p) / log_T
                if freq_m >= 0.5:
                    break
                amp_m = np.log(p) / p ** m
                model += amp_m * np.cos(2 * np.pi * k_arr * freq_m)
    return (2.0 / log_T) * model


# The predicted model with alpha = 1 (Montgomery/Selberg)
pred_montgomery = montgomery_prediction(LOG_T_HIGH, MAX_LAG, primes_all)

# Fit a single scale + short-range to the data
X_mont = np.column_stack([pred_montgomery] + short_3)
amps_mont, _, _, _ = np.linalg.lstsq(X_mont, excess, rcond=None)
pred_mont_fit = X_mont @ amps_mont
R2_mont = 1 - np.sum((excess - pred_mont_fit) ** 2) / ss_tot
R2_adj_mont = 1 - (1 - R2_mont) * (MAX_LAG - 1) / (MAX_LAG - 5)

print(f'\nMontgomery prediction (alpha=1, 2/logT normalization):')
print(f'  Fitted scale: {amps_mont[0]:+.4f} (theory predicts 1.0)')
print(f'  R2 = {R2_mont:.4f}, R2_adj = {R2_adj_mont:.4f}')

# Compare to our best constrained model
def fit_constrained_alpha(alpha, log_T, primes, excess, max_lag, short_cols):
    k_arr = np.arange(1, max_lag + 1, dtype=float)
    model = np.zeros(max_lag)
    for p in primes:
        freq = np.log(p) / log_T
        if freq >= 0.45: continue
        model += np.log(p) / p ** alpha * np.cos(2 * np.pi * k_arr * freq)
    X = np.column_stack([model] + short_cols)
    a, _, _, _ = np.linalg.lstsq(X, excess, rcond=None)
    ss_res = np.sum((excess - X @ a) ** 2)
    return ss_res / np.sum(excess ** 2)

res_opt = minimize_scalar(lambda a: fit_constrained_alpha(a, LOG_T_HIGH, primes_all,
                          excess, MAX_LAG, short_3), bounds=(0.3, 1.5), method='bounded')
alpha_opt = res_opt.x
R2_opt = 1 - res_opt.fun
R2_adj_opt = 1 - (1 - R2_opt) * (MAX_LAG - 1) / (MAX_LAG - 5 - 1)

print(f'\nBest constrained model:')
print(f'  alpha = {alpha_opt:.4f}, R2_adj = {R2_adj_opt:.4f}')
print(f'\nMontgomery (alpha=1) vs optimal (alpha={alpha_opt:.3f}):')
print(f'  Delta R2_adj = {R2_adj_mont - R2_adj_opt:+.4f}')

# ============================================================
# STEP 2: WHY alpha(T) < 1 — the finite-T correction
# ============================================================
print('\n' + '=' * 70)
print('STEP 2: FINITE-T CORRECTION TO THE FORM FACTOR')
print('=' * 70)

# The form factor at finite T involves a SMOOTH CUTOFF on the prime sum.
# Instead of summing over ALL primes (giving alpha=1), the effective sum is:
#
#   sum_p log(p) * w(p, T) * cos(...)
#
# where w(p, T) is a weight that:
#   - equals 1/p for p << T^{1/2}  (the Selberg regime)
#   - decays faster for p approaching T^{1/2}
#   - is 0 for p > T
#
# The explicit formula weight from the Guinand-Weil formula is:
#   w(p, T) = (1/p) * Phi(log(p) / log(T))
#
# where Phi is related to the test function in the explicit formula.
# For a sharp cutoff at X = T^{1/2}: Phi(u) = 1 for u < 1/2, 0 otherwise.
# For a smooth cutoff: Phi decays smoothly.
#
# When we fit sum_p log(p)/p^alpha * cos(...), the effective alpha absorbs
# the shape of Phi. If Phi(u) ~ exp(-c*u) for some c, then:
#   log(p)/p * Phi(log(p)/logT) ≈ log(p)/p * exp(-c*log(p)/logT)
#                                = log(p) / (p * p^{c/logT})
#                                = log(p) / p^{1 + c/logT}
#
# So the effective alpha is:
#   alpha(T) = 1 + c / log(T)   (!) --- this INCREASES with T, wrong sign
#
# Hmm, that gives alpha > 1 and increasing. But we observe alpha < 1 and
# increasing toward 1. Let me reconsider.

# Actually, the issue is different. The INCOMPLETE prime sum at finite T
# is missing the large primes (p > T^{1/2}). These missing primes would
# contribute with weight 1/p (small). Their absence means the effective
# amplitude for small primes is RELATIVELY larger than 1/p compared to
# the full sum.
#
# More precisely: the form factor at finite T is
#   F(tau, T) = sum_{p<=X} log(p)/p * cos(2*pi*tau*logp) / sum_{p<=X} log(p)/p
# (normalized by the partial prime sum)
#
# versus the asymptotic:
#   F(tau) = sum_p log(p)/p * cos(2*pi*tau*logp) / sum_p log(p)/p
#
# The normalization sum_{p<=X} log(p)/p ≈ log(X) ≈ (1/2)*log(T).
# The full sum diverges as log(X) -> infinity.
# So the finite-T form factor has normalization 1/logX ≈ 2/logT.
#
# But this is just the overall SCALE, not the shape (alpha).
#
# The alpha < 1 arises from a different effect. Let me think about this
# from the DATA side: what does alpha < 1 mean?

# alpha < 1 means small primes (p=2,3,5) are OVERWEIGHTED relative to 1/p.
# log(2)/2^0.83 = 0.012  vs  log(2)/2^1.0 = 0.0069
# The data gives larger amplitudes for small primes than Selberg predicts.

# This could arise from:
# 1. The form factor F(tau) is not exactly |tau| but has a correction
# 2. The short-range (exp(-k)) component contaminates the oscillatory fit
# 3. The conversion from R_2(x) to ACF(k) introduces a T-dependent kernel

# Let me test hypothesis 1: fit the EFFECTIVE form factor from data.

print('\nEffective form factor measurement:')
print('  If A(p) = log(p)/p^alpha, the effective form factor at tau=logp/logT is:')
print('  F_eff(tau) ~ tau^{1-alpha} * (logT)^{1-alpha}')
print(f'  At alpha = {alpha_opt:.3f}: F_eff(tau) ~ tau^{{{1-alpha_opt:.3f}}}')
print(f'  Montgomery predicts: F(tau) = tau^1')
print(f'  Deviation from Montgomery: exponent = {1-alpha_opt:.3f} (should be 1.0)')

# ============================================================
# STEP 3: Derive alpha(T) from the explicit formula truncation
# ============================================================
print('\n' + '=' * 70)
print('STEP 3: THEORETICAL DERIVATION OF alpha(T)')
print('=' * 70)

# The key derivation:
#
# The pair correlation of zeros at height T involves the sum:
#   S(k, T) = sum_{p<=P_max} Lambda(p) * g(p, T) * cos(2*pi*k*logp/logT)
#
# where g(p, T) is the pair correlation weight for prime p at height T.
#
# From Bogomolny-Keating (1996), for the diagonal approximation:
#   g(p, T) = 1/p   (this gives alpha = 1, the Selberg result)
#
# From the off-diagonal correction (Bogomolny-Keating 1996, eq. 20):
#   g(p, T) = 1/p + correction terms involving T and p
#
# The leading correction is:
#   g(p, T) ≈ (1/p) * (1 + beta * log(p) / log(T))
#
# for some constant beta related to the number of off-diagonal terms.
# When fit as log(p)/p^alpha:
#   log(p) * g(p,T) = log(p)/p * (1 + beta*logp/logT)
#                   ≈ log(p) / p^{1 - beta*logp/(p*logT)}
#
# For small primes where log(p)/logT << 1:
#   effective alpha ≈ 1 - beta / logT * (weighted average of logp/p)
#
# This gives alpha(T) = 1 - c/logT for some c > 0, which:
#   - Is less than 1 (correct!)
#   - Increases toward 1 as T -> inf (correct!)
#   - The rate is 1/logT (testable!)

# Alternative derivation from the PRIME NUMBER THEOREM correction:
# The prime sum sum_{p<=X} log(p)/p = log(X) + M + O(exp(-c*sqrt(logX)))
# where M is the Meissel-Mertens constant.
# The NORMALIZED amplitude for prime p relative to the full sum is:
#   A(p, X) = (log(p)/p) / (sum_{q<=X} log(q)/q)
#           ≈ (log(p)/p) / log(X)
# vs the asymptotic:
#   A(p, inf) = log(p)/p / infinity = 0
# So the finite-X normalization is what gives the 1/logT overall scale.
# But this doesn't explain alpha < 1.

# The ACTUAL derivation: consider the explicit formula for the ACF.
# The ACF at lag k is related to the pair correlation via a kernel:
#   ACF(k) - ACF_GUE(k) = integral R_2(x) * K(x, k) dx
#
# where K(x, k) depends on the spacing density and the lag.
# For the Montgomery form, this integral involves:
#   integral cos(2*pi*f*x) * K(x, k) dx
# which selects specific frequencies from R_2.
#
# At finite T, the integral is over a finite range, introducing a
# SPECTRAL LEAKAGE effect: the cos(2*pi*f_p*x) term at frequency f_p
# leaks into nearby frequencies, effectively broadening the peak.
# When we fit a PURE cosine at frequency f_p, the fitted amplitude
# absorbs both the true amplitude and the leakage from nearby primes.
# Small primes (well-separated frequencies) absorb more leakage from
# large primes, INFLATING their apparent amplitude.

# Let's test this numerically: does the inflation of small-prime amplitudes
# match the alpha < 1 observation?

print('\nNumerical test: spectral leakage effect')
print('  If the ACF excess were EXACTLY Montgomery (alpha=1),')
print('  what alpha would we fit from a finite lag range?')

# Generate the "true" Montgomery ACF excess for many lags
N_LAGS_TRUE = 10000  # much more than 400
k_true = np.arange(1, N_LAGS_TRUE + 1, dtype=float)
true_excess = np.zeros(N_LAGS_TRUE)
for p in primes_all:
    freq = np.log(p) / LOG_T_HIGH
    if freq >= 0.5: continue
    true_excess += (2.0 / LOG_T_HIGH) * (np.log(p) / p) * np.cos(2 * np.pi * k_true * freq)

# Now TRUNCATE to 400 lags (mimicking our data) and fit alpha
truncated = true_excess[:MAX_LAG]
ss_trunc = np.sum(truncated ** 2)
short_trunc = [np.exp(-k_arr / 1.0), np.exp(-k_arr / 3.0), 1.0 / k_arr ** 2]

def neg_R2_trunc(alpha):
    model = np.zeros(MAX_LAG)
    for p in primes_all:
        freq = np.log(p) / LOG_T_HIGH
        if freq >= 0.45: continue
        model += np.log(p) / p ** alpha * np.cos(2 * np.pi * k_arr * freq)
    X = np.column_stack([model] + short_trunc)
    a, _, _, _ = np.linalg.lstsq(X, truncated, rcond=None)
    return np.sum((truncated - X @ a) ** 2) / ss_trunc

res_trunc = minimize_scalar(neg_R2_trunc, bounds=(0.3, 1.5), method='bounded')
alpha_trunc = res_trunc.x
R2_trunc = 1 - res_trunc.fun

print(f'  True model: alpha = 1.0 (Montgomery)')
print(f'  Fitted from 400 lags: alpha = {alpha_trunc:.4f}')
print(f'  R2 = {R2_trunc:.4f}')

if abs(alpha_trunc - 1.0) < 0.05:
    print('  -> Truncation does NOT explain alpha < 1')
    print('  -> The deviation is REAL, not an artifact of finite lags')
else:
    print(f'  -> Truncation shifts alpha from 1.0 to {alpha_trunc:.3f}!')
    print('  -> Part of the deviation is a spectral leakage artifact')

# ============================================================
# STEP 4: Predict alpha(T) from the form factor derivative
# ============================================================
print('\n' + '=' * 70)
print('STEP 4: PREDICTION FROM THE PAIR CORRELATION FORM FACTOR')
print('=' * 70)

# The Montgomery form factor F(tau) = |tau| for |tau| < 1.
# The amplitude of prime p in the pair correlation is proportional to:
#   a_p = (d/dtau) F(tau)|_{tau = log(p)/logT}
#        = 1     for tau > 0  (constant, giving 1/p after integration)
#
# But the MEASURED quantity is the ACF, not R_2 directly.
# The ACF at lag k is:
#   ACF(k) = (1/N) sum_n s_n * s_{n+k} / var(s)
#
# The connection to R_2 involves the LOCAL pair correlation at height T:
#   R_2(x; T) ≈ 1 - (sin(pi*x)/(pi*x))^2 + (2/logT) * sum_p (logp/p) * cos(2pi*x*logp/logT)
#
# This is the DIAGONAL approximation. Off-diagonal terms contribute:
#   R_2^{off}(x; T) ~ (2/logT)^2 * sum_{p,q: p!=q} (logp*logq)/(p*q) * cos(...)
#
# These off-diagonal terms are suppressed by 1/logT relative to diagonal.
# Their effect on the fitted alpha: they modify the amplitude of small primes
# by mixing in contributions from other primes.

# The effective amplitude INCLUDING off-diagonal (to leading order):
#   A_eff(p, T) = log(p)/p * (1 + sum_{q!=p} (logq/q) * K(p,q,T) / log(T))
#
# where K(p,q,T) involves the pair correlation of primes p,q.
# This correction term is:
#   - Positive (enhances small prime amplitudes)
#   - Scales as 1/logT (vanishes as T -> inf)
#   - Larger for small p (because more q's contribute)

# The fit alpha < 1 absorbs this correction:
#   log(p)/p * (1 + epsilon(p,T)) ≈ log(p) / p^{1-delta(p)}
# where delta(p) ≈ epsilon(p,T) * p / (p*log(p))... this is getting circular.

# Let me try a DIFFERENT approach: direct numerical prediction.

# APPROACH: Compute R_2(x;T) from the explicit formula at MULTIPLE heights,
# convert to ACF, and fit alpha at each height.

print('\nDirect computation: predicted alpha(T) from explicit formula')

def predict_alpha_at_height(T, max_lag=400, n_primes=200):
    """Compute the THEORETICAL ACF excess from the explicit formula
    at height T, then fit alpha."""
    log_T = np.log(T / (2 * np.pi))
    k = np.arange(1, max_lag + 1, dtype=float)
    primes = list(primerange(2, 500))[:n_primes]

    # The pair correlation contribution to the ACF excess:
    # From Bogomolny-Keating (1996), the connected 2-point function gives:
    #   ACF_excess(k) ≈ -1/(2*pi^2) * sum_p (log^2(p)/p) * cos(2*pi*k*logp/logT) / logT
    #
    # Note: the log^2(p)/p comes from the SQUARE of the von Mangoldt function
    # in the pair correlation (because R_2 involves |sum Lambda(n) n^{-it}|^2).
    #
    # More precisely, the oscillatory part of the pair correlation is:
    #   delta R_2(x) = -(1/2*pi^2*logT) * sum_p log^2(p)/p * cos(2*pi*x*logp/logT)
    #
    # When this is sampled at lag k (x = k in normalized spacing units):
    #   ACF_excess(k) ≈ c * sum_p log^2(p)/p * cos(2*pi*k*logp/logT) / logT

    # Model 1: log(p)/p (form factor derivative — Selberg)
    model_selberg = np.zeros(max_lag)
    for p in primes:
        freq = np.log(p) / log_T
        if freq >= 0.45: continue
        model_selberg += np.log(p) / p * np.cos(2 * np.pi * k * freq)

    # Model 2: log^2(p)/p (pair correlation — Bogomolny-Keating)
    model_bk = np.zeros(max_lag)
    for p in primes:
        freq = np.log(p) / log_T
        if freq >= 0.45: continue
        model_bk += np.log(p) ** 2 / p * np.cos(2 * np.pi * k * freq)

    # The THEORETICAL prediction: use the BK model as "truth"
    theoretical = model_bk / log_T

    # Now fit alpha to this theoretical prediction
    ss_th = np.sum(theoretical ** 2)
    if ss_th < 1e-30:
        return None

    sr = [np.exp(-k / 1.0), np.exp(-k / 3.0), 1.0 / k ** 2]

    def neg_R2(alpha):
        m = np.zeros(max_lag)
        for p in primes:
            freq = np.log(p) / log_T
            if freq >= 0.45: continue
            m += np.log(p) / p ** alpha * np.cos(2 * np.pi * k * freq)
        X = np.column_stack([m] + sr)
        a, _, _, _ = np.linalg.lstsq(X, theoretical, rcond=None)
        return np.sum((theoretical - X @ a) ** 2) / ss_th

    res = minimize_scalar(neg_R2, bounds=(0.3, 1.5), method='bounded')
    return res.x, 1 - res.fun


# Test at multiple heights
print(f'\n{"T":>15} {"logT":>8} {"loglogT":>8} {"alpha_pred":>10} {"R2":>7}')
print('-' * 55)

T_values = [1e4, 1e6, 1e8, 1e10, T_HIGH, 1e14, 1e16, 1e18, 1e20, 1e30, 1e50, 1e100]
pred_alphas = []
pred_loglogT = []

for T in T_values:
    result = predict_alpha_at_height(T)
    if result is None:
        continue
    alpha_pred, R2 = result
    log_T = np.log(T / (2 * np.pi))
    llt = np.log(log_T)
    pred_alphas.append(alpha_pred)
    pred_loglogT.append(llt)
    print(f'{T:>15.2e} {log_T:>8.2f} {llt:>8.3f} {alpha_pred:>10.4f} {R2:>7.4f}')

# Fit the convergence law
pred_alphas = np.array(pred_alphas)
pred_loglogT = np.array(pred_loglogT)

# Test: alpha = 1 - c / logT
log_T_arr = np.array([np.log(T / (2*np.pi)) for T in T_values[:len(pred_alphas)]])
inv_logT = 1.0 / log_T_arr

# Fit alpha = 1 - c / logT
from numpy.polynomial import polynomial as P
# alpha = a + b * (1/logT)
fit_invlogT = np.polyfit(inv_logT, pred_alphas, 1)
print(f'\nFit: alpha = {fit_invlogT[1]:.4f} + {fit_invlogT[0]:+.4f} / logT')
print(f'  Predicted intercept (T->inf): {fit_invlogT[1]:.4f} (should be ~1.0)')

# Fit alpha = 1 - c / log(logT)
inv_loglogT = 1.0 / pred_loglogT
fit_invllt = np.polyfit(inv_loglogT, pred_alphas, 1)
print(f'\nFit: alpha = {fit_invllt[1]:.4f} + {fit_invllt[0]:+.4f} / log(logT)')
print(f'  Predicted intercept (T->inf): {fit_invllt[1]:.4f}')

# Fit alpha = a + b * loglogT
fit_llt = np.polyfit(pred_loglogT, pred_alphas, 1)
print(f'\nFit: alpha = {fit_llt[1]:.4f} + {fit_llt[0]:+.4f} * log(logT)')
print(f'  Predicted at T~2.7e11 (loglogT=3.20): alpha = {fit_llt[1] + fit_llt[0]*3.20:.4f}')
print(f'  Observed at T~2.7e11: alpha = {alpha_opt:.4f}')

# ============================================================
# STEP 5: Compare PREDICTED vs OBSERVED alpha
# ============================================================
print('\n' + '=' * 70)
print('STEP 5: PREDICTED vs OBSERVED')
print('=' * 70)

# Get predicted alpha at our two data heights
for T_label, T_val, alpha_obs in [('T~2.7e11', T_HIGH, 0.833), ('T~1.4e20', 1.44e20, 0.893)]:
    result = predict_alpha_at_height(T_val)
    if result:
        alpha_pred, R2 = result
        print(f'{T_label}: predicted alpha = {alpha_pred:.4f}, observed = {alpha_obs:.3f}, '
              f'delta = {alpha_pred - alpha_obs:+.4f}')

# ============================================================
# STEP 6: The log^2(p)/p amplitude law
# ============================================================
print('\n' + '=' * 70)
print('STEP 6: ALTERNATIVE AMPLITUDE LAW — log^2(p)/p')
print('=' * 70)

# If the BK pair correlation formula gives log^2(p)/p instead of log(p)/p,
# the effective alpha when fitting log(p)/p^alpha would be:
#   log^2(p)/p = log(p) * (log(p)/p) = log(p) / p * log(p)
#
# Fitting this as log(p)/p^alpha:
#   log(p)/p^alpha = log^2(p)/p
#   => 1/p^alpha = log(p)/p
#   => p^{1-alpha} = log(p)
#   => (1-alpha)*log(p) = log(log(p))
#   => alpha = 1 - log(log(p))/log(p)
#
# This is a PRIME-DEPENDENT alpha! For p=2: alpha = 1 - log(log2)/log2 = 1 + 0.528 = 1.53
# That's too high. For p=100: alpha = 1 - log(log100)/log100 = 1 - 1.527/4.605 = 0.668
# Hmm, this doesn't match either.

# Actually, let me think more carefully. The BK formula gives:
#   R_2(x) - R_2^GUE(x) = -(1/logT) * sum_p log^2(p)/p * cos(2pi*x*logp/logT) + O(1/logT^2)
#
# But this is the PAIR CORRELATION R_2(x), not the ACF.
# The ACF is related to R_2 by an integral transform.
# When we fit the ACF excess, we're fitting a CONVOLVED version of R_2.

# Let me just test: does log^2(p)/p fit the DATA better than log(p)/p^alpha?

model_log2 = np.zeros(MAX_LAG)
for p in primes_all:
    freq = np.log(p) / LOG_T_HIGH
    if freq >= 0.45: continue
    model_log2 += np.log(p) ** 2 / p * np.cos(2 * np.pi * k_arr * freq)

X_log2 = np.column_stack([model_log2] + short_3)
amps_log2, _, _, _ = np.linalg.lstsq(X_log2, excess, rcond=None)
R2_log2 = 1 - np.sum((excess - X_log2 @ amps_log2) ** 2) / ss_tot
R2_adj_log2 = 1 - (1 - R2_log2) * (MAX_LAG - 1) / (MAX_LAG - 5)

# Also test log(p)/p (pure Selberg)
model_logp = np.zeros(MAX_LAG)
for p in primes_all:
    freq = np.log(p) / LOG_T_HIGH
    if freq >= 0.45: continue
    model_logp += np.log(p) / p * np.cos(2 * np.pi * k_arr * freq)

X_logp = np.column_stack([model_logp] + short_3)
amps_logp, _, _, _ = np.linalg.lstsq(X_logp, excess, rcond=None)
R2_logp = 1 - np.sum((excess - X_logp @ amps_logp) ** 2) / ss_tot
R2_adj_logp = 1 - (1 - R2_logp) * (MAX_LAG - 1) / (MAX_LAG - 5)

# And log(p)/p^0.833
model_opt = np.zeros(MAX_LAG)
for p in primes_all:
    freq = np.log(p) / LOG_T_HIGH
    if freq >= 0.45: continue
    model_opt += np.log(p) / p ** 0.833 * np.cos(2 * np.pi * k_arr * freq)

X_opt_m = np.column_stack([model_opt] + short_3)
amps_opt_m, _, _, _ = np.linalg.lstsq(X_opt_m, excess, rcond=None)
R2_opt_m = 1 - np.sum((excess - X_opt_m @ amps_opt_m) ** 2) / ss_tot
R2_adj_opt_m = 1 - (1 - R2_opt_m) * (MAX_LAG - 1) / (MAX_LAG - 5)

print(f'\n{"Amplitude law":<25} {"R2":>7} {"R2_adj":>7} {"scale":>10}')
print('-' * 55)
print(f'{"log(p)/p (Selberg)":<25} {R2_logp:>7.4f} {R2_adj_logp:>7.4f} {amps_logp[0]:>+10.5f}')
print(f'{"log^2(p)/p (BK)":<25} {R2_log2:>7.4f} {R2_adj_log2:>7.4f} {amps_log2[0]:>+10.5f}')
print(f'{"log(p)/p^0.833 (fitted)":<25} {R2_opt_m:>7.4f} {R2_adj_opt_m:>7.4f} {amps_opt_m[0]:>+10.5f}')

# ============================================================
# VERDICT
# ============================================================
print('\n' + '=' * 70)
print('VERDICT: THEORETICAL STATUS OF alpha(T)')
print('=' * 70)

print(f'\n1. Montgomery/Selberg (alpha=1, log(p)/p):')
print(f'   R2_adj = {R2_adj_logp:.4f}')
print(f'   This IS the asymptotic prediction. It\'s correct as T -> inf.')

print(f'\n2. Bogomolny-Keating (log^2(p)/p):')
print(f'   R2_adj = {R2_adj_log2:.4f}')
print(f'   The BK pair correlation formula gives a different weight.')

print(f'\n3. Data-optimal (alpha={alpha_opt:.3f}, log(p)/p^alpha):')
print(f'   R2_adj = {R2_adj_opt:.4f}')
print(f'   Best fit absorbs finite-T effects into the exponent.')

winner = max(
    [('Selberg log(p)/p', R2_adj_logp),
     ('BK log^2(p)/p', R2_adj_log2),
     ('Fitted log(p)/p^0.833', R2_adj_opt_m)],
    key=lambda x: x[1]
)
print(f'\n>>> BEST FIT: {winner[0]} (R2_adj = {winner[1]:.4f})')

if abs(R2_adj_log2 - R2_adj_opt_m) < 0.01:
    print('\n>>> KEY FINDING: log^2(p)/p matches the data as well as the fitted alpha!')
    print('>>> This means: alpha(T) < 1 arises because the TRUE amplitude law')
    print('>>> is log^2(p)/p (from the pair correlation), not log(p)/p (from the form factor).')
    print('>>> When fit as log(p)/p^alpha, the extra log(p) factor mimics a smaller alpha.')
    print('>>> The "convergence" alpha -> 1 is actually log^2(p)/p -> log(p)/p as the')
    print('>>> off-diagonal terms in the pair correlation become negligible.')
elif R2_adj_log2 > R2_adj_logp + 0.01:
    print(f'\n>>> log^2(p)/p BEATS log(p)/p by {R2_adj_log2 - R2_adj_logp:.4f}')
    print('>>> The Bogomolny-Keating weight is the correct finite-T amplitude law.')
else:
    print(f'\n>>> No clear winner between Selberg and BK.')
    print('>>> The finite-T correction is more subtle than a simple amplitude law change.')

print(f'\nTotal time: {time.time() - t0:.1f}s')
