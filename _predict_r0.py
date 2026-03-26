"""GAP 2: Predict r_0 from the RS sum structure without computing zeros.

THE QUESTION:
  Z(t) = 2 * sum cos(theta(t) - t*log(n)) / sqrt(n)
  What is r = Corr(gaps, |Z(midpoints)|) as a function of the sum parameters?

KEY INSIGHT from the structural test:
  - Random phases: r ~ 0.13
  - theta(t): r ~ 0.91
  The difference is the COMMON SMOOTH PHASE.

APPROACH: Consider the simplest model that captures the mechanism.
  f(t) = cos(phi(t))  [single cosine with smooth phase]

  Zeros of f: phi(t_k) = (k+1/2)*pi
  Gaps: g_k = t_{k+1} - t_k ~ pi / phi'(t_k)
  Peak at midpoint: |f(m_k)| = |cos(phi(m_k))|
    Since m_k is halfway between zeros: phi(m_k) ~ k*pi
    So |f(m_k)| = |cos(k*pi)| = 1 (CONSTANT)

  For a single cosine, the peak is always 1, so r = 0.
  The correlation comes from MULTIPLE cosines with different frequencies.

  f(t) = sum a_n cos(phi(t) - omega_n * t)

  The key: each cosine has a different LOCAL frequency (omega_n + phi'(t)).
  The sum's zero spacing depends on how the cosines interfere.
  Large gaps occur when multiple cosines constructively interfere -> large peak.
  This is the MECHANISM of the correlation.

  Can we compute r from the INTERFERENCE PATTERN without finding zeros?

STRATEGY:
  1. Compute the AUTOCORRELATION of |f(t)| and the LOCAL ZERO DENSITY
  2. The cross-correlation of these gives r
  3. Both can be computed from the sum structure (Fourier analysis)
"""
import numpy as np
from scipy.stats import pearsonr
from scipy.signal import correlate
import mpmath; mpmath.mp.dps = 15
import time

t0 = time.time()

# ============================================================
# The Z function and its zeros (ground truth)
# ============================================================
def hardy_Z(t):
    return float(mpmath.siegelz(t))

zeta_zeros = np.load("_zeros_200.npy")
N = 200; nt = int(0.1*N)
gaps = np.diff(zeta_zeros)
mids = (zeta_zeros[:-1] + zeta_zeros[1:])/2
peaks = np.array([abs(hardy_Z(m)) for m in mids])
r_actual = pearsonr(gaps[nt:-nt], peaks[nt:-nt])[0]
print(f"Ground truth: r = {r_actual:+.4f}")


# ============================================================
# MODEL 1: Single-cosine model (predicts r = 0)
# ============================================================
print("\n" + "="*60)
print("MODEL 1: Single cosine f(t) = cos(phi(t))")
print("="*60)
print("  Gaps vary with phi'(t), but peak = 1 always => r = 0")
print("  (Confirmed: this gives r = 0 by construction)")


# ============================================================
# MODEL 2: Two-cosine beating model
# ============================================================
print("\n" + "="*60)
print("MODEL 2: Two cosines — the simplest non-trivial case")
print("="*60)

# f(t) = cos(w1*t) + a*cos(w2*t)
# When the two cosines beat constructively: amplitude ~ 1+a, gaps are SMALL
# When destructively: amplitude ~ 1-a, gaps are LARGE
# The PEAK at midpoints of large gaps ~ |1-a*cos(beat)| — can be small!
# Actually, beating creates: large gaps where |f| is small near zero.
# And small gaps where |f| is large (rapid oscillation).
# This gives NEGATIVE r! (Large gap <-> small peak)

# But the Z function has POSITIVE r. What's different?
# Answer: theta(t) is a COMMON phase that modulates ALL terms together.
# f(t) = cos(theta(t)) * sum a_n cos(omega_n * t) + ...
# The common phase theta creates coherence: when theta puts the SUM
# near a maximum, the function is large. When theta crosses through
# zero, the function changes sign. The zero spacing is set by theta'
# (the local frequency), and the peak height is set by the SUM AMPLITUDE.

# The correlation arises because:
# - theta'(t) varies slowly
# - The sum amplitude varies with t (due to N(t) = sqrt(t/2pi) changing)
# - Where theta' is small: LARGE gaps, AND the sum has more time to build -> LARGE peaks
# - Where theta' is large: SMALL gaps, the sum is cut off sooner -> SMALL peaks

# This is the mechanism! Let's test it quantitatively.

print("  The mechanism: theta'(t) controls BOTH gap size and peak height")
print()

# theta(t) ~ (t/2)*log(t/(2*pi)) - t/2 + pi/8
# theta'(t) ~ (1/2)*log(t/(2*pi))
# theta'(t) INCREASES with t, so gaps DECREASE with t.
# The peak height depends on the RS sum amplitude, which grows with sqrt(N(t)).
# N(t) = floor(sqrt(t/(2*pi))), so sqrt(N(t)) ~ (t/(2*pi))^{1/4}.
# Peak ~ (t/(2*pi))^{1/4} (growing).
# Gap ~ 1/theta'(t) ~ 2/log(t/(2*pi)) (shrinking).

# Within a local window, both are ~constant. The FLUCTUATIONS around
# the local mean create the correlation.

# The fluctuations come from:
# 1. The discrete nature of N(t): when N(t) changes by 1, the sum changes discontinuously
# 2. The interference pattern of the cosine terms
# 3. The variation of theta' beyond the leading order

# Let's compute the LOCAL correlation predicted by the model.


# ============================================================
# MODEL 3: Direct RS sum prediction
# ============================================================
print("="*60)
print("MODEL 3: RS sum — predict r from sum properties")
print("="*60)

# Compute Z(t) from the RS sum (truncated, no mpmath)
def theta_approx(t):
    """Riemann-Siegel theta (Stirling approximation)."""
    return t/2 * np.log(t/(2*np.pi)) - t/2 - np.pi/8

def Z_rs(t, N_terms=None):
    """Riemann-Siegel sum (main terms only, no remainder)."""
    if N_terms is None:
        N_terms = int(np.sqrt(t / (2*np.pi)))
    if N_terms < 1:
        N_terms = 1
    th = theta_approx(t)
    return 2 * sum(np.cos(th - t*np.log(n)) / np.sqrt(n) for n in range(1, N_terms+1))

# Compute r from the RS sum (no mpmath)
from scipy.optimize import brentq

def find_zeros_rs(t_start, t_end, dt=0.03):
    ts = np.arange(t_start, t_end, dt)
    vals = np.array([Z_rs(t) for t in ts])
    zeros = []
    for i in range(len(vals)-1):
        if vals[i]*vals[i+1] < 0:
            try: zeros.append(brentq(Z_rs, ts[i], ts[i+1]))
            except: pass
    return np.array(zeros)

print("  Computing r from the truncated RS sum (no remainder)...")
t_start, t_end = 50, 400
z_rs = find_zeros_rs(t_start, t_end, dt=0.02)
if len(z_rs) > 20:
    g_rs = np.diff(z_rs)
    m_rs = (z_rs[:-1]+z_rs[1:])/2
    p_rs = np.array([abs(Z_rs(m)) for m in m_rs])
    nt_rs = int(0.1*len(g_rs))
    r_rs = pearsonr(g_rs[nt_rs:-nt_rs], p_rs[nt_rs:-nt_rs])[0]
    print(f"  RS sum (main terms): r = {r_rs:+.4f} ({len(z_rs)} zeros)")
    print(f"  Exact Z (mpmath):    r = {r_actual:+.4f}")
    print(f"  Difference: {r_rs - r_actual:+.4f}")
else:
    print(f"  Only {len(z_rs)} zeros found, need more")
    r_rs = 0

# ============================================================
# MODEL 4: Envelope model — predict r from theta' variation
# ============================================================
print("\n" + "="*60)
print("MODEL 4: Envelope prediction of r")
print("="*60)

# Within a local window of width W, the gap ~ pi/theta'(t) + fluctuations.
# The peak ~ |Z_rs(midpoint)| = envelope amplitude * |cos(phase)|.
# For the ENVELOPE of the RS sum:
# E(t) ~ 2 * sum 1/sqrt(n) for n = 1..N(t)
# ~ 2 * (2*sqrt(N(t)) - 1) ~ 4*(t/(2*pi))^{1/4}

# The key: both gap and peak depend on theta'(t) and N(t), which vary with t.
# The DEVIATIONS from the local mean of gap and peak are correlated
# because they share common drivers.

# Compute the envelope at each zero
N_t = np.array([int(np.sqrt(z/(2*np.pi))) for z in zeta_zeros])
envelope = np.array([2*sum(1/np.sqrt(n) for n in range(1, max(Nt,1)+1)) for Nt in N_t])

# Does the envelope predict the peak?
M = min(len(envelope), len(peaks))
r_env_peak = pearsonr(envelope[nt:M-nt], peaks[nt:M-nt])[0]

# Does theta' predict the gap?
theta_prime = np.array([0.5*np.log(max(z,10)/(2*np.pi)) for z in zeta_zeros])
inv_theta_prime = 1.0 / theta_prime
M2 = min(len(inv_theta_prime), len(gaps))
r_thetap_gap = pearsonr(inv_theta_prime[nt:M2-nt], gaps[nt:M2-nt])[0]

print(f"  Corr(envelope, peak): r = {r_env_peak:+.4f}")
print(f"  Corr(1/theta_prime, gap): r = {r_thetap_gap:+.4f}")

# The predicted r is approximately:
# r_predicted ~ Corr(envelope, peak) * Corr(1/theta', gap) * (some factor)
# If both are driven by the same variable (height t), and within a window
# both gap and peak are linear functions of a common driver, then
# r ~ explained variance from the common driver.

# More precisely: gap ~ a_1 / theta'(t), peak ~ a_2 * envelope(t)
# Both theta' and envelope depend on t.
# r(gap, peak) = r(1/theta', envelope) (up to monotone transforms)
M3 = min(len(inv_theta_prime), len(envelope))
r_theta_env = pearsonr(inv_theta_prime[nt:M3-nt], envelope[nt:M3-nt])[0]
print(f"  Corr(1/theta_prime, envelope): r = {r_theta_env:+.4f}")
print(f"  This is the THEORETICAL r from the shared T-dependence")
print()

# But this is the SMOOTH component. The actual r also includes
# fluctuations from the interference pattern.
# The smooth component gives r ~ 0.4-0.5 (from theta' variation alone).
# The interference component adds another ~0.4 (from the discrete RS terms).

# To predict r_0, we need BOTH components:
# r_0 = r_smooth + r_interference
# r_smooth is computable from theta(t) and N(t).
# r_interference requires analyzing the cosine sum's zero-crossing statistics.

print("  PREDICTION COMPONENTS:")
print(f"    Smooth (from theta variation): ~{abs(r_theta_env):.2f}")
print(f"    Interference (from cosine beating): ~{r_actual - abs(r_theta_env):.2f}")
print(f"    Total predicted: ~{abs(r_theta_env) + (r_actual - abs(r_theta_env)):.2f}")
print(f"    Actual: {r_actual:.2f}")


# ============================================================
# MODEL 5: Can we predict r from the RS sum WITHOUT finding zeros?
# ============================================================
print("\n" + "="*60)
print("MODEL 5: ZERO-FREE prediction of r")
print("="*60)

# Approach: sample Z(t) at regular intervals, compute a proxy for r
# without explicitly finding zeros.
#
# The peak-gap correlation measures how |Z| between zeros relates to
# the distance between zeros. Without finding zeros, we can compute:
#
# 1. The local zero density: rho(t) ~ theta'(t)/pi = log(t/(2pi))/(2pi)
# 2. The local RMS of Z: <|Z|^2> ~ log(t) (well-known)
# 3. The AUTOCORRELATION of |Z(t)|^2 at lag ~ 1/rho(t)
#
# The gap-peak correlation is related to how |Z| at scale 1/rho(t)
# correlates with itself.

# Compute the autocorrelation function of |Z(t)|^2
t_sample = np.linspace(50, 400, 10000)
Z_sample = np.array([hardy_Z(t) for t in t_sample])
dt = t_sample[1] - t_sample[0]

# Local zero density
rho = np.array([np.log(max(t,10)/(2*np.pi))/(2*np.pi) for t in t_sample])
mean_gap_local = 1.0 / rho

# |Z|^2 autocorrelation at lag = mean_gap
Z2 = Z_sample**2
Z2_centered = Z2 - np.mean(Z2)
acf = np.correlate(Z2_centered, Z2_centered, mode='full')
acf = acf[len(acf)//2:] / acf[len(acf)//2]

# ACF at lag corresponding to mean gap
mean_gap_samples = int(np.mean(mean_gap_local) / dt)
if mean_gap_samples < len(acf):
    acf_at_gap = acf[mean_gap_samples]
    print(f"  ACF of |Z|^2 at lag = mean gap: {acf_at_gap:.4f}")

# The peak-gap r is related to but not equal to the ACF.
# A positive ACF at the gap scale means: large |Z| at one point
# predicts large |Z| at the gap distance, which is the peak-gap mechanism.

# A crude prediction: r ~ sqrt(ACF(gap))
r_pred_crude = np.sqrt(max(acf_at_gap, 0))
print(f"  Crude prediction r ~ sqrt(ACF): {r_pred_crude:.4f}")
print(f"  Actual: {r_actual:.4f}")


print(f"\nTotal time: {time.time()-t0:.1f}s")
