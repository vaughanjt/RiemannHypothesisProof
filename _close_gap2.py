"""CLOSING GAP 2: Bound the remainder's effect on r, then compute r_RS.

THE STRATEGY:
  Z(t) = Z_main(t) + R(t)   where R(t) = O(t^{-1/4})
  r_full = Corr(gaps_of_Z, |Z(mids)|)
  r_main = Corr(gaps_of_Z_main, |Z_main(mids)|)

  We measured: r_full = 0.88, r_main = 0.80. Difference = 0.08.

  Question 1: Does the difference shrink with T?
  If yes: at large T, r_main -> r_full, and we only need to compute r_main.

  Question 2: Can we BOUND |r_full - r_main| in terms of ||R||/||Z||?
  If yes: this gives an analytic convergence rate.

  Question 3: Can we compute r_main for the truncated sum?
  The truncated sum has N(t) = floor(sqrt(t/2pi)) terms.
  At T=100: N=3. At T=400: N=5. At T=10000: N=13.
  For small N, the interference pattern is simple enough to analyze.

ALSO: Test with higher zeros to see if the gap narrows.
"""
import numpy as np
from scipy.stats import pearsonr, linregress
from scipy.optimize import brentq
import mpmath; mpmath.mp.dps = 15
import time

t0 = time.time()

def hardy_Z(t):
    return float(mpmath.siegelz(t))

def theta_approx(t):
    return t/2 * np.log(t/(2*np.pi)) - t/2 - np.pi/8

def Z_main(t, N_terms=None):
    """RS main sum (no remainder)."""
    if N_terms is None:
        N_terms = int(np.sqrt(t / (2*np.pi)))
    if N_terms < 1: N_terms = 1
    th = theta_approx(t)
    return 2 * sum(np.cos(th - t*np.log(n)) / np.sqrt(n)
                   for n in range(1, N_terms+1))

def find_zeros_func(f, t_start, t_end, dt=0.02):
    ts = np.arange(t_start, t_end, dt)
    vals = np.array([f(t) for t in ts])
    zeros = []
    for i in range(len(vals)-1):
        if vals[i]*vals[i+1] < 0:
            try: zeros.append(brentq(f, ts[i], ts[i+1]))
            except: pass
    return np.array(zeros)

def compute_r(zeros, f, trim_frac=0.1):
    if len(zeros) < 20: return 0., 0
    gaps = np.diff(zeros)
    mids = (zeros[:-1]+zeros[1:])/2
    peaks = np.array([abs(f(m)) for m in mids])
    nt = int(trim_frac * len(gaps))
    if nt > 0 and len(gaps) > 2*nt:
        return pearsonr(gaps[nt:-nt], peaks[nt:-nt])[0], len(gaps)-2*nt
    return pearsonr(gaps, peaks)[0], len(gaps)


# ============================================================
# TEST 1: How does r_main - r_full scale with T?
# ============================================================
print("="*60)
print("TEST 1: GAP r_full - r_main vs T (sliding windows)")
print("="*60)

zeta_zeros = np.load("_zeros_500.npy")

print(f"\n  {'window':>12} {'T_mid':>8} {'r_full':>8} {'r_main':>8} "
      f"{'gap':>8} {'N(T)':>6}")
print(f"  {'-'*54}")

for t_lo, t_hi in [(50,150), (100,200), (150,250), (200,300),
                    (300,450), (400,550), (500,700), (600,811)]:
    # Full Z
    z_full = find_zeros_func(hardy_Z, t_lo, t_hi, dt=0.02)
    r_full, n_full = compute_r(z_full, hardy_Z)

    # Main RS sum
    z_main = find_zeros_func(Z_main, t_lo, t_hi, dt=0.02)
    r_main, n_main = compute_r(z_main, Z_main)

    T_mid = (t_lo + t_hi) / 2
    N_T = int(np.sqrt(T_mid / (2*np.pi)))

    if n_full > 0 and n_main > 0:
        print(f"  {f'{t_lo}-{t_hi}':>12} {T_mid:>8.0f} {r_full:>+8.4f} {r_main:>+8.4f} "
              f"{r_full-r_main:>+8.4f} {N_T:>6}")


# ============================================================
# TEST 2: Perturbation theory — how does small additive noise affect r?
# ============================================================
print(f"\n{'='*60}")
print("TEST 2: PERTURBATION THEORY — additive noise effect on r")
print("="*60)

# If Z_pert(t) = Z(t) + epsilon * noise(t), how does r change?
# This models the remainder R(t) as noise of size epsilon.

# Use the exact Z zeros and peaks as baseline
N_use = 200
zeros_exact = zeta_zeros[:N_use]
gaps_exact = np.diff(zeros_exact)
mids_exact = (zeros_exact[:-1]+zeros_exact[1:])/2
peaks_exact = np.array([abs(hardy_Z(m)) for m in mids_exact])
nt = int(0.1*len(gaps_exact))
r_base = pearsonr(gaps_exact[nt:-nt], peaks_exact[nt:-nt])[0]

rng = np.random.default_rng(42)
print(f"\n  Baseline r = {r_base:+.4f}")
print(f"\n  {'epsilon':>10} {'r_pert':>10} {'delta_r':>10} {'|delta_r/eps|':>14}")
print(f"  {'-'*48}")

for eps in [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
    # Add smooth noise to Z at the midpoints
    # Model: R(t) ~ eps * cos(random_phase + t*random_freq)
    noise_phase = rng.uniform(0, 2*np.pi)
    noise_freq = rng.uniform(0.5, 2.0)
    peaks_pert = np.array([abs(hardy_Z(m) + eps*np.cos(noise_phase + m*noise_freq))
                           for m in mids_exact])
    r_pert = pearsonr(gaps_exact[nt:-nt], peaks_pert[nt:-nt])[0]
    delta = r_pert - r_base
    ratio = abs(delta/eps) if eps > 0 else 0
    print(f"  {eps:>10.3f} {r_pert:>+10.4f} {delta:>+10.4f} {ratio:>14.4f}")


# ============================================================
# TEST 3: The ACTUAL remainder R(t) = Z(t) - Z_main(t)
# ============================================================
print(f"\n{'='*60}")
print("TEST 3: ACTUAL REMAINDER R(t) = Z_exact - Z_main")
print("="*60)

# Compute R(t) at sample points
t_sample = np.linspace(50, 400, 1000)
R_vals = np.array([hardy_Z(t) - Z_main(t) for t in t_sample])
Z_vals = np.array([hardy_Z(t) for t in t_sample])

print(f"\n  ||R||_rms = {np.sqrt(np.mean(R_vals**2)):.4f}")
print(f"  ||Z||_rms = {np.sqrt(np.mean(Z_vals**2)):.4f}")
print(f"  ||R||/||Z|| = {np.sqrt(np.mean(R_vals**2))/np.sqrt(np.mean(Z_vals**2)):.4f}")

# R(t) scaling with t
for t_lo, t_hi in [(50,100), (100,200), (200,300), (300,400)]:
    mask = (t_sample >= t_lo) & (t_sample < t_hi)
    R_rms = np.sqrt(np.mean(R_vals[mask]**2))
    Z_rms = np.sqrt(np.mean(Z_vals[mask]**2))
    T_mid = (t_lo+t_hi)/2
    t_quarter = T_mid**(-0.25)
    print(f"  T~{T_mid:.0f}: ||R||={R_rms:.4f}, ||Z||={Z_rms:.4f}, "
          f"R/Z={R_rms/Z_rms:.4f}, T^{{-1/4}}={t_quarter:.4f}")


# ============================================================
# TEST 4: Correct r_main by adding the KNOWN R(t)
# ============================================================
print(f"\n{'='*60}")
print("TEST 4: RS sum with FIRST correction term")
print("="*60)

# The RS formula: Z(t) = Z_main(t) + R_0(t) + R_1(t) + ...
# R_0(t) = (-1)^{N-1} * (t/(2pi))^{-1/4} * cos(2*pi*(p^2 - p - 1/16))
# where p = sqrt(t/(2pi)) - N, N = floor(sqrt(t/(2pi)))

def Z_corrected(t, n_corrections=1):
    """RS sum with correction terms."""
    N_t = int(np.sqrt(t / (2*np.pi)))
    if N_t < 1: N_t = 1
    th = theta_approx(t)
    main = 2 * sum(np.cos(th - t*np.log(n)) / np.sqrt(n)
                   for n in range(1, N_t+1))

    if n_corrections >= 1:
        p = np.sqrt(t / (2*np.pi)) - N_t
        C0 = np.cos(2*np.pi*(p**2 - p - 1.0/16)) / np.cos(2*np.pi*p)
        R0 = (-1)**(N_t - 1) * (t / (2*np.pi))**(-0.25) * C0
        main += R0

    return main

# Compare: main only vs main + R0 vs exact
for label, f in [("Main only", Z_main),
                  ("Main + R0", Z_corrected),
                  ("Exact Z", hardy_Z)]:
    zeros = find_zeros_func(f, 50, 400, dt=0.02)
    r_val, n_pts = compute_r(zeros, f)
    print(f"  {label:>15}: r = {r_val:+.4f} ({len(zeros)} zeros)")


# ============================================================
# TEST 5: Extrapolate r_main and r_full to large T
# ============================================================
print(f"\n{'='*60}")
print("TEST 5: CONVERGENCE — does r_main -> r_full at large T?")
print("="*60)

# We can only test up to T~800 with our data.
# But we can check: is the gap (r_full - r_main) shrinking?
print(f"\n  {'T_mid':>8} {'r_full':>8} {'r_main':>8} {'r_corr':>8} "
      f"{'gap_main':>10} {'gap_corr':>10}")
print(f"  {'-'*58}")

for t_lo, t_hi in [(50,200), (100,300), (200,400), (300,500),
                    (400,600), (500,700), (600,811)]:
    z_f = find_zeros_func(hardy_Z, t_lo, t_hi, dt=0.02)
    z_m = find_zeros_func(Z_main, t_lo, t_hi, dt=0.02)
    z_c = find_zeros_func(Z_corrected, t_lo, t_hi, dt=0.02)

    r_f, _ = compute_r(z_f, hardy_Z)
    r_m, _ = compute_r(z_m, Z_main)
    r_c, _ = compute_r(z_c, Z_corrected)

    T_mid = (t_lo+t_hi)/2
    print(f"  {T_mid:>8.0f} {r_f:>+8.4f} {r_m:>+8.4f} {r_c:>+8.4f} "
          f"{r_f-r_m:>+10.4f} {r_f-r_c:>+10.4f}")


# ============================================================
# VERDICT
# ============================================================
print(f"\n{'='*60}")
print("VERDICT: GAP 2 STATUS")
print("="*60)
print(f"\nTotal time: {time.time()-t0:.1f}s")
