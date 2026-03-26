"""GAP 1: Is r a structural property of the RS sum form?

THE TEST:
  Z(t) = 2 * sum_{n=1}^{N(t)} cos(theta(t) - t*log(n)) / sqrt(n)

  The peak-gap correlation r = +0.88 could arise from:
  (A) The FORM of the sum (1/sqrt(n) weights, log(n) frequencies) -- STRUCTURAL
  (B) The SPECIFIC phase theta(t) from the functional equation -- ACCIDENTAL
  (C) Something special about the zeta zeros -- CIRCULAR

  If (A): random phases should also give high r. r is generic to the sum class.
  If (B): only theta gives high r. r depends on the functional equation.
  If (C): the argument is circular.

  Paper 2 already showed (A) dominates: generic smooth phase gives r = 0.670.
  Here we push further with systematic tests.

APPROACH:
  1. Generate f(t) = sum a_n cos(phi_n + omega_n * t) with:
     - omega_n = log(n) (RS frequencies)
     - a_n = 1/sqrt(n) (RS weights)
     - phi_n = random (NOT theta)
  2. Find zeros of f numerically
  3. Compute gap-peak correlation
  4. Repeat for many random phase realizations
  5. Also vary: amplitude law, frequency law, number of terms
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import pearsonr
import time

t0 = time.time()
rng = np.random.default_rng(42)


def find_zeros(f, t_start, t_end, dt=0.05):
    """Find zeros of f in [t_start, t_end] by sign-change detection + Brent."""
    ts = np.arange(t_start, t_end, dt)
    vals = np.array([f(t) for t in ts])
    zeros = []
    for i in range(len(vals) - 1):
        if vals[i] * vals[i+1] < 0:
            try:
                z = brentq(f, ts[i], ts[i+1])
                zeros.append(z)
            except:
                pass
    return np.array(zeros)


def compute_r(zeros, f):
    """Compute gap-peak correlation for a zero sequence and function f."""
    if len(zeros) < 20:
        return 0.0, 0
    gaps = np.diff(zeros)
    mids = (zeros[:-1] + zeros[1:]) / 2
    peaks = np.array([abs(f(m)) for m in mids])
    nt = int(0.1 * len(gaps))
    if nt > 0 and len(gaps) > 2*nt:
        r, _ = pearsonr(gaps[nt:-nt], peaks[nt:-nt])
        return r, len(gaps) - 2*nt
    elif len(gaps) >= 10:
        r, _ = pearsonr(gaps, peaks)
        return r, len(gaps)
    return 0.0, 0


# ============================================================
# TEST 1: Random-phase RS sums
# ============================================================
print("="*70)
print("TEST 1: RANDOM-PHASE RS SUMS")
print("  f(t) = sum cos(phi_n + t*log(n)) / sqrt(n)")
print("  phi_n = random uniform [0, 2*pi)")
print("="*70)

N_terms = 30   # number of terms in the sum
t_start = 50   # start looking for zeros
t_end = 300    # end of window

n_vals = np.arange(1, N_terms + 1)
omega = np.log(n_vals)       # frequencies = log(n)
amp = 1.0 / np.sqrt(n_vals)  # amplitudes = 1/sqrt(n)

print(f"\n  N_terms={N_terms}, window=[{t_start}, {t_end}]")
print(f"\n  {'trial':>6} {'n_zeros':>8} {'r':>10} {'mean_gap':>10} {'mean_peak':>10}")
print(f"  {'-'*48}")

r_values = []
for trial in range(30):
    phases = rng.uniform(0, 2*np.pi, N_terms)

    def f(t, _phases=phases):
        return np.sum(amp * np.cos(_phases + omega * t))

    zeros = find_zeros(f, t_start, t_end, dt=0.03)
    r_val, n_pts = compute_r(zeros, f)

    if n_pts > 0:
        r_values.append(r_val)
        gaps = np.diff(zeros)
        mids = (zeros[:-1] + zeros[1:]) / 2
        peaks = np.array([abs(f(m)) for m in mids])
        if trial < 10 or trial == 29:
            print(f"  {trial:>6} {len(zeros):>8} {r_val:>+10.4f} "
                  f"{np.mean(gaps):>10.4f} {np.mean(peaks):>10.4f}")

r_arr = np.array(r_values)
print(f"\n  SUMMARY ({len(r_arr)} trials):")
print(f"    mean r = {np.mean(r_arr):+.4f}")
print(f"    std r  = {np.std(r_arr):.4f}")
print(f"    min r  = {np.min(r_arr):+.4f}")
print(f"    max r  = {np.max(r_arr):+.4f}")
print(f"    P(r > 0.5) = {np.mean(r_arr > 0.5)*100:.0f}%")
print(f"    P(r > 0.7) = {np.mean(r_arr > 0.7)*100:.0f}%")


# ============================================================
# TEST 2: Vary the amplitude law
# ============================================================
print("\n" + "="*70)
print("TEST 2: VARY AMPLITUDE LAW a_n = 1/n^alpha")
print("="*70)

print(f"\n  {'alpha':>8} {'mean_r':>10} {'std_r':>8} {'n_zeros':>8}")
print(f"  {'-'*38}")

for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5]:
    amp_a = 1.0 / n_vals**alpha
    r_trials = []
    for trial in range(20):
        phases = rng.uniform(0, 2*np.pi, N_terms)
        def f(t, _p=phases, _a=amp_a):
            return np.sum(_a * np.cos(_p + omega * t))
        zeros = find_zeros(f, t_start, t_end, dt=0.03)
        r_val, n_pts = compute_r(zeros, f)
        if n_pts > 10:
            r_trials.append(r_val)
    if r_trials:
        n_z = len(find_zeros(lambda t: np.sum(amp_a * np.cos(rng.uniform(0,2*np.pi,N_terms) + omega*t)), t_start, t_end, dt=0.03))
        print(f"  {alpha:>8.1f} {np.mean(r_trials):>+10.4f} {np.std(r_trials):>8.4f} {n_z:>8}")


# ============================================================
# TEST 3: Vary the frequency law
# ============================================================
print("\n" + "="*70)
print("TEST 3: VARY FREQUENCY LAW")
print("="*70)

print(f"\n  {'freq_law':>20} {'mean_r':>10} {'std_r':>8}")
print(f"  {'-'*42}")

freq_laws = {
    "log(n) [RS]": np.log(n_vals),
    "n": n_vals.astype(float),
    "sqrt(n)": np.sqrt(n_vals),
    "n^0.3": n_vals**0.3,
    "log(n)^2": np.log(n_vals + 1)**2,
    "random": np.sort(rng.uniform(0.1, 5, N_terms)),
    "harmonic n*w0": n_vals * 0.3,
    "prime-like": np.array([np.log(p) for p in [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113]][:N_terms]),
}

for name, freq in freq_laws.items():
    r_trials = []
    for trial in range(20):
        phases = rng.uniform(0, 2*np.pi, N_terms)
        def f(t, _p=phases, _f=freq):
            return np.sum(amp * np.cos(_p + _f * t))
        zeros = find_zeros(f, t_start, t_end, dt=0.02)
        r_val, n_pts = compute_r(zeros, f)
        if n_pts > 10:
            r_trials.append(r_val)
    if r_trials:
        print(f"  {name:>20} {np.mean(r_trials):>+10.4f} {np.std(r_trials):>8.4f}")


# ============================================================
# TEST 4: Scale with number of terms
# ============================================================
print("\n" + "="*70)
print("TEST 4: SCALE WITH NUMBER OF TERMS")
print("="*70)

print(f"\n  {'N_terms':>8} {'mean_r':>10} {'std_r':>8} {'n_zeros':>8}")
print(f"  {'-'*38}")

for N_t in [3, 5, 10, 20, 30, 50, 80]:
    nv = np.arange(1, N_t + 1)
    om = np.log(nv)
    am = 1.0 / np.sqrt(nv)
    r_trials = []
    nz_avg = 0
    for trial in range(20):
        phases = rng.uniform(0, 2*np.pi, N_t)
        def f(t, _p=phases, _o=om, _a=am):
            return np.sum(_a * np.cos(_p + _o * t))
        zeros = find_zeros(f, t_start, t_end, dt=0.02)
        nz_avg += len(zeros)
        r_val, n_pts = compute_r(zeros, f)
        if n_pts > 10:
            r_trials.append(r_val)
    if r_trials:
        print(f"  {N_t:>8} {np.mean(r_trials):>+10.4f} {np.std(r_trials):>8.4f} "
              f"{nz_avg//20:>8}")


# ============================================================
# TEST 5: The actual Z function vs random phases (direct comparison)
# ============================================================
print("\n" + "="*70)
print("TEST 5: ACTUAL Z FUNCTION vs RANDOM-PHASE SUMS")
print("="*70)

import mpmath
mpmath.mp.dps = 15

def hardy_Z(t):
    return float(mpmath.siegelz(t))

# Compute r for the actual Z function in [50, 300]
z_zeros = find_zeros(hardy_Z, 50, 300, dt=0.03)
r_Z, n_Z = compute_r(z_zeros, hardy_Z)
print(f"\n  Actual Z function: r = {r_Z:+.4f} ({len(z_zeros)} zeros)")
print(f"  Random-phase RS sums: r = {np.mean(r_arr):+.4f} +/- {np.std(r_arr):.4f} ({len(r_arr)} trials)")
print(f"  Difference: {r_Z - np.mean(r_arr):+.4f}")
print(f"  Z-score: {(r_Z - np.mean(r_arr))/np.std(r_arr):.1f} sigma")

# ============================================================
# VERDICT
# ============================================================
print("\n" + "="*70)
print("VERDICT: IS r STRUCTURAL?")
print("="*70)

print(f"""
  Random-phase RS sums (1/sqrt(n) weights, log(n) frequencies):
    mean r = {np.mean(r_arr):+.4f} +/- {np.std(r_arr):.4f}
    Actual Z: r = {r_Z:+.4f}

  If random-phase r is comparable to actual Z r:
    -> r is STRUCTURAL (depends on sum form, not specific phase)
    -> Gap 1 is resolved: r_0 is determined by the RS sum class

  If random-phase r is much lower:
    -> r depends on theta(t) specifically
    -> Gap 1 requires proving theta gives maximal r
""")

print(f"Total time: {time.time()-t0:.1f}s")
