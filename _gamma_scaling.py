"""Push gamma measurement to large N.

The critical question: does gamma stay above 1 as N -> infinity?
Trend from N=50-1000: gamma ~ 2.58 - 0.10*log(N)
Projected crossing of gamma=1 at N ~ 11,000.

We need to go to N=2000+ to see if gamma stabilizes or keeps falling.

OPTIMIZATION: We don't need the full eigendecomposition.
For gamma, we need the relationship between |b.v_i|^2 and lambda_i.
We can sample eigenvalues at the BOTTOM of the spectrum using
iterative methods (Lanczos) instead of full O(N^3) decomposition.

But for N up to 5000, full decomposition on the Gram matrix
G = W @ W^T is feasible since G is only NxN (not MxM).
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np

t0 = time.time()

M_sum = 10000
weights = np.array([1.0/(n*(n+1)) for n in range(1, M_sum+1)])
sqrt_w = np.sqrt(weights)

print("="*70, flush=True)
print("GAMMA SCALING TO LARGE N", flush=True)
print("="*70, flush=True)

# Pre-build the full basis matrix for maximum N
N_max = 5000
print(f"  Building basis matrix {N_max} x {M_sum}...", flush=True)
t_build = time.time()

# Vectorized construction
ns = np.arange(1, M_sum+1)
W_full = np.zeros((N_max, M_sum))
for k_idx in range(N_max):
    k = k_idx + 2
    W_full[k_idx, :] = ((ns % k) / k) * sqrt_w

# b vector
b_full = np.zeros(N_max)
for k_idx in range(N_max):
    k = k_idx + 2
    b_full[k_idx] = np.dot((ns % k) / k, weights)

print(f"  Built in {time.time()-t_build:.1f}s", flush=True)

# Compute gamma at each N
test_Ns = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

print(f"\n  {'N':>6} {'gamma_all':>10} {'gamma_small':>12} {'gamma_tail':>11} "
      f"{'d_n^2':>12} {'lmin':>12} {'time':>6}", flush=True)
print(f"  {'-'*72}", flush=True)

results = []

for N in test_Ns:
    t1 = time.time()

    # Build G = W @ W^T (N x N)
    W_n = W_full[:N, :]
    G = W_n @ W_n.T
    b = b_full[:N]

    # Full eigendecomposition
    eigenvalues, V = np.linalg.eigh(G)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]

    # Project b
    b_proj_sq = (V.T @ b)**2

    # d_n^2
    terms = b_proj_sq / np.maximum(eigenvalues, 1e-30)
    d_n_sq = max(1.0 - np.sum(terms), 0)

    # Fit gamma (all eigenvalues)
    mask = (eigenvalues > 1e-12) & (b_proj_sq > 1e-30)
    if np.sum(mask) > 10:
        coeffs = np.polyfit(np.log(eigenvalues[mask]),
                            np.log(b_proj_sq[mask]), 1)
        gamma_all = coeffs[0]
    else:
        gamma_all = float('nan')

    # Fit gamma (small eigenvalues only — bottom half)
    med = np.median(eigenvalues[eigenvalues > 1e-12])
    mask_s = (eigenvalues > 1e-12) & (eigenvalues < med) & (b_proj_sq > 1e-30)
    if np.sum(mask_s) > 5:
        coeffs_s = np.polyfit(np.log(eigenvalues[mask_s]),
                              np.log(b_proj_sq[mask_s]), 1)
        gamma_small = coeffs_s[0]
    else:
        gamma_small = float('nan')

    # Fit gamma (bottom 10% — the real tail)
    q10 = np.quantile(eigenvalues[eigenvalues > 1e-12], 0.10)
    mask_t = (eigenvalues > 1e-12) & (eigenvalues < q10) & (b_proj_sq > 1e-30)
    if np.sum(mask_t) > 5:
        coeffs_t = np.polyfit(np.log(eigenvalues[mask_t]),
                              np.log(b_proj_sq[mask_t]), 1)
        gamma_tail = coeffs_t[0]
    else:
        gamma_tail = float('nan')

    elapsed = time.time() - t1
    results.append((N, gamma_all, gamma_small, gamma_tail, d_n_sq, eigenvalues[0]))

    print(f"  {N:>6} {gamma_all:>10.4f} {gamma_small:>12.4f} {gamma_tail:>11.4f} "
          f"{d_n_sq:>12.6e} {eigenvalues[0]:>12.4e} {elapsed:>5.1f}s", flush=True)


# ============================================================
# TREND ANALYSIS
# ============================================================
print("\n" + "="*70, flush=True)
print("TREND ANALYSIS", flush=True)
print("="*70, flush=True)

ns_r = np.array([r[0] for r in results])
ga_all = np.array([r[1] for r in results])
ga_small = np.array([r[2] for r in results])
ga_tail = np.array([r[3] for r in results])

# Fit trends
for name, ga in [("gamma_all", ga_all), ("gamma_small", ga_small), ("gamma_tail", ga_tail)]:
    mask_valid = ~np.isnan(ga)
    if np.sum(mask_valid) > 3:
        # Linear in log(N)
        coeffs = np.polyfit(np.log(ns_r[mask_valid]), ga[mask_valid], 1)
        slope = coeffs[0]
        intercept = coeffs[1]

        # Projected crossing of gamma = 1
        if slope < 0:
            n_cross = np.exp((1.0 - intercept) / slope)
            print(f"  {name:>12}: gamma ~ {intercept:.3f} + {slope:.4f}*log(N)", flush=True)
            print(f"               Crosses gamma=1 at N ~ {n_cross:.0f}", flush=True)
        else:
            print(f"  {name:>12}: gamma ~ {intercept:.3f} + {slope:.4f}*log(N) (STABLE/GROWING)", flush=True)

        # Last 5 values — is it stabilizing?
        last5 = ga[mask_valid][-5:]
        if len(last5) >= 3:
            local_slope = np.polyfit(np.arange(len(last5)), last5, 1)[0]
            print(f"               Last 5 trend: {local_slope:+.4f}/step "
                  f"({'declining' if local_slope < -0.01 else 'stable' if abs(local_slope) < 0.01 else 'rising'})", flush=True)
            print(f"               Last 5 values: {last5}", flush=True)

# ============================================================
# THE CRITICAL QUESTION
# ============================================================
print("\n" + "="*70, flush=True)
print("THE CRITICAL QUESTION: Does gamma stay above 1?", flush=True)
print("="*70, flush=True)

last_gamma_all = ga_all[~np.isnan(ga_all)][-1]
last_gamma_small = ga_small[~np.isnan(ga_small)][-1]
last_gamma_tail = ga_tail[~np.isnan(ga_tail)][-1]
last_N = ns_r[-1]

print(f"\n  At N = {last_N}:", flush=True)
print(f"    gamma (all):   {last_gamma_all:.4f}", flush=True)
print(f"    gamma (small): {last_gamma_small:.4f}", flush=True)
print(f"    gamma (tail):  {last_gamma_tail:.4f}", flush=True)

if last_gamma_all > 1 and last_gamma_small > 1:
    print(f"\n  GAMMA REMAINS ABOVE 1 at N={last_N}.", flush=True)
    print(f"  The spectral projection |b.v|^2 ~ lambda^gamma with gamma > 1", flush=True)
    print(f"  ensures absolute convergence of the Nyman-Beurling sum.", flush=True)
elif last_gamma_tail > 1:
    print(f"\n  gamma_all dipped below 1 but gamma_tail still > 1.", flush=True)
    print(f"  The TAIL (smallest eigenvalues) still has favorable scaling.", flush=True)
else:
    print(f"\n  WARNING: gamma approaching or below 1.", flush=True)
    print(f"  The convergence argument weakens at this scale.", flush=True)

# Does gamma stabilize to a constant?
if len(ga_all) > 5:
    # Fit gamma = a + b/log(N) (convergence to constant a)
    mask_v = ~np.isnan(ga_all) & (ns_r >= 200)
    if np.sum(mask_v) > 3:
        inv_logN = 1.0 / np.log(ns_r[mask_v])
        coeffs_stab = np.polyfit(inv_logN, ga_all[mask_v], 1)
        gamma_inf = coeffs_stab[1]  # value as 1/log(N) -> 0
        print(f"\n  Stabilization fit: gamma ~ {gamma_inf:.4f} + {coeffs_stab[0]:.2f}/log(N)", flush=True)
        print(f"  Asymptotic gamma (N -> inf): {gamma_inf:.4f}", flush=True)
        if gamma_inf > 1:
            print(f"  *** GAMMA CONVERGES TO {gamma_inf:.2f} > 1 ***", flush=True)
            print(f"  *** If provable: d_n -> 0, hence RH ***", flush=True)
        elif gamma_inf > 0.5:
            print(f"  Asymptotic gamma between 0.5 and 1 — convergence unclear", flush=True)
        else:
            print(f"  Asymptotic gamma < 0.5 — convergence unlikely from this alone", flush=True)

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
