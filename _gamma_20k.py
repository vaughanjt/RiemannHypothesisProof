"""Push gamma measurement to N=20000 for publication-grade evidence.

Uses chunked Gram matrix construction to manage memory.
At N=20000, G is 20000x20000 = 3.2GB in float64.
We use the identity G = W @ W^T and compute eigendecomposition
of the smaller dimension if possible.

Key optimization: M_sum can be reduced since weights decay as 1/(n(n+1)).
For n > 20000, the cumulative weight is < 1e-4. M_sum = 30000 is sufficient.
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np

t0 = time.time()

M_sum = 20000  # sufficient for N=20k (tail < 1e-4)
weights = np.array([1.0/(n*(n+1)) for n in range(1, M_sum+1)])
sqrt_w = np.sqrt(weights)
ns = np.arange(1, M_sum+1)

# Test points
test_Ns = [1000, 2000, 3000, 5000, 7500, 10000, 12500, 15000, 17500, 20000]

print("="*70, flush=True)
print("GAMMA SCALING TO N=20000 (publication grade)", flush=True)
print(f"M_sum = {M_sum}", flush=True)
print("="*70, flush=True)

print(f"\n  {'N':>6} {'gamma_all':>10} {'gamma_small':>12} {'d_n^2':>12} "
      f"{'lmin':>12} {'lmin*n^2':>10} {'time':>6}", flush=True)
print(f"  {'-'*72}", flush=True)

results = []

for N in test_Ns:
    t1 = time.time()

    # Build W in chunks to manage memory
    W = np.zeros((N, M_sum), dtype=np.float32)  # float32 to save memory
    for k_idx in range(N):
        k = k_idx + 2
        W[k_idx, :] = ((ns % k) / k) * sqrt_w

    # G = W @ W^T (NxN, float64 for accuracy)
    G = (W.astype(np.float64)) @ (W.astype(np.float64)).T

    # b vector
    b = np.zeros(N)
    for k_idx in range(N):
        k = k_idx + 2
        b[k_idx] = np.dot((ns % k) / k, weights)

    # Free W to save memory
    del W

    # Eigendecomposition
    eigenvalues, V = np.linalg.eigh(G)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]

    # Project b
    b_proj_sq = (V.T @ b)**2

    # d_n^2
    terms = b_proj_sq / np.maximum(eigenvalues, 1e-30)
    d_n_sq = max(1.0 - np.sum(terms), 0)

    # lambda_min
    lmin = eigenvalues[0]
    lmin_n2 = lmin * N**2

    # Free V to save memory
    del V, G

    # Fit gamma (all)
    mask = (eigenvalues > 1e-15) & (b_proj_sq > 1e-30)
    if np.sum(mask) > 10:
        coeffs = np.polyfit(np.log(eigenvalues[mask]),
                            np.log(b_proj_sq[mask]), 1)
        gamma_all = coeffs[0]
    else:
        gamma_all = float('nan')

    # Fit gamma (small eigenvalues)
    med = np.median(eigenvalues[eigenvalues > 1e-15])
    mask_s = (eigenvalues > 1e-15) & (eigenvalues < med) & (b_proj_sq > 1e-30)
    if np.sum(mask_s) > 5:
        coeffs_s = np.polyfit(np.log(eigenvalues[mask_s]),
                              np.log(b_proj_sq[mask_s]), 1)
        gamma_small = coeffs_s[0]
    else:
        gamma_small = float('nan')

    elapsed = time.time() - t1
    results.append((N, gamma_all, gamma_small, d_n_sq, lmin, lmin_n2))

    print(f"  {N:>6} {gamma_all:>10.4f} {gamma_small:>12.4f} {d_n_sq:>12.6e} "
          f"{lmin:>12.4e} {lmin_n2:>10.4f} {elapsed:>5.1f}s", flush=True)
    sys.stdout.flush()

# ============================================================
# ANALYSIS
# ============================================================
print("\n" + "="*70, flush=True)
print("ANALYSIS", flush=True)
print("="*70, flush=True)

ns_r = np.array([r[0] for r in results])
ga_all = np.array([r[1] for r in results])
ga_small = np.array([r[2] for r in results])
d2_arr = np.array([r[3] for r in results])
lmin_arr = np.array([r[4] for r in results])

# Gamma trend
mask_v = ~np.isnan(ga_all) & (ns_r >= 1000)
if np.sum(mask_v) > 3:
    # Linear in log(N)
    coeffs = np.polyfit(np.log(ns_r[mask_v]), ga_all[mask_v], 1)
    print(f"\n  gamma_all trend: {coeffs[1]:.4f} + {coeffs[0]:.4f}*log(N)", flush=True)
    if coeffs[0] < 0:
        n_cross = np.exp((1.0 - coeffs[1]) / coeffs[0])
        print(f"  Crosses gamma=1 at N ~ {n_cross:.2e}", flush=True)

    # Stabilization: gamma = a + b/log(N)
    inv_logN = 1.0 / np.log(ns_r[mask_v])
    coeffs_stab = np.polyfit(inv_logN, ga_all[mask_v], 1)
    gamma_inf = coeffs_stab[1]
    print(f"  Stabilization: gamma -> {gamma_inf:.4f} as N -> inf", flush=True)

# d_n^2 trend
mask_d = d2_arr > 0
if np.sum(mask_d) > 3:
    coeffs_d = np.polyfit(np.log(np.log(ns_r[mask_d])), np.log(d2_arr[mask_d]), 1)
    beta_d = -coeffs_d[0]
    print(f"\n  d_n^2 ~ C / (log n)^{{{beta_d:.2f}}}", flush=True)
    print(f"  (RH prediction: beta = 2)", flush=True)

# lambda_min trend
mask_l = lmin_arr > 0
if np.sum(mask_l) > 3:
    coeffs_l = np.polyfit(np.log(ns_r[mask_l]), np.log(lmin_arr[mask_l]), 1)
    alpha_l = -coeffs_l[0]
    print(f"\n  lambda_min ~ n^(-{alpha_l:.2f})", flush=True)

# Summary table for the paper
print("\n" + "="*70, flush=True)
print("TABLE FOR PUBLICATION", flush=True)
print("="*70, flush=True)
print(f"  {'N':>6} | {'d_n^2':>12} | {'lambda_min':>12} | {'gamma':>8} | {'gamma_small':>12}", flush=True)
print(f"  {'-'*60}", flush=True)
for N, ga, gs, d2, lm, _ in results:
    print(f"  {N:>6} | {d2:>12.6e} | {lm:>12.4e} | {ga:>8.4f} | {gs:>12.4f}", flush=True)

print(f"\n  Asymptotic gamma: {gamma_inf:.4f}", flush=True)
print(f"  RH prediction: 2.0", flush=True)

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
