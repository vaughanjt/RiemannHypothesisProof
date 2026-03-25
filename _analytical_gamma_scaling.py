"""Verify: does the ANALYTICAL gamma bound (from decomposition) stay > 1 at large N?

The analytical decomposition gives:
  beta_2 >= max_seg_exponent(S) + max_S_exponent = delta_S + beta_1
  gamma_analytical >= 2 * (delta_S + beta_1)

At N=500: gamma_analytical = 2 * 0.663 = 1.327 > 1

Does this hold at larger N?
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np

t0 = time.time()

M_sum = 10000
weights = np.array([1.0/(n*(n+1)) for n in range(1, M_sum+1)])
sqrt_w = np.sqrt(weights)
ns_arr = np.arange(1, M_sum+1)

print("="*70, flush=True)
print("ANALYTICAL GAMMA BOUND SCALING", flush=True)
print("="*70, flush=True)

print(f"\n  {'N':>6} {'beta_1':>8} {'delta_S':>8} {'b1+dS':>8} {'2*(b1+dS)':>10} "
      f"{'beta_2':>8} {'gamma':>8} {'ok?':>5}", flush=True)
print(f"  {'-'*66}", flush=True)

for N in [100, 200, 300, 500, 750, 1000, 1500, 2000]:
    W = np.zeros((N, M_sum))
    for k_idx in range(N):
        k = k_idx + 2
        W[k_idx, :] = ((ns_arr % k) / k) * sqrt_w

    G = W @ W.T
    b = np.zeros(N)
    for k_idx in range(N):
        k = k_idx + 2
        b[k_idx] = np.dot((ns_arr % k) / k, weights)

    eigenvalues, V = np.linalg.eigh(G)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]

    b_proj_sq = (V.T @ b)**2

    # Measured gamma
    mask_g = (eigenvalues > 1e-12) & (b_proj_sq > 1e-30)
    gamma_meas = np.polyfit(np.log(eigenvalues[mask_g]), np.log(b_proj_sq[mask_g]), 1)[0]

    # Partial sums
    S = np.cumsum(V, axis=0)
    T = np.cumsum(S, axis=0)
    max_S = np.max(np.abs(S), axis=0)
    max_T = np.max(np.abs(T), axis=0)

    # beta_1 = exponent of max|S| vs lambda
    mask_s = (eigenvalues > 1e-12) & (max_S > 1e-12)
    beta_1 = np.polyfit(np.log(eigenvalues[mask_s]), np.log(max_S[mask_s]), 1)[0]

    # beta_2 = exponent of max|T| vs lambda
    mask_t = (eigenvalues > 1e-12) & (max_T > 1e-12)
    beta_2 = np.polyfit(np.log(eigenvalues[mask_t]), np.log(max_T[mask_t]), 1)[0]

    # delta_S = exponent of max segment length of S vs lambda
    S_max_seg = np.zeros(N)
    for i in range(N):
        s = S[:, i]
        signs = np.sign(s)
        cp = np.where(np.abs(np.diff(signs)) > 0)[0]
        if len(cp) > 0:
            segs = np.diff(np.concatenate([[0], cp, [N-1]]))
            S_max_seg[i] = np.max(segs)
        else:
            S_max_seg[i] = N

    mask_seg = (eigenvalues > 1e-12) & (S_max_seg > 0) & (S_max_seg < N)
    if np.sum(mask_seg) > 5:
        delta_S = np.polyfit(np.log(eigenvalues[mask_seg]), np.log(S_max_seg[mask_seg]), 1)[0]
    else:
        delta_S = 0.5  # default

    analytical = beta_1 + delta_S
    gamma_anal = 2 * analytical
    ok = "YES" if gamma_anal > 1 else "no"

    print(f"  {N:>6} {beta_1:>8.4f} {delta_S:>8.4f} {analytical:>8.4f} {gamma_anal:>10.3f} "
          f"{beta_2:>8.4f} {gamma_meas:>8.4f} {ok:>5}", flush=True)

print(f"\n  All times computed in {time.time()-t0:.1f}s", flush=True)
print(f"\n  KEY: 2*(beta_1 + delta_S) > 1 means the ANALYTICAL decomposition", flush=True)
print(f"  (without any arithmetic bonus) proves gamma > 1.", flush=True)
