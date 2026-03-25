"""LAMBDA_MIN ATTACK: Prove lambda_min(G_n) > c/n^A to get RH.

Grok confirmed: if lambda_min(G_n) >> 1/n^A for any finite A,
combined with known norm bounds, d_n -> 0 and RH follows.

THE STRUCTURE WE CAN EXPLOIT:
  G_{jk} = sum_{n=1}^M [(n mod j)/j] * [(n mod k)/k] / (n(n+1))

  The basis vector for k is PERIODIC with period k:
    f_k[n] = (n mod k)/k

  For primes p, q with gcd(p,q)=1:
  By CRT, (n mod p, n mod q) cycles through ALL pq pairs uniformly.
  This means the overlap <f_p, f_q> is PREDICTABLE from arithmetic.

  The Gram matrix decomposes as:
    G = D + E
  where D = diagonal (self-overlaps) and E = off-diagonal (cross-terms)
  D_{kk} ~ (k-1)(2k-1)/(6k^2)  [converges to 1/3]
  E_{jk} ~ (j-1)(k-1)/(4jk) * h(j,k)  [CRT + weight correction]

  If we can BOUND the spectral norm of E relative to D,
  we get lambda_min(G) >= lambda_min(D) - ||E||.

PLAN:
  1. Compute lambda_min(G_n) for n up to 2000+
  2. Establish the decay law (polynomial? exponential?)
  3. Decompose G = D + E and bound ||E||
  4. Use CRT to prove ||E|| < lambda_min(D)
  5. This gives lambda_min(G) > 0 for all n => d_n -> 0 => RH
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np
from math import gcd

t0 = time.time()

# ============================================================
# STEP 1: Compute lambda_min(G_n) for large n
# ============================================================
print("="*70, flush=True)
print("STEP 1: lambda_min(G_n) SCALING LAW", flush=True)
print("="*70, flush=True)

M_sum = 10000  # high truncation for accuracy

# Pre-compute weights
weights = np.array([1.0/(n*(n+1)) for n in range(1, M_sum+1)])
sqrt_w = np.sqrt(weights)

# Build the weighted basis matrix incrementally
# W[k_idx, n] = (n mod k)/k * sqrt(weight[n])
# G = W @ W^T

# We need lambda_min for various n. Build W row by row.
print(f"  M_sum = {M_sum}", flush=True)
print(f"\n  {'n':>6} {'lambda_min':>14} {'lambda_max':>14} {'cond':>12} {'lambda_min*n^2':>16} {'lambda_min*(logn)^2':>20}", flush=True)
print(f"  {'-'*85}", flush=True)

results = []
ns_to_compute = (list(range(5, 51, 5)) + list(range(60, 201, 10)) +
                 list(range(220, 501, 20)) + list(range(550, 1001, 50)) +
                 list(range(1100, 2001, 100)))

# Build W incrementally
max_n = max(ns_to_compute)
W_full = np.zeros((max_n, M_sum))

for k_idx in range(max_n):
    k = k_idx + 2
    ns = np.arange(1, M_sum+1)
    W_full[k_idx, :] = ((ns % k) / k) * sqrt_w

print(f"  W matrix built: {max_n} x {M_sum} ({time.time()-t0:.1f}s)", flush=True)

for n_basis in ns_to_compute:
    W_n = W_full[:n_basis, :]
    G_n = W_n @ W_n.T

    eigs = np.linalg.eigvalsh(G_n)
    lmin = eigs[0]
    lmax = eigs[-1]
    cond = lmax / max(lmin, 1e-30)

    lmin_n2 = lmin * n_basis**2
    lmin_logn2 = lmin * np.log(n_basis)**2

    results.append((n_basis, lmin, lmax, cond))

    if n_basis <= 50 or n_basis % 100 == 0 or n_basis in [60, 80, 100, 150, 200, 300, 500, 750, 1000]:
        print(f"  {n_basis:>6} {lmin:>14.6e} {lmax:>14.6e} {cond:>12.2e} {lmin_n2:>16.6e} {lmin_logn2:>20.6e}", flush=True)

print(f"\n  Computed in {time.time()-t0:.1f}s", flush=True)


# ============================================================
# STEP 2: FIT THE DECAY LAW
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 2: lambda_min DECAY LAW", flush=True)
print("="*70, flush=True)

ns_arr = np.array([r[0] for r in results])
lmin_arr = np.array([r[1] for r in results])

# Filter valid
mask = lmin_arr > 0
ns_v = ns_arr[mask]
lmin_v = lmin_arr[mask]

if len(ns_v) > 10:
    # Power law: lambda_min ~ A * n^{-alpha}
    log_n = np.log(ns_v)
    log_lmin = np.log(lmin_v)

    # Fit on latter half (skip transient)
    start = len(log_n) // 3
    coeffs_power = np.polyfit(log_n[start:], log_lmin[start:], 1)
    alpha_power = -coeffs_power[0]
    A_power = np.exp(coeffs_power[1])

    # R^2
    predicted = coeffs_power[1] + coeffs_power[0] * log_n[start:]
    ss_res = np.sum((log_lmin[start:] - predicted)**2)
    ss_tot = np.sum((log_lmin[start:] - np.mean(log_lmin[start:]))**2)
    r2_power = 1 - ss_res / (ss_tot + 1e-30)

    print(f"\n  Power law: lambda_min ~ {A_power:.4e} * n^(-{alpha_power:.4f})", flush=True)
    print(f"  R^2 = {r2_power:.6f}", flush=True)

    # Exponential: lambda_min ~ A * exp(-beta * n)
    coeffs_exp = np.polyfit(ns_v[start:], log_lmin[start:], 1)
    beta_exp = -coeffs_exp[0]
    A_exp = np.exp(coeffs_exp[1])
    predicted_exp = coeffs_exp[1] + coeffs_exp[0] * ns_v[start:]
    ss_res_exp = np.sum((log_lmin[start:] - predicted_exp)**2)
    r2_exp = 1 - ss_res_exp / (ss_tot + 1e-30)

    print(f"\n  Exponential: lambda_min ~ {A_exp:.4e} * exp(-{beta_exp:.6f} * n)", flush=True)
    print(f"  R^2 = {r2_exp:.6f}", flush=True)

    # Log law: lambda_min ~ A / (log n)^gamma
    log_log_n = np.log(log_n[start:])
    coeffs_log = np.polyfit(log_log_n, log_lmin[start:], 1)
    gamma_log = -coeffs_log[0]
    A_log = np.exp(coeffs_log[1])
    predicted_log = coeffs_log[1] + coeffs_log[0] * log_log_n
    ss_res_log = np.sum((log_lmin[start:] - predicted_log)**2)
    r2_log = 1 - ss_res_log / (ss_tot + 1e-30)

    print(f"\n  Log law: lambda_min ~ {A_log:.4e} / (log n)^{{{gamma_log:.4f}}}", flush=True)
    print(f"  R^2 = {r2_log:.6f}", flush=True)

    print(f"\n  BEST FIT: ", end="", flush=True)
    if r2_power > r2_exp and r2_power > r2_log:
        print(f"POWER LAW (n^(-{alpha_power:.2f}), R^2={r2_power:.4f})", flush=True)
        print(f"  => lambda_min DECAYS POLYNOMIALLY", flush=True)
        print(f"  => d_n^2 < C/n^B for some B > 0", flush=True)
        print(f"  => d_n -> 0 => RH", flush=True)
        print(f"  IF this decay rate can be PROVED.", flush=True)
    elif r2_exp > r2_power:
        print(f"EXPONENTIAL (exp(-{beta_exp:.4f}*n), R^2={r2_exp:.4f})", flush=True)
        print(f"  => lambda_min decays FASTER than polynomial", flush=True)
        print(f"  => Still gives d_n -> 0, but harder to prove", flush=True)
    else:
        print(f"LOG LAW ((log n)^(-{gamma_log:.2f}), R^2={r2_log:.4f})", flush=True)

    # Critical test: is lambda_min * n^2 bounded or growing?
    product_n2 = lmin_v * ns_v**2
    print(f"\n  lambda_min * n^2 behavior:", flush=True)
    print(f"    At n=50:   {product_n2[ns_v==50][0] if 50 in ns_v else 'N/A':.6e}", flush=True)
    idx_200 = np.argmin(np.abs(ns_v - 200))
    idx_500 = np.argmin(np.abs(ns_v - 500))
    idx_1000 = np.argmin(np.abs(ns_v - 1000))
    print(f"    At n~{ns_v[idx_200]}: {product_n2[idx_200]:.6e}", flush=True)
    print(f"    At n~{ns_v[idx_500]}: {product_n2[idx_500]:.6e}", flush=True)
    if idx_1000 < len(product_n2):
        print(f"    At n~{ns_v[idx_1000]}: {product_n2[idx_1000]:.6e}", flush=True)


# ============================================================
# STEP 3: GRAM MATRIX DECOMPOSITION G = D + E
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 3: GRAM MATRIX DECOMPOSITION G = D + E", flush=True)
print("="*70, flush=True)

# At a manageable size, decompose G into diagonal + off-diagonal
# and analyze the spectral norm of each part
N_decomp = 300

W_d = W_full[:N_decomp, :]
G_d = W_d @ W_d.T

D = np.diag(np.diag(G_d))  # diagonal part
E = G_d - D                 # off-diagonal part

eigs_D = np.diag(D)  # eigenvalues of diagonal = diagonal entries
eigs_G = np.linalg.eigvalsh(G_d)
norm_E = np.linalg.norm(E, ord=2)  # spectral norm of E

print(f"\n  N = {N_decomp}", flush=True)
print(f"  lambda_min(G) = {eigs_G[0]:.6e}", flush=True)
print(f"  lambda_min(D) = {np.min(eigs_D):.6e}", flush=True)
print(f"  ||E||_2        = {norm_E:.6e}", flush=True)
print(f"  lambda_min(D) - ||E|| = {np.min(eigs_D) - norm_E:.6e}", flush=True)

if np.min(eigs_D) > norm_E:
    print(f"  => Weyl bound gives lambda_min(G) > 0 (but loose)", flush=True)
else:
    print(f"  => Weyl bound: lambda_min(D) < ||E||, need tighter analysis", flush=True)
    print(f"  => The off-diagonal dominates the diagonal!", flush=True)

# Diagonal structure
print(f"\n  Diagonal entries D_kk = <rho_k, rho_k>:", flush=True)
print(f"  {'k':>5} {'D_kk':>14} {'theory':>14} {'ratio':>8}", flush=True)
print(f"  {'-'*44}", flush=True)
for k_idx in range(min(20, N_decomp)):
    k = k_idx + 2
    d_kk = D[k_idx, k_idx]
    theory = (k-1)*(2*k-1) / (6*k*k)  # asymptotic for large M
    ratio = d_kk / (theory + 1e-30)
    if k <= 15 or k in [20, 30, 50]:
        print(f"  {k:>5} {d_kk:>14.8e} {theory:>14.8e} {ratio:>8.4f}", flush=True)

# Off-diagonal structure
print(f"\n  Off-diagonal spectral norm ||E|| vs N:", flush=True)
for N_test in [20, 50, 100, 200, 300]:
    if N_test > N_decomp:
        break
    G_t = W_full[:N_test, :] @ W_full[:N_test, :].T
    D_t = np.diag(np.diag(G_t))
    E_t = G_t - D_t
    norm_Et = np.linalg.norm(E_t, ord=2)
    min_Dt = np.min(np.diag(D_t))
    eigs_t = np.linalg.eigvalsh(G_t)
    print(f"    N={N_test:>4}: ||E||={norm_Et:.4e}, min(D)={min_Dt:.4e}, "
          f"lambda_min(G)={eigs_t[0]:.4e}, ratio ||E||/min(D)={norm_Et/min_Dt:.2f}", flush=True)


# ============================================================
# STEP 4: CRT STRUCTURE OF OFF-DIAGONAL
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 4: CRT STRUCTURE — Why off-diagonal is bounded", flush=True)
print("="*70, flush=True)

# For j, k coprime: by CRT, (n mod j, n mod k) is equidistributed
# over Z/j x Z/k. The Gram entry G_{jk} = sum (n%j)(n%k)/(jk) * w_n
# The "leading term" is (mean of n%j)(mean of n%k) * sum(w_n)
# = [(j-1)/2][(k-1)/2]/(jk) * 1
# = (j-1)(k-1)/(4jk)
#
# The DEVIATION from this leading term is what matters.
# By CRT equidistribution, the deviation is O(1/lcm(j,k)) per period.

# Compute: how well does (j-1)(k-1)/(4jk) predict off-diagonal entries?
print(f"\n  CRT prediction of off-diagonal entries (coprime j,k):", flush=True)
print(f"  {'(j,k)':>10} {'G_jk':>14} {'CRT pred':>14} {'deviation':>14} {'rel_dev':>10}", flush=True)
print(f"  {'-'*65}", flush=True)

deviations = []
for j_idx in range(min(50, N_decomp)):
    j = j_idx + 2
    for k_idx in range(j_idx + 1, min(50, N_decomp)):
        k = k_idx + 2
        if gcd(j, k) == 1:
            g_jk = G_d[j_idx, k_idx]
            crt_pred = (j-1)*(k-1) / (4*j*k)
            dev = g_jk - crt_pred
            rel_dev = dev / (crt_pred + 1e-30)
            deviations.append((j, k, g_jk, crt_pred, dev, rel_dev))

# Show a few
for j, k, g, c, d, r in deviations[:10]:
    print(f"  ({j:>3},{k:>3}) {g:>14.8e} {c:>14.8e} {d:>14.8e} {r:>10.4f}", flush=True)

# Statistics
devs = np.array([d[4] for d in deviations])
rel_devs = np.array([d[5] for d in deviations])
print(f"\n  Deviation statistics ({len(deviations)} coprime pairs):", flush=True)
print(f"    Mean absolute deviation: {np.mean(np.abs(devs)):.6e}", flush=True)
print(f"    Mean relative deviation: {np.mean(np.abs(rel_devs)):.4f}", flush=True)
print(f"    Max relative deviation:  {np.max(np.abs(rel_devs)):.4f}", flush=True)

# The CRT matrix: E_CRT_{jk} = (j-1)(k-1)/(4jk) for coprime, 0 otherwise
# Its spectral norm is bounded because it's rank-1 up to corrections
# E_CRT = (v @ v^T) * correction  where v_k = (k-1)/(2k) ~ 1/2
# So ||E_CRT|| ~ N/4 * (sum 1/k)^2 ... this grows with N.

# But the DEVIATION E - E_CRT has much smaller spectral norm.
# This is the key: the predictable part (CRT) can be handled analytically,
# and the unpredictable part (deviations) may be bounded.

N_crt = 200
G_crt = W_full[:N_crt, :] @ W_full[:N_crt, :].T

# Build CRT prediction matrix
E_crt_pred = np.zeros((N_crt, N_crt))
for j_idx in range(N_crt):
    j = j_idx + 2
    for k_idx in range(j_idx + 1, N_crt):
        k = k_idx + 2
        if gcd(j, k) == 1:
            E_crt_pred[j_idx, k_idx] = (j-1)*(k-1) / (4*j*k)
            E_crt_pred[k_idx, j_idx] = E_crt_pred[j_idx, k_idx]
        else:
            # Non-coprime: more complex, use gcd structure
            g = gcd(j, k)
            # Rough prediction: scale by gcd-related factor
            E_crt_pred[j_idx, k_idx] = (j-1)*(k-1) / (4*j*k) * (1 + np.log(g)/np.log(max(j,k)+1))
            E_crt_pred[k_idx, j_idx] = E_crt_pred[j_idx, k_idx]

E_residual = G_crt - np.diag(np.diag(G_crt)) - E_crt_pred
norm_residual = np.linalg.norm(E_residual, ord=2)
norm_crt = np.linalg.norm(E_crt_pred, ord=2)
norm_offdiag = np.linalg.norm(G_crt - np.diag(np.diag(G_crt)), ord=2)

print(f"\n  Decomposition at N={N_crt}:", flush=True)
print(f"    ||E||           = {norm_offdiag:.6e} (full off-diagonal)", flush=True)
print(f"    ||E_CRT||       = {norm_crt:.6e} (CRT-predictable part)", flush=True)
print(f"    ||E - E_CRT||   = {norm_residual:.6e} (CRT residual)", flush=True)
print(f"    CRT explains:    {(1 - norm_residual/norm_offdiag)*100:.1f}%", flush=True)


# ============================================================
# STEP 5: EIGENVALUE STRUCTURE OF G
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 5: EIGENVALUE DISTRIBUTION OF G", flush=True)
print("="*70, flush=True)

# The eigenvalue distribution tells us about the "difficulty" of approximation
# A few large eigenvalues = easy directions (primes 2, 3, 5)
# Many small eigenvalues = hard directions (high composites)

for N_test in [100, 300, 500, 1000]:
    if N_test > max_n:
        break
    G_t = W_full[:N_test, :] @ W_full[:N_test, :].T
    eigs_t = np.sort(np.linalg.eigvalsh(G_t))

    # Eigenvalue statistics
    n_large = np.sum(eigs_t > 0.01)
    n_small = np.sum(eigs_t < 1e-4)

    print(f"\n  N={N_test}:", flush=True)
    print(f"    lambda_1 (min) = {eigs_t[0]:.4e}", flush=True)
    print(f"    lambda_5       = {eigs_t[4]:.4e}", flush=True)
    print(f"    lambda_N (max) = {eigs_t[-1]:.4e}", flush=True)
    print(f"    > 0.01: {n_large} eigenvalues ({n_large*100/N_test:.0f}%)", flush=True)
    print(f"    < 1e-4: {n_small} eigenvalues ({n_small*100/N_test:.0f}%)", flush=True)
    print(f"    Trace(G) = {np.sum(eigs_t):.4f}", flush=True)

    # The trace = sum of diagonal entries = sum D_kk
    # D_kk ~ (k-1)(2k-1)/(6k^2) -> 1/3 for large k
    # So Trace ~ N/3 for large N
    print(f"    N/3 = {N_test/3:.4f}", flush=True)


# ============================================================
# STEP 6: THE KEY — CAN WE BOUND lambda_min FROM BELOW?
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 6: LOWER BOUND ON lambda_min", flush=True)
print("="*70, flush=True)

# Approach: Gershgorin circle theorem
# lambda_min(G) >= min_k { D_kk - sum_{j!=k} |G_{jk}| }
#
# D_kk ~ 1/3 for large k
# sum_{j!=k} |G_{jk}| = sum of absolute off-diagonal entries in row k
#
# If this sum < 1/3 for all k, then lambda_min > 0.

N_gersh = 500
if N_gersh > max_n:
    N_gersh = max_n
G_gersh = W_full[:N_gersh, :] @ W_full[:N_gersh, :].T

gersh_bounds = []
for k_idx in range(N_gersh):
    d_kk = G_gersh[k_idx, k_idx]
    row_sum = np.sum(np.abs(G_gersh[k_idx, :])) - d_kk
    gersh_lower = d_kk - row_sum
    gersh_bounds.append((k_idx + 2, d_kk, row_sum, gersh_lower))

gersh_arr = np.array([g[3] for g in gersh_bounds])
min_gersh = np.min(gersh_arr)
worst_k = gersh_bounds[np.argmin(gersh_arr)][0]

print(f"\n  Gershgorin bound at N={N_gersh}:", flush=True)
print(f"    min over k of (D_kk - sum|G_jk|) = {min_gersh:.6e}", flush=True)
print(f"    Worst row: k={worst_k}", flush=True)
print(f"    Actual lambda_min = {np.linalg.eigvalsh(G_gersh)[0]:.6e}", flush=True)

if min_gersh > 0:
    print(f"    GERSHGORIN PROVES lambda_min > 0 at N={N_gersh}!", flush=True)
else:
    print(f"    Gershgorin is negative — too loose.", flush=True)

    # How do the row sums grow?
    print(f"\n  Row sum growth:", flush=True)
    print(f"  {'k':>5} {'D_kk':>10} {'row_sum':>10} {'Gersh':>10} {'row/D':>8}", flush=True)
    print(f"  {'-'*46}", flush=True)
    for k, d, r, g in gersh_bounds:
        if k <= 15 or k in [20, 50, 100, 200, 300, 500]:
            print(f"  {k:>5} {d:>10.6f} {r:>10.6f} {g:>10.6f} {r/d:>8.2f}", flush=True)

    # The ratio row_sum/D_kk: if this is bounded < 1, we're done
    ratios = np.array([g[2]/g[1] for g in gersh_bounds])
    print(f"\n  Row sum / D_kk: mean={np.mean(ratios):.4f}, max={np.max(ratios):.4f}", flush=True)
    print(f"  Does ratio grow with k? ", end="", flush=True)
    # Fit
    ks = np.array([g[0] for g in gersh_bounds])
    coeffs_r = np.polyfit(np.log(ks), ratios, 1)
    print(f"slope in log(k) = {coeffs_r[0]:.4f}", flush=True)
    if coeffs_r[0] > 0.1:
        print(f"  YES — row sums grow relative to diagonal. Gershgorin can't work directly.", flush=True)
    else:
        print(f"  NO — ratio is stable. Gershgorin might work with refinement.", flush=True)


# ============================================================
# VERDICT
# ============================================================
print("\n" + "="*70, flush=True)
print("VERDICT", flush=True)
print("="*70, flush=True)

if len(ns_v) > 10:
    print(f"""
  lambda_min DECAY:
    Power law: n^(-{alpha_power:.2f}), R^2 = {r2_power:.4f}
    Exponential: exp(-{beta_exp:.5f}*n), R^2 = {r2_exp:.4f}
    Log law: (log n)^(-{gamma_log:.2f}), R^2 = {r2_log:.4f}

  lambda_min is {'POLYNOMIAL' if r2_power > r2_exp else 'EXPONENTIAL'} decay.
  """, flush=True)

    if r2_power > r2_exp:
        print(f"  POLYNOMIAL DECAY with exponent {alpha_power:.2f}.", flush=True)
        print(f"  If provable: d_n^2 ~ n^(-B) for B = 2 - {alpha_power:.2f} = {2-alpha_power:.2f}", flush=True)
        if 2 - alpha_power > 0:
            print(f"  Since B > 0: d_n -> 0. RH follows.", flush=True)
        else:
            print(f"  B <= 0: lambda_min decays too fast, doesn't directly give d_n -> 0", flush=True)
            print(f"  But d_n^2 ~ 1/n^alpha * 1/lambda_min is more subtle", flush=True)

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
