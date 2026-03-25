"""THE PROOF PATH: Why beta_2 > 0.5 via eigenvalue equation.

From the eigenvalue equation Gv = lambda*v:
  S(k) = <v, R_k> / lambda    where R_k(m) = sum_{j<=k} G_{jm}
  T(k) = <v, Q_k> / lambda    where Q_k(m) = sum_{j<=k} R_j(m)

So max|T_i| = max_k |<v_i, Q_k>| / lambda_i.

For beta_2 = 0.75: max|T| ~ lambda^{0.75}
=> max_k |<v_i, Q_k>| ~ lambda^{0.75} * lambda = lambda^{1.75}

This means Q_k has the SAME spectral concentration property as b:
its projection onto small-eigenvalue eigenvectors decays as lambda^{1.75}.

THE SELF-CONSISTENT ARGUMENT:
If we can show Q_k (the double cumulative row sum of G) has the same
smoothness as b, then the gamma > 1 for b IMPLIES gamma > 1 for Q_k,
which IMPLIES beta_2 > 0.5, which gives ANOTHER proof that gamma > 1.

This is a BOOTSTRAP: the smoothness of b and Q_k reinforce each other
through the Gram matrix structure.
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np

t0 = time.time()

M_sum = 10000
weights = np.array([1.0/(n*(n+1)) for n in range(1, M_sum+1)])
sqrt_w = np.sqrt(weights)
ns_arr = np.arange(1, M_sum+1)

N = 500
print(f"Building system at N={N}...", flush=True)

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

# ============================================================
# STEP 1: Compute Q_k and its spectral projection
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 1: Spectral projection of Q_k (double row sum of G)", flush=True)
print("="*70, flush=True)

# R_k(m) = sum_{j=1}^{k} G_{jm} = cumulative row sum
# Q_k(m) = sum_{j=1}^{k} R_j(m) = double cumulative row sum

# R_k = G^T @ e_k where e_k = [1,...,1,0,...,0] (k ones)
# Actually R_k = sum of first k rows of G

# Compute Q_k for several k values and measure spectral projection
print(f"\n  Measuring |<Q_k, v_i>| scaling with lambda for various k:", flush=True)

gamma_Q_values = []
for k_test in [N//4, N//2, 3*N//4, N]:
    # R(m) = sum_{j=1}^{k} G_{jm}
    R = np.sum(G[:k_test, :], axis=0)
    # Q = sum_{j=1}^{k} R_j  where R_j = sum_{i=1}^{j} G_{im}
    # More efficiently: Q(m) = sum_{j=1}^{k} sum_{i=1}^{j} G_{im}
    #                        = sum_{i=1}^{k} (k-i+1) * G_{im}
    Q = np.zeros(N)
    for j in range(k_test):
        Q += np.sum(G[:j+1, :], axis=0)

    # Spectral projection of Q
    Q_proj = V.T @ Q
    Q_proj_sq = Q_proj**2

    # Fit gamma_Q
    mask = (eigenvalues > 1e-10) & (Q_proj_sq > 1e-30)
    if np.sum(mask) > 10:
        coeffs = np.polyfit(np.log(eigenvalues[mask]),
                            np.log(Q_proj_sq[mask]), 1)
        gamma_Q = coeffs[0]
        gamma_Q_values.append((k_test, gamma_Q))

        print(f"  k={k_test:>4}: gamma_Q = {gamma_Q:.4f} "
              f"(b has gamma = {np.polyfit(np.log(eigenvalues[b_proj_sq > 1e-30]), np.log(b_proj_sq[b_proj_sq > 1e-30]), 1)[0]:.4f})", flush=True)

        # Compare: where is Q's weight concentrated?
        Q_top30 = np.sum(Q_proj_sq[-30:]) / np.sum(Q_proj_sq) * 100
        b_top30 = np.sum(b_proj_sq[-30:]) / np.sum(b_proj_sq) * 100
        print(f"         Q: {Q_top30:.2f}% in top 30 eigenvecs (b: {b_top30:.2f}%)", flush=True)


# ============================================================
# STEP 2: Is Q_k "smooth" like b?
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 2: Is Q_k smooth like b?", flush=True)
print("="*70, flush=True)

# Compute Q at k=N/2 and compare to b
k_mid = N // 2
Q_mid = np.zeros(N)
for j in range(k_mid):
    Q_mid += np.sum(G[:j+1, :], axis=0)

# Compare Q_mid and b entry by entry
print(f"\n  Comparing Q_{{N/2}} and b (both normalized):", flush=True)
Q_norm = Q_mid / np.linalg.norm(Q_mid)
b_norm = b / np.linalg.norm(b)
corr_Qb = np.corrcoef(Q_norm, b_norm)[0, 1]
print(f"  Correlation: {corr_Qb:.6f}", flush=True)

# Is Q smooth? Check total variation
tv_Q = np.sum(np.abs(np.diff(Q_mid)))
tv_b = np.sum(np.abs(np.diff(b)))
print(f"  TV(Q) = {tv_Q:.4f}, TV(b) = {tv_b:.4f}", flush=True)
print(f"  TV(Q)/||Q|| = {tv_Q/np.linalg.norm(Q_mid):.4f}, "
      f"TV(b)/||b|| = {tv_b/np.linalg.norm(b):.4f}", flush=True)

# Does Q_k look like log(k)/k?
ks = np.arange(2, N+2)
Q_theory = np.log(ks) / ks  # same asymptotic as b
Q_scaled = Q_mid / Q_mid[0] * Q_theory[0]
corr_theory = np.corrcoef(Q_mid[:50], Q_theory[:50])[0, 1]
print(f"  Correlation Q vs log(k)/k (first 50): {corr_theory:.4f}", flush=True)

# Power law decay of Q_k entries
mask_Q = Q_mid > 1e-15
if np.sum(mask_Q) > 20:
    coeffs_Q = np.polyfit(np.log(ks[mask_Q][:200]),
                           np.log(Q_mid[mask_Q][:200]), 1)
    print(f"  Q_k ~ k^{{{coeffs_Q[0]:.3f}}} (b ~ k^{{-0.81}})", flush=True)


# ============================================================
# STEP 3: THE BOOTSTRAP ARGUMENT
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 3: THE BOOTSTRAP ARGUMENT", flush=True)
print("="*70, flush=True)

# The key identity: T_i(k) = <v_i, Q_k> / lambda_i
# max|T_i| = max_k |<v_i, Q_k>| / lambda_i
#
# If Q_k has spectral concentration gamma_Q:
#   |<v_i, Q_k>| ~ lambda_i^{gamma_Q/2} * ||Q_k||
#   (from the spectral projection bound)
#
# Then: max|T_i| ~ lambda_i^{gamma_Q/2} * ||Q|| / lambda_i
#                = ||Q|| * lambda_i^{gamma_Q/2 - 1}
#
# For beta_2 = gamma_Q/2 - 1 > 0, we need gamma_Q > 2.
# For beta_2 > 0.5, we need gamma_Q > 3.
# We measured gamma_Q ~ 1.6-2.4 depending on k.

# Wait -- let me be more careful. The spectral projection of Q_k is:
# |<Q_k, v_i>|^2 ~ lambda_i^{gamma_Q}
# So |<Q_k, v_i>| ~ lambda_i^{gamma_Q/2}
# And max|T_i| = max_k |<v_i, Q_k>| / lambda_i ~ lambda_i^{gamma_Q/2 - 1}
# beta_2 = gamma_Q/2 - 1

# From our measurement: gamma_Q ~ 1.75 (at k=N/2)
# beta_2 = 1.75/2 - 1 = -0.125

# That's NEGATIVE! This contradicts the measured beta_2 = 0.75.
# The issue: max over k means we take the WORST case k,
# and gamma_Q varies with k.

# Let me recompute: what is max_k |<v_i, Q_k>| as a function of lambda_i?
print(f"\n  Computing max_k |<v_i, Q_k>| for each eigenvector...", flush=True)

# Build Q_k for ALL k
max_inner_Q = np.zeros(N)
cumulative_row_sum = np.zeros(N)  # R_k
cumulative_Q = np.zeros(N)  # Q_k (running sum of R_k's)

best_Q_proj = np.zeros(N)

for k in range(N):
    cumulative_row_sum += G[k, :]  # R_k = R_{k-1} + G[k,:]
    cumulative_Q += cumulative_row_sum  # Q_k = Q_{k-1} + R_k

    # <v_i, Q_k>
    inner = V.T @ cumulative_Q  # N-vector of inner products
    # Update maximum
    for i in range(N):
        if abs(inner[i]) > abs(best_Q_proj[i]):
            best_Q_proj[i] = inner[i]
    max_inner_Q = np.maximum(max_inner_Q, np.abs(inner))

# Now max|T_i| = max_inner_Q[i] / lambda[i]
# And we measured beta_2 from max|T_i| ~ lambda^{beta_2}
# So max_inner_Q[i] ~ lambda^{beta_2} * lambda = lambda^{beta_2 + 1} = lambda^{1.75}

# Check: does max_inner_Q scale as lambda^{1.75}?
mask_mQ = (eigenvalues > 1e-10) & (max_inner_Q > 1e-30)
if np.sum(mask_mQ) > 10:
    coeffs_mQ = np.polyfit(np.log(eigenvalues[mask_mQ]),
                            np.log(max_inner_Q[mask_mQ]), 1)
    gamma_max_Q = coeffs_mQ[0]
    print(f"\n  max_k |<v_i, Q_k>| ~ lambda^{{{gamma_max_Q:.4f}}}", flush=True)
    print(f"  Expected: lambda^{{{0.75 + 1:.3f}}} = lambda^1.75", flush=True)
    print(f"  Actual:   lambda^{{{gamma_max_Q:.3f}}}", flush=True)

    # And from the Abel relation:
    # max|T_i| = max_k |<v_i, Q_k>| / lambda_i
    # ~ lambda^{gamma_max_Q - 1}
    beta_from_Q = gamma_max_Q - 1
    print(f"\n  beta_2 (from Q analysis): {beta_from_Q:.4f}", flush=True)
    print(f"  beta_2 (measured):         0.7429", flush=True)
    print(f"  Agreement:                 {'GOOD' if abs(beta_from_Q - 0.743) < 0.1 else 'POOR'}", flush=True)


# ============================================================
# STEP 4: Self-consistency check
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 4: SELF-CONSISTENCY — Does the argument close?", flush=True)
print("="*70, flush=True)

print(f"""
  THE PROOF CHAIN (with measured exponents):

  GIVEN: b_k ~ log(k)/k (smooth, bounded variation)
  GIVEN: G is the Nyman-Beurling Gram matrix

  STEP A: Spectral projection of b
    |<b, v_i>|^2 ~ lambda_i^{{gamma_b}}     gamma_b = 1.94

  STEP B: Abel summation (order 1)
    |<b, v_i>| <= TV(b) * max_k |S_i(k)|
    max|S_i| = max_k |<v_i, R_k>| / lambda_i

  STEP C: Spectral projection of R_k (cumulative row sum of G)
    max_k |<v_i, R_k>| ~ lambda_i^{{gamma_R}}
    => max|S_i| ~ lambda_i^{{gamma_R - 1}} = lambda_i^{{beta_1}}
    beta_1 = gamma_R - 1 = 0.243

  STEP D: Abel summation (order 2)
    |<b, v_i>| <= TV^2(b) * max_k |T_i(k)|
    max|T_i| = max_k |<v_i, Q_k>| / lambda_i

  STEP E: Spectral projection of Q_k (double cumulative row sum)
    max_k |<v_i, Q_k>| ~ lambda_i^{{gamma_Q}}     gamma_Q = {gamma_max_Q:.3f}
    => max|T_i| ~ lambda_i^{{gamma_Q - 1}} = lambda_i^{{beta_2}}
    beta_2 = gamma_Q - 1 = {beta_from_Q:.3f}

  STEP F: Double Abel bound
    |<b, v_i>|^2 <= C * lambda_i^{{2*beta_2}} = C * lambda_i^{{{2*beta_from_Q:.3f}}}
    gamma >= 2*beta_2 = {2*beta_from_Q:.3f}

  SELF-CONSISTENCY:
    gamma_b (measured directly) = 1.94
    gamma (from Abel bound)    = {2*beta_from_Q:.3f}
    The Abel bound captures {2*beta_from_Q/1.94*100:.0f}% of the measured gamma.
""", flush=True)

# The KEY question: can we PROVE gamma_Q > 2 (equivalently beta_2 > 0.5)?
# gamma_Q = spectral concentration of Q_k

# Q_k involves double cumulative sums of G rows.
# G rows are "arithmetic" (involve modular arithmetic).
# Q_k = double-integrated arithmetic function.
# Integration makes things SMOOTHER.
# So Q_k should be SMOOTHER than G rows, which should be smoother than raw basis functions.

# This suggests: gamma_Q > gamma_R > gamma_0 = 0 (raw basis functions have no spectral concentration)
# And we need gamma_Q > 2.

# From the data:
# gamma_b = 1.94 (projection of b, which is smooth)
# gamma_Q ~ 1.74 (projection of Q, which is also smooth)
# gamma_R ~ 1.24 (projection of R, less smooth)

# The PATTERN: each integration step adds ~0.5 to gamma.
# gamma_0 (raw) ~ 0
# gamma_R (single integral) ~ 1.24
# gamma_Q (double integral) ~ 1.74

# If this pattern continues, triple integral gives gamma ~ 2.24.
# This matches the triple Abel beta_3 ~ 1.0 (gamma_U - 1 = 1.0, so gamma_U ~ 2.0).

# CAN WE PROVE THIS PATTERN?
# Each integration of a vector in R^N adds ~0.5 to its spectral concentration
# with respect to the Gram matrix eigenbasis.
# This is because integration is a SMOOTHING operator, and the Gram matrix
# separates smooth from rough.

# The smoothing theorem would be:
# If f has spectral concentration gamma_f (meaning |<f, v_i>| ~ lambda^{gamma_f/2}),
# and F = cumsum(f) is its antiderivative,
# then F has spectral concentration gamma_F >= gamma_f + delta for some delta > 0.

# If delta = 0.5 (as suggested by data), then:
# Starting from gamma_0 = 0 (no concentration for raw eigenvectors)
# After k integrations: gamma_k ~ 0.5*k
# gamma_Q (k=2) ~ 1.0, giving beta_2 = gamma_Q - 1 = 0... hmm, that's 0.

# Wait, the issue is that the RAW row sums R_k are NOT the antiderivative of
# the eigenvectors. They're the antiderivative of the WEIGHTED eigenvectors
# (weighted by the G structure).

# Let me reconsider. Actually R_k(m) = sum_{j<=k} G_{jm}. This is the partial
# sum of column m of G over its rows. This IS related to the eigenvector
# structure through the eigendecomposition.

# I think the cleanest path is:
# 1. R_k is a linear functional of G (partial row sum)
# 2. G = V Lambda V^T
# 3. R_k = sum_{j<=k} (V Lambda V^T)_{j,:} = (sum_{j<=k} V_{j,:}) Lambda V^T
# 4. So <R_k, v_i> = lambda_i * S_i(k)

# THIS IS EXACTLY THE EIGENVALUE EQUATION!
# <R_k, v_i> = lambda_i * S_i(k)

# And <Q_k, v_i> = sum_{j<=k} <R_j, v_i> = lambda_i * T_i(k)

# So: max_k |<v_i, Q_k>| = lambda_i * max|T_i|

# This is a TAUTOLOGY! It doesn't help — it just restates the definition.

# The Abel bound works the OTHER direction:
# From |<b,v>| = |sum b_k v_k|, use summation by parts.
# This DOES give information because b is KNOWN to be smooth.

# The challenge is that we need an INDEPENDENT bound on max|T_i| that
# doesn't go through the spectral projection of Q_k.

print(f"""
  RESOLUTION: The eigenvalue equation gives <Q_k, v_i> = lambda_i * T_i(k),
  which is a tautology. The spectral projection of Q_k doesn't give
  INDEPENDENT information about max|T_i|.

  The Abel bound works because b is externally known to be smooth
  (b ~ log(k)/k), NOT because of any property of Q_k.

  THE INDEPENDENT BOUND must come from the STRUCTURE of the eigenvectors
  themselves — specifically, the cancellation in their partial sums.

  From the data:
  - Single partial sums: max|S_i| ~ lambda^0.24 (weak cancellation)
  - Double partial sums: max|T_i| ~ lambda^0.75 (strong cancellation)
  - The jump from 0.24 to 0.75 is a factor of 3x in exponent

  The ARITHMETIC CANCELLATION (the 0.33 bonus beyond random walk)
  is the irreducible number-theoretic content.

  WHAT WOULD PROVE IT:
  If we could show that eigenvectors of G with eigenvalue lambda have
  at least C * lambda^{{-0.37}} sign changes (measured: lambda^{{-0.37}}),
  then standard results on integrated oscillatory functions give
  max|T_i| ~ lambda^{{0.37 * 2}} = lambda^{{0.74}} ~ lambda^{{beta_2}}.

  The sign change bound follows from: small eigenvalue => v is "rough"
  in the discrete Sobolev sense => many zero crossings.
  This is a DISCRETE NODAL DOMAIN theorem for the Gram matrix.
""", flush=True)


# ============================================================
# STEP 5: DISCRETE NODAL DOMAIN THEOREM
# ============================================================
print("="*70, flush=True)
print("STEP 5: NODAL DOMAIN THEOREM — Sign changes vs eigenvalue", flush=True)
print("="*70, flush=True)

# The classical Courant nodal domain theorem: the k-th eigenfunction
# of a Sturm-Liouville operator has at most k nodal domains.
# For MATRICES: Fiedler's theorem gives similar bounds.

# For our Gram matrix: eigenvector i (sorted by eigenvalue) should have
# approximately i sign changes (by Courant's theorem for matrices).

# But our eigenvalues are sorted ASCENDING, so eigenvector 1 (smallest lambda)
# should have the MOST sign changes (it's the highest "frequency" mode).

# Let's verify: sign changes vs eigenvalue INDEX
sign_changes = np.zeros(N)
for i in range(N):
    v = V[:, i]
    signs = np.sign(v)
    signs_nz = signs[signs != 0]
    sign_changes[i] = np.sum(np.abs(np.diff(signs_nz)) > 0)

# Also: sign changes of partial sums S
S = np.cumsum(V, axis=0)
S_sign_changes = np.zeros(N)
for i in range(N):
    signs_S = np.sign(S[:, i])
    signs_S_nz = signs_S[signs_S != 0]
    S_sign_changes[i] = np.sum(np.abs(np.diff(signs_S_nz)) > 0)

print(f"\n  Eigenvector sign changes (direct and partial sums):", flush=True)
print(f"  {'i':>5} {'lambda':>12} {'v sign chg':>11} {'S sign chg':>11} {'ratio S/v':>10}", flush=True)
print(f"  {'-'*52}", flush=True)

for i in range(N):
    if i < 5 or i >= N-3 or i % (N//10) == 0:
        ratio = S_sign_changes[i] / (sign_changes[i] + 1e-10)
        print(f"  {i+1:>5} {eigenvalues[i]:>12.4e} {sign_changes[i]:>11.0f} "
              f"{S_sign_changes[i]:>11.0f} {ratio:>10.4f}", flush=True)

# The ratio S_sign_changes / v_sign_changes tells us how much smoothing
# the partial sum operation does.
# For highly oscillatory v: S has fewer crossings (smoothing).
# The ratio should decrease for more oscillatory (small lambda) eigenvectors.

# Fit: S_sign_changes vs lambda
mask_ss = (eigenvalues > 1e-10) & (S_sign_changes > 0)
coeffs_ss = np.polyfit(np.log(eigenvalues[mask_ss]),
                        np.log(S_sign_changes[mask_ss]), 1)
print(f"\n  S zero crossings ~ lambda^{{{coeffs_ss[0]:.4f}}}", flush=True)

# If S has M crossings, T (integral of S) has max ~ max|S| * N/M
# max|T| ~ max|S| * N / S_crossings
# If S_crossings ~ lambda^alpha and max|S| ~ lambda^{beta_1}:
# max|T| ~ lambda^{beta_1} * N / lambda^alpha = N * lambda^{beta_1 - alpha}

# For max|T| ~ lambda^{beta_2} independent of N:
# beta_2 = beta_1 - alpha (and the N cancels if everything is properly normalized)

beta_1_meas = 0.243
alpha_S_cross = coeffs_ss[0]
beta_2_pred = beta_1_meas - alpha_S_cross
print(f"\n  Predicted beta_2 = beta_1 - alpha_S = {beta_1_meas:.3f} - ({alpha_S_cross:.3f}) = {beta_2_pred:.3f}", flush=True)
print(f"  Measured beta_2 = 0.743", flush=True)
print(f"  Agreement: {'GOOD' if abs(beta_2_pred - 0.743) < 0.15 else 'REASONABLE' if abs(beta_2_pred - 0.743) < 0.3 else 'POOR'}", flush=True)

print(f"""
  THE NODAL THEOREM CONNECTION:

  IF we can prove: sign_changes(S_i) ~ lambda^{{alpha_S}} with alpha_S < 0
  (partial sums of small-eigenvalue eigenvectors have FEWER crossings)

  AND: max|T_i| ~ max|S_i| / sign_changes(S_i)^c for some c > 0

  THEN: max|T_i| ~ lambda^{{beta_1}} * lambda^{{-alpha_S * c}}
                  = lambda^{{beta_1 - alpha_S * c}}

  For beta_2 = beta_1 - alpha_S * c > 0.5, we need the nodal bound.

  The discrete nodal domain theorem (Fiedler) applied to G gives:
  eigenvector i has at most i+1 nodal domains.
  For the SMALLEST eigenvalue eigenvector: up to N nodal domains (N/2 sign changes).
  This is consistent with our observation (~360 sign changes for N=500).

  The KEY: the sign changes of the PARTIAL SUM S also follow a nodal pattern.
  This requires extending the nodal theorem to ANTIDERIVATIVES of eigenvectors.
""", flush=True)

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
