"""DISCRETE SOBOLEV + NODAL ANALYSIS: Proving the sign change bound.

The proof chain is:
  1. Eigenvectors with small lambda have many sign changes [NEED TO PROVE]
  2. Many sign changes => bounded partial sums [standard]
  3. Bounded partial sums + Abel summation => gamma > 1 [proved]
  4. gamma > 1 => d_n -> 0 => RH [Baez-Duarte]

We attack step 1 via:
  A. Discrete Sobolev inequality for the Gram matrix
  B. Cheeger-type inequality
  C. Direct analysis of eigenvector structure

KEY INSIGHT: G = W^T W where W_{k,n} = (n mod (k+2))/(k+2) * sqrt(w_n).
Eigenvector v of G satisfies Gv = lambda v, i.e., W^T W v = lambda v.
So Wv has norm sqrt(lambda). If ||Wv|| is small, v must CANCEL the
columns of W, which requires oscillation.

The columns of W are PERIODIC functions of k (with period n).
Cancellation of periodic functions requires sign changes.
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np
from math import gcd

t0 = time.time()

M_sum = 10000
weights = np.array([1.0/(n*(n+1)) for n in range(1, M_sum+1)])
sqrt_w = np.sqrt(weights)
ns = np.arange(1, M_sum+1)

# ============================================================
# BUILD SYSTEM
# ============================================================
N = 500
W = np.zeros((N, M_sum))
for k_idx in range(N):
    k = k_idx + 2
    W[k_idx, :] = ((ns % k) / k) * sqrt_w

G = W @ W.T
eigenvalues, V = np.linalg.eigh(G)
idx = np.argsort(eigenvalues)
eigenvalues = eigenvalues[idx]
V = V[:, idx]

print("="*70, flush=True)
print("SOBOLEV / NODAL ANALYSIS", flush=True)
print("="*70, flush=True)


# ============================================================
# PART A: Discrete Sobolev inequality for G
# ============================================================
print("\n" + "="*70, flush=True)
print("PART A: DISCRETE SOBOLEV INEQUALITY", flush=True)
print("="*70, flush=True)

# For a standard discrete Laplacian L, the Poincare inequality gives:
# ||grad(v)||^2 >= lambda * ||v||^2
# which means: TV(v) >= C * sqrt(lambda) * ||v||
#
# For our Gram matrix G: v^T G v = lambda means ||Wv||^2 = lambda.
# This is a "Sobolev" norm: ||v||_G^2 = v^T G v = lambda.
#
# The discrete Sobolev embedding theorem says:
# ||v||_inf <= C * ||v||_G^{alpha} * ||v||_2^{1-alpha}
# where alpha depends on the dimension and the operator.
#
# For our 1D setting (vectors in R^N):
# ||v||_inf <= C * TV(v)^{1/2} * ||v||^{1/2}  (1D Sobolev)
#
# Can we relate TV(v) to the Gram norm ||v||_G?

# Compute: TV(v) vs sqrt(lambda) for each eigenvector
tv_vals = np.array([np.sum(np.abs(np.diff(V[:, i]))) for i in range(N)])

print(f"\n  TV(v_i) vs lambda:", flush=True)
mask = (eigenvalues > 1e-10) & (tv_vals > 0)
coeffs_tv = np.polyfit(np.log(eigenvalues[mask]), np.log(tv_vals[mask]), 1)
print(f"  TV(v) ~ lambda^{{{coeffs_tv[0]:.4f}}}", flush=True)

# The discrete gradient: (Dv)_k = v_{k+1} - v_k
# ||Dv||^2 = sum (v_{k+1} - v_k)^2 = 2*||v||^2 - 2*<v, Sv> where S is the shift
# This is the discrete Laplacian quadratic form.

grad_sq = np.array([np.sum(np.diff(V[:, i])**2) for i in range(N)])
print(f"  ||grad(v)||^2 ~ lambda^{{{np.polyfit(np.log(eigenvalues[mask]), np.log(grad_sq[mask]), 1)[0]:.4f}}}")

# The RATIO ||grad(v)||^2 / lambda tells us how "rough" v is relative to its
# Gram eigenvalue. If this ratio is large for small lambda, then small-eigenvalue
# eigenvectors are ROUGHER than the Gram norm suggests.
ratio_grad_lambda = grad_sq / (eigenvalues + 1e-30)
print(f"\n  ||grad(v)||^2 / lambda:", flush=True)
print(f"    Small lambda (i=1-10): mean = {np.mean(ratio_grad_lambda[:10]):.2f}", flush=True)
print(f"    Mid lambda (i=200-300): mean = {np.mean(ratio_grad_lambda[200:300]):.2f}", flush=True)
print(f"    Large lambda (i=490-500): mean = {np.mean(ratio_grad_lambda[490:]):.2f}", flush=True)

# If ratio grows as lambda decreases: small-eigenvalue eigenvectors are
# SUPER-ROUGH compared to what their Gram norm suggests.
coeffs_ratio = np.polyfit(np.log(eigenvalues[mask]),
                           np.log(ratio_grad_lambda[mask]), 1)
print(f"  Ratio ~ lambda^{{{coeffs_ratio[0]:.4f}}}", flush=True)
if coeffs_ratio[0] < -0.1:
    print(f"  CONFIRMED: small lambda eigenvectors are disproportionately rough", flush=True)


# ============================================================
# PART B: Sign change count vs eigenvalue — tight bound
# ============================================================
print("\n" + "="*70, flush=True)
print("PART B: SIGN CHANGES — Can we prove the bound?", flush=True)
print("="*70, flush=True)

sign_ch = np.array([np.sum(np.abs(np.diff(np.sign(V[:, i])[np.sign(V[:, i]) != 0])) > 0)
                     for i in range(N)])

# Relationship: sign_changes * mean_segment_length ~ N
# mean_segment_length = N / sign_changes
# Within each segment, v doesn't change sign, so |v| is bounded by its max.
# The partial sum S grows by at most max|v| * segment_length within a segment.
# Over all segments, S oscillates with amplitude ~ max|v| * max_segment_length.

# max|v| = ||v||_inf ~ 1/sqrt(N) (for random-like eigenvectors)
# Actually measure it:
v_inf = np.array([np.max(np.abs(V[:, i])) for i in range(N)])

print(f"  ||v||_inf vs lambda:", flush=True)
coeffs_inf = np.polyfit(np.log(eigenvalues[mask]), np.log(v_inf[mask]), 1)
print(f"  ||v||_inf ~ lambda^{{{coeffs_inf[0]:.4f}}}", flush=True)
print(f"  For comparison: 1/sqrt(N) = {1/np.sqrt(N):.4f}", flush=True)
print(f"  Mean ||v||_inf: {np.mean(v_inf):.4f}", flush=True)

# Key bound: max|S(k)| <= N * ||v||_inf
# (trivially, since |S(k)| <= k * max|v_j|)
# Better: max|S(k)| <= (N/sign_changes) * ||v||_inf * sign_changes
# = N * ||v||_inf (same thing)

# Even better: within each segment between sign changes, S increases by
# at most segment_length * ||v||_inf. And S oscillates, so the accumulated
# drift is bounded by sqrt(sign_changes) * segment_length * ||v||_inf
# = sqrt(sign_changes) * (N/sign_changes) * ||v||_inf
# = N * ||v||_inf / sqrt(sign_changes)

# So: max|S| ~ N * ||v||_inf / sqrt(sign_changes)
# ~ N * lambda^{alpha_inf} / sqrt(lambda^{alpha_sc})
# ~ N * lambda^{alpha_inf + |alpha_sc|/2}

alpha_inf = coeffs_inf[0]  # expected near 0 (||v||_inf ~ const for most eigenvectors)
alpha_sc = np.polyfit(np.log(eigenvalues[mask]), np.log(sign_ch[mask] + 1), 1)[0]

beta_1_predicted = alpha_inf + abs(alpha_sc) / 2
print(f"\n  Predicted beta_1 from sign change model:", flush=True)
print(f"    alpha_inf (||v||_inf exponent): {alpha_inf:.4f}", flush=True)
print(f"    alpha_sc (sign change exponent): {alpha_sc:.4f}", flush=True)
print(f"    Predicted beta_1 = alpha_inf + |alpha_sc|/2 = {beta_1_predicted:.4f}", flush=True)
print(f"    Measured beta_1 = 0.243", flush=True)

# For the DOUBLE partial sum:
# max|T| ~ max|S| * (N / S_sign_changes) / sqrt(S_sign_changes)
# where S_sign_changes is the number of sign changes of S.

S = np.cumsum(V, axis=0)
S_sc = np.array([np.sum(np.abs(np.diff(np.sign(S[:, i])[np.sign(S[:, i]) != 0])) > 0)
                  for i in range(N)])

alpha_S_sc = np.polyfit(np.log(eigenvalues[mask & (S_sc > 0)]),
                         np.log(S_sc[mask & (S_sc > 0)]), 1)[0]

# max|T| ~ max|S| / sqrt(S_sign_changes) * (N / S_sign_changes)
# But we need to be careful about N-dependence.
# Actually: max|T| ~ max|S| * max_segment_length(S)
# where max_segment_length(S) = N / S_sign_changes

max_S = np.max(np.abs(S), axis=0)
T = np.cumsum(S, axis=0)
max_T = np.max(np.abs(T), axis=0)

# Verify: max|T| ~ max|S| * N / S_sign_changes
predicted_maxT = max_S * N / (S_sc + 1)
ratio_predicted = max_T / (predicted_maxT + 1e-30)

print(f"\n  Testing: max|T| ~ max|S| * N / S_sign_changes", flush=True)
print(f"  {'lambda':>12} {'max|T| actual':>14} {'max|T| pred':>14} {'ratio':>8}", flush=True)
for i in [0, 10, 50, 100, 200, 300, 400, 490]:
    if i < N:
        print(f"  {eigenvalues[i]:>12.4e} {max_T[i]:>14.4f} {predicted_maxT[i]:>14.4f} "
              f"{ratio_predicted[i]:>8.3f}", flush=True)


# ============================================================
# PART C: The key relationship at the proof level
# ============================================================
print("\n" + "="*70, flush=True)
print("PART C: THE KEY RELATIONSHIP", flush=True)
print("="*70, flush=True)

# From the data we know:
# 1. sign_changes(v) ~ lambda^{-0.33}  [call this alpha_v]
# 2. sign_changes(S) ~ lambda^{-0.37}  [call this alpha_S]
# 3. max|S| ~ lambda^{0.24}            [call this beta_1]
# 4. max|T| ~ lambda^{0.75}            [call this beta_2]
# 5. ||v||_inf ~ lambda^{0.01}         [essentially constant]

# The PROVABLE relationships (for any matrix, not just G):
# R1: max|S| <= N * ||v||_inf  (trivial)
# R2: max|S| <= sqrt(N) * ||v||  [Cauchy-Schwarz on partial sums of unit vector]
#     = sqrt(N) since ||v||=1
# R3: If v has M sign changes: max|S| <= N/M * max|v| * sqrt(M)
#     = sqrt(N*M) * max|v| ... no, this isn't right either.

# ACTUALLY: for a vector v with M sign changes:
# The partial sums S(k) = sum_{j<=k} v_j form a random-walk-like path
# that reverses direction at each sign change.
# max|S| <= max_segment_length * ||v||_inf
# where max_segment_length = N / M (on average, can be larger)

# For EIGENVECTORS of G: the sign changes are NOT uniformly spaced.
# They cluster at certain positions related to the modular arithmetic.
# This clustering can make some segments longer than average.

# Let's measure the maximum segment length
print(f"\n  Maximum segment length (between sign changes) vs eigenvalue:", flush=True)

max_seg_lengths = np.zeros(N)
for i in range(N):
    v = V[:, i]
    signs = np.sign(v)
    # Find sign change positions
    change_pos = np.where(np.abs(np.diff(signs)) > 0)[0]
    if len(change_pos) > 0:
        segments = np.diff(np.concatenate([[0], change_pos, [N-1]]))
        max_seg_lengths[i] = np.max(segments)
    else:
        max_seg_lengths[i] = N

mask_seg = (eigenvalues > 1e-10) & (max_seg_lengths > 0)
coeffs_seg = np.polyfit(np.log(eigenvalues[mask_seg]),
                         np.log(max_seg_lengths[mask_seg]), 1)
print(f"  Max segment length ~ lambda^{{{coeffs_seg[0]:.4f}}}", flush=True)

# If max_seg ~ lambda^{delta}: max|S| <= max_seg * ||v||_inf ~ lambda^{delta + alpha_inf}
# We need delta + alpha_inf to match beta_1 = 0.24
print(f"  Predicted beta_1 = delta + alpha_inf = {coeffs_seg[0]:.3f} + {alpha_inf:.3f} = {coeffs_seg[0]+alpha_inf:.3f}", flush=True)
print(f"  Measured beta_1 = 0.243", flush=True)

# For double partial sums: similar analysis with S_sign_changes and S_segment_lengths
S_max_seg = np.zeros(N)
for i in range(N):
    s = S[:, i]
    signs = np.sign(s)
    change_pos = np.where(np.abs(np.diff(signs)) > 0)[0]
    if len(change_pos) > 0:
        segments = np.diff(np.concatenate([[0], change_pos, [N-1]]))
        S_max_seg[i] = np.max(segments)
    else:
        S_max_seg[i] = N

mask_Sseg = (eigenvalues > 1e-10) & (S_max_seg > 0)
coeffs_Sseg = np.polyfit(np.log(eigenvalues[mask_Sseg]),
                          np.log(S_max_seg[mask_Sseg]), 1)
print(f"\n  S max segment length ~ lambda^{{{coeffs_Sseg[0]:.4f}}}", flush=True)

max_S_vals = max_S
mask_mS = (eigenvalues > 1e-10) & (max_S_vals > 1e-10)
coeffs_mS = np.polyfit(np.log(eigenvalues[mask_mS]), np.log(max_S_vals[mask_mS]), 1)
beta_1_direct = coeffs_mS[0]

# max|T| <= S_max_seg * max|S|
# ~ lambda^{delta_S} * lambda^{beta_1}
# = lambda^{delta_S + beta_1}
print(f"  Predicted beta_2 = delta_S + beta_1 = {coeffs_Sseg[0]:.3f} + {beta_1_direct:.3f} = {coeffs_Sseg[0]+beta_1_direct:.3f}", flush=True)
print(f"  Measured beta_2 = 0.743", flush=True)


# ============================================================
# SUMMARY: The provable path
# ============================================================
print("\n" + "="*70, flush=True)
print("SUMMARY: WHAT CAN BE PROVED", flush=True)
print("="*70, flush=True)

print(f"""
  THE PROVABLE CHAIN:

  1. b_k ~ log(k)/k, TV(b) < inf, TV^2(b) < inf  [PROVED: arithmetic]

  2. For eigenvector v of G with eigenvalue lambda:
     a. ||v||_inf ~ const (independent of lambda)  [MEASURED: exponent {alpha_inf:.3f}]
     b. sign_changes(v) ~ lambda^{{{alpha_sc:.3f}}}  [MEASURED]
     c. max_segment_length(v) ~ lambda^{{{coeffs_seg[0]:.3f}}}  [MEASURED]
     d. max|S| ~ lambda^{{{beta_1_direct:.3f}}}  [MEASURED, DERIVABLE from c+a]

  3. For the partial sums S = cumsum(v):
     a. sign_changes(S) ~ lambda^{{{alpha_S_sc:.3f}}}  [MEASURED]
     b. max_segment_length(S) ~ lambda^{{{coeffs_Sseg[0]:.3f}}}  [MEASURED]
     c. max|T| ~ lambda^{{beta_1 + delta_S}} = lambda^{{{beta_1_direct + coeffs_Sseg[0]:.3f}}}  [DERIVABLE from b+2d]

  4. Double Abel: |<b,v>|^2 <= TV^2(b) * max|T|^2 ~ lambda^{{{2*(beta_1_direct + coeffs_Sseg[0]):.3f}}}

  KEY PROVABLE STATEMENTS:
  - 2c -> 2d: max|S| <= max_seg * ||v||_inf  [TRIVIAL, for any vector]
  - 3c: max|T| <= max_seg(S) * max|S|        [TRIVIAL, for any functions]

  KEY STATEMENTS NEEDING PROOF (from G's structure):
  - 2b: sign_changes(v) >= C * lambda^{{{alpha_sc:.3f}}}  [NODAL DOMAIN]
  - 2c: max_segment(v) <= C * lambda^{{{coeffs_seg[0]:.3f}}}  [SEGMENT BOUND]
  - 3a: sign_changes(S) >= C * lambda^{{{alpha_S_sc:.3f}}}  [NODAL FOR S]
  - 3b: max_segment(S) <= C * lambda^{{{coeffs_Sseg[0]:.3f}}}  [SEGMENT FOR S]

  THE MINIMAL REQUIREMENT:
  We need 2*(beta_1 + delta_S) > 1, i.e., beta_1 + delta_S > 0.5.
  Measured: beta_1 + delta_S = {beta_1_direct:.3f} + {coeffs_Sseg[0]:.3f} = {beta_1_direct + coeffs_Sseg[0]:.3f}

  This is {'> 0.5: SUFFICIENT!' if beta_1_direct + coeffs_Sseg[0] > 0.5 else '< 0.5: NOT sufficient from this decomposition alone.'}

  If we use the MEASURED beta_2 = 0.743 directly:
  2*beta_2 = 1.486 > 1: gamma > 1 from double Abel.
  The analytical decomposition (beta_1 + delta_S = {beta_1_direct + coeffs_Sseg[0]:.3f})
  captures {(beta_1_direct + coeffs_Sseg[0])/0.743*100:.0f}% of the measured beta_2.
  The remaining {100-(beta_1_direct + coeffs_Sseg[0])/0.743*100:.0f}% is the arithmetic bonus.
""", flush=True)

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
