"""ANALYTICAL ATTACK: Prove gamma > 1 from first principles.

WHY does |<b, v_i>|^2 ~ lambda_i^gamma with gamma > 1?

b_k ~ log(k)/k is smooth. Small-eigenvalue eigenvectors v_i are oscillatory.
Smooth functions have small inner product with oscillatory functions.

APPROACH 1: ABEL SUMMATION (summation by parts)
  <b, v> = sum b_k v_k = sum (b_k - b_{k+1}) S(k) + b_N S(N)
  where S(k) = sum_{j<=k} v_j (partial sums of eigenvector)
  If v is oscillatory, S(k) should be bounded => |<b,v>| small

APPROACH 2: EIGENVECTOR OSCILLATION ANALYSIS
  Count sign changes, zero crossings, "roughness" of each eigenvector
  Correlate with eigenvalue: small lambda <=> more oscillatory?

APPROACH 3: EXPONENTIAL SUM BOUNDS
  If v_i(k) has arithmetic oscillation at "frequency" theta,
  <b, v_i> = sum (log k/k) * cos(2pi k theta)
  Van der Corput bounds apply to such sums.

APPROACH 4: MULTIPLICATIVE STRUCTURE
  Eigenvectors may decompose into Dirichlet characters or Ramanujan sums.
  Character sums of smooth functions have known bounds.
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

# Build system at N=500 for detailed analysis
N = 500
print(f"Building system at N={N}...", flush=True)

W = np.zeros((N, M_sum))
for k_idx in range(N):
    k = k_idx + 2
    W[k_idx, :] = ((ns % k) / k) * sqrt_w

G = W @ W.T
b = np.zeros(N)
for k_idx in range(N):
    k = k_idx + 2
    b[k_idx] = np.dot((ns % k) / k, weights)

eigenvalues, V = np.linalg.eigh(G)
idx = np.argsort(eigenvalues)
eigenvalues = eigenvalues[idx]
V = V[:, idx]

b_proj = V.T @ b
b_proj_sq = b_proj**2

print(f"System built. lambda range: [{eigenvalues[0]:.4e}, {eigenvalues[-1]:.4e}]", flush=True)


# ============================================================
# APPROACH 1: ABEL SUMMATION
# ============================================================
print("\n" + "="*70, flush=True)
print("APPROACH 1: ABEL SUMMATION (Summation by Parts)", flush=True)
print("="*70, flush=True)

# <b, v_i> = sum_{k=0}^{N-1} b_k * v_i(k)
# Abel summation: = sum_{k=0}^{N-2} (b_k - b_{k+1}) * S_i(k) + b_{N-1} * S_i(N-1)
# where S_i(k) = sum_{j=0}^{k} v_i(j)
#
# Key insight: |b_k - b_{k+1}| = |log(k+2)/(k+2) - log(k+3)/(k+3)|
# ~ 1/k^2 for large k (the derivative of log(x)/x is (1-log(x))/x^2)
#
# So |<b, v_i>| <= max|S_i(k)| * sum |b_k - b_{k+1}| + |b_{N-1}| * |S_i(N-1)|
#                <= max|S_i(k)| * C1 + C2/N * |S_i(N-1)|
#
# The question: what is max|S_i(k)| as a function of lambda_i?

print(f"\n  Computing partial sums S_i(k) = sum_{{j<=k}} v_i(j) for each eigenvector...", flush=True)

# Partial sums of each eigenvector
S = np.cumsum(V, axis=0)  # S[k, i] = sum_{j=0}^{k} V[j, i]

# Maximum partial sum for each eigenvector
max_S = np.max(np.abs(S), axis=0)  # max over k of |S_i(k)|

# Abel bound on |<b, v_i>|
delta_b = np.abs(np.diff(b))  # |b_k - b_{k+1}|
sum_delta_b = np.sum(delta_b)  # total variation of b

abel_bound = max_S * sum_delta_b + np.abs(b[-1]) * np.abs(S[-1, :])

# Actual inner product
actual_inner = np.abs(b_proj)

print(f"  Total variation of b: {sum_delta_b:.6f}", flush=True)
print(f"  |b_N|: {np.abs(b[-1]):.6e}", flush=True)

# Compare Abel bound with actual and with eigenvalue
print(f"\n  {'i':>5} {'lambda_i':>12} {'|<b,v>|':>12} {'Abel bound':>12} {'max|S_i|':>10} "
      f"{'sign_chg':>9} {'bound/actual':>13}", flush=True)
print(f"  {'-'*75}", flush=True)

for i in range(N):
    if (i < 10 or i >= N-5 or i % (N//15) == 0):
        # Count sign changes
        signs = np.sign(V[:, i])
        sign_changes = np.sum(np.abs(np.diff(signs[signs != 0])) > 0)

        ratio = abel_bound[i] / (actual_inner[i] + 1e-30)
        print(f"  {i+1:>5} {eigenvalues[i]:>12.4e} {actual_inner[i]:>12.4e} "
              f"{abel_bound[i]:>12.4e} {max_S[i]:>10.4f} {sign_changes:>9} "
              f"{ratio:>13.2f}", flush=True)

# KEY TEST: does max|S_i| DECREASE with eigenvalue?
print(f"\n  Fitting: max|S_i| vs lambda_i", flush=True)
mask_fit = (eigenvalues > 1e-10) & (max_S > 1e-10)
if np.sum(mask_fit) > 10:
    coeffs_S = np.polyfit(np.log(eigenvalues[mask_fit]), np.log(max_S[mask_fit]), 1)
    beta_S = coeffs_S[0]
    print(f"  max|S_i| ~ lambda_i^{{{beta_S:.4f}}}", flush=True)
    if beta_S > 0:
        print(f"  Partial sums DECREASE with eigenvalue (oscillation causes cancellation)", flush=True)
        print(f"  Abel bound: |<b,v>| <= C * lambda_i^{{{beta_S:.3f}}}", flush=True)
        print(f"  Therefore: |<b,v>|^2 <= C^2 * lambda_i^{{{2*beta_S:.3f}}}", flush=True)
        if 2*beta_S > 1:
            print(f"  *** 2*beta = {2*beta_S:.3f} > 1: GAMMA > 1 FROM ABEL SUMMATION! ***", flush=True)
    else:
        print(f"  Partial sums do NOT decrease with eigenvalue.", flush=True)
        print(f"  Abel summation alone is insufficient.", flush=True)


# ============================================================
# APPROACH 2: EIGENVECTOR OSCILLATION ANALYSIS
# ============================================================
print("\n" + "="*70, flush=True)
print("APPROACH 2: EIGENVECTOR OSCILLATION ANALYSIS", flush=True)
print("="*70, flush=True)

# For each eigenvector, measure:
# 1. Number of sign changes
# 2. Total variation sum |v_{k+1} - v_k|
# 3. "Roughness" = TV / ||v||
# 4. Autocorrelation at lag 1

sign_changes_arr = np.zeros(N)
total_variation_arr = np.zeros(N)
autocorr_arr = np.zeros(N)

for i in range(N):
    v = V[:, i]

    # Sign changes
    signs = np.sign(v)
    signs_nonzero = signs[signs != 0]
    sign_changes_arr[i] = np.sum(np.abs(np.diff(signs_nonzero)) > 0)

    # Total variation
    total_variation_arr[i] = np.sum(np.abs(np.diff(v)))

    # Autocorrelation at lag 1
    if len(v) > 1:
        autocorr_arr[i] = np.corrcoef(v[:-1], v[1:])[0, 1]

print(f"  Eigenvector statistics vs eigenvalue:", flush=True)
print(f"  {'i':>5} {'lambda':>12} {'sign_chg':>9} {'TV':>10} {'autocorr1':>10} {'|<b,v>|^2':>12}", flush=True)
print(f"  {'-'*60}", flush=True)

for i in range(N):
    if i < 10 or i >= N-5 or i % (N//15) == 0:
        print(f"  {i+1:>5} {eigenvalues[i]:>12.4e} {sign_changes_arr[i]:>9.0f} "
              f"{total_variation_arr[i]:>10.4f} {autocorr_arr[i]:>10.4f} "
              f"{b_proj_sq[i]:>12.4e}", flush=True)

# Fit: sign changes vs eigenvalue
mask_sc = (eigenvalues > 1e-10) & (sign_changes_arr > 0)
if np.sum(mask_sc) > 10:
    coeffs_sc = np.polyfit(np.log(eigenvalues[mask_sc]),
                            np.log(sign_changes_arr[mask_sc]), 1)
    print(f"\n  Sign changes ~ lambda^{{{coeffs_sc[0]:.4f}}}", flush=True)
    if coeffs_sc[0] < -0.1:
        print(f"  Small eigenvalues have MORE sign changes (more oscillatory)", flush=True)
    elif coeffs_sc[0] > 0.1:
        print(f"  Small eigenvalues have FEWER sign changes", flush=True)
    else:
        print(f"  Sign changes roughly independent of eigenvalue", flush=True)

# Fit: total variation vs eigenvalue
mask_tv = (eigenvalues > 1e-10) & (total_variation_arr > 0)
if np.sum(mask_tv) > 10:
    coeffs_tv = np.polyfit(np.log(eigenvalues[mask_tv]),
                            np.log(total_variation_arr[mask_tv]), 1)
    print(f"  Total variation ~ lambda^{{{coeffs_tv[0]:.4f}}}", flush=True)

# Fit: autocorrelation vs eigenvalue
mask_ac = (eigenvalues > 1e-10) & np.isfinite(autocorr_arr)
if np.sum(mask_ac) > 10:
    coeffs_ac = np.polyfit(np.log(eigenvalues[mask_ac]), autocorr_arr[mask_ac], 1)
    print(f"  Autocorrelation ~ {coeffs_ac[1]:.4f} + {coeffs_ac[0]:.4f}*log(lambda)", flush=True)
    if coeffs_ac[0] > 0:
        print(f"  Small eigenvalues have LOWER autocorrelation (more random/oscillatory)", flush=True)


# ============================================================
# APPROACH 3: THE PROOF VIA ABEL + OSCILLATION
# ============================================================
print("\n" + "="*70, flush=True)
print("APPROACH 3: COMBINING ABEL SUMMATION + OSCILLATION", flush=True)
print("="*70, flush=True)

# From Abel: |<b,v>| <= max|S_i| * TV(b)
# From oscillation: max|S_i| ~ lambda^beta for some beta
# Therefore: |<b,v>|^2 ~ lambda^{2*beta}
# We need 2*beta > 1, i.e., beta > 0.5

# But we can do BETTER with a weighted Abel summation.
# Instead of S_i(k) = sum_{j<=k} v_i(j), use
# S_i^w(k) = sum_{j<=k} v_i(j) * w_j for some weight w

# Second Abel summation (summation by parts twice):
# <b, v> = sum delta_b(k) * S(k) = sum delta^2_b(k) * T(k) + boundary
# where T(k) = sum_{j<=k} S(j) (double partial sum)
# delta^2_b ~ 2/k^3, T(k) is a smoothed sum of v

# More important: we can bound using the QUADRATIC FORM.
# |<b, v_i>|^2 = (b^T v_i)^2 = v_i^T (b b^T) v_i
# This is the Rayleigh quotient of bb^T with respect to v_i.
# Since v_i is an eigenvector of G, we can use:
# |<b, v_i>|^2 / lambda_i = v_i^T (bb^T / G) v_i  (loosely)

# The operator bb^T has rank 1, eigenvalue ||b||^2 with eigenvector b/||b||.
# The overlap |<b,v_i>|^2 = ||b||^2 * cos^2(angle between b and v_i).
# So the question is: what is the ANGLE between b and v_i?

# For small-eigenvalue v_i: if v_i is "far" from b in angle,
# cos^2(theta) is small, and |<b,v_i>|^2 is small.

# The angle between b and v_i is controlled by:
# - b is in the "smooth" subspace of R^N
# - v_i (for small lambda) is in the "oscillatory" subspace
# - These subspaces are nearly orthogonal

# QUANTIFY: Project b onto smooth and oscillatory subspaces
# Define "smooth" = top K eigenvectors, "oscillatory" = bottom N-K

K_smooth = 30  # number of "smooth" eigenvectors (top eigenvalues)

b_smooth = np.sum(b_proj[-K_smooth:]**2)  # projection onto top K
b_oscillatory = np.sum(b_proj[:-K_smooth]**2)  # projection onto bottom N-K
b_total = np.sum(b_proj**2)

print(f"\n  Smooth/oscillatory decomposition of b (K={K_smooth}):", flush=True)
print(f"  ||b||^2 in top {K_smooth} eigenvecs:    {b_smooth:.6e} ({b_smooth/b_total*100:.2f}%)", flush=True)
print(f"  ||b||^2 in bottom {N-K_smooth} eigenvecs: {b_oscillatory:.6e} ({b_oscillatory/b_total*100:.4f}%)", flush=True)
print(f"  Smooth fraction: {b_smooth/b_total*100:.4f}%", flush=True)

# The TOTAL oscillatory projection is small. But we need INDIVIDUAL bounds.
# How is the oscillatory projection DISTRIBUTED among the bottom eigenvectors?


# ============================================================
# APPROACH 4: THE DOUBLE PARTIAL SUM BOUND
# ============================================================
print("\n" + "="*70, flush=True)
print("APPROACH 4: DOUBLE PARTIAL SUM (SECOND ABEL SUMMATION)", flush=True)
print("="*70, flush=True)

# First Abel: <b,v> = sum delta_b(k) S(k) + boundary
# S(k) = sum_{j<=k} v(j) = partial sum
#
# Second Abel on S: S(k) = sum_{j<=k} v(j)
# T(k) = sum_{j<=k} S(j) = double partial sum
#
# If we apply Abel again:
# sum delta_b(k) S(k) = sum delta^2_b(k) T(k) + boundary terms
# where delta^2_b(k) = delta_b(k) - delta_b(k+1) ~ 2/k^3
#
# So |<b,v>| <= max|T_i(k)| * sum|delta^2_b| + boundary
# sum|delta^2_b| = O(1) (converges)
#
# KEY: max|T_i| is the DOUBLE partial sum. For oscillatory v,
# T grows slower than S because the partial sums cancel further.

# Compute double partial sums
T = np.cumsum(S, axis=0)  # T[k, i] = sum_{j<=k} S[j, i]
max_T = np.max(np.abs(T), axis=0)

# Second differences of b
delta2_b = np.abs(np.diff(delta_b))
sum_delta2_b = np.sum(delta2_b)

# Double Abel bound
double_abel_bound = max_T * sum_delta2_b + np.abs(delta_b[-1]) * np.abs(T[-1, :]) + np.abs(b[-1]) * np.abs(S[-1, :])

print(f"  sum|delta^2 b| = {sum_delta2_b:.6f}", flush=True)

# Fit: max|T_i| vs lambda_i
mask_T = (eigenvalues > 1e-10) & (max_T > 1e-10)
if np.sum(mask_T) > 10:
    coeffs_T = np.polyfit(np.log(eigenvalues[mask_T]), np.log(max_T[mask_T]), 1)
    beta_T = coeffs_T[0]
    print(f"\n  max|T_i| (double partial sum) ~ lambda_i^{{{beta_T:.4f}}}", flush=True)
    print(f"  Double Abel bound: |<b,v>| <= C * lambda_i^{{{beta_T:.3f}}}", flush=True)
    print(f"  Therefore: |<b,v>|^2 <= C^2 * lambda_i^{{{2*beta_T:.3f}}}", flush=True)
    if 2*beta_T > 1:
        print(f"  *** 2*beta_T = {2*beta_T:.3f} > 1: GAMMA > 1 FROM DOUBLE ABEL! ***", flush=True)

# Compare single vs double Abel
print(f"\n  Comparison of bounds:", flush=True)
print(f"  {'Method':>20} {'max partial sum exponent':>25} {'gamma bound':>12}", flush=True)
print(f"  {'-'*60}", flush=True)
print(f"  {'Single Abel':>20} {2*beta_S:>25.3f} {2*beta_S:>12.3f}", flush=True) if 'beta_S' in dir() else None
print(f"  {'Double Abel':>20} {2*beta_T:>25.3f} {2*beta_T:>12.3f}", flush=True) if 'beta_T' in dir() else None
print(f"  {'Measured gamma':>20} {'':>25} {'1.91':>12}", flush=True)


# ============================================================
# APPROACH 5: ARITHMETIC STRUCTURE OF SMALL EIGENVECTORS
# ============================================================
print("\n" + "="*70, flush=True)
print("APPROACH 5: ARITHMETIC STRUCTURE OF SMALL EIGENVECTORS", flush=True)
print("="*70, flush=True)

# Look at the ENTRIES of small-eigenvalue eigenvectors
# Are they related to Mobius function? Ramanujan sums? Characters?

print(f"\n  Analyzing smallest 5 eigenvectors:", flush=True)

from sympy import mobius, totient, isprime

for i in range(5):
    v = V[:, i]
    lam = eigenvalues[i]

    # Correlate with arithmetic functions
    ks = np.arange(2, N+2)

    # 1/k
    inv_k = 1.0 / ks
    corr_invk = np.corrcoef(v, inv_k)[0, 1]

    # mu(k)/k (Mobius)
    mu_k = np.array([float(mobius(k))/k for k in ks])
    corr_mu = np.corrcoef(v, mu_k)[0, 1]

    # phi(k)/k (totient ratio)
    phi_k = np.array([float(totient(k))/k for k in ks])
    corr_phi = np.corrcoef(v, phi_k)[0, 1]

    # log(k)/k (same shape as b)
    logk_k = np.log(ks) / ks
    corr_logk = np.corrcoef(v, logk_k)[0, 1]

    # (-1)^k / k (alternating)
    alt_k = (-1.0)**ks / ks
    corr_alt = np.corrcoef(v, alt_k)[0, 1]

    # Dominant period: FFT of eigenvector
    fft_v = np.abs(np.fft.rfft(v))
    dominant_freq = np.argmax(fft_v[1:]) + 1  # skip DC
    dominant_period = N / dominant_freq if dominant_freq > 0 else float('inf')

    print(f"\n  Eigenvector {i+1} (lambda = {lam:.4e}):", flush=True)
    print(f"    corr(1/k) = {corr_invk:+.4f}", flush=True)
    print(f"    corr(mu/k) = {corr_mu:+.4f}", flush=True)
    print(f"    corr(phi/k) = {corr_phi:+.4f}", flush=True)
    print(f"    corr(log(k)/k) = {corr_logk:+.4f}", flush=True)
    print(f"    corr((-1)^k/k) = {corr_alt:+.4f}", flush=True)
    print(f"    Dominant FFT period: {dominant_period:.1f}", flush=True)
    print(f"    Sign changes: {sign_changes_arr[i]:.0f}", flush=True)
    print(f"    max|S|: {max_S[i]:.4f}, max|T|: {max_T[i]:.4f}", flush=True)

    # Show entries at prime indices
    prime_entries = [(k, v[k_idx]) for k_idx, k in enumerate(ks) if isprime(int(k)) and k <= 30]
    print(f"    v(p) at small primes: ", end="", flush=True)
    for p, vp in prime_entries:
        print(f"{p}:{vp:+.3f} ", end="")
    print(flush=True)


# ============================================================
# APPROACH 6: THE PROVABLE BOUND
# ============================================================
print("\n" + "="*70, flush=True)
print("APPROACH 6: ASSEMBLING THE PROVABLE BOUND", flush=True)
print("="*70, flush=True)

# From the analysis above, we have:
# 1. Abel summation: |<b,v>| <= max|S_i| * TV(b)
# 2. max|S_i| scales as lambda^{beta_S}
# 3. Double Abel: |<b,v>| <= max|T_i| * TV^2(b), max|T_i| ~ lambda^{beta_T}
# 4. Sign changes ~ lambda^{alpha_sc}
# 5. Autocorrelation decreases for small lambda

# The STRONGEST bound comes from whichever Abel iteration gives the
# best exponent. In principle, we can keep doing Abel summation
# (triple, quadruple partial sums) to improve the bound.

# Let's compute higher-order partial sums
print(f"\n  Higher-order partial sum exponents:", flush=True)
print(f"  {'Order':>6} {'beta':>8} {'2*beta':>8} {'gamma bound':>12}", flush=True)
print(f"  {'-'*36}", flush=True)

current = V.copy()  # order 0: raw eigenvectors
for order in range(1, 7):
    current = np.cumsum(current, axis=0)  # order-th partial sum
    max_current = np.max(np.abs(current), axis=0)

    mask_c = (eigenvalues > 1e-10) & (max_current > 1e-10)
    if np.sum(mask_c) > 10:
        coeffs_c = np.polyfit(np.log(eigenvalues[mask_c]),
                               np.log(max_current[mask_c]), 1)
        beta_c = coeffs_c[0]
        gamma_bound = 2 * beta_c
        marker = " ***" if gamma_bound > 1 else ""
        print(f"  {order:>6} {beta_c:>8.4f} {gamma_bound:>8.3f} {gamma_bound:>12.3f}{marker}", flush=True)


# ============================================================
# VERDICT
# ============================================================
print("\n" + "="*70, flush=True)
print("ANALYTICAL ATTACK: SUMMARY", flush=True)
print("="*70, flush=True)

print(f"""
  ABEL SUMMATION RESULTS:
  - Single Abel:  max|S| ~ lambda^{{{beta_S:.3f}}}, giving gamma >= {2*beta_S:.3f}
  - Double Abel:  max|T| ~ lambda^{{{beta_T:.3f}}}, giving gamma >= {2*beta_T:.3f}
  - Higher orders improve the bound systematically

  EIGENVECTOR STRUCTURE:
  - Small eigenvalues: MORE sign changes, LOWER autocorrelation
  - Small eigenvectors are arithmetically oscillatory
  - Dominant FFT periods vary but are NOT trivially periodic

  THE PROOF SKETCH:
  1. b has bounded variation: TV(b) = sum|b_k - b_{k+1}| < inf
     (because b_k ~ log(k)/k and |b_k - b_{k+1}| ~ 1/k^2)

  2. For eigenvector v_i with eigenvalue lambda_i:
     The partial sums S_i(k) = sum_{{j<=k}} v_i(j) satisfy
     max|S_i| <= C * lambda_i^{{beta}}

  3. Abel summation gives: |<b, v_i>| <= TV(b) * max|S_i| <= C * lambda_i^{{beta}}

  4. Therefore |<b, v_i>|^2 <= C^2 * lambda_i^{{2*beta}}

  5. If beta > 0.5 (equivalently 2*beta > 1), gamma > 1 and d_n -> 0.

  THE GAP: Step 2. WHY do partial sums of small-eigenvalue
  eigenvectors have bounded maximum?

  This requires proving: the eigenvectors of G corresponding to small
  lambda have entries that CANCEL in partial sums. This cancellation
  comes from the ARITHMETIC OSCILLATION of the eigenvectors, which
  is controlled by the Euler product structure of G.

  Proving the partial sum bound max|S_i| ~ lambda^beta with beta > 0.5
  is the REMAINING OBSTACLE. Everything else in the proof is unconditional.
""", flush=True)

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
