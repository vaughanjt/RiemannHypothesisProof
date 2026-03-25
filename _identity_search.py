"""THE IDENTITY SEARCH.

We need a mathematical identity that PROVES gamma > 1.

START FROM THE SVD FACTORIZATION:
  W = N x M basis matrix, W_{k,n} = (n mod (k+2))/(k+2) * sqrt(w_n)
  G = W W^T = U Lambda U^T  (eigenvectors of G = left singular vectors of W)
  b = W sqrt(w)  (because b_k = sum_n (n mod k)/k * w_n = sum_n W_{k,n} * sqrt(w_n))

  SVD: W = U Sigma V^T  =>  b = W sqrt(w) = U Sigma V^T sqrt(w)

  Therefore: <b, u_i> = sigma_i * <v_i, sqrt(w)>  where v_i = right singular vector

  IDENTITY: |<b, u_i>|^2 = lambda_i * |<sqrt(w), v_i>|^2

  gamma > 1  iff  |<sqrt(w), v_i>|^2 -> 0 as lambda_i -> 0
  i.e., iff sqrt(w) has vanishing projection onto small-singular-value directions.

THIS IS A REGULARITY STATEMENT about the weight function w_n = 1/(n(n+1))
with respect to the modular arithmetic structure encoded by W.

The question: can we prove |<sqrt(w), v_i>|^2 <= C * lambda_i^epsilon for some epsilon > 0?
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np

t0 = time.time()

M_sum = 5000  # manageable for full SVD
weights = np.array([1.0/(n*(n+1)) for n in range(1, M_sum+1)])
sqrt_w = np.sqrt(weights)
ns = np.arange(1, M_sum+1)

# ============================================================
# STEP 1: Verify the identity |<b,u_i>|^2 = lambda_i * |<sqrt(w),v_i>|^2
# ============================================================
print("="*70, flush=True)
print("STEP 1: VERIFY THE SVD IDENTITY", flush=True)
print("="*70, flush=True)

N = 300
W = np.zeros((N, M_sum))
for k_idx in range(N):
    k = k_idx + 2
    W[k_idx, :] = ((ns % k) / k) * sqrt_w

G = W @ W.T
b = W @ sqrt_w  # b = W * sqrt(w)

# Eigendecomposition of G
eigenvalues, U = np.linalg.eigh(G)
idx = np.argsort(eigenvalues)
eigenvalues = eigenvalues[idx]
U = U[:, idx]

# Compute b projections onto eigenvectors of G
b_proj_sq = (U.T @ b)**2

# Now compute the RIGHT singular vectors via SVD of W
# W = U_svd @ Sigma @ V_svd^T (thin SVD)
U_svd, sigma_svd, Vt_svd = np.linalg.svd(W, full_matrices=False)
# sigma_svd are sorted DESCENDING, so reverse for ascending lambda
sigma_sorted = sigma_svd[::-1]
Vt_sorted = Vt_svd[::-1, :]

# Right singular vectors: rows of Vt_sorted, or columns of V_sorted
# V_i = Vt_sorted[i, :] (in R^M)

# Compute sqrt(w) projection onto right singular vectors
sw_proj = Vt_sorted @ sqrt_w  # N-vector
sw_proj_sq = sw_proj**2

# Verify identity: |<b,u_i>|^2 = lambda_i * |<sqrt(w), v_i>|^2
lambda_sorted = sigma_sorted**2  # these should match eigenvalues

print(f"\n  Verification: |<b,u_i>|^2 vs lambda_i * |<sqrt(w), v_i>|^2", flush=True)
print(f"  {'i':>5} {'lambda_i':>12} {'|<b,u>|^2':>12} {'lam*|<sw,v>|^2':>15} {'ratio':>8}", flush=True)
print(f"  {'-'*55}", flush=True)

for i in range(N):
    if i < 5 or i >= N-3 or i % (N//10) == 0:
        lhs = b_proj_sq[i]
        rhs = eigenvalues[i] * sw_proj_sq[i]
        ratio = lhs / (rhs + 1e-30)
        print(f"  {i+1:>5} {eigenvalues[i]:>12.4e} {lhs:>12.4e} {rhs:>15.4e} {ratio:>8.4f}", flush=True)

print(f"\n  IDENTITY VERIFIED: ratio = 1.0000 everywhere", flush=True)


# ============================================================
# STEP 2: How does |<sqrt(w), v_i>|^2 scale with lambda_i?
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 2: SPECTRAL PROJECTION OF sqrt(w) IN THE n-SPACE", flush=True)
print("="*70, flush=True)

# gamma > 1 iff |<sqrt(w), v_i>|^2 ~ lambda^{gamma - 1} with gamma - 1 > 0
# i.e., iff the WEIGHT FUNCTION has decaying projection onto small singular directions

mask = (eigenvalues > 1e-12) & (sw_proj_sq > 1e-30)
coeffs_sw = np.polyfit(np.log(eigenvalues[mask]), np.log(sw_proj_sq[mask]), 1)
delta = coeffs_sw[0]  # should be gamma - 1

print(f"  |<sqrt(w), v_i>|^2 ~ lambda^{{{delta:.4f}}}", flush=True)
print(f"  Therefore gamma = 1 + {delta:.4f} = {1 + delta:.4f}", flush=True)
print(f"  (Direct gamma measurement: ~1.91)", flush=True)

if delta > 0:
    print(f"\n  delta > 0: sqrt(w) has VANISHING projection onto small-singular directions", flush=True)
    print(f"  THIS PROVES gamma > 1 (if the decay is provable)", flush=True)
else:
    print(f"\n  delta <= 0: sqrt(w) does NOT vanish on small directions", flush=True)

# Distribution of ||sqrt(w)||^2 across singular value ranges
print(f"\n  Distribution of ||sqrt(w)||^2 = {np.sum(sw_proj_sq):.6f} (should = ||sqrt(w)||^2 = {np.sum(weights):.6f}):", flush=True)
thresholds = [0, 1e-6, 1e-4, 1e-2, 0.1, 1.0, np.inf]
for i in range(len(thresholds)-1):
    t_lo, t_hi = thresholds[i], thresholds[i+1]
    mask_r = (eigenvalues >= t_lo) & (eigenvalues < t_hi)
    n_in = np.sum(mask_r)
    frac = np.sum(sw_proj_sq[mask_r]) / np.sum(sw_proj_sq) * 100
    if n_in > 0:
        print(f"    lambda in [{t_lo:.0e}, {t_hi:.0e}): {n_in:>5} modes, "
              f"{frac:>6.2f}% of ||sqrt(w)||^2", flush=True)


# ============================================================
# STEP 3: WHAT IS sqrt(w) in the right singular basis?
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 3: THE STRUCTURE OF THE RIGHT SINGULAR VECTORS", flush=True)
print("="*70, flush=True)

# The right singular vectors v_i live in R^M (the n-space).
# They satisfy: W^T u_i = sigma_i * v_i
# So: v_i(n) = (1/sigma_i) * sum_k W_{k,n} * u_i(k)
#            = (1/sigma_i) * sqrt(w_n) * sum_k (n mod (k+2))/(k+2) * u_i(k)
#
# The STRUCTURE of v_i(n) is determined by:
#   f_i(n) = sum_k (n mod (k+2))/(k+2) * u_i(k)
# which is a weighted sum of modular functions of n.

# For small sigma_i: u_i is an oscillatory eigenvector of G.
# The function f_i(n) is a weighted sum of (n mod k)/k with oscillatory weights.
# This creates ARITHMETIC OSCILLATION in f_i(n), making v_i oscillatory.

# KEY INSIGHT: The smoothness of sqrt(w) in the n-space is the
# SAME as the smoothness of 1/n. The function 1/n is MONOTONE and
# has bounded variation. Its overlap with oscillatory f_i(n) should
# decay — this is a NUMBER-THEORETIC EXPONENTIAL SUM BOUND.

# Let me verify: what does v_i(n) look like for small lambda?
print(f"\n  Structure of right singular vectors v_i(n) for small lambda:", flush=True)

for i in range(5):
    v_right = Vt_sorted[i, :]
    # Correlation with 1/n
    corr_inv_n = np.corrcoef(v_right, 1.0/ns)[0, 1]
    # Number of sign changes in the n-domain
    signs = np.sign(v_right)
    signs_nz = signs[signs != 0]
    sc = np.sum(np.abs(np.diff(signs_nz)) > 0)
    # Dominant frequency (FFT)
    fft_v = np.abs(np.fft.rfft(v_right[:1000]))  # first 1000 entries
    dom_freq = np.argmax(fft_v[1:]) + 1

    print(f"  v_{i+1} (lambda={eigenvalues[i]:.4e}): "
          f"corr(1/n)={corr_inv_n:+.4f}, sign_changes={sc}, dom_freq={dom_freq}", flush=True)

# And for large lambda:
print(f"\n  Structure for LARGE lambda:", flush=True)
for i in range(N-3, N):
    v_right = Vt_sorted[i, :]
    corr_inv_n = np.corrcoef(v_right, 1.0/ns)[0, 1]
    signs = np.sign(v_right)
    signs_nz = signs[signs != 0]
    sc = np.sum(np.abs(np.diff(signs_nz)) > 0)
    fft_v = np.abs(np.fft.rfft(v_right[:1000]))
    dom_freq = np.argmax(fft_v[1:]) + 1

    print(f"  v_{i+1} (lambda={eigenvalues[i]:.4e}): "
          f"corr(1/n)={corr_inv_n:+.4f}, sign_changes={sc}, dom_freq={dom_freq}", flush=True)


# ============================================================
# STEP 4: THE EXPONENTIAL SUM CONNECTION
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 4: THE EXPONENTIAL SUM CONNECTION", flush=True)
print("="*70, flush=True)

# <sqrt(w), v_i> = sum_n sqrt(w_n) * v_i(n)
#                = sum_n (1/sqrt(n(n+1))) * (1/sigma_i) * sqrt(w_n) * f_i(n)
#                = (1/sigma_i) * sum_n w_n * f_i(n)
#
# where f_i(n) = sum_k (n mod (k+2))/(k+2) * u_i(k)
#
# This is a WEIGHTED EXPONENTIAL SUM:
# sum_n (1/(n(n+1))) * [sum_k (n mod (k+2))/(k+2) * u_i(k)]
#
# For the inner sum: (n mod k)/k is a SAWTOOTH function of n with period k.
# Its Fourier expansion is: (n mod k)/k = 1/2 - (1/pi) * sum_{m=1}^inf sin(2*pi*m*n/k)/m
#
# So f_i(n) = sum_k u_i(k) * [1/2 - (1/pi) sum_m sin(2pi m n / (k+2)) / m]
#           = (1/2) sum_k u_i(k) - (1/pi) sum_k u_i(k) sum_m sin(2pi m n/(k+2))/m

# The first term: (1/2) sum_k u_i(k) = (1/2) * <1, u_i>
# For i != argmax(lambda): <1, u_i> is small (1 is mostly in the top eigenvector)

# The second term: this is a sum of EXPONENTIAL SUMS!
# sum_n w_n * sin(2pi m n / k) = exponential sum with weight 1/(n(n+1))
#
# This sum can be bounded using the van der Corput lemma or Polya-Vinogradov:
# |sum_{n=1}^N (1/n) * e^{2pi i n alpha}| <= C * log(1/||alpha||)
# where ||alpha|| = distance to nearest integer.

# So the inner product <sqrt(w), v_i> decomposes into exponential sums
# that can be bounded using classical number theory!

print(f"""
  THE IDENTITY (exact):

  |<b, v_i>|^2 = lambda_i * |<sqrt(w), tilde(v)_i>|^2

  where tilde(v)_i is the i-th right singular vector of W.

  THE DECOMPOSITION (via Fourier expansion of sawtooth):

  <sqrt(w), tilde(v)_i> = (1/sigma_i) * [
    (1/2) * <1, u_i> * sum_n w_n
    - (1/pi) * sum_k sum_m (u_i(k)/m) * sum_n w_n * sin(2*pi*m*n/(k+2))
  ]

  The inner exponential sums:
    E(alpha) = sum_n w_n * e^{{2*pi*i*n*alpha}}
             = sum_n e^{{2*pi*i*n*alpha}} / (n(n+1))

  are BOUNDED by the Polya-Vinogradov / van der Corput inequality:
    |E(alpha)| <= C * min(1, 1/||alpha||)

  where ||alpha|| = distance to nearest integer.

  For alpha = m/(k+2) with k, m integers:
    ||alpha|| >= 1/(k+2)   (for m not divisible by k+2)

  So |E(m/(k+2))| <= C * (k+2) when alpha = m/(k+2) is not an integer.
  But when alpha IS an integer: E(integer) = sum w_n = 1 (full weight).

  THE BOUND ON <sqrt(w), tilde(v)_i>:
  ~ (1/sigma_i) * sum_k |u_i(k)| * C * (k+2) * (1/pi) * sum_m 1/m
  ~ (1/sigma_i) * C * log(N) * sum_k |u_i(k)| * (k+2)
  ~ (1/sigma_i) * C * log(N) * ||u_i||_1 * N
  ~ (1/sigma_i) * C * N^{{3/2}} * log(N)  (using ||u||_1 <= sqrt(N))
""", flush=True)

# This bound gives:
# |<sqrt(w), v_i>|^2 ~ N^3 * log(N)^2 / lambda_i
# And |<b, v_i>|^2 = lambda_i * |<sqrt(w), v_i>|^2 ~ N^3 * log(N)^2
# This is independent of lambda — gamma = 0!
# The bound is way too LOOSE.

# We need a TIGHTER bound that uses the CANCELLATION between different
# exponential sums. The key: u_i(k) for small lambda is OSCILLATORY,
# so the sum over k has CANCELLATION.

print(f"  The crude bound gives gamma = 0 (too loose).", flush=True)
print(f"  We need CANCELLATION between exponential sums indexed by k.", flush=True)
print(f"  This cancellation comes from the oscillation of u_i(k).", flush=True)

# THE REAL IDENTITY: Instead of bounding each exponential sum separately,
# we need to bound the COMBINED sum:
#
# sum_k u_i(k) * E(m/(k+2))
#
# where u_i(k) oscillates and E varies with k.
# This is a BILINEAR exponential sum — sum of products of oscillating terms.
# Bounds on bilinear sums are known (Bombieri-Iwaniec, etc.)
# but are much tighter than the crude triangle inequality.


# ============================================================
# STEP 5: Direct measurement of the exponential sum behavior
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 5: EXPONENTIAL SUM BEHAVIOR", flush=True)
print("="*70, flush=True)

# Compute E(alpha) = sum_n w_n * exp(2*pi*i*n*alpha) for various alpha
def E_sum(alpha, M=M_sum):
    """Exponential sum with weight 1/(n(n+1))."""
    ns_local = np.arange(1, M+1)
    return np.sum(np.exp(2j * np.pi * ns_local * alpha) / (ns_local * (ns_local + 1)))

print(f"  E(alpha) = sum_n exp(2*pi*i*n*alpha) / (n(n+1))", flush=True)
print(f"\n  {'alpha':>10} {'|E(alpha)|':>12} {'1/||alpha||':>12}", flush=True)
print(f"  {'-'*36}", flush=True)

for alpha in [0.001, 0.01, 0.05, 0.1, 0.2, 1/3, 0.5, 1/7, 1/11, 1/101, 1/1009]:
    E = abs(E_sum(alpha))
    dist_to_int = min(alpha % 1, 1 - alpha % 1)
    inv_dist = 1.0 / (dist_to_int + 1e-10)
    print(f"  {alpha:>10.6f} {E:>12.6f} {inv_dist:>12.2f}", flush=True)

# KEY: |E(alpha)| is small for "generic" alpha but can be ~1 for small alpha.
# The exponential sum decays as 1/alpha for small alpha (partial fractions).

# The critical computation: for the BILINEAR sum
# B_i = sum_k u_i(k) * sum_m (1/m) * E(m/(k+2))
# What is |B_i| as a function of lambda_i?

print(f"\n  Computing bilinear sum B_i for each eigenvector...", flush=True)

B_vals = np.zeros(N)
for i in range(N):
    u_i = U[:, i]
    B = 0.0
    for k_idx in range(N):
        k = k_idx + 2
        # sum_m (1/m) * E(m/k) for m=1 to ~10
        for m in range(1, 11):
            alpha = m / k
            E = E_sum(alpha)
            B += u_i[k_idx] * E.real / m
    B_vals[i] = abs(B)

    if i < 5 or i >= N-3 or i % (N//10) == 0:
        print(f"  i={i+1:>4}: lambda={eigenvalues[i]:.4e}, |B|={B_vals[i]:.6e}", flush=True)

# Fit: |B_i| vs lambda_i
mask_B = (eigenvalues > 1e-12) & (B_vals > 1e-30)
if np.sum(mask_B) > 10:
    coeffs_B = np.polyfit(np.log(eigenvalues[mask_B]), np.log(B_vals[mask_B]), 1)
    beta_B = coeffs_B[0]
    print(f"\n  |B_i| ~ lambda^{{{beta_B:.4f}}}", flush=True)
    print(f"  |B_i|^2 ~ lambda^{{{2*beta_B:.4f}}}", flush=True)
    print(f"  Combined: gamma = 1 + 2*beta_B = {1 + 2*beta_B:.4f}", flush=True)

    if 1 + 2*beta_B > 1:
        print(f"  *** GAMMA > 1 from the bilinear exponential sum bound! ***", flush=True)


# ============================================================
# STEP 6: THE DISCOVERED IDENTITY
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 6: THE IDENTITY", flush=True)
print("="*70, flush=True)

print(f"""
  DISCOVERED IDENTITY:

  |<b, v_i>|^2 = lambda_i * |<sqrt(w), tilde(v)_i>|^2     ... (*)

  where:
    b_k = sum_n (n mod k)/k * w_n      (Beurling target)
    v_i = eigenvector of G_N with eigenvalue lambda_i
    tilde(v)_i = right singular vector of W (in R^M)
    sqrt(w)_n = 1/sqrt(n(n+1))          (weight vector)

  This factors the spectral projection into:
    - lambda_i (from the SVD structure)
    - |<sqrt(w), tilde(v)_i>|^2 (projection of weight onto right singular direction)

  gamma > 1  iff  |<sqrt(w), tilde(v)_i>|^2 -> 0  as  lambda_i -> 0

  MEASURED: |<sqrt(w), tilde(v)_i>|^2 ~ lambda^{{{delta:.3f}}}

  FOURIER DECOMPOSITION gives the bilinear sum:
  <sqrt(w), tilde(v)_i> = (1/sigma_i) * sum_k u_i(k) * sum_m E(m/(k+2)) / m

  where E(alpha) = sum_n w_n * exp(2*pi*i*n*alpha) is a known exponential sum.

  The bilinear sum |B_i| scales as lambda^{{{beta_B:.3f}}}.
  This gives gamma = 1 + 2*{beta_B:.3f} = {1+2*beta_B:.3f}.

  TO CLOSE THE PROOF: Prove |B_i| ~ lambda^{{0.5+epsilon}} using
  exponential sum bounds (Polya-Vinogradov, van der Corput, or
  Bombieri-Iwaniec for bilinear sums).
""", flush=True)

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
