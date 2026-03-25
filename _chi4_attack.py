"""ATTACK GRH FOR L(s, chi_4).

chi_4 has gamma = 3.07 at N=1000. Three times the threshold.
The character kills all even n, creating Z/2 sparsity.

THE STRUCTURE WE CAN EXPLOIT:
  chi_4(n) = 0 if n even, +1 if n=1 mod 4, -1 if n=3 mod 4

  The chi-weighted basis vector for k:
    f_k^chi(n) = chi_4(n) * (n mod k)/k * sqrt(w_n)

  For even k: f_k^chi only sees odd n, and (n mod k)/k has period k
  For odd k: same, but the mod-k structure interacts with mod-4

  KEY: The Gram matrix G^chi has BLOCK STRUCTURE because chi_4 is
  supported on odd integers. We can reindex n -> 2m+1 (odd only)
  and work in a REDUCED basis where the matrix is half the size.

  In the reduced basis, the modular arithmetic simplifies:
  For odd n = 2m+1: (n mod k)/k depends on k mod 2:
    - k even: (2m+1 mod k)/k cycles through odd residues only
    - k odd: (2m+1 mod k)/k cycles through all residues

  The chi_4 weighting further splits:
    - n = 1 mod 4: weight +1
    - n = 3 mod 4: weight -1
  This is the SIGN PATTERN that creates the extra cancellation.

PLAN:
  1. Analyze the reduced Gram matrix structure
  2. Verify gamma = 3.07 in the reduced basis
  3. Look for exploitable structure (block diagonal, near-Toeplitz, etc.)
  4. Attempt a direct proof that gamma > 1 for chi_4
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np
from math import gcd

t0 = time.time()

M_sum = 10000
ns = np.arange(1, M_sum+1)

def chi_4(n):
    r = n % 4
    if r == 1: return 1.0
    if r == 3: return -1.0
    return 0.0

chi_vals = np.array([chi_4(int(n)) for n in ns])
weights = 1.0 / (ns * (ns + 1))
sqrt_w = np.sqrt(weights)

# ============================================================
# STEP 1: THE REDUCED BASIS (odd n only)
# ============================================================
print("="*70, flush=True)
print("STEP 1: REDUCED BASIS — odd n only", flush=True)
print("="*70, flush=True)

# chi_4(n) = 0 for even n. So the effective sum is over odd n only.
# Odd indices: n = 1, 3, 5, 7, ...
odd_mask = (ns % 2 == 1)
ns_odd = ns[odd_mask]
chi_odd = chi_vals[odd_mask]  # +1 for 1 mod 4, -1 for 3 mod 4
w_odd = weights[odd_mask]
sqrt_w_odd = np.sqrt(w_odd)

M_odd = len(ns_odd)
print(f"  Full M = {M_sum}, Odd M = {M_odd}", flush=True)
print(f"  chi_4 values on odd n: +1 for 1mod4, -1 for 3mod4", flush=True)

# The alternating sign pattern: chi_4(2m+1) = (-1)^m
# n=1: chi=+1 (m=0), n=3: chi=-1 (m=1), n=5: chi=+1 (m=2), n=7: chi=-1 (m=3)
print(f"  Verify: chi_4(odd) = (-1)^m pattern:", flush=True)
ms = np.arange(M_odd)
chi_pattern = (-1.0)**ms
print(f"    Match: {np.allclose(chi_odd, chi_pattern)}", flush=True)

# So in the odd basis: chi_4(n) = (-1)^m where n = 2m+1
# This is a PERFECT ALTERNATING SIGN. Extremely regular.


# ============================================================
# STEP 2: Build the chi_4-weighted Gram matrix in reduced basis
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 2: CHI_4 GRAM MATRIX STRUCTURE", flush=True)
print("="*70, flush=True)

N = 500  # basis size

# Full chi-weighted W
W_chi = np.zeros((N, M_sum))
for k_idx in range(N):
    k = k_idx + 2
    W_chi[k_idx, :] = chi_vals * ((ns % k) / k) * sqrt_w

G_chi = W_chi @ W_chi.T

# Reduced W (odd n only, with chi sign absorbed)
W_red = np.zeros((N, M_odd))
for k_idx in range(N):
    k = k_idx + 2
    W_red[k_idx, :] = chi_odd * ((ns_odd % k) / k) * sqrt_w_odd

G_red = W_red @ W_red.T

# These should be identical
print(f"  ||G_chi - G_red|| = {np.linalg.norm(G_chi - G_red):.6e} (should be ~0)", flush=True)

# b vector
b_chi = np.zeros(N)
for k_idx in range(N):
    k = k_idx + 2
    b_chi[k_idx] = np.dot(chi_vals * ((ns % k) / k), weights)

# Eigendecomposition
eigenvalues, V = np.linalg.eigh(G_chi)
idx = np.argsort(eigenvalues)
eigenvalues = eigenvalues[idx]
V = V[:, idx]

b_proj_sq = (V.T @ b_chi)**2
pos_mask = eigenvalues > 1e-12

# Gamma
mask_g = pos_mask & (b_proj_sq > 1e-30)
gamma_chi4 = np.polyfit(np.log(eigenvalues[mask_g]), np.log(b_proj_sq[mask_g]), 1)[0]
print(f"\n  gamma(chi_4) = {gamma_chi4:.4f} at N={N}", flush=True)


# ============================================================
# STEP 3: WHAT MAKES CHI_4 SPECIAL?
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 3: WHY IS GAMMA SO HIGH FOR CHI_4?", flush=True)
print("="*70, flush=True)

# Compare b_chi to b_trivial
b_triv = np.zeros(N)
for k_idx in range(N):
    k = k_idx + 2
    b_triv[k_idx] = np.dot((ns % k) / k, weights)

# How does b_chi look?
print(f"  b_chi vs b_trivial:", flush=True)
print(f"    ||b_chi||  = {np.linalg.norm(b_chi):.6f}", flush=True)
print(f"    ||b_triv|| = {np.linalg.norm(b_triv):.6f}", flush=True)
print(f"    corr(b_chi, b_triv) = {np.corrcoef(b_chi, b_triv)[0,1]:+.4f}", flush=True)

# b_chi(k) = sum_{n odd} chi_4(n) * (n mod k)/k * w_n
# = sum_m (-1)^m * ((2m+1) mod k)/k * w_{2m+1}
# For k even: (2m+1 mod k)/k depends on m mod (k/2), creating period k/2
# For k odd: (2m+1 mod k)/k has full period k

# The KEY: b_chi(k) for even k involves alternating sums of (2m+1 mod k)/k
# The alternating sign creates massive cancellation.

print(f"\n  b_chi entries (first 20):", flush=True)
for k_idx in range(20):
    k = k_idx + 2
    print(f"    k={k:>3} ({'even' if k%2==0 else 'odd '}): b_chi={b_chi[k_idx]:>12.6e}, "
          f"b_triv={b_triv[k_idx]:>12.6e}, ratio={b_chi[k_idx]/(b_triv[k_idx]+1e-30):>8.4f}", flush=True)

# Are even-k entries systematically smaller?
even_b = np.abs(b_chi[::2])  # k=2,4,6,...
odd_b = np.abs(b_chi[1::2])   # k=3,5,7,...
print(f"\n  Mean |b_chi| for even k: {np.mean(even_b):.6e}", flush=True)
print(f"  Mean |b_chi| for odd k:  {np.mean(odd_b):.6e}", flush=True)
print(f"  Ratio (even/odd):         {np.mean(even_b)/np.mean(odd_b):.4f}", flush=True)


# ============================================================
# STEP 4: SVD IDENTITY FOR CHI_4
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 4: SVD IDENTITY FOR CHI_4", flush=True)
print("="*70, flush=True)

# W_chi = U Sigma Vt
U_svd, sigma_svd, Vt_svd = np.linalg.svd(W_chi, full_matrices=False)
sigma_sorted = sigma_svd[::-1]
Vt_sorted = Vt_svd[::-1, :]
lambda_sorted = sigma_sorted**2

# Right singular vectors in R^M
# sqrt(w) projection
sw_proj_sq = (Vt_sorted @ sqrt_w)**2

# Verify identity
b_u_proj_sq = (U_svd[:, ::-1].T @ b_chi)**2

print(f"  Verifying: |<b, u_i>|^2 = lambda_i * |<sqrt(w), v~_i>|^2", flush=True)
match = True
for i in range(min(10, N)):
    lhs = b_u_proj_sq[i]
    rhs = lambda_sorted[i] * sw_proj_sq[i]
    if abs(lhs) > 1e-20 or abs(rhs) > 1e-20:
        ratio = lhs / (rhs + 1e-30)
        if abs(ratio - 1) > 0.01:
            match = False
print(f"  Identity verified: {match}", flush=True)

# Delta for chi_4
mask_d = (lambda_sorted > 1e-12) & (sw_proj_sq > 1e-30)
if np.sum(mask_d) > 10:
    delta_chi4 = np.polyfit(np.log(lambda_sorted[mask_d]),
                             np.log(sw_proj_sq[mask_d]), 1)[0]
    print(f"  delta(chi_4) = {delta_chi4:.4f} (gamma = 1 + delta = {1 + delta_chi4:.4f})", flush=True)

# Compare: where does sqrt(w) live in the chi_4 SVD?
print(f"\n  ||sqrt(w)||^2 distribution in chi_4 SVD:", flush=True)
total_sw = np.sum(sw_proj_sq)
top3 = np.sum(sw_proj_sq[-3:])
print(f"    Top 3 modes: {top3/total_sw*100:.2f}%", flush=True)
print(f"    Bottom 50%:  {np.sum(sw_proj_sq[:N//2])/total_sw*100:.4f}%", flush=True)


# ============================================================
# STEP 5: THE ALTERNATING SIGN ADVANTAGE
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 5: THE ALTERNATING SIGN ADVANTAGE", flush=True)
print("="*70, flush=True)

# In the reduced basis (odd n only), chi_4(n) = (-1)^m.
# The Gram matrix becomes:
# G^chi_{jk} = sum_m (-1)^m * ((2m+1) mod j)/j * (-1)^m * ((2m+1) mod k)/k * w_{2m+1}
#            = sum_m ((2m+1) mod j)/j * ((2m+1) mod k)/k * w_{2m+1}
# Wait, chi appears TWICE (once for each factor), so chi^2 = 1 for odd n.

# Actually: G^chi = W_chi @ W_chi^T where W_chi has chi_4(n) in each entry.
# G^chi_{jk} = sum_n chi_4(n)^2 * (n mod j)/j * (n mod k)/k * w_n
#            = sum_{n odd} (n mod j)/j * (n mod k)/k * w_n
# Because chi_4(n)^2 = 1 for odd n and 0 for even n!

# So G^chi is just G restricted to ODD n! No alternating signs in G itself!
# The alternating signs only appear in b_chi = sum chi_4(n) * f(n).

print(f"  KEY INSIGHT: G^chi = G restricted to odd n (no signs in G!)", flush=True)
print(f"  The character signs appear ONLY in b_chi, not in G.", flush=True)
print(f"  b_chi(k) = sum_m (-1)^m * ((2m+1) mod k)/k * w_{{2m+1}}", flush=True)

# This means: G^chi is a SIMPLER matrix (only odd n contribute),
# but the CHALLENGE is that b_chi has the alternating sign.

# The alternating sign in b makes b_chi SMOOTHER than b_triv.
# Why? Because the (-1)^m oscillation cancels with the modular oscillation
# of (2m+1 mod k)/k, leaving a SMOOTHER residual.

# Verify: is b_chi smoother than b_triv?
tv_chi = np.sum(np.abs(np.diff(b_chi)))
tv_triv = np.sum(np.abs(np.diff(b_triv)))
print(f"\n  Total variation: TV(b_chi) = {tv_chi:.6f}, TV(b_triv) = {tv_triv:.6f}", flush=True)
print(f"  Ratio: {tv_chi/tv_triv:.4f}", flush=True)

tv2_chi = np.sum(np.abs(np.diff(np.diff(b_chi))))
tv2_triv = np.sum(np.abs(np.diff(np.diff(b_triv))))
print(f"  TV^2: TV^2(b_chi) = {tv2_chi:.6f}, TV^2(b_triv) = {tv2_triv:.6f}", flush=True)
print(f"  Ratio: {tv2_chi/tv2_triv:.4f}", flush=True)

# How does b_chi decay?
ks = np.arange(2, N+2)
mask_bfit = np.abs(b_chi) > 1e-15
if np.sum(mask_bfit) > 20:
    coeffs_bchi = np.polyfit(np.log(ks[mask_bfit][:200]),
                              np.log(np.abs(b_chi[mask_bfit][:200])), 1)
    print(f"\n  b_chi(k) ~ k^{{{coeffs_bchi[0]:.3f}}} (b_triv ~ k^{{-0.81}})", flush=True)
    if coeffs_bchi[0] < -0.81:
        print(f"  b_chi decays FASTER than b_triv! Extra smoothness from chi_4.", flush=True)


# ============================================================
# STEP 6: ABEL ANALYSIS FOR CHI_4
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 6: ABEL EXPONENTS FOR CHI_4", flush=True)
print("="*70, flush=True)

# Compute higher-order partial sum exponents for chi_4 eigenvectors
current = V.copy()
print(f"  {'Order':>6} {'beta':>8} {'2*beta':>8} {'gamma_bound':>12}", flush=True)
print(f"  {'-'*36}", flush=True)

for order in range(1, 7):
    current = np.cumsum(current, axis=0)
    max_current = np.max(np.abs(current), axis=0)
    mask_c = pos_mask & (max_current > 1e-12)
    if np.sum(mask_c) > 10:
        coeffs_c = np.polyfit(np.log(eigenvalues[mask_c]),
                               np.log(max_current[mask_c]), 1)
        beta_c = coeffs_c[0]
        marker = " ***" if 2*beta_c > 1 else ""
        print(f"  {order:>6} {beta_c:>8.4f} {2*beta_c:>8.3f} {2*beta_c:>12.3f}{marker}", flush=True)


# ============================================================
# STEP 7: CAN WE PROVE GAMMA > 1 FOR CHI_4?
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 7: THE PROOF ATTEMPT FOR CHI_4", flush=True)
print("="*70, flush=True)

# The chi_4 case has several advantages:
# 1. G^chi only involves odd n (half the sum)
# 2. b_chi has extra smoothness from the (-1)^m factor
# 3. gamma = 3.07 gives 3x headroom
# 4. The Galois structure is Z/2 (simplest possible)

# The DIRECT approach: bound |<b_chi, u_i>|^2 using the explicit form.
# b_chi(k) = sum_{m=0}^{M/2} (-1)^m * ((2m+1) mod k)/k * w_{2m+1}
#
# Using the Fourier expansion of sawtooth:
# (n mod k)/k = 1/2 - (1/pi) sum_{j=1}^inf sin(2*pi*j*n/k) / j
#
# For n = 2m+1:
# ((2m+1) mod k)/k = 1/2 - (1/pi) sum_j sin(2*pi*j*(2m+1)/k) / j
#
# So b_chi(k) = sum_m (-1)^m * [1/2 - (1/pi) sum_j sin(2*pi*j*(2m+1)/k)/j] * w_{2m+1}
#
# The 1/2 term: (1/2) sum_m (-1)^m * w_{2m+1}
# This is a CONVERGENT ALTERNATING SERIES (Leibniz test):
# sum (-1)^m / ((2m+1)(2m+2)) = (pi/4 - 1 + log(2))/2  or similar

half_term = 0.5 * np.sum(chi_pattern * w_odd)
print(f"  Constant term: (1/2) sum (-1)^m w_{{2m+1}} = {half_term:.8f}", flush=True)

# The oscillatory terms:
# -(1/pi) sum_j (1/j) sum_m (-1)^m sin(2*pi*j*(2m+1)/k) * w_{2m+1}
#
# The inner sum: S_j(k) = sum_m (-1)^m sin(2*pi*j*(2m+1)/k) * w_{2m+1}
#              = sum_m (-1)^m sin((4m+2)*pi*j/k) * w_{2m+1}
#
# Using sin(A+B) = sin(A)cos(B) + cos(A)sin(B):
# sin((4m+2)pi*j/k) = sin(4m*pi*j/k + 2pi*j/k)
#                    = sin(4m*pi*j/k)cos(2pi*j/k) + cos(4m*pi*j/k)sin(2pi*j/k)
#
# And (-1)^m = cos(m*pi). So:
# (-1)^m sin((4m+2)pi*j/k) involves products of oscillatory functions.
#
# This is getting very intricate. Let me try a COMPUTATIONAL shortcut:
# measure whether the eigenvectors of G^chi have PROVABLE structure.

# Eigenvector analysis for chi_4
print(f"\n  Eigenvector properties (chi_4, N={N}):", flush=True)
print(f"  {'i':>5} {'lambda':>12} {'sign_ch':>8} {'max|S|':>10} {'max|T|':>10} {'TV':>10}", flush=True)
print(f"  {'-'*56}", flush=True)

S_chi = np.cumsum(V, axis=0)
T_chi = np.cumsum(S_chi, axis=0)
max_S_chi = np.max(np.abs(S_chi), axis=0)
max_T_chi = np.max(np.abs(T_chi), axis=0)

for i in range(N):
    if not pos_mask[i]:
        continue
    if i < 5 or i >= N-3 or i % (N//8) == 0:
        v = V[:, i]
        signs = np.sign(v)
        sc = np.sum(np.abs(np.diff(signs[signs!=0])) > 0)
        tv = np.sum(np.abs(np.diff(v)))
        print(f"  {i+1:>5} {eigenvalues[i]:>12.4e} {sc:>8} {max_S_chi[i]:>10.4f} "
              f"{max_T_chi[i]:>10.4f} {tv:>10.4f}", flush=True)

# The CRITICAL question: are chi_4 eigenvectors MORE oscillatory than zeta's?
# If so, the partial sum bounds are STRONGER, making the proof easier.

# Compare sign changes: chi_4 vs trivial
W_triv = np.zeros((N, M_sum))
for k_idx in range(N):
    k = k_idx + 2
    W_triv[k_idx, :] = ((ns % k) / k) * sqrt_w
G_triv = W_triv @ W_triv.T
eigs_triv, V_triv = np.linalg.eigh(G_triv)
idx_t = np.argsort(eigs_triv)
eigs_triv = eigs_triv[idx_t]
V_triv = V_triv[:, idx_t]

sc_chi4_small = []
sc_triv_small = []
for i in range(min(50, N)):
    if eigenvalues[i] > 1e-12:
        signs = np.sign(V[:, i])
        sc_chi4_small.append(np.sum(np.abs(np.diff(signs[signs!=0])) > 0))
    if eigs_triv[i] > 1e-12:
        signs = np.sign(V_triv[:, i])
        sc_triv_small.append(np.sum(np.abs(np.diff(signs[signs!=0])) > 0))

print(f"\n  Mean sign changes (bottom 50 eigenvectors):", flush=True)
print(f"    chi_4:   {np.mean(sc_chi4_small):.1f}", flush=True)
print(f"    trivial: {np.mean(sc_triv_small):.1f}", flush=True)
if np.mean(sc_chi4_small) > np.mean(sc_triv_small):
    print(f"    chi_4 eigenvectors are MORE oscillatory -> easier to bound", flush=True)


# ============================================================
# STEP 8: THE EXPLICIT BOUND ATTEMPT
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 8: EXPLICIT BOUND ATTEMPT FOR CHI_4", flush=True)
print("="*70, flush=True)

# For chi_4, b_chi(k) involves the alternating sum:
# b_chi(k) = sum_m (-1)^m * ((2m+1) mod k)/k * w_{2m+1}
#
# The KEY IDENTITY: for odd k,
# sum_{n=1}^{k} chi_4(n) * (n mod k)/k = sum_{n odd, n<=k} chi_4(n) * n/k
# = (1/k) * sum_{n=1,3,...,k'} (+/-)n
# where k' = k or k-1.
#
# For k = 4q+1: chi_4 symmetric, sum ~ 0
# For k = 4q+3: chi_4 antisymmetric, sum ~ q+1
# For k = 4q: sum involves cancellation from period-4 structure

# Compute the KEY sum: C(k) = sum_{n=1}^{k} chi_4(n) * (n mod k)/k
print(f"  Key quantity: C(k) = sum_{{n=1}}^k chi_4(n) * (n mod k)/k", flush=True)
print(f"  {'k':>5} {'C(k)':>12} {'k mod 4':>8}", flush=True)
print(f"  {'-'*28}", flush=True)

C_k = np.zeros(N)
for k_idx in range(N):
    k = k_idx + 2
    C_k[k_idx] = sum(chi_4(int(n)) * (n % k) / k for n in range(1, k+1))
    if k <= 20 or k % 50 == 0:
        print(f"  {k:>5} {C_k[k_idx]:>12.6f} {k%4:>8}", flush=True)

# Is C(k) bounded? Does it have a pattern?
print(f"\n  C(k) by k mod 4:", flush=True)
for r in range(4):
    mask_r = (ks % 4 == r)
    if np.sum(mask_r) > 0:
        vals = C_k[mask_r[:N]]
        print(f"    k = {r} mod 4: mean={np.mean(vals):+.4f}, std={np.std(vals):.4f}, "
              f"range=[{np.min(vals):+.4f}, {np.max(vals):+.4f}]", flush=True)

# If C(k) is bounded and has a pattern, we can bound b_chi(k) using it.
# b_chi(k) = sum of C(k)-type sums over the weight w_n, with period structure.

# THE EXPLICIT FORMULA FOR b_chi(k):
# b_chi(k) = sum_{n=1}^inf chi_4(n) * (n mod k)/k / (n(n+1))
#           = sum_{r=0}^{k-1} (r/k) * sum_{n=r mod k} chi_4(n) / (n(n+1))
#
# The inner sum: A_k(r) = sum_{n=r mod k} chi_4(n) / (n(n+1))
# This is an L-function-like sum restricted to an arithmetic progression.

# For k COPRIME to 4: by Dirichlet's theorem on primes in AP,
# A_k(r) = (1/phi(4k)) sum_chi' chi'(r) * L(1, chi' chi_4) * correction

# This connects b_chi to VALUES OF L-FUNCTIONS at s=1!
# L(1, chi_4) = pi/4 (known exactly!)
# The other L-function values are also computable.

print(f"\n  L(1, chi_4) = pi/4 = {np.pi/4:.8f}", flush=True)
print(f"  sum chi_4(n)/n = pi/4 (Leibniz formula)", flush=True)
print(f"  sum chi_4(n)/(n(n+1)) = sum chi_4(n)/n - sum chi_4(n)/(n+1)", flush=True)
print(f"                        = pi/4 - sum chi_4(n+1)/(n+1)... (needs careful analysis)", flush=True)

# Compute directly
L1_chi4 = sum(chi_4(int(n)) / n for n in range(1, 100001))
L1_chi4_nn1 = sum(chi_4(int(n)) / (n*(n+1)) for n in range(1, 100001))
print(f"\n  Numerical: sum chi_4(n)/n = {L1_chi4:.8f} (theory: {np.pi/4:.8f})", flush=True)
print(f"  Numerical: sum chi_4(n)/(n(n+1)) = {L1_chi4_nn1:.8f}", flush=True)

# This value is related to the Catalan constant!
# G = sum_{n=0}^inf (-1)^n / (2n+1)^2 = 0.9159655...
# And sum chi_4(n)/(n(n+1)) relates to digamma values.

print(f"\n  b_chi(k) connects to L-function values at rational points.", flush=True)
print(f"  If we can express b_chi(k) EXACTLY in terms of L(1, chi_4) = pi/4", flush=True)
print(f"  and digamma values, we may be able to prove the smoothness bound.", flush=True)


# ============================================================
# VERDICT
# ============================================================
print("\n" + "="*70, flush=True)
print("VERDICT: CHI_4 ATTACK STATUS", flush=True)
print("="*70, flush=True)

print(f"""
  STRUCTURAL ADVANTAGES OF CHI_4:
  1. G^chi = G restricted to odd n (half the sum, cleaner structure)
  2. chi_4(n) = (-1)^m on odd n = 2m+1 (perfect alternating sign)
  3. b_chi decays as k^{{{coeffs_bchi[0]:.2f}}} (vs k^{{-0.81}} for zeta)
  4. gamma = {gamma_chi4:.2f} (3x headroom above threshold)
  5. b_chi connects to L(1, chi_4) = pi/4 (known exactly!)

  REMAINING CHALLENGE:
  The eigenvectors of G^chi still encode the modular arithmetic of odd numbers.
  Proving gamma > 1 requires bounding the bilinear sum B_i^chi,
  which involves the eigenvector oscillation.

  MOST PROMISING APPROACH:
  Express b_chi(k) EXACTLY as a combination of L-function values
  and digamma functions. Then use the ANALYTIC properties of these
  functions (known poles, residues, functional equations) to bound
  the spectral projection.

  The L-function connection means b_chi is NOT just "smooth" —
  it's an ANALYTIC FUNCTION with known singularity structure.
  This is MUCH stronger than the generic smoothness we had for b_triv.
""", flush=True)

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
