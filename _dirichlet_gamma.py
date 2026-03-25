"""DIRICHLET L-FUNCTION GAMMA TEST.

The Galois structure is simpler for individual Dirichlet L-functions.
For a character chi mod q, the analogue of the Nyman-Beurling criterion
uses twisted Beurling functions rho_{chi, alpha}(x) = chi(floor(alpha/x))...

Actually, the SIMPLEST approach: instead of changing the Beurling functions,
change the WEIGHT function to encode the character.

For chi_4 (the non-trivial character mod 4):
  chi_4(n) = 0 if n even, 1 if n=1 mod 4, -1 if n=3 mod 4
  L(s, chi_4) = sum chi_4(n) / n^s = 1 - 1/3^s + 1/5^s - 1/7^s + ...

The GRH for L(s, chi_4) states all non-trivial zeros have Re(s) = 1/2.

APPROACH: Modify the Gram matrix to use chi-weighted inner products.
  G_{jk}^chi = sum_n chi(n) * (n mod j)/j * (n mod k)/k * w_n
  b_k^chi = sum_n chi(n) * (n mod k)/k * w_n

The gamma for G^chi measures the GRH for L(s, chi).

KEY QUESTION: Is gamma for chi_4 LARGER than gamma for zeta?
The simpler Galois structure (only 2 classes: 1 mod 4 and 3 mod 4)
might make the bound easier to prove.
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np
from math import gcd

t0 = time.time()

M_sum = 10000
ns = np.arange(1, M_sum+1)

# ============================================================
# Define Dirichlet characters
# ============================================================

def chi_4(n):
    """Non-trivial character mod 4: chi_4(1)=1, chi_4(3)=-1, chi_4(even)=0."""
    r = n % 4
    if r == 1: return 1
    if r == 3: return -1
    return 0

def chi_3(n):
    """Non-trivial character mod 3: chi_3(1)=1, chi_3(2)=-1, chi_3(3)=0."""
    r = n % 3
    if r == 1: return 1
    if r == 2: return -1
    return 0

def chi_trivial(n):
    """Trivial character (= zeta function case)."""
    return 1

characters = {
    'trivial (zeta)': chi_trivial,
    'chi_4 (L mod 4)': chi_4,
    'chi_3 (L mod 3)': chi_3,
}

# ============================================================
# For each character: build Gram matrix and measure gamma
# ============================================================
print("="*70, flush=True)
print("DIRICHLET L-FUNCTION GAMMA COMPARISON", flush=True)
print("="*70, flush=True)

test_Ns = [100, 200, 300, 500, 750, 1000]

for chi_name, chi_func in characters.items():
    print(f"\n  === {chi_name} ===", flush=True)

    # Precompute chi values
    chi_vals = np.array([chi_func(int(n)) for n in ns], dtype=float)

    # Weight function: w_n * chi(n)^2 for the Gram matrix
    # Actually, for L-functions we want:
    # G_{jk} = sum_n |chi(n)|^2 * (n mod j)/j * (n mod k)/k / (n(n+1))
    # b_k = sum_n chi(n) * (n mod k)/k / (n(n+1))
    #
    # Wait, this isn't quite right. The Nyman-Beurling criterion for L(s,chi)
    # uses a DIFFERENT set of approximants. Let me use the simplest version:
    # just weight the existing Gram matrix by chi.

    # The CORRECT approach for L(s, chi):
    # The analogue of d_n^2 = inf || chi_{character} - sum c_k rho_k ||^2
    # where the target function changes based on the character.
    # For simplicity, let's just look at the chi-WEIGHTED Gram matrix:

    # Method: Weight the basis functions by chi(n)
    # W^chi_{k,n} = chi(n) * (n mod (k+2))/(k+2) * sqrt(w_n)
    # G^chi = W^chi @ (W^chi)^T
    # b^chi_k = sum_n chi(n) * (n mod (k+2))/(k+2) * w_n

    weights_chi = np.array([1.0/(n*(n+1)) for n in ns])
    sqrt_w_chi = np.sqrt(weights_chi)

    print(f"  {'N':>6} {'gamma':>8} {'gamma_sm':>9} {'d_n^2':>12} {'lmin':>12}", flush=True)
    print(f"  {'-'*50}", flush=True)

    for N in test_Ns:
        # Build chi-weighted basis matrix
        W_chi = np.zeros((N, M_sum))
        for k_idx in range(N):
            k = k_idx + 2
            W_chi[k_idx, :] = chi_vals * ((ns % k) / k) * sqrt_w_chi

        G_chi = W_chi @ W_chi.T

        # b vector: chi-weighted target
        b_chi = np.zeros(N)
        for k_idx in range(N):
            k = k_idx + 2
            b_chi[k_idx] = np.dot(chi_vals * ((ns % k) / k), weights_chi)

        # Check if G is trivially zero (for non-trivial characters, many entries vanish)
        if np.max(np.abs(G_chi)) < 1e-15:
            print(f"  {N:>6} {'(zero)':>8}", flush=True)
            continue

        # Eigendecomposition
        eigenvalues, V = np.linalg.eigh(G_chi)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        V = V[:, idx]

        # Filter positive eigenvalues (chi-weighted G may not be positive definite
        # if chi has negative values... actually G = W^T W so it IS positive semi-def)
        pos_mask = eigenvalues > 1e-12

        if np.sum(pos_mask) < 10:
            print(f"  {N:>6} {'(rank<10)':>8}", flush=True)
            continue

        b_proj_sq = (V.T @ b_chi)**2
        lmin = eigenvalues[pos_mask][0]

        # Fit gamma
        mask_g = pos_mask & (b_proj_sq > 1e-30)
        if np.sum(mask_g) > 10:
            coeffs = np.polyfit(np.log(eigenvalues[mask_g]),
                                np.log(b_proj_sq[mask_g]), 1)
            gamma = coeffs[0]
        else:
            gamma = float('nan')

        # Fit gamma on small eigenvalues
        med = np.median(eigenvalues[pos_mask])
        mask_s = pos_mask & (eigenvalues < med) & (b_proj_sq > 1e-30)
        if np.sum(mask_s) > 5:
            gamma_sm = np.polyfit(np.log(eigenvalues[mask_s]),
                                   np.log(b_proj_sq[mask_s]), 1)[0]
        else:
            gamma_sm = float('nan')

        # d_n^2
        terms = b_proj_sq[pos_mask] / eigenvalues[pos_mask]
        norm_b = np.dot(b_chi, b_chi)
        if norm_b > 1e-15:
            d_n_sq = max(1.0 - np.sum(terms) / norm_b, 0) * norm_b
        else:
            d_n_sq = 0

        print(f"  {N:>6} {gamma:>8.4f} {gamma_sm:>9.4f} {d_n_sq:>12.4e} {lmin:>12.4e}", flush=True)


# ============================================================
# COMPARISON: Does simpler Galois structure give higher gamma?
# ============================================================
print("\n" + "="*70, flush=True)
print("COMPARISON ACROSS CHARACTERS", flush=True)
print("="*70, flush=True)

# Recompute at N=500 for all characters and compare
N_comp = 500
print(f"\n  At N = {N_comp}:", flush=True)

results_comp = {}
for chi_name, chi_func in characters.items():
    chi_vals = np.array([chi_func(int(n)) for n in ns], dtype=float)
    weights_chi = np.array([1.0/(n*(n+1)) for n in ns])
    sqrt_w_chi = np.sqrt(weights_chi)

    W_chi = np.zeros((N_comp, M_sum))
    for k_idx in range(N_comp):
        k = k_idx + 2
        W_chi[k_idx, :] = chi_vals * ((ns % k) / k) * sqrt_w_chi

    G_chi = W_chi @ W_chi.T
    b_chi = np.zeros(N_comp)
    for k_idx in range(N_comp):
        k = k_idx + 2
        b_chi[k_idx] = np.dot(chi_vals * ((ns % k) / k), weights_chi)

    eigenvalues, V = np.linalg.eigh(G_chi)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]

    pos_mask = eigenvalues > 1e-12
    b_proj_sq = (V.T @ b_chi)**2

    mask_g = pos_mask & (b_proj_sq > 1e-30)
    if np.sum(mask_g) > 10:
        gamma = np.polyfit(np.log(eigenvalues[mask_g]),
                            np.log(b_proj_sq[mask_g]), 1)[0]
    else:
        gamma = float('nan')

    # Also verify the SVD identity for this character
    U_svd, sigma_svd, Vt_svd = np.linalg.svd(W_chi, full_matrices=False)
    sigma_sorted = sigma_svd[::-1]
    Vt_sorted = Vt_svd[::-1, :]
    sw_proj_sq = (Vt_sorted @ sqrt_w_chi)**2
    lambda_sorted = sigma_sorted**2

    # delta = exponent of |<sqrt(w), v_tilde>|^2 vs lambda
    mask_d = (lambda_sorted > 1e-12) & (sw_proj_sq > 1e-30)
    if np.sum(mask_d) > 10:
        delta = np.polyfit(np.log(lambda_sorted[mask_d]),
                            np.log(sw_proj_sq[mask_d]), 1)[0]
    else:
        delta = float('nan')

    # Number of nonzero eigenvalues (effective rank)
    eff_rank = np.sum(eigenvalues > 1e-10)

    results_comp[chi_name] = {
        'gamma': gamma, 'delta': delta, 'eff_rank': eff_rank,
        'lmin': eigenvalues[pos_mask][0] if np.any(pos_mask) else 0
    }

    print(f"  {chi_name:>25}: gamma={gamma:.4f}, delta={delta:.4f}, "
          f"rank={eff_rank}, lmin={eigenvalues[pos_mask][0] if np.any(pos_mask) else 0:.4e}", flush=True)

# Which character has the highest gamma?
best_chi = max(results_comp.items(), key=lambda x: x[1]['gamma'] if not np.isnan(x[1]['gamma']) else -1)
print(f"\n  HIGHEST gamma: {best_chi[0]} with gamma = {best_chi[1]['gamma']:.4f}", flush=True)

# Does simpler Galois structure give higher gamma?
print(f"""
  GALOIS COMPLEXITY:
    trivial (zeta): full Galois group, all primes relevant
    chi_4:          Z/2 structure, only odd primes, 2 classes
    chi_3:          Z/2 structure, only primes != 3, 2 classes

  If simpler Galois => higher gamma, we might be able to PROVE
  gamma > 1 for a specific L-function first (easier than full RH).
""", flush=True)

# Also check: Abel exponents for each character
print("="*70, flush=True)
print("ABEL EXPONENTS BY CHARACTER", flush=True)
print("="*70, flush=True)

N_abel = 300
for chi_name, chi_func in characters.items():
    chi_vals_a = np.array([chi_func(int(n)) for n in ns], dtype=float)
    W_a = np.zeros((N_abel, M_sum))
    for k_idx in range(N_abel):
        k = k_idx + 2
        W_a[k_idx, :] = chi_vals_a * ((ns % k) / k) * sqrt_w_chi

    G_a = W_a @ W_a.T
    eigenvalues_a, V_a = np.linalg.eigh(G_a)
    idx_a = np.argsort(eigenvalues_a)
    eigenvalues_a = eigenvalues_a[idx_a]
    V_a = V_a[:, idx_a]

    pos = eigenvalues_a > 1e-12
    if np.sum(pos) < 20:
        print(f"  {chi_name}: insufficient rank", flush=True)
        continue

    # Higher-order partial sum exponents
    current = V_a.copy()
    betas = []
    for order in range(1, 4):
        current = np.cumsum(current, axis=0)
        max_current = np.max(np.abs(current), axis=0)
        mask_c = pos & (max_current > 1e-12)
        if np.sum(mask_c) > 10:
            coeffs_c = np.polyfit(np.log(eigenvalues_a[mask_c]),
                                   np.log(max_current[mask_c]), 1)
            betas.append(coeffs_c[0])
        else:
            betas.append(float('nan'))

    while len(betas) < 3:
        betas.append(float('nan'))

    print(f"  {chi_name:>25}: beta_1={betas[0]:.3f}, beta_2={betas[1]:.3f}, "
          f"beta_3={betas[2]:.3f}  |  2*b2={2*betas[1]:.3f}", flush=True)

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
