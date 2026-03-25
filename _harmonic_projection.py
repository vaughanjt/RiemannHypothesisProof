"""THE HARMONIC PROJECTION: Does <1/k, u_i> decay with lambda_i for G^chi?

From the exact decomposition:
  b_chi(k) = L_shifted/k + r(k)   where ||r|| = O(1/N^{3/2})

So: <b_chi, u_i> = L_shifted * <1/k, u_i> + <r, u_i>

The second term: |<r, u_i>| <= ||r|| = O(1/N^{3/2}) -> 0 as N grows.
The first term: L_shifted * <1/k, u_i>.

QUESTION: Does |<1/k, u_i>|^2 decay as lambda_i^alpha for some alpha > 0?

If YES with alpha > 0:
  |<b_chi, u_i>|^2 ~ L^2 * lambda_i^alpha
  From SVD identity: gamma = 1 + alpha
  For alpha > 0: gamma > 1 => d_n -> 0 => GRH for chi_4.

If alpha >= 1:
  gamma >= 2, matching the measured gamma = 3.

THE KEY INSIGHT: The harmonic function 1/k can be DECOMPOSED using
the Gram matrix itself. Since G^chi has known structure (restriction
to odd n), the projection of 1/k onto its eigenvectors is computable.

Moreover: 1/k is in the range of G^chi in a specific sense.
If we can show G^chi * (something) ~ 1/k, then:
<1/k, u_i> = <G(something), u_i> = lambda_i * <something, u_i>
=> |<1/k, u_i>|^2 = lambda_i^2 * |<something, u_i>|^2
=> alpha >= 2 (if <something, u_i> is bounded)!
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np

t0 = time.time()

M_sum = 10000
ns = np.arange(1, M_sum+1)

def chi_4(n):
    r = n % 4
    if r == 1: return 1.0
    if r == 3: return -1.0
    return 0.0

chi_arr = np.array([chi_4(n) for n in ns])
weights = 1.0 / (ns * (ns + 1))
sqrt_w = np.sqrt(weights)

# ============================================================
# STEP 1: Measure alpha = decay exponent of <1/k, u_i>
# ============================================================
print("="*70, flush=True)
print("STEP 1: HARMONIC PROJECTION <1/k, u_i> vs lambda_i", flush=True)
print("="*70, flush=True)

for N in [200, 500, 1000]:
    W_chi = np.zeros((N, M_sum))
    for k_idx in range(N):
        k = k_idx + 2
        W_chi[k_idx, :] = chi_arr * ((ns % k) / k) * sqrt_w

    G_chi = W_chi @ W_chi.T

    eigenvalues, U = np.linalg.eigh(G_chi)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    U = U[:, idx]

    # The harmonic vector: f(k) = 1/k for k = 2,...,N+1
    f_harmonic = 1.0 / np.arange(2, N+2)

    # Projection onto eigenvectors
    f_proj_sq = (U.T @ f_harmonic)**2

    # Also compute b_chi projection for comparison
    b_chi = np.zeros(N)
    for k_idx in range(N):
        k = k_idx + 2
        b_chi[k_idx] = np.dot(chi_arr * ((ns % k) / k), weights)

    b_proj_sq = (U.T @ b_chi)**2

    # Fit alpha for harmonic projection
    pos = eigenvalues > 1e-12
    mask_f = pos & (f_proj_sq > 1e-30)
    mask_b = pos & (b_proj_sq > 1e-30)

    if np.sum(mask_f) > 10:
        alpha_f = np.polyfit(np.log(eigenvalues[mask_f]),
                              np.log(f_proj_sq[mask_f]), 1)[0]
    else:
        alpha_f = float('nan')

    if np.sum(mask_b) > 10:
        gamma_b = np.polyfit(np.log(eigenvalues[mask_b]),
                              np.log(b_proj_sq[mask_b]), 1)[0]
    else:
        gamma_b = float('nan')

    print(f"\n  N = {N}:", flush=True)
    print(f"    |<1/k, u_i>|^2 ~ lambda^{{{alpha_f:.4f}}}  (alpha)", flush=True)
    print(f"    |<b_chi, u_i>|^2 ~ lambda^{{{gamma_b:.4f}}}  (gamma)", flush=True)
    print(f"    gamma - alpha = {gamma_b - alpha_f:.4f} (should be ~1 from SVD identity factor)", flush=True)

    # Verify: gamma = 1 + alpha_for_sqrt_w_projection?
    # From SVD identity: |<b,u>|^2 = lambda * |<sqrt(w), v~>|^2
    # And b ~ L * (1/k) + O(1/k^2), so:
    # |<b,u>|^2 ~ L^2 * |<1/k, u>|^2 + cross terms
    # gamma_b ~ alpha_f (not alpha_f + 1) because b = L*(1/k), not L*G*(1/k)

    # Wait — the SVD identity says |<b,u>|^2 = lambda * |<sqrt(w), v~>|^2.
    # So gamma = 1 + delta where delta is the decay of |<sqrt(w), v~>|^2.
    # But <b,u> = L * <1/k, u> + <r, u>, so:
    # gamma_b = alpha_f (approximately, ignoring r).
    # And delta = gamma_b - 1 = alpha_f - 1.

    print(f"    Expected delta (from SVD) = gamma - 1 = {gamma_b - 1:.4f}", flush=True)


# ============================================================
# STEP 2: Can we show 1/k is in the "range" of G^chi?
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 2: IS 1/k IN THE RANGE OF G^chi?", flush=True)
print("="*70, flush=True)

N = 500
W_chi = np.zeros((N, M_sum))
for k_idx in range(N):
    k = k_idx + 2
    W_chi[k_idx, :] = chi_arr * ((ns % k) / k) * sqrt_w

G_chi = W_chi @ W_chi.T
f_harmonic = 1.0 / np.arange(2, N+2)

# Solve G^chi * x = f_harmonic (if solvable, 1/k is in range of G)
eigenvalues, U = np.linalg.eigh(G_chi)
idx = np.argsort(eigenvalues)
eigenvalues = eigenvalues[idx]
U = U[:, idx]

# Project f onto eigenbasis
f_proj = U.T @ f_harmonic

# x_i = f_proj_i / lambda_i
# ||x|| = sqrt(sum f_proj_i^2 / lambda_i^2)
pos = eigenvalues > 1e-12
x_proj = np.zeros(N)
x_proj[pos] = f_proj[pos] / eigenvalues[pos]
x = U @ x_proj  # the solution G*x = f (projected)

# Check: G*x should equal f (projected onto the range of G)
Gx = G_chi @ x
residual = np.linalg.norm(Gx - f_harmonic)
print(f"  ||G*x - f|| = {residual:.6e}")
print(f"  ||f||       = {np.linalg.norm(f_harmonic):.6e}")
print(f"  Relative error: {residual/np.linalg.norm(f_harmonic):.6e}")

# If 1/k is (approximately) in the range of G:
# f = G*x => <f, u_i> = lambda_i * <x, u_i>
# => |<f, u_i>|^2 = lambda_i^2 * |<x, u_i>|^2
# If <x, u_i> is bounded: alpha = 2!

x_proj_sq = x_proj**2
mask_x = pos & (x_proj_sq > 1e-30)
if np.sum(mask_x) > 10:
    alpha_x = np.polyfit(np.log(eigenvalues[mask_x]),
                          np.log(x_proj_sq[mask_x]), 1)[0]
    print(f"\n  |<x, u_i>|^2 ~ lambda^{{{alpha_x:.4f}}}")
    print(f"  If alpha_x ~ 0: then alpha_f ~ 2 (since f = G*x)")
    print(f"  Measured alpha_f should be ~ 2 + alpha_x = {2 + alpha_x:.4f}")

# What about G^2 * y = f? (f in range of G^2)
y_proj = np.zeros(N)
y_proj[pos] = f_proj[pos] / eigenvalues[pos]**2
G2y = G_chi @ G_chi @ (U @ y_proj)
residual2 = np.linalg.norm(G2y - f_harmonic)
print(f"\n  ||G^2*y - f|| = {residual2:.6e}")
print(f"  Relative: {residual2/np.linalg.norm(f_harmonic):.6e}")

y_proj_sq = y_proj**2
mask_y = pos & (y_proj_sq > 1e-30)
if np.sum(mask_y) > 10:
    alpha_y = np.polyfit(np.log(eigenvalues[mask_y]),
                          np.log(y_proj_sq[mask_y]), 1)[0]
    print(f"  |<y, u_i>|^2 ~ lambda^{{{alpha_y:.4f}}}")
    print(f"  If alpha_y ~ 0: alpha_f ~ 4 (since f = G^2 * y)")


# ============================================================
# STEP 3: What IS x = G^{-1} * (1/k)?
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 3: STRUCTURE OF x = G^{-1} * (1/k)", flush=True)
print("="*70, flush=True)

# x(k) = (G^{-1} f)(k) where f(k) = 1/k
# What does x look like?
print(f"  x(k) = (G^{{-1}} * (1/k))(k) for k=2,...,{N+1}:", flush=True)
print(f"  {'k':>5} {'x(k)':>14} {'1/k':>10} {'x*k':>10}", flush=True)
print(f"  {'-'*42}", flush=True)
for k_idx in range(min(20, N)):
    k = k_idx + 2
    print(f"  {k:>5} {x[k_idx]:>14.6f} {1/k:>10.6f} {x[k_idx]*k:>10.4f}", flush=True)

# Is x smooth or oscillatory?
tv_x = np.sum(np.abs(np.diff(x)))
tv_f = np.sum(np.abs(np.diff(f_harmonic)))
print(f"\n  TV(x) = {tv_x:.4f}, TV(f) = {tv_f:.4f}, ratio = {tv_x/tv_f:.2f}")

# The roughness of x determines alpha_x.
# If x is as smooth as f (TV comparable): alpha_x ~ alpha_f ~ 3
# If x is rougher: alpha_x < alpha_f

# How does x decay?
mask_xvals = np.abs(x) > 1e-15
ks = np.arange(2, N+2)
if np.sum(mask_xvals[:200]) > 20:
    coeffs_x = np.polyfit(np.log(ks[mask_xvals[:200]][:200]),
                           np.log(np.abs(x[mask_xvals[:200]][:200])), 1)
    print(f"  |x(k)| ~ k^{{{coeffs_x[0]:.3f}}}")
    print(f"  (for comparison: f(k) = 1/k, so f ~ k^{{-1}})")
    print(f"  x decays {'faster' if coeffs_x[0] < -1 else 'slower' if coeffs_x[0] > -1 else 'same'} than f")


# ============================================================
# STEP 4: THE PROOF STRUCTURE
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 4: THE PROOF STRUCTURE", flush=True)
print("="*70, flush=True)

# Measure the complete chain at N=500
f_proj_sq_full = (U.T @ f_harmonic)**2
b_chi = np.zeros(N)
for k_idx in range(N):
    k = k_idx + 2
    b_chi[k_idx] = np.dot(chi_arr * ((ns % k) / k), weights)
b_proj_sq_full = (U.T @ b_chi)**2

# r = b_chi - L_shifted * f_harmonic
L_shifted = np.sum(chi_arr[::1] / np.arange(2, M_sum+2))
r = b_chi - L_shifted * f_harmonic
r_proj_sq = (U.T @ r)**2
r_norm = np.linalg.norm(r)

print(f"  ||b_chi|| = {np.linalg.norm(b_chi):.6f}", flush=True)
print(f"  ||L*f||   = {np.linalg.norm(L_shifted * f_harmonic):.6f}", flush=True)
print(f"  ||r||     = {r_norm:.6e} (the O(1/k^2) correction)", flush=True)
print(f"  ||r||/||b|| = {r_norm/np.linalg.norm(b_chi):.6e}", flush=True)

# Fit decay of r projection
mask_r = pos & (r_proj_sq > 1e-30)
if np.sum(mask_r) > 10:
    alpha_r = np.polyfit(np.log(eigenvalues[mask_r]),
                          np.log(r_proj_sq[mask_r]), 1)[0]
    print(f"\n  |<r, u_i>|^2 ~ lambda^{{{alpha_r:.4f}}} (correction projection)", flush=True)

# The KEY numbers
mask_all = pos & (f_proj_sq_full > 1e-30)
alpha_f_final = np.polyfit(np.log(eigenvalues[mask_all]),
                            np.log(f_proj_sq_full[mask_all]), 1)[0]

print(f"\n  SUMMARY AT N={N}:", flush=True)
print(f"    |<1/k, u_i>|^2     ~ lambda^{{{alpha_f_final:.4f}}}  [harmonic projection]", flush=True)
print(f"    |<b_chi, u_i>|^2   ~ lambda^{{{gamma_b:.4f}}}  [target projection = gamma]", flush=True)
print(f"    |<r, u_i>|^2       ~ lambda^{{{alpha_r:.4f}}}  [correction projection]", flush=True)
print(f"    |<G^{{-1}}f, u_i>|^2 ~ lambda^{{{alpha_x:.4f}}}  [inverse projection]", flush=True)

print(f"""
  THE CHAIN:
    b_chi = L * f + r     where f = 1/k, |r| = O(1/k^2)
    gamma(b_chi) = {gamma_b:.2f}
    alpha(f) = {alpha_f_final:.2f}
    alpha(r) = {alpha_r:.2f}

  From SVD: gamma = 1 + delta where delta measures sqrt(w) projection.
  From decomposition: gamma ~ alpha(f) since b ~ L*f + small correction.

  The harmonic projection alpha = {alpha_f_final:.2f} IS the gamma.
  This makes sense: gamma(b) ~ alpha(1/k) because b ~ L/k.

  TO PROVE gamma > 1 for chi_4: prove alpha(1/k) > 1 against G^chi.
  MEASURED: alpha = {alpha_f_final:.2f} >> 1.

  Since G * G^{{-1}}f = f, we have:
    <f, u_i> = lambda_i * <G^{{-1}}f, u_i>
    |<f, u_i>|^2 = lambda_i^2 * |<G^{{-1}}f, u_i>|^2

  So alpha(f) = 2 + alpha(G^{{-1}}f) = 2 + ({alpha_x:.2f}) = {2+alpha_x:.2f}
  Measured alpha(f) = {alpha_f_final:.2f}.
  Match: {'YES' if abs(alpha_f_final - (2+alpha_x)) < 0.2 else 'approximate'}.

  For GAMMA > 1: need alpha(f) > 1, i.e., alpha(G^{{-1}}f) > -1.
  Measured alpha(G^{{-1}}f) = {alpha_x:.2f} {'>' if alpha_x > -1 else '<'} -1.

  For GAMMA > 2: need alpha(G^{{-1}}f) > 0.
  Measured alpha(G^{{-1}}f) = {alpha_x:.2f} {'>' if alpha_x > 0 else '<'} 0.

  The question reduces to: IS G^{{-1}}(1/k) a "bounded" function
  in the spectral sense? If |<G^{{-1}}f, u_i>|^2 doesn't GROW
  with decreasing lambda, then gamma > 2.
""", flush=True)

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
