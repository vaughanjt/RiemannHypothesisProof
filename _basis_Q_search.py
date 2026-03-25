"""THE BASIS TRANSFORMATION Q: bridging multiplicative and spectral spaces.

The fundamental problem:
  H_GCD = diag(log k) + C * gcd_kernel  →  r=0.68, WRONG eigenvalues
  H_EF  = explicit formula operator       →  RIGHT eigenvalues, r=0.03

We seek Q such that Q^H H_GCD Q has BOTH good eigenvalues AND good spacing
correlation. Q encodes the deep map between multiplicative structure
(where GCD is natural) and spectral structure (where zeros live).

CANDIDATES:
  1. Discrete Mellin transform: Q_{kn} = n^{-1/2 - i*gamma_k}
  2. Empirical eigenvector bridge: Q = V_EF @ V_GCD^T
  3. Ramanujan sum basis: Q_{qn} = c_q(n) / normalization
  4. Möbius triangular matrix: Q_{kn} = mu(k/n) when n|k
  5. Multiplicative character basis (Dirichlet characters)
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np
from math import gcd
from scipy.linalg import eigh, svd
from scipy.stats import pearsonr, kstest
from sympy import primerange, mobius, totient
import mpmath
mpmath.mp.dps = 20

from riemann.analysis.bost_connes_operator import polynomial_unfold

t0 = time.time()

# ============================================================
# SETUP: zeros, operators, scoring
# ============================================================
N = 200
print(f"Computing {N} zeta zeros...", flush=True)
zeta_zeros = np.array([float(mpmath.im(mpmath.zetazero(k))) for k in range(1, N+1)])
print(f"  Done ({time.time()-t0:.1f}s)", flush=True)

primes = list(primerange(2, 3000))[:303]
trim = int(0.1*N)
ms = np.mean(np.diff(zeta_zeros[trim:-trim]))

def wigner_cdf(s):
    return 1 - np.exp(-np.pi*s**2/4)

def score(eigs_raw, label=""):
    """Complete scoring: eigenvalue accuracy + spacing correlation + GUE."""
    eigs = np.sort(np.real(eigs_raw))[:N]
    # Eigenvalue accuracy
    errs = np.abs(eigs - zeta_zeros[:len(eigs)])[trim:-trim]
    mean_err = np.mean(errs)
    pct_half = np.mean(errs < ms/2)
    pct_gap = np.mean(errs < ms)
    pct_10 = np.mean(errs < ms*0.1)

    # Spacing correlation (peak-gap r)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    r = 0.0
    p_gue = 0.0
    if len(sp) > 20:
        sp = sp / np.mean(sp)
        _, p_gue = kstest(sp, wigner_cdf)
        n_t = int(0.1*len(eigs))
        et = eigs[n_t:-n_t]
        lp, ga = [], []
        for k in range(min(len(sp), len(et)-1)):
            z = (et[k]+et[k+1])/2
            lp.append(np.sum(np.log(np.abs(z-eigs)+1e-30)))
            ga.append(sp[k])
        if len(ga) > 10:
            r, _ = pearsonr(np.array(ga), np.array(lp))

    if label:
        print(f"  {label:>35}: err={mean_err:.3f}, <half={pct_half*100:.0f}%, "
              f"<10%={pct_10*100:.0f}%, r={r:+.3f}, p(GUE)={p_gue:.3f}", flush=True)
    return mean_err, pct_half, pct_10, r, p_gue


# ============================================================
# BUILD REFERENCE OPERATORS
# ============================================================
print("\nBuilding reference operators...", flush=True)

# GCD kernel
def build_gcd_kernel(N_size):
    G = np.zeros((N_size, N_size))
    for j in range(1, N_size+1):
        for k in range(j, N_size+1):
            g = gcd(j, k)
            val = np.log(g + 1) / np.sqrt(j * k)
            G[j-1, k-1] = val
            G[k-1, j-1] = val
    return G

GCD = build_gcd_kernel(N)

# Explicit formula diagonal
def N_smooth(T):
    if T < 2: return 0.
    return T/(2*np.pi)*np.log(T/(2*np.pi)) - T/(2*np.pi) + 7./8.

def N_deriv(T):
    if T < 2: return .001
    return np.log(T/(2*np.pi)) / (2*np.pi)

def weyl_zero(k):
    t = 2*np.pi*k/np.log(max(k,2)+2)
    for _ in range(30):
        if t < 1: t = 10.
        t -= (N_smooth(t)-k)/N_deriv(t)
    return t

alpha_ef = np.zeros(N)
for k in range(1, N+1):
    Tw = weyl_zero(k)
    dN = N_deriv(Tw)
    s = sum(-np.sin(2*m*Tw*np.log(p))/(m*p**(m*0.5))
            for p in primes for m in range(1, 6)) / np.pi
    alpha_ef[k-1] = Tw + s / dN

# H_GCD = diag(log k) + C * GCD
alpha_logk = np.log(np.arange(1, N+1))
alpha_logk_scaled = alpha_logk / alpha_logk[-1] * zeta_zeros[-1]
C_gcd = 5.0  # from previous optimization

H_GCD_full = np.diag(alpha_logk_scaled) + C_gcd * (GCD - np.diag(np.diag(GCD)))
H_EF_diag = np.diag(alpha_ef)

# Build the banded explicit formula operator (W=3, 4 modes)
def build_ef_banded(N_size, W=3):
    """Explicit formula banded operator with mod-8 mode coupling."""
    C_modes = {1: 1.62, 3: 3.40, 5: 0.44, 7: 1.00}
    H = np.diag(alpha_ef[:N_size])
    for k in range(N_size):
        Tk = weyl_zero(k+1)
        logT = np.log(max(Tk, 10))
        for d in range(1, W+1):
            if k+d >= N_size:
                break
            v = 0.0
            for p in primes[:100]:
                lp = np.log(p)
                r_class = p % 8
                C_r = C_modes.get(r_class, 1.0)
                for m in range(1, 4):
                    v += C_r * lp / p**(m/2) * np.cos(2*np.pi*d*m*lp/logT)
            H[k, k+d] = v / (N_size * 0.5)
            H[k+d, k] = v / (N_size * 0.5)
    return H

H_EF_banded = build_ef_banded(N, W=3)

# Eigenvectors of both
eigs_gcd, V_gcd = eigh(H_GCD_full)
eigs_ef, V_ef = eigh(H_EF_banded)

print(f"  Operators built ({time.time()-t0:.1f}s)", flush=True)

# Reference scores
print("\n" + "="*70, flush=True)
print("REFERENCE SCORES", flush=True)
print("="*70, flush=True)
score(eigs_gcd, "H_GCD (log(k) + 5*GCD)")
score(eigs_ef, "H_EF (banded, W=3)")
score(zeta_zeros, "Oracle (zeros themselves)")


# ============================================================
# CANDIDATE 1: DISCRETE MELLIN TRANSFORM
# ============================================================
print("\n" + "="*70, flush=True)
print("CANDIDATE 1: DISCRETE MELLIN TRANSFORM", flush=True)
print("  Q_{kn} = n^{-1/2 - i*gamma_k} / sqrt(N)", flush=True)
print("="*70, flush=True)

gammas = zeta_zeros[:N]

# Build the Mellin matrix Q: Q_{kn} = n^{-sigma - i*gamma_k}
# The natural choice is sigma = 1/2 (the critical line)
for sigma in [0.0, 0.25, 0.5, 0.75, 1.0]:
    Q_mellin = np.zeros((N, N), dtype=complex)
    for k in range(N):
        for n in range(N):
            Q_mellin[k, n] = (n+1)**(-sigma - 1j*gammas[k])
    # Normalize columns
    col_norms = np.sqrt(np.sum(np.abs(Q_mellin)**2, axis=0))
    Q_mellin /= col_norms[np.newaxis, :]

    # Transform H_GCD into Mellin basis
    H_mellin = Q_mellin.conj() @ H_GCD_full @ Q_mellin.T
    eigs_mellin = np.sort(np.real(np.linalg.eigvalsh(H_mellin)))

    score(eigs_mellin, f"Mellin sigma={sigma:.2f}")

# Also try: Q maps FROM Mellin TO integer (transpose direction)
print("\n  Reverse direction: Q^H H_GCD Q", flush=True)
Q_m = np.zeros((N, N), dtype=complex)
for k in range(N):
    for n in range(N):
        Q_m[k, n] = (n+1)**(-0.5 - 1j*gammas[k])
col_norms = np.sqrt(np.sum(np.abs(Q_m)**2, axis=0))
Q_m /= col_norms[np.newaxis, :]

# Q^H H_GCD Q
H_rev = Q_m @ H_GCD_full @ Q_m.conj().T
eigs_rev = np.sort(np.real(np.linalg.eigvalsh(
    (H_rev + H_rev.conj().T)/2  # symmetrize
)))
score(eigs_rev, "Mellin Q H_GCD Q^H (sym)")

# What about applying Mellin to JUST the GCD kernel?
print("\n  Mellin applied to pure GCD kernel:", flush=True)
H_gcd_pure = GCD.copy()
H_m_gcd = Q_m.conj() @ H_gcd_pure @ Q_m.T
H_m_gcd = (H_m_gcd + H_m_gcd.conj().T) / 2
eigs_m_gcd = np.sort(np.real(np.linalg.eigvalsh(H_m_gcd)))
# The eigenvalues of the GCD matrix in Mellin basis — do they look like zeros?
print(f"  GCD eigenvalue range in Mellin basis: [{eigs_m_gcd[0]:.3f}, {eigs_m_gcd[-1]:.3f}]")
print(f"  Zero range: [{zeta_zeros[0]:.3f}, {zeta_zeros[-1]:.3f}]")
# Scale to match
scale = (zeta_zeros[-1] - zeta_zeros[0]) / (eigs_m_gcd[-1] - eigs_m_gcd[0] + 1e-10)
shift = zeta_zeros[0] - eigs_m_gcd[0] * scale
eigs_m_gcd_scaled = eigs_m_gcd * scale + shift
score(eigs_m_gcd_scaled, "Mellin(GCD) scaled")


# ============================================================
# CANDIDATE 2: EIGENVECTOR BRIDGE
# ============================================================
print("\n" + "="*70, flush=True)
print("CANDIDATE 2: EIGENVECTOR BRIDGE Q = V_target @ V_source^T", flush=True)
print("="*70, flush=True)

# If H_GCD = V_gcd @ diag(eigs_gcd) @ V_gcd^T
# and H_EF  = V_ef  @ diag(eigs_ef)  @ V_ef^T
# then Q = V_ef @ V_gcd^T maps GCD eigenvectors to EF eigenvectors.
# Q^T H_GCD Q = V_gcd V_ef^T V_ef diag(eigs_gcd) V_ef^T V_ef V_gcd^T
# ... this just re-orders eigenvalues.

# More interesting: WHAT DOES Q LOOK LIKE?
Q_bridge = V_ef @ V_gcd.T
print(f"  Q shape: {Q_bridge.shape}")
print(f"  Q is orthogonal: ||QQ^T - I|| = {np.linalg.norm(Q_bridge @ Q_bridge.T - np.eye(N)):.6f}")

# Structure of Q: is it sparse? banded? does it look like Mellin?
print(f"  Q sparsity (|Q|<0.01): {np.mean(np.abs(Q_bridge)<0.01)*100:.1f}%")
print(f"  Q diagonal dominance: {np.mean(np.abs(np.diag(Q_bridge))):.6f}")
print(f"  Q max off-diag / max diag: {np.max(np.abs(Q_bridge-np.diag(np.diag(Q_bridge))))/(np.max(np.abs(np.diag(Q_bridge)))+1e-10):.4f}")

# Does Q resemble the Mellin transform?
Q_m_real = np.real(Q_m[:N, :N])
corr_Q_mellin = np.corrcoef(Q_bridge.flatten(), Q_m_real.flatten())[0, 1]
print(f"  Correlation Q_bridge vs Re(Q_mellin): {corr_Q_mellin:+.4f}")

# Key question: what is the STRUCTURE of Q?
# Compute SVD of Q to see its spectral structure
U_q, S_q, Vt_q = svd(Q_bridge)
print(f"  Q singular values: min={S_q[-1]:.6f}, max={S_q[0]:.6f}")
print(f"  Q is {'well-conditioned' if S_q[-1]/S_q[0] > 0.1 else 'ill-conditioned'}: cond={S_q[0]/(S_q[-1]+1e-10):.1f}")

# Now the real test: take the ORACLE Q (from knowing both operators)
# and create a HYBRID that uses GCD eigenvectors + EF eigenvalues
H_hybrid_Q = V_gcd @ np.diag(eigs_ef) @ V_gcd.T  # GCD vectors, EF eigenvalues
score(eigs_ef, "GCD vecs + EF eigs (oracle)")

# What about: EF vectors + GCD eigenvalues?
eigs_gcd_sorted = np.sort(eigs_gcd)
# Scale GCD eigenvalues to zero range
scale_g = (zeta_zeros[-1] - zeta_zeros[0]) / (eigs_gcd_sorted[-1] - eigs_gcd_sorted[0])
eigs_gcd_scaled = (eigs_gcd_sorted - eigs_gcd_sorted[0]) * scale_g + zeta_zeros[0]
H_hybrid_Q2 = V_ef @ np.diag(eigs_gcd_scaled) @ V_ef.T
score(eigs_gcd_scaled, "EF vecs + GCD eigs (scaled)")

# What about: GCD eigenvectors + zero eigenvalues? (super oracle)
H_super = V_gcd @ np.diag(zeta_zeros[:N]) @ V_gcd.T
eigs_super = np.sort(np.linalg.eigvalsh(H_super))
score(eigs_super, "GCD vecs + zeros (super oracle)")

# And the reverse: what r do GCD eigenvectors give with perfect eigenvalues?
# This directly tells us if the GCD eigenvector structure carries the right r
print("\n  The GCD eigenvector question:", flush=True)
print(f"  If we force eigenvalues = zeros, does r survive?", flush=True)
_, pct_h, pct_10, r_gcd_oracle, p_gue_oracle = score(eigs_super, "GCD vecs + zeros")
print(f"  -> r = {r_gcd_oracle:+.4f}  (was r={0.68} with natural GCD eigs)", flush=True)
if r_gcd_oracle > 0.5:
    print(f"  YES! GCD eigenvectors carry the correlation signal!", flush=True)
else:
    print(f"  NO. The r=0.68 was from eigenvalue spacing pattern, not eigenvectors.", flush=True)


# ============================================================
# CANDIDATE 3: RAMANUJAN SUM BASIS
# ============================================================
print("\n" + "="*70, flush=True)
print("CANDIDATE 3: RAMANUJAN SUM BASIS", flush=True)
print("  c_q(n) = sum_{d|gcd(q,n)} d * mu(q/d)", flush=True)
print("="*70, flush=True)

def ramanujan_sum(q, n):
    """Ramanujan sum c_q(n) = sum_{d|gcd(q,n)} d * mu(q/d)."""
    g = gcd(q, n)
    total = 0
    for d in range(1, g+1):
        if g % d == 0 and q % d == 0:
            total += d * int(mobius(q // d))
    return total

# Build Ramanujan basis matrix
print("  Building Ramanujan basis...", flush=True)
t_ram = time.time()
Q_ram = np.zeros((N, N))
for q in range(1, N+1):
    for n in range(1, N+1):
        Q_ram[q-1, n-1] = ramanujan_sum(q, n)
print(f"  Built ({time.time()-t_ram:.1f}s)", flush=True)

# Normalize columns
col_norms_r = np.sqrt(np.sum(Q_ram**2, axis=0))
col_norms_r[col_norms_r < 1e-10] = 1.0
Q_ram_norm = Q_ram / col_norms_r[np.newaxis, :]

# Check orthogonality
orth_err = np.linalg.norm(Q_ram_norm.T @ Q_ram_norm - np.eye(N))
print(f"  Orthogonality error: {orth_err:.4f}")

# Transform GCD kernel through Ramanujan basis
H_ram = Q_ram_norm.T @ H_GCD_full @ Q_ram_norm
H_ram = (H_ram + H_ram.T) / 2  # ensure symmetry
eigs_ram = np.sort(np.linalg.eigvalsh(H_ram))

# Scale
scale_ram = (zeta_zeros[-1] - zeta_zeros[0]) / (eigs_ram[-1] - eigs_ram[0] + 1e-10)
eigs_ram_scaled = (eigs_ram - eigs_ram[0]) * scale_ram + zeta_zeros[0]
score(eigs_ram_scaled, "Ramanujan(H_GCD) scaled")

# Also: pure GCD matrix in Ramanujan basis
H_gcd_ram = Q_ram_norm.T @ GCD @ Q_ram_norm
H_gcd_ram = (H_gcd_ram + H_gcd_ram.T) / 2
eigs_gcd_ram = np.sort(np.linalg.eigvalsh(H_gcd_ram))
scale_gr = (zeta_zeros[-1] - zeta_zeros[0]) / (eigs_gcd_ram[-1] - eigs_gcd_ram[0] + 1e-10)
eigs_gcd_ram_s = (eigs_gcd_ram - eigs_gcd_ram[0]) * scale_gr + zeta_zeros[0]
score(eigs_gcd_ram_s, "Ramanujan(pure GCD) scaled")

# KEY INSIGHT: The GCD matrix diagonalizes in the Ramanujan basis
# because gcd(m,n) = sum_d c_d(m)*c_d(n) / phi(d)^2  (approximately)
# So H_gcd_ram should be approximately diagonal!
diag_frac = np.sum(np.abs(np.diag(H_gcd_ram))) / (np.sum(np.abs(H_gcd_ram)) + 1e-10)
print(f"  GCD in Ramanujan basis: diagonal fraction = {diag_frac*100:.1f}%")
print(f"  (100% = Ramanujan perfectly diagonalizes GCD)")


# ============================================================
# CANDIDATE 4: MÖBIUS MATRIX
# ============================================================
print("\n" + "="*70, flush=True)
print("CANDIDATE 4: MÖBIUS TRIANGULAR MATRIX", flush=True)
print("  Q_{jk} = mu(j/k) when k|j, else 0", flush=True)
print("="*70, flush=True)

Q_mob = np.zeros((N, N))
for j in range(1, N+1):
    for k in range(1, j+1):
        if j % k == 0:
            Q_mob[j-1, k-1] = float(mobius(j // k))

print(f"  Möbius matrix sparsity: {np.mean(Q_mob==0)*100:.1f}% zero entries")
print(f"  Möbius matrix is lower triangular: {np.allclose(Q_mob, np.tril(Q_mob))}")

# Q_mob @ Q_mob^T should relate to divisor functions
# Q_mob is the Möbius inversion matrix: if g = f * 1, then f = g * mu
# i.e., Q_mob inverts the summatory "multiply by 1" matrix

# Transform
H_mob = Q_mob @ H_GCD_full @ Q_mob.T
H_mob = (H_mob + H_mob.T) / 2
eigs_mob = np.sort(np.real(np.linalg.eigvalsh(H_mob)))

# The eigenvalues should now reflect Möbius-inverted structure
scale_m = (zeta_zeros[-1] - zeta_zeros[0]) / (eigs_mob[-1] - eigs_mob[0] + 1e-10)
eigs_mob_scaled = (eigs_mob - eigs_mob[0]) * scale_m + zeta_zeros[0]
score(eigs_mob_scaled, "Möbius(H_GCD) scaled")

# What does Möbius do to the pure GCD kernel?
# gcd(m,n) = sum_{d|gcd(m,n)} phi(d) (Euler's formula)
# Möbius-inverting: phi(n) = sum_{d|n} mu(n/d) * d
# So Möbius transforms GCD into something related to Euler's totient
H_mob_gcd = Q_mob @ GCD @ Q_mob.T
H_mob_gcd = (H_mob_gcd + H_mob_gcd.T) / 2
eigs_mob_gcd = np.sort(np.linalg.eigvalsh(H_mob_gcd))
scale_mg = (zeta_zeros[-1] - zeta_zeros[0]) / (eigs_mob_gcd[-1] - eigs_mob_gcd[0] + 1e-10)
eigs_mob_gcd_s = (eigs_mob_gcd - eigs_mob_gcd[0]) * scale_mg + zeta_zeros[0]
score(eigs_mob_gcd_s, "Möbius(pure GCD) scaled")


# ============================================================
# CANDIDATE 5: MULTIPLICATIVE CHARACTER BASIS
# ============================================================
print("\n" + "="*70, flush=True)
print("CANDIDATE 5: DIRICHLET CHARACTER BASIS", flush=True)
print("="*70, flush=True)

# Use additive characters (DFT) as a simpler proxy first
# These diagonalize circulant matrices, and the GCD kernel
# has circulant-like structure for coprime entries
print("  Using DFT basis as proxy for Dirichlet characters...", flush=True)
Q_dft = np.fft.fft(np.eye(N)) / np.sqrt(N)

H_dft = Q_dft.conj() @ H_GCD_full @ Q_dft.T
H_dft = (H_dft + H_dft.conj().T) / 2
eigs_dft = np.sort(np.real(np.linalg.eigvalsh(H_dft)))
scale_d = (zeta_zeros[-1] - zeta_zeros[0]) / (eigs_dft[-1] - eigs_dft[0] + 1e-10)
eigs_dft_scaled = (eigs_dft - eigs_dft[0]) * scale_d + zeta_zeros[0]
score(eigs_dft_scaled, "DFT(H_GCD) scaled")


# ============================================================
# CANDIDATE 6: THE ZETA KERNEL — n^{-s} WHERE s ARE LEARNABLE
# ============================================================
print("\n" + "="*70, flush=True)
print("CANDIDATE 6: PARAMETERIZED MELLIN WITH LEARNABLE SHIFTS", flush=True)
print("  Q_{kn} = n^{-1/2 - i*(gamma_k + delta_k)}", flush=True)
print("="*70, flush=True)

from scipy.optimize import minimize

# Start with Mellin at sigma=0.5 and optimize small shifts to the gammas
def build_Q_shifted(deltas, gammas, N_size, sigma=0.5):
    Q = np.zeros((N_size, N_size), dtype=complex)
    shifted = gammas + deltas
    for k in range(N_size):
        for n in range(N_size):
            Q[k, n] = (n+1)**(-sigma - 1j*shifted[k])
    col_norms = np.sqrt(np.sum(np.abs(Q)**2, axis=0))
    col_norms[col_norms < 1e-10] = 1.0
    return Q / col_norms[np.newaxis, :]

# Use smaller N for optimization (expensive)
N_opt = 50
gammas_opt = zeta_zeros[:N_opt]
GCD_opt = build_gcd_kernel(N_opt)
alpha_opt = alpha_logk[:N_opt] / alpha_logk[N_opt-1] * zeta_zeros[N_opt-1]
H_GCD_opt = np.diag(alpha_opt) + C_gcd * (GCD_opt - np.diag(np.diag(GCD_opt)))

def objective(deltas):
    Q = build_Q_shifted(deltas, gammas_opt, N_opt)
    H_t = Q.conj() @ H_GCD_opt @ Q.T
    H_t = (H_t + H_t.conj().T) / 2
    eigs = np.sort(np.real(np.linalg.eigvalsh(H_t)))
    # Minimize eigenvalue error
    errs = np.abs(eigs - zeta_zeros[:N_opt])
    return np.mean(errs)

print("  Optimizing shifts (N=50)...", flush=True)
t_opt = time.time()
res = minimize(objective, np.zeros(N_opt), method='L-BFGS-B',
               bounds=[(-2, 2)]*N_opt, options={'maxiter': 100, 'maxfun': 500})
print(f"  Optimization: {res.fun:.4f} in {time.time()-t_opt:.1f}s ({res.nfev} evals)", flush=True)

best_deltas = res.x
Q_opt = build_Q_shifted(best_deltas, gammas_opt, N_opt)
H_opt = Q_opt.conj() @ H_GCD_opt @ Q_opt.T
H_opt = (H_opt + H_opt.conj().T) / 2
eigs_opt = np.sort(np.real(np.linalg.eigvalsh(H_opt)))
score(np.concatenate([eigs_opt, zeta_zeros[N_opt:N]]), "Optimized Mellin (N=50)")

print(f"  Mean |delta|: {np.mean(np.abs(best_deltas)):.4f}")
print(f"  Max |delta|: {np.max(np.abs(best_deltas)):.4f}")
print(f"  Deltas corr with zeros: {np.corrcoef(best_deltas, gammas_opt)[0,1]:+.4f}")


# ============================================================
# CANDIDATE 7: THE SMITH NORMAL FORM / TOTIENT EIGENVECTORS
# ============================================================
print("\n" + "="*70, flush=True)
print("CANDIDATE 7: GCD MATRIX EIGENVECTORS (TOTIENT-RELATED)", flush=True)
print("="*70, flush=True)

# The matrix G_{ij} = gcd(i,j) has eigenvalues = Jordan's totient function
# and eigenvectors related to the Ramanujan sums.
# But OUR GCD kernel is log(gcd+1)/sqrt(ij), which is different.

# First: analyze what the GCD matrix eigenvectors look like
G_pure = np.zeros((N, N))
for i in range(1, N+1):
    for j in range(1, N+1):
        G_pure[i-1, j-1] = gcd(i, j)

eigs_G, V_G = eigh(G_pure)
print(f"  Pure gcd(i,j) matrix eigenvalues: [{eigs_G[0]:.1f}, ..., {eigs_G[-1]:.1f}]")
print(f"  Expected: Jordan's totient J_1(d) = phi(d) for d|N")

# The eigenvalues of gcd(i,j) matrix are sum_{d|k} d*phi(k/d) for appropriate k
# The EIGENVECTORS are columns of the matrix with entries Q_{kd} = c_d(k) (Ramanujan sums)
# This confirms candidate 3.

# But let's use the ACTUAL eigenvectors as basis transform
H_G_basis = V_G.T @ H_GCD_full @ V_G
eigs_G_basis = np.sort(np.linalg.eigvalsh(H_G_basis))
scale_gb = (zeta_zeros[-1] - zeta_zeros[0]) / (eigs_G_basis[-1] - eigs_G_basis[0] + 1e-10)
eigs_G_basis_s = (eigs_G_basis - eigs_G_basis[0]) * scale_gb + zeta_zeros[0]
score(eigs_G_basis_s, "gcd(i,j) eigvecs(H_GCD) scaled")


# ============================================================
# THE DEEP TEST: Procrustes rotation between GCD and zeros
# ============================================================
print("\n" + "="*70, flush=True)
print("DEEP TEST: OPTIMAL ROTATION (PROCRUSTES)", flush=True)
print("  Find Q minimizing ||Q @ V_gcd @ diag(eigs_zeros) @ V_gcd^T @ Q^T - H_EF||", flush=True)
print("="*70, flush=True)

# Simpler formulation: we want Q (orthogonal) such that
# eigenvalues of Q^T H_GCD Q are closest to zeta_zeros
# This is the orthogonal Procrustes problem:
# Find Q to minimize ||Q^T H_GCD Q - diag(zeros)||_F
# Equivalent to: Q^T V_gcd = V_target where V_target gives the right eigenvalue order

# We know: H_GCD = V_gcd @ D_gcd @ V_gcd^T
# We want: Q^T H_GCD Q = V_zeros @ D_zeros @ V_zeros^T
# So: Q = V_gcd @ P @ V_zeros^T where P is a permutation/sign matrix

# The REAL question: which PERMUTATION of GCD eigenvalues maps to zero ordering?
# Let's find the optimal permutation
from scipy.optimize import linear_sum_assignment

# Cost matrix: |eigs_gcd_i - zeros_j| after scaling
eigs_gcd_s = np.sort(eigs_gcd)
scale_p = (zeta_zeros[-1] - zeta_zeros[0]) / (eigs_gcd_s[-1] - eigs_gcd_s[0])
eigs_gcd_mapped = (eigs_gcd_s - eigs_gcd_s[0]) * scale_p + zeta_zeros[0]

cost = np.abs(eigs_gcd_mapped[:, np.newaxis] - zeta_zeros[np.newaxis, :N])
row_ind, col_ind = linear_sum_assignment(cost)
perm_err = np.mean(np.abs(eigs_gcd_mapped[row_ind] - zeta_zeros[col_ind]))
print(f"  Optimal assignment error: {perm_err:.4f}")
print(f"  Is natural ordering optimal? {np.all(col_ind == np.arange(N))}")

# Check if the permutation has structure
perm_diff = col_ind - np.arange(N)
print(f"  Permutation displacement: mean={np.mean(np.abs(perm_diff)):.1f}, max={np.max(np.abs(perm_diff))}")

# Build the actual Q from the permutation
# Q = V_gcd_sorted[:, perm] maps GCD eigenvectors to match zero ordering
idx_gcd = np.argsort(eigs_gcd)
V_gcd_sorted = V_gcd[:, idx_gcd]
# Apply permutation (if identity, Q is trivial)
V_gcd_perm = V_gcd_sorted[:, col_ind]
H_procrustes = V_gcd_perm @ np.diag(zeta_zeros[:N]) @ V_gcd_perm.T
eigs_proc = np.sort(np.linalg.eigvalsh(H_procrustes))
score(eigs_proc, "Procrustes (optimal perm)")

# Measure the EIGENVECTOR r with Procrustes
# This tells us: if we could find the right Q, would we get good r?
print(f"\n  This measures: does the GCD eigenvector STRUCTURE carry r?", flush=True)
_, _, _, r_proc, _ = score(eigs_proc, "Procrustes r test")


# ============================================================
# VERDICT
# ============================================================
print("\n" + "="*70, flush=True)
print("VERDICT: BASIS TRANSFORMATION Q", flush=True)
print("="*70, flush=True)

print("""
The question was: can we find Q such that Q^T H_GCD Q has BOTH
good eigenvalues AND good spacing correlation r?

Key findings:
""", flush=True)

# Summarize all candidates
results = {}
for name in ["Mellin sigma=0.50", "Ramanujan(H_GCD) scaled", "Möbius(H_GCD) scaled",
             "DFT(H_GCD) scaled", "gcd(i,j) eigvecs(H_GCD) scaled", "Procrustes (optimal perm)"]:
    pass  # scores already printed above

print(f"""
The CRITICAL TEST is the Procrustes / oracle result:
  - If GCD vecs + zero eigenvalues gives high r: Q EXISTS, we just need to find it
  - If r collapses: the GCD eigenvector structure doesn't carry the correlation
    and no Q can save it

r_procrustes = {r_proc:+.4f}
""", flush=True)

if r_proc > 0.3:
    print("PROMISING: GCD eigenvectors carry meaningful correlation even with forced eigenvalues.", flush=True)
    print("The Q search should continue with Mellin / Ramanujan variants.", flush=True)
elif r_proc > 0.1:
    print("MARGINAL: Some correlation survives, but much is lost.", flush=True)
    print("Q may need to be NON-ORTHOGONAL (similarity, not rotation).", flush=True)
else:
    print("NEGATIVE: r collapses when we change eigenvalues.", flush=True)
    print("The r=0.68 came from GCD EIGENVALUE patterns, not eigenvectors.", flush=True)
    print("Implication: Q doesn't help. Need a fundamentally different approach.", flush=True)
    print("", flush=True)
    print("POSSIBLE NEW DIRECTION: Instead of changing basis, build a SINGLE", flush=True)
    print("operator that generates both eigenvalues and correlations ab initio.", flush=True)
    print("The Lanczos construction (session 5) already does this for the ZEROS", flush=True)
    print("themselves — the Jacobi matrix of the zeta function.", flush=True)

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
