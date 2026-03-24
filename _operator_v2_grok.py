"""Operator v2: incorporating Grok's feedback.

CHANGES FROM V1:
1. Stronger coupling: try C=40-80 and raw (no ||V|| normalization)
2. Bandwidth-dependent C: C(d) = C_0 / d^alpha
3. Non-circular Ulam weights from L-function values L(1,chi) mod 8
4. Add -pi/8 phase shift (functional equation fingerprint)
5. TARGET: r > 0.65 AND KS p > 0.01 simultaneously
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np
from scipy.linalg import eigh
from scipy.stats import pearsonr, kstest
from scipy.optimize import minimize_scalar, minimize
from sympy import primerange
import mpmath
mpmath.mp.dps = 20

t0 = time.time()

N = 200
print("Computing zeros + Z'...", flush=True)
zeta_zeros = np.array([float(mpmath.im(mpmath.zetazero(k))) for k in range(1, N+1)])
primes_all = list(primerange(2, 3000))

trim = int(0.1*N)
ms = np.mean(np.diff(zeta_zeros[trim:-trim]))

def N_smooth(T):
    if T < 2: return 0.
    return T/(2*np.pi)*np.log(T/(2*np.pi)) - T/(2*np.pi) + 7./8.

def N_deriv(T):
    if T < 2: return .001
    return np.log(T/(2*np.pi)) / (2*np.pi)

def weyl_zero(k):
    t = 2*np.pi*k / np.log(max(k,2)+2)
    for _ in range(30):
        if t < 1: t = 10.
        t -= (N_smooth(t)-k) / N_deriv(t)
    return t

from riemann.analysis.bost_connes_operator import polynomial_unfold

def measure_peak_gap(eigs_raw):
    eigs = np.sort(eigs_raw)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(sp) < 20: return 0., 0
    sp = sp / np.mean(sp)
    n_t = int(0.1*len(eigs)); eigs_t = eigs[n_t:-n_t]
    lp, ga = [], []
    for k in range(min(len(sp), len(eigs_t)-1)):
        z = (eigs_t[k]+eigs_t[k+1])/2
        lp.append(np.sum(np.log(np.abs(z-eigs)+1e-30)))
        ga.append(sp[k])
    if len(ga) < 10: return 0., 0
    return pearsonr(np.array(ga), np.array(lp))[0], len(ga)

def wigner_cdf(s):
    return 1 - np.exp(-np.pi*s**2/4)


# ============================================================
# Non-circular Ulam weights from L-function values
# ============================================================
print("Computing L-function values for characters mod 8...", flush=True)

# Characters of (Z/8Z)* = {1,3,5,7} ~ Z/2 x Z/2
# chi_0: (1,1,1,1) — principal
# chi_a: (1,-1,-1,1) — Kronecker (-4/n)
# chi_b: (1,-1,1,-1) — Kronecker (8/n)
# chi_c: (1,1,-1,-1) — Kronecker (-8/n)

chars_mod8 = {
    'chi_0': {1:1, 3:1, 5:1, 7:1},
    'chi_a': {1:1, 3:-1, 5:-1, 7:1},   # (-4/n)
    'chi_b': {1:1, 3:-1, 5:1, 7:-1},   # (8/n)
    'chi_c': {1:1, 3:1, 5:-1, 7:-1},   # (-8/n)
}

# Compute L(1, chi) for each character
L_values = {}
for name, chi in chars_mod8.items():
    if name == 'chi_0':
        L_values[name] = float('inf')  # pole
        continue
    # L(1, chi) = sum_{n=1}^{inf} chi(n)/n
    L = 0.0
    for n in range(1, 10000):
        r = n % 8
        if r in chi:
            L += chi[r] / n
    L_values[name] = L

print(f"  L(1, chi_a) = {L_values['chi_a']:.6f}  ((-4/n) character)")
print(f"  L(1, chi_b) = {L_values['chi_b']:.6f}  ((8/n) character)")
print(f"  L(1, chi_c) = {L_values['chi_c']:.6f}  ((-8/n) character)")

# Inverse Fourier: mode weight for residue r = sum_chi chi(r) * L(1,chi)
# (excluding principal character)
L_mode_weights = {}
for r in [1, 3, 5, 7]:
    w = 0.0
    for name, chi in chars_mod8.items():
        if name == 'chi_0': continue
        w += chi[r] * abs(L_values[name])
    L_mode_weights[r] = abs(w)

# Normalize so max = 3.47 (to compare with optimized)
max_w = max(L_mode_weights.values())
L_mode_normalized = {r: v/max_w * 3.47 for r, v in L_mode_weights.items()}

print(f"\n  L-function derived mode weights:")
for r in [1,3,5,7]:
    print(f"    r={r} mod 8: raw={L_mode_weights[r]:.4f}, "
          f"normalized={L_mode_normalized[r]:.4f} "
          f"(optimized was {[1.22, 3.47, 0.001, 1.61][[1,3,5,7].index(r)]:.3f})")


# ============================================================
# Build operator with all Grok corrections
# ============================================================

def build_v2(N_size, n_primes=168, W=3, C_raw=None, C_normalized=15.0,
             use_normalization=True, mode_weights=None, phase_shift=0.0,
             C_decay=0.0, sigma=0.5):
    """V2 operator with Grok's corrections.

    New parameters:
        C_raw: if set, use this as raw coupling (no ||V|| normalization)
        use_normalization: if False, skip V/||V|| step
        phase_shift: add this to the kernel phase (e.g., -pi/8)
        C_decay: if > 0, coupling decays as C/d^C_decay with distance
    """
    primes = primes_all[:n_primes]
    if mode_weights is None:
        mode_weights = {1:1.22, 3:3.47, 5:0.001, 7:1.61}

    prime_classes = {r: [p for p in primes if p%8==r] for r in mode_weights}

    # Diagonal
    alpha = np.zeros(N_size)
    for k in range(1, N_size+1):
        Tw = weyl_zero(k); dN = N_deriv(Tw)
        s = sum(-np.sin(2*m*Tw*np.log(p))/(m*p**(m*sigma))
                for p in primes for m in range(1,6)) / np.pi
        alpha[k-1] = Tw + s / dN

    # Off-diagonal
    H = np.diag(alpha)
    for ki in range(N_size):
        Tk = alpha[ki]
        logT = max(np.log(max(Tk,10)/(2*np.pi)), 0.1)
        for d in range(1, W+1):
            if ki+d >= N_size: continue
            # Distance-dependent coupling
            d_factor = 1.0 / (d ** C_decay) if C_decay > 0 else 1.0
            val = 0.0
            for r, w_r in mode_weights.items():
                for p in prime_classes.get(r, []):
                    lp = np.log(p)
                    for m in range(1, 3):
                        # WITH phase shift
                        val += w_r * lp/(p**(m*sigma)*logT) * \
                               np.cos(2*np.pi*d*m*lp/logT + phase_shift)
            H[ki, ki+d] = val * d_factor
            H[ki+d, ki] = val * d_factor

    # Coupling
    if use_normalization:
        V = H - np.diag(np.diag(H))
        vn = np.linalg.norm(V, ord=2)
        if vn > 0.01:
            H = np.diag(alpha) + V/vn * C_normalized
    elif C_raw is not None:
        V = H - np.diag(np.diag(H))
        H = np.diag(alpha) + V * C_raw

    return H, alpha


def score_all(H, alpha):
    """Compute all six metrics."""
    eigs = np.sort(np.linalg.eigvalsh(H))

    # 1. Eigenvalue error
    errs = np.abs(eigs - zeta_zeros[:len(eigs)])[trim:-trim]
    mean_err = np.mean(errs)
    pct_half = np.mean(errs < ms/2)

    # 2. Peak-gap r
    r_pg, _ = measure_peak_gap(eigs)

    # 3. KS test vs Wigner
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(sp) > 20:
        sp = sp / np.mean(sp)
        _, p_gue = kstest(sp, wigner_cdf)
    else:
        p_gue = 0.0

    # 4. Trace formula (Lorentzian)
    def lor(t): return 25./((t-100)**2+25)
    tr_H = np.sum(lor(eigs))
    tr_z = np.sum(lor(zeta_zeros[:len(eigs)]))
    trace_err = abs(tr_H - tr_z) / tr_z if tr_z > 0 else 0

    return {
        'mean_err': mean_err,
        'pct_half': pct_half,
        'r': r_pg,
        'p_gue': p_gue,
        'trace_err': trace_err,
    }


# ============================================================
# SWEEP 1: Coupling strength (C_normalized and C_raw)
# ============================================================
print("\n" + "="*70, flush=True)
print("SWEEP 1: COUPLING STRENGTH", flush=True)
print("="*70, flush=True)

print(f"\n  {'Method':>20} {'C':>8} {'mean_err':>10} {'r':>8} {'p(GUE)':>8} "
      f"{'trace%':>8} {'<half':>8}", flush=True)
print(f"  {'-'*68}", flush=True)

# Normalized
for C in [5, 10, 15, 20, 30, 50, 80]:
    H, a = build_v2(N, C_normalized=C)
    s = score_all(H, a)
    print(f"  {'normalized':>20} {C:>8} {s['mean_err']:>10.4f} {s['r']:>+8.4f} "
          f"{s['p_gue']:>8.4f} {s['trace_err']*100:>7.1f}% {s['pct_half']*100:>7.1f}%", flush=True)

# Raw (no normalization)
for C in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    H, a = build_v2(N, C_raw=C, use_normalization=False)
    s = score_all(H, a)
    print(f"  {'raw':>20} {C:>8.1f} {s['mean_err']:>10.4f} {s['r']:>+8.4f} "
          f"{s['p_gue']:>8.4f} {s['trace_err']*100:>7.1f}% {s['pct_half']*100:>7.1f}%", flush=True)


# ============================================================
# SWEEP 2: Phase shift (-pi/8 and variants)
# ============================================================
print("\n" + "="*70, flush=True)
print("SWEEP 2: PHASE SHIFT (functional equation)", flush=True)
print("="*70, flush=True)

print(f"\n  {'phase':>12} {'mean_err':>10} {'r':>8} {'p(GUE)':>8} {'<half':>8}", flush=True)
print(f"  {'-'*50}", flush=True)

for phi in [0, -np.pi/8, -np.pi/4, np.pi/8, -np.pi/16, -3*np.pi/8]:
    H, a = build_v2(N, C_normalized=15, phase_shift=phi)
    s = score_all(H, a)
    print(f"  {phi/np.pi:>+10.4f}*pi {s['mean_err']:>10.4f} {s['r']:>+8.4f} "
          f"{s['p_gue']:>8.4f} {s['pct_half']*100:>7.1f}%", flush=True)


# ============================================================
# SWEEP 3: L-function derived mode weights
# ============================================================
print("\n" + "="*70, flush=True)
print("SWEEP 3: L-FUNCTION vs OPTIMIZED MODE WEIGHTS", flush=True)
print("="*70, flush=True)

weight_sets = {
    "optimized": {1:1.22, 3:3.47, 5:0.001, 7:1.61},
    "L-function": L_mode_normalized,
    "uniform": {1:1, 3:1, 5:1, 7:1},
    "inert_only": {1:0, 3:3.47, 5:0, 7:1.61},
}

print(f"\n  {'Weights':>15} {'mean_err':>10} {'r':>8} {'p(GUE)':>8} {'<half':>8}", flush=True)
print(f"  {'-'*52}", flush=True)

for wname, weights in weight_sets.items():
    H, a = build_v2(N, C_normalized=15, mode_weights=weights)
    s = score_all(H, a)
    print(f"  {wname:>15} {s['mean_err']:>10.4f} {s['r']:>+8.4f} "
          f"{s['p_gue']:>8.4f} {s['pct_half']*100:>7.1f}%", flush=True)


# ============================================================
# SWEEP 4: Bandwidth-dependent coupling C(d) = C_0 / d^alpha
# ============================================================
print("\n" + "="*70, flush=True)
print("SWEEP 4: BANDWIDTH-DEPENDENT COUPLING", flush=True)
print("="*70, flush=True)

print(f"\n  {'W':>4} {'C_decay':>8} {'C':>6} {'mean_err':>10} {'r':>8} {'p(GUE)':>8} {'<half':>8}", flush=True)
print(f"  {'-'*56}", flush=True)

for W in [3, 5, 10]:
    for decay in [0, 0.5, 1.0, 1.5]:
        for C in [15, 30, 50]:
            H, a = build_v2(N, W=W, C_normalized=C, C_decay=decay)
            s = score_all(H, a)
            if s['r'] > 0.3 or (W==3 and decay==0):  # only show interesting
                print(f"  {W:>4} {decay:>8.1f} {C:>6} {s['mean_err']:>10.4f} "
                      f"{s['r']:>+8.4f} {s['p_gue']:>8.4f} {s['pct_half']*100:>7.1f}%", flush=True)


# ============================================================
# SWEEP 5: Joint optimization (Nelder-Mead over C, phase, decay)
# ============================================================
print("\n" + "="*70, flush=True)
print("SWEEP 5: JOINT OPTIMIZATION", flush=True)
print("="*70, flush=True)

def joint_objective(params):
    """Optimize for r > 0.65 AND p_gue > 0.01 AND low error."""
    C, phi, decay = np.exp(params[0]), params[1], max(params[2], 0)
    H, a = build_v2(N, C_normalized=C, phase_shift=phi, C_decay=decay, W=3)
    s = score_all(H, a)
    # Multi-objective: minimize error, penalize low r and low p_gue
    penalty = 0
    if s['r'] < 0.65: penalty += (0.65 - s['r']) * 10
    if s['p_gue'] < 0.01: penalty += (0.01 - s['p_gue']) * 100
    return s['mean_err'] + penalty

print("  Optimizing (C, phase, decay) jointly...", flush=True)
rng = np.random.default_rng(42)
best_params = None
best_score = 1e10

for trial in range(30):
    x0 = [rng.uniform(1, 4.5), rng.uniform(-np.pi/4, np.pi/4), rng.uniform(0, 1.5)]
    res = minimize(joint_objective, x0, method='Nelder-Mead',
                   options={'maxiter': 100, 'xatol': 0.01})
    if res.fun < best_score:
        best_score = res.fun
        best_params = res.x

C_best = np.exp(best_params[0])
phi_best = best_params[1]
decay_best = max(best_params[2], 0)

H_best, a_best = build_v2(N, C_normalized=C_best, phase_shift=phi_best,
                            C_decay=decay_best, W=3)
s_best = score_all(H_best, a_best)

print(f"\n  BEST JOINT RESULT:", flush=True)
print(f"    C = {C_best:.2f}", flush=True)
print(f"    phase = {phi_best/np.pi:+.4f}*pi", flush=True)
print(f"    decay = {decay_best:.4f}", flush=True)
print(f"    mean_err = {s_best['mean_err']:.4f}", flush=True)
print(f"    r = {s_best['r']:+.4f}  (target: > 0.65)", flush=True)
print(f"    p(GUE) = {s_best['p_gue']:.4f}  (target: > 0.01)", flush=True)
print(f"    trace_err = {s_best['trace_err']*100:.1f}%", flush=True)
print(f"    <half_gap = {s_best['pct_half']*100:.1f}%", flush=True)

# Also try with L-function weights
H_L, a_L = build_v2(N, C_normalized=C_best, phase_shift=phi_best,
                      C_decay=decay_best, W=3, mode_weights=L_mode_normalized)
s_L = score_all(H_L, a_L)
print(f"\n  Same params with L-function weights:", flush=True)
print(f"    r = {s_L['r']:+.4f}, p(GUE) = {s_L['p_gue']:.4f}, "
      f"err = {s_L['mean_err']:.4f}", flush=True)


# ============================================================
# FINAL: Best configuration details
# ============================================================
print("\n" + "="*70, flush=True)
print("FINAL OPERATOR SPECIFICATION", flush=True)
print("="*70, flush=True)

eigs_final = np.sort(np.linalg.eigvalsh(H_best))
print(f"\n  First 15 eigenvalues vs zeros:", flush=True)
print(f"  {'k':>4} {'Eigenvalue':>12} {'Zero':>12} {'Error':>10}", flush=True)
for i in range(15):
    err = abs(eigs_final[i] - zeta_zeros[i])
    tag = " <<<" if err < 0.3 else ""
    print(f"  {i+1:>4} {eigs_final[i]:>12.4f} {zeta_zeros[i]:>12.4f} "
          f"{err:>10.4f}{tag}", flush=True)

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
