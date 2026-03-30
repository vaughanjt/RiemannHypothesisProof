"""
Session 13: RALPH MODE — Close the Cov(g, h(g)) > 0 gap
==========================================================
Attack strategy: monotone decomposition + tail bound

h(g) = sqrt(2V(g)/pi) where V(g) = 2*Var_p[cos(w*g/2)] / (1+C(g))

Proof structure:
  h = h_core (monotone) + delta (small tail oscillation)
  Cov(g, h) = Cov(g, h_core) + Cov(g, delta)
  Cov(g, h_core) > 0  [Chebyshev]
  |Cov(g, delta)| <= sigma_g * eps * sqrt(P_tail)  [Cauchy-Schwarz]
  => Cov(g, h) > 0  if margin positive
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import pearsonr
import warnings, time
warnings.filterwarnings('ignore')

t0 = time.time()

# ============================================================
# SPECTRAL DENSITY FUNCTIONS
# ============================================================

def rs_weights(N):
    """p_n = (1/n) / H_N"""
    inv_n = 1.0 / np.arange(1, N+1)
    return inv_n / inv_n.sum()

def rs_frequencies(N):
    """omega_n = log(n+1)"""
    return np.log(np.arange(2, N+2))

def spectral_moments(N):
    """Compute m_0, m_2, m_4"""
    p = rs_weights(N)
    w = rs_frequencies(N)
    return np.dot(p, w**0), np.dot(p, w**2), np.dot(p, w**4)

def C_func(tau, p, w):
    """C(tau) = sum p_n cos(w_n tau) — vectorized"""
    tau = np.atleast_1d(tau)
    return np.dot(np.cos(np.outer(tau, w)), p)

def V_func(g, p, w):
    """V(g) = 1 - 2C(g/2)^2 / (1+C(g)) — vectorized"""
    g = np.atleast_1d(g)
    Cg = C_func(g, p, w)
    Cg2 = C_func(g/2, p, w)
    return 1.0 - 2.0*Cg2**2 / (1.0 + Cg)

def F_func(g, p, w):
    """F(g) = 2*Var_p[cos(w*g/2)] = numerator of V (before dividing by 1+C)"""
    g = np.atleast_1d(g)
    cos_vals = np.cos(np.outer(g, w) / 2)  # shape (len(g), N)
    E_cos = cos_vals @ p       # E_p[cos]
    E_cos2 = (cos_vals**2) @ p  # E_p[cos^2]
    return 2.0 * (E_cos2 - E_cos**2)

def h_func(g, p, w):
    """h(g) = sqrt(2V(g)/pi)"""
    V = V_func(g, p, w)
    return np.sqrt(np.maximum(2.0*V/np.pi, 0.0))

def V_deriv(g, p, w, eps=1e-7):
    """Numerical V'(g)"""
    return (V_func(g+eps, p, w) - V_func(g-eps, p, w)) / (2*eps)


# ============================================================
# PART 1: V(g) LANDSCAPE AND MONOTONICITY BREAKDOWN
# ============================================================
print("="*70)
print("PART 1: V(g) LANDSCAPE")
print("="*70)

for N in [50, 100, 200, 500, 1000]:
    p = rs_weights(N)
    w = rs_frequencies(N)
    m0, m2, m4 = spectral_moments(N)
    g_bar = np.pi / np.sqrt(m2)
    alpha2 = m0*m4/m2**2 - 1

    # Compute V on fine grid
    g_grid = np.linspace(1e-4, 5*g_bar, 50000)
    V_vals = V_func(g_grid, p, w)

    # Find first local max (where V' goes from + to -)
    dV = np.diff(V_vals)
    sign_changes = np.where((dV[:-1] > 0) & (dV[1:] < 0))[0]

    if len(sign_changes) > 0:
        g_star = g_grid[sign_changes[0]+1]
        V_star = V_vals[sign_changes[0]+1]
    else:
        g_star = g_grid[-1]
        V_star = V_vals[-1]

    # Oscillation in tail
    tail_mask = g_grid >= g_star
    V_tail = V_vals[tail_mask]
    V_min_tail = np.min(V_tail)
    V_max_tail = np.max(V_tail)
    h_at_star = np.sqrt(max(2*V_star/np.pi, 0))
    h_min_tail = np.sqrt(max(2*V_min_tail/np.pi, 0))
    h_max_tail = np.sqrt(max(2*V_max_tail/np.pi, 0))
    eps_h = max(abs(h_max_tail - h_at_star), abs(h_min_tail - h_at_star))

    # V at mean gap
    V_at_gbar = V_func(np.array([g_bar]), p, w)[0]

    print(f"\n  N={N:>4}  m2={m2:.4f}  alpha^2={alpha2:.4f}  g_bar={g_bar:.5f}")
    print(f"    g*/g_bar = {g_star/g_bar:.4f}  V(g*) = {V_star:.6f}")
    print(f"    V(g_bar) = {V_at_gbar:.6f}")
    print(f"    V tail range: [{V_min_tail:.6f}, {V_max_tail:.6f}]")
    print(f"    h oscillation eps_h = {eps_h:.6f}")
    print(f"    h(g*) = {h_at_star:.6f}")

    # Check: is V monotone up to g*?
    V_to_star = V_vals[g_grid <= g_star]
    dV_core = np.diff(V_to_star)
    n_neg = np.sum(dV_core < -1e-10)
    print(f"    V monotone in core [0, g*]: {'YES' if n_neg == 0 else f'NO ({n_neg} violations)'}")


# ============================================================
# PART 2: GP SIMULATION + GAP DISTRIBUTION
# ============================================================
print(f"\n{'='*70}")
print("PART 2: GP SIMULATION — GAP DISTRIBUTION & COVARIANCE")
print("="*70)

def simulate_gp_gaps(N, L=10000, dt=0.005, n_trials=100, seed=42):
    """
    Simulate f_N(t) = sum (1/sqrt(n)) cos(w_n t + phi_n) / sigma_N
    Return all gaps and midpoint peaks.
    Process in chunks to avoid memory errors.
    """
    rng = np.random.default_rng(seed)
    omega = rs_frequencies(N)
    amp = 1.0 / np.sqrt(np.arange(1, N+1))
    sigma_N = np.sqrt(np.sum(1.0/np.arange(1, N+1)))

    all_gaps = []
    all_peaks = []

    chunk_size = 200000  # points per chunk

    for trial in range(n_trials):
        phi = rng.uniform(0, 2*np.pi, N)

        trial_f = []
        trial_t = []
        n_pts = int(L / dt)

        for start in range(0, n_pts, chunk_size):
            end = min(start + chunk_size + 1, n_pts)  # +1 for overlap
            t_chunk = np.arange(start, end) * dt
            phases = np.outer(t_chunk, omega) + phi[np.newaxis, :]
            f_chunk = (np.cos(phases) @ amp) / sigma_N
            trial_t.append(t_chunk)
            trial_f.append(f_chunk)

        t = np.concatenate(trial_t)
        f = np.concatenate(trial_f)
        # Remove duplicate overlap points
        _, unique_idx = np.unique(t, return_index=True)
        t = t[unique_idx]
        f = f[unique_idx]

        # Find zeros by sign change
        signs = np.sign(f)
        changes = np.where(signs[:-1] * signs[1:] < 0)[0]

        if len(changes) < 20:
            continue

        # Refine zeros by linear interpolation
        zeros = t[changes] - f[changes] * dt / (f[changes+1] - f[changes])

        gaps = np.diff(zeros)
        mid_idx = ((zeros[:-1] + zeros[1:]) / (2*dt)).astype(int)
        mid_idx = np.clip(mid_idx, 0, len(f)-1)
        peaks = np.abs(f[mid_idx])

        # Trim edges (10% each side)
        trim = max(5, int(0.1*len(gaps)))
        all_gaps.extend(gaps[trim:-trim].tolist())
        all_peaks.extend(peaks[trim:-trim].tolist())

    return np.array(all_gaps), np.array(all_peaks)


results = {}

for N in [50, 200, 500]:
    print(f"\n  --- N = {N} ---")
    p = rs_weights(N)
    w = rs_frequencies(N)
    m0, m2, m4 = spectral_moments(N)
    g_bar = np.pi / np.sqrt(m2)

    print(f"  Simulating GP (100 trials, L=30000)...")
    gaps, peaks = simulate_gp_gaps(N, L=30000, dt=0.001, n_trials=100)
    n_gaps = len(gaps)
    print(f"  Collected {n_gaps} gaps")

    mu_g = np.mean(gaps)
    sigma_g = np.std(gaps)
    skew_g = np.mean(((gaps - mu_g)/sigma_g)**3)
    print(f"  mu_g = {mu_g:.6f}  (theory g_bar = {g_bar:.6f})")
    print(f"  sigma_g = {sigma_g:.6f}  (CV = {sigma_g/mu_g:.4f})")
    print(f"  skewness = {skew_g:.4f}")

    # Compute h(g) for each gap
    h_vals = h_func(gaps, p, w)
    mu_h = np.mean(h_vals)
    sigma_h = np.std(h_vals)
    print(f"  mu_h = {mu_h:.6f}  sigma_h = {sigma_h:.6f}")

    # Actual covariance
    cov_gh = np.mean((gaps - mu_g) * (h_vals - mu_h))
    r_gh = cov_gh / (sigma_g * sigma_h)
    print(f"  Cov(g, h(g)) = {cov_gh:.8f}")
    print(f"  r(g, h(g)) = {r_gh:.6f}")

    # Actual peak-gap correlation
    cov_gP = np.mean((gaps - np.mean(gaps)) * (peaks - np.mean(peaks)))
    r_gP = cov_gP / (np.std(gaps) * np.std(peaks))
    print(f"  r(g, P) = {r_gP:.6f}  (full, including residual noise)")

    # Find g* for this N
    g_fine = np.linspace(1e-4, 5*g_bar, 50000)
    V_fine = V_func(g_fine, p, w)
    dV = np.diff(V_fine)
    sign_changes = np.where((dV[:-1] > 0) & (dV[1:] < 0))[0]
    g_star = g_fine[sign_changes[0]+1] if len(sign_changes) > 0 else g_fine[-1]

    results[N] = {
        'gaps': gaps, 'peaks': peaks, 'h_vals': h_vals,
        'mu_g': mu_g, 'sigma_g': sigma_g,
        'mu_h': mu_h, 'sigma_h': sigma_h,
        'cov_gh': cov_gh, 'r_gh': r_gh,
        'g_star': g_star, 'g_bar': g_bar,
        'p': p, 'w': w
    }


# ============================================================
# PART 3: DECOMPOSITION PROOF — h = h_core + delta
# ============================================================
print(f"\n{'='*70}")
print("PART 3: MONOTONE DECOMPOSITION PROOF")
print("="*70)

print("""
  Strategy:
    h_core(g) = h(min(g, g*))    [monotone nondecreasing]
    delta(g) = h(g) - h_core(g)   [small, supported on (g*, inf)]

    Cov(g, h) = Cov(g, h_core) + Cov(g, delta)

    LOWER BOUND on Cov(g, h_core): Chebyshev's inequality => > 0
    UPPER BOUND on |Cov(g, delta)|: Cauchy-Schwarz => sigma_g * eps * sqrt(P_tail)

    PROOF SUCCEEDS if: Cov(g, h_core) > sigma_g * eps * sqrt(P_tail)
""")

for N in [50, 200, 500]:
    R = results[N]
    gaps = R['gaps']
    h_vals = R['h_vals']
    g_star = R['g_star']
    p, w = R['p'], R['w']

    print(f"\n  === N = {N} ===")
    print(f"  g* = {g_star:.6f}  (= {g_star/R['g_bar']:.3f} g_bar)")

    # h_core: monotone (h on [0,g*], then flat at h(g*))
    h_star = h_func(np.array([g_star]), p, w)[0]
    h_core = np.where(gaps <= g_star, h_vals, h_star)
    delta = h_vals - h_core

    # Covariances
    mu_g = R['mu_g']
    cov_core = np.mean((gaps - mu_g) * (h_core - np.mean(h_core)))
    cov_delta = np.mean((gaps - mu_g) * (delta - np.mean(delta)))
    cov_total = np.mean((gaps - mu_g) * (h_vals - np.mean(h_vals)))

    print(f"  Cov(g, h)      = {cov_total:.8f}")
    print(f"  Cov(g, h_core) = {cov_core:.8f}")
    print(f"  Cov(g, delta)  = {cov_delta:.8f}")
    print(f"  Sum check:       {cov_core + cov_delta:.8f}")

    # Tail bound
    sigma_g = R['sigma_g']
    eps_h = np.max(np.abs(delta))
    P_tail = np.mean(gaps > g_star)
    tail_bound = sigma_g * eps_h * np.sqrt(P_tail)

    print(f"\n  eps_h = {eps_h:.6f}")
    print(f"  P(g > g*) = {P_tail:.6f}")
    print(f"  sigma_g = {sigma_g:.6f}")
    print(f"  Tail bound = sigma_g * eps_h * sqrt(P_tail) = {tail_bound:.8f}")

    margin = cov_core - tail_bound
    ratio = cov_core / tail_bound if tail_bound > 0 else float('inf')
    print(f"\n  MARGIN = Cov_core - tail_bound = {margin:.8f}")
    print(f"  RATIO  = Cov_core / tail_bound = {ratio:.2f}x")
    print(f"  PROOF WORKS: {'YES' if margin > 0 else 'NO'}")


# ============================================================
# PART 4: COVARIANCE INTEGRAL DECOMPOSITION BY REGION
# ============================================================
print(f"\n{'='*70}")
print("PART 4: REGIONAL DECOMPOSITION OF Cov(g, h(g))")
print("="*70)

N = 200
R = results[N]
gaps = R['gaps']
h_vals = R['h_vals']
mu_g = R['mu_g']
mu_h = R['mu_h']
g_bar = R['g_bar']

# Decompose into regions: [0, 0.5*g_bar], [0.5, 1.0], [1.0, 1.5], [1.5, 2.0], [2.0, inf]
boundaries = [0, 0.5*g_bar, 1.0*g_bar, 1.5*g_bar, 2.0*g_bar, np.inf]
print(f"\n  N={N}, g_bar={g_bar:.6f}, mu_g={mu_g:.6f}")
print(f"\n  {'Region':>20} {'P(region)':>10} {'Contrib':>12} {'Frac of Cov':>12}")
print(f"  {'-'*56}")

total_cov = np.mean((gaps - mu_g) * (h_vals - mu_h))
for i in range(len(boundaries)-1):
    lo, hi = boundaries[i], boundaries[i+1]
    mask = (gaps >= lo) & (gaps < hi)
    n_in = np.sum(mask)
    if n_in == 0:
        continue
    # This region's contribution to the TOTAL covariance
    # Cov = E[(g-mu)(h-mu_h)] = sum over all gaps / N_total
    contrib = np.sum((gaps[mask] - mu_g) * (h_vals[mask] - mu_h)) / len(gaps)
    P_region = n_in / len(gaps)
    frac = contrib / total_cov if abs(total_cov) > 1e-15 else 0

    label = f"[{lo/g_bar:.1f}, {hi/g_bar:.1f})*g_bar"
    print(f"  {label:>20} {P_region:>10.4f} {contrib:>12.8f} {frac:>11.1%}")

print(f"  {'TOTAL':>20} {'1.0000':>10} {total_cov:>12.8f} {'100.0%':>12}")


# ============================================================
# PART 5: TAYLOR EXPANSION APPROACH
# ============================================================
print(f"\n{'='*70}")
print("PART 5: TAYLOR EXPANSION Cov(g,h) = h'(mu)*Var(g) + corrections")
print("="*70)

for N in [50, 200, 500]:
    R = results[N]
    gaps = R['gaps']
    p, w = R['p'], R['w']
    mu_g = R['mu_g']

    # h'(g) = V'(g) / sqrt(2*pi*V(g))
    eps = 1e-7
    V_at_mu = V_func(np.array([mu_g]), p, w)[0]
    Vp_at_mu = (V_func(np.array([mu_g+eps]), p, w)[0] - V_func(np.array([mu_g-eps]), p, w)[0]) / (2*eps)
    Vpp_at_mu = (V_func(np.array([mu_g+eps]), p, w)[0] - 2*V_at_mu + V_func(np.array([mu_g-eps]), p, w)[0]) / eps**2

    h_at_mu = np.sqrt(max(2*V_at_mu/np.pi, 1e-30))
    hp_at_mu = Vp_at_mu / (np.sqrt(2*np.pi*V_at_mu)) if V_at_mu > 0 else 0
    hpp_at_mu = (Vpp_at_mu / np.sqrt(2*np.pi*V_at_mu) - Vp_at_mu**2 / (2*V_at_mu*np.sqrt(2*np.pi*V_at_mu))) if V_at_mu > 0 else 0

    var_g = np.var(gaps)
    mu3 = np.mean((gaps - mu_g)**3)
    mu4 = np.mean((gaps - mu_g)**4)

    # Taylor: Cov(g,h) ~ h'*Var(g) + h''/2 * mu3 + h'''/6 * mu4 + ...
    term1 = hp_at_mu * var_g
    term2 = hpp_at_mu * mu3 / 2

    actual_cov = R['cov_gh']

    print(f"\n  N={N}:")
    print(f"    V(mu_g) = {V_at_mu:.6f},  V'(mu_g) = {Vp_at_mu:.6f},  V''(mu_g) = {Vpp_at_mu:.6f}")
    print(f"    h'(mu_g) = {hp_at_mu:.6f},  h''(mu_g) = {hpp_at_mu:.6f}")
    print(f"    Var(g) = {var_g:.8f},  mu3 = {mu3:.8f}")
    print(f"    Taylor term 1 (h'*Var): {term1:.8f}")
    print(f"    Taylor term 2 (h''*mu3/2): {term2:.8f}")
    print(f"    Taylor sum (1+2): {term1+term2:.8f}")
    print(f"    Actual Cov(g,h): {actual_cov:.8f}")
    print(f"    h'(mu) > 0: {'YES' if hp_at_mu > 0 else 'NO'}")
    print(f"    Leading term DOMINATES: {'YES' if abs(term1) > abs(term2) else 'NO'}")
    print(f"    Leading term POSITIVE: {'YES' if term1 > 0 else 'NO'}")


# ============================================================
# PART 6: ALTERNATIVE — PROVE h(g)*g IS INCREASING
# ============================================================
print(f"\n{'='*70}")
print("PART 6: IS g*h(g) ALWAYS INCREASING? (bypasses h monotonicity)")
print("="*70)

for N in [50, 200, 500]:
    p = rs_weights(N)
    w = rs_frequencies(N)
    m0, m2, m4 = spectral_moments(N)
    g_bar = np.pi / np.sqrt(m2)

    g_grid = np.linspace(1e-4, 5*g_bar, 100000)
    gh = g_grid * h_func(g_grid, p, w)
    d_gh = np.diff(gh)
    n_neg = np.sum(d_gh < -1e-12)

    print(f"  N={N}: g*h(g) decreasing at {n_neg}/{len(d_gh)} points "
          f"({'ALWAYS INCREASING' if n_neg == 0 else 'NOT monotone'})")

    if n_neg > 0:
        # Where does it fail?
        fail_idx = np.where(d_gh < -1e-12)[0]
        first_fail = g_grid[fail_idx[0]]
        print(f"    First failure at g = {first_fail:.6f} = {first_fail/g_bar:.3f} g_bar")


# ============================================================
# PART 7: THE KENDALL TAU APPROACH
# ============================================================
print(f"\n{'='*70}")
print("PART 7: CONCORDANCE ANALYSIS (coupling representation)")
print("="*70)
print("""
  Cov(g,h) = (1/2) E[(g2-g1)(h(g2)-h(g1))]  for iid g1, g2
  Positive iff more concordant than discordant pairs.
""")

for N in [50, 200, 500]:
    R = results[N]
    gaps = R['gaps']
    h_vals = R['h_vals']

    # Sample pairs (too many for all pairs)
    rng = np.random.default_rng(123)
    n_pairs = min(500000, len(gaps)*(len(gaps)-1)//2)
    idx1 = rng.integers(0, len(gaps), n_pairs)
    idx2 = rng.integers(0, len(gaps), n_pairs)
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]

    dg = gaps[idx2] - gaps[idx1]
    dh = h_vals[idx2] - h_vals[idx1]
    product = dg * dh

    n_concordant = np.sum(product > 0)
    n_discordant = np.sum(product < 0)
    n_tied = np.sum(product == 0)
    tau = (n_concordant - n_discordant) / (n_concordant + n_discordant)

    print(f"  N={N}: concordant={n_concordant}  discordant={n_discordant}  "
          f"tau={tau:+.6f}  ratio={n_concordant/max(n_discordant,1):.2f}:1")


# ============================================================
# PART 8: SCAN FOR OPTIMAL g* (MAXIMIZE MARGIN)
# ============================================================
print(f"\n{'='*70}")
print("PART 8: OPTIMAL g* SCAN (maximize proof margin)")
print("="*70)

for N in [50, 200, 500]:
    R = results[N]
    gaps = R['gaps']
    h_vals = R['h_vals']
    p, w = R['p'], R['w']
    mu_g = R['mu_g']
    sigma_g = R['sigma_g']
    g_bar = R['g_bar']

    best_margin = -np.inf
    best_g_star = 0
    best_ratio = 0

    # Scan g* from 0.5*g_bar to 3*g_bar
    for g_trial in np.linspace(0.5*g_bar, 3.0*g_bar, 200):
        h_star_val = h_func(np.array([g_trial]), p, w)[0]
        h_core = np.where(gaps <= g_trial, h_vals, h_star_val)
        delta = h_vals - h_core

        cov_core = np.mean((gaps - mu_g) * (h_core - np.mean(h_core)))
        eps_h = np.max(np.abs(delta)) if np.any(np.abs(delta) > 0) else 0
        P_tail = np.mean(gaps > g_trial)

        tail_bound = sigma_g * eps_h * np.sqrt(max(P_tail, 1e-30))
        margin = cov_core - tail_bound

        if margin > best_margin:
            best_margin = margin
            best_g_star = g_trial
            best_ratio = cov_core / max(tail_bound, 1e-30)
            best_cov_core = cov_core
            best_tail_bound = tail_bound
            best_P_tail = P_tail
            best_eps = eps_h

    print(f"\n  N={N}:")
    print(f"    Optimal g* = {best_g_star:.6f} = {best_g_star/g_bar:.3f} g_bar")
    print(f"    Cov_core = {best_cov_core:.8f}")
    print(f"    Tail bound = {best_tail_bound:.8f}")
    print(f"    MARGIN = {best_margin:.8f}")
    print(f"    RATIO = {best_ratio:.2f}x")
    print(f"    P(tail) = {best_P_tail:.4f}")
    print(f"    eps_h = {best_eps:.6f}")
    print(f"    PROOF: {'WORKS' if best_margin > 0 else 'FAILS'}")


# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*70}")
print(f"ELAPSED: {time.time()-t0:.1f}s")
print("="*70)
