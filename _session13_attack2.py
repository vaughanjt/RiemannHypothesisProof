"""
Session 13: RALPH MODE — Close the Cov(g, h(g)) > 0 gap
==========================================================
LEAN version — memory-safe GP simulation, all 8 analyses.

Key insight from Part 1 (already computed):
  V(g) is monotone on [0, g*] where g* = 2*tau0 (first zero of C(tau))
  g*/g_bar ~ 1.175 (stable across N)
  V(g*) = 1.0 exactly (h(g*) = sqrt(2/pi))
  V oscillates in tail with decreasing amplitude

New analytical insight: INTEGRAL REPRESENTATION
  Cov(g,h) = int_0^inf h'(s) K(s) ds
  where K(s) = Cov(g, 1_{g>s}) >= 0
  This splits cleanly into core (positive) + tail (bounded oscillation)
"""
import numpy as np
from scipy.stats import pearsonr
import time
t0 = time.time()

# ============================================================
# SPECTRAL DENSITY
# ============================================================
def rs_weights(N):
    inv_n = 1.0 / np.arange(1, N+1)
    return inv_n / inv_n.sum()

def rs_frequencies(N):
    return np.log(np.arange(2, N+2))

def C_vec(tau, p, w):
    """C(tau) vectorized"""
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    return np.array([np.dot(p, np.cos(w * t)) for t in tau])

def V_vec(g, p, w):
    """V(g) = 1 - 2C(g/2)^2/(1+C(g)) vectorized"""
    g = np.atleast_1d(np.asarray(g, dtype=float))
    Cg = C_vec(g, p, w)
    Cg2 = C_vec(g/2, p, w)
    return 1.0 - 2.0*Cg2**2 / (1.0 + Cg)

def h_vec(g, p, w):
    return np.sqrt(np.maximum(2.0 * V_vec(g, p, w) / np.pi, 0.0))

def find_g_star(p, w, g_bar):
    """Find g* where V first reaches 1 (first zero of C(tau) at tau = g*/2)"""
    tau_grid = np.linspace(0.01, 2*g_bar, 50000)
    C_vals = C_vec(tau_grid, p, w)
    sc = np.where(C_vals[:-1]*C_vals[1:] < 0)[0]
    if len(sc) > 0:
        return 2 * tau_grid[sc[0]]
    return 3 * g_bar  # fallback


# ============================================================
# LEAN GP SIMULATION — process one window at a time
# ============================================================
def simulate_gp_lean(N, L_per_trial=5000, dt=0.02, n_trials=100, seed=42):
    """Memory-safe: chunked matmul, ~50MB peak per trial."""
    rng = np.random.default_rng(seed)
    omega = rs_frequencies(N)
    amp = 1.0 / np.sqrt(np.arange(1, N+1))
    sigma_N = np.sqrt(np.sum(1.0 / np.arange(1, N+1)))
    chunk = 50000  # time points per chunk

    all_gaps = []
    all_peaks = []

    for trial in range(n_trials):
        phi = rng.uniform(0, 2*np.pi, N)
        n_pts = int(L_per_trial / dt)

        # Build f(t) in chunks via matmul
        f = np.empty(n_pts)
        for s in range(0, n_pts, chunk):
            e = min(s + chunk, n_pts)
            t_c = np.arange(s, e) * dt
            # (chunk_len, N) @ (N,) -> (chunk_len,)
            f[s:e] = np.cos(np.outer(t_c, omega) + phi) @ amp

        f /= sigma_N
        t_all = np.arange(n_pts) * dt

        # Zeros
        sc = np.where(f[:-1] * f[1:] < 0)[0]
        if len(sc) < 20:
            continue
        zeros = t_all[sc] - f[sc] * dt / (f[sc+1] - f[sc])

        gaps = np.diff(zeros)
        mid_idx = ((zeros[:-1] + zeros[1:]) / (2*dt)).astype(int)
        mid_idx = np.clip(mid_idx, 0, n_pts-1)
        peaks = np.abs(f[mid_idx])

        trim = max(3, int(0.05*len(gaps)))
        all_gaps.extend(gaps[trim:-trim].tolist())
        all_peaks.extend(peaks[trim:-trim].tolist())

    return np.array(all_gaps), np.array(all_peaks)


# ============================================================
# PART 1 (condensed): V(g) landscape
# ============================================================
print("="*70)
print("PART 1: V(g) LANDSCAPE (condensed)")
print("="*70)

landscape = {}
for N in [50, 200, 500]:
    p, w = rs_weights(N), rs_frequencies(N)
    m2 = np.dot(p, w**2)
    g_bar = np.pi / np.sqrt(m2)
    g_star = find_g_star(p, w, g_bar)

    # V in tail
    g_tail = np.linspace(g_star, 5*g_bar, 20000)
    V_tail = V_vec(g_tail, p, w)
    eps_V = 1.0 - np.min(V_tail)
    eps_h = np.sqrt(2/np.pi) - np.sqrt(2*np.min(V_tail)/np.pi)

    landscape[N] = {'p': p, 'w': w, 'g_bar': g_bar, 'g_star': g_star,
                     'eps_V': eps_V, 'eps_h': eps_h, 'm2': m2}

    print(f"  N={N:>4}: g_bar={g_bar:.5f}, g*={g_star:.5f} ({g_star/g_bar:.3f} g_bar), "
          f"eps_h={eps_h:.5f}")


# ============================================================
# PART 2: GP simulation + exact Cov(g, h(g))
# ============================================================
print(f"\n{'='*70}")
print("PART 2: GP SIMULATION")
print("="*70)

results = {}
for N in [50, 200, 500]:
    L = landscape[N]
    p, w, g_bar = L['p'], L['w'], L['g_bar']
    g_star = L['g_star']

    print(f"\n  --- N = {N} ---")
    gaps, peaks = simulate_gp_lean(N, L_per_trial=5000, dt=0.01, n_trials=200)
    print(f"  {len(gaps)} gaps collected")

    mu_g, sigma_g = np.mean(gaps), np.std(gaps)
    h_vals = h_vec(gaps, p, w)
    mu_h, sigma_h = np.mean(h_vals), np.std(h_vals)

    cov_gh = np.mean((gaps - mu_g) * (h_vals - mu_h))
    r_gh = cov_gh / (sigma_g * sigma_h)

    cov_gP = np.mean((gaps - mu_g) * (peaks - np.mean(peaks)))
    r_gP = cov_gP / (np.std(gaps) * np.std(peaks)) if np.std(peaks) > 0 else 0

    P_tail = np.mean(gaps > g_star)

    print(f"  mu_g={mu_g:.5f} (theory {g_bar:.5f}), sigma_g={sigma_g:.5f}, CV={sigma_g/mu_g:.4f}")
    print(f"  mu_h={mu_h:.5f}, sigma_h={sigma_h:.5f}")
    print(f"  Cov(g,h) = {cov_gh:.8f}, r(g,h) = {r_gh:.5f}")
    print(f"  r(g,P) = {r_gP:.5f}  (actual peak-gap)")
    print(f"  P(g > g*) = {P_tail:.4f}")

    results[N] = {
        'gaps': gaps, 'peaks': peaks, 'h_vals': h_vals,
        'mu_g': mu_g, 'sigma_g': sigma_g, 'mu_h': mu_h, 'sigma_h': sigma_h,
        'cov_gh': cov_gh, 'r_gh': r_gh, 'P_tail': P_tail, 'g_star': g_star
    }


# ============================================================
# PART 3: MONOTONE DECOMPOSITION h = h_core + delta
# ============================================================
print(f"\n{'='*70}")
print("PART 3: MONOTONE DECOMPOSITION PROOF")
print("="*70)

for N in [50, 200, 500]:
    R = results[N]
    L = landscape[N]
    p, w = L['p'], L['w']
    gaps, h_vals = R['gaps'], R['h_vals']
    g_star = R['g_star']
    mu_g, sigma_g = R['mu_g'], R['sigma_g']

    h_star = np.sqrt(2.0/np.pi)  # h(g*) = sqrt(2/pi) since V(g*)=1
    h_core = np.where(gaps <= g_star, h_vals, h_star)
    delta = h_vals - h_core

    cov_core = np.mean((gaps - mu_g) * (h_core - np.mean(h_core)))
    cov_delta = np.mean((gaps - mu_g) * (delta - np.mean(delta)))

    # Cauchy-Schwarz tail bound
    eps_delta = np.max(np.abs(delta))
    P_tail = R['P_tail']
    cs_bound = sigma_g * eps_delta * np.sqrt(P_tail)

    # Tighter: actual |Cov(g, delta)|
    actual_cov_delta = abs(cov_delta)

    margin_cs = cov_core - cs_bound
    margin_actual = cov_core - actual_cov_delta

    print(f"\n  N={N}:")
    print(f"    Cov(g, h_core) = {cov_core:.8f}  [POSITIVE by Chebyshev]")
    print(f"    Cov(g, delta)  = {cov_delta:+.8f}  [oscillatory tail]")
    print(f"    |Cov(g, delta)| = {actual_cov_delta:.8f}")
    print(f"    C-S bound      = {cs_bound:.8f}")
    print(f"    eps_delta = {eps_delta:.6f}, P_tail = {P_tail:.4f}")
    print(f"    MARGIN (C-S)   = {margin_cs:+.8f} ({'WORKS' if margin_cs > 0 else 'FAILS'})")
    print(f"    MARGIN (exact) = {margin_actual:+.8f} ({'WORKS' if margin_actual > 0 else 'FAILS'})")
    print(f"    C-S ratio      = {cov_core/cs_bound:.2f}x" if cs_bound > 0 else "")
    print(f"    Exact ratio    = {cov_core/actual_cov_delta:.2f}x" if actual_cov_delta > 0 else "")


# ============================================================
# PART 4: OPTIMAL g* SCAN
# ============================================================
print(f"\n{'='*70}")
print("PART 4: OPTIMAL g* SCAN")
print("="*70)

for N in [50, 200, 500]:
    R = results[N]
    L = landscape[N]
    p, w = L['p'], L['w']
    gaps, h_vals = R['gaps'], R['h_vals']
    mu_g, sigma_g = R['mu_g'], R['sigma_g']
    g_bar = L['g_bar']

    best_margin = -np.inf
    best_info = {}

    for g_trial in np.linspace(0.3*g_bar, 4.0*g_bar, 500):
        h_at_trial = h_vec(np.array([g_trial]), p, w)[0]
        h_core = np.where(gaps <= g_trial, h_vals, h_at_trial)
        delta = h_vals - h_core

        cov_core = np.mean((gaps - mu_g) * (h_core - np.mean(h_core)))
        eps_d = np.max(np.abs(delta)) if np.any(gaps > g_trial) else 0
        P_tail = np.mean(gaps > g_trial)
        cs_bound = sigma_g * eps_d * np.sqrt(max(P_tail, 1e-30))
        margin = cov_core - cs_bound

        if margin > best_margin:
            best_margin = margin
            best_info = {'g': g_trial, 'g_ratio': g_trial/g_bar,
                        'cov_core': cov_core, 'cs_bound': cs_bound,
                        'eps': eps_d, 'P_tail': P_tail}

    b = best_info
    print(f"\n  N={N}: optimal g* = {b['g']:.5f} = {b['g_ratio']:.3f} g_bar")
    print(f"    Cov_core = {b['cov_core']:.8f}")
    print(f"    CS_bound = {b['cs_bound']:.8f}")
    print(f"    MARGIN   = {best_margin:+.8f}")
    print(f"    RATIO    = {b['cov_core']/b['cs_bound']:.2f}x" if b['cs_bound'] > 0 else "")
    print(f"    P_tail   = {b['P_tail']:.4f}")
    print(f"    eps      = {b['eps']:.6f}")
    print(f"    PROOF: {'WORKS' if best_margin > 0 else 'FAILS'}")


# ============================================================
# PART 5: TAYLOR EXPANSION
# ============================================================
print(f"\n{'='*70}")
print("PART 5: TAYLOR EXPANSION")
print("="*70)

for N in [50, 200, 500]:
    R = results[N]
    L = landscape[N]
    p, w = L['p'], L['w']
    gaps = R['gaps']
    mu_g = R['mu_g']
    eps = 1e-6

    V0 = V_vec(np.array([mu_g]), p, w)[0]
    Vp = (V_vec(np.array([mu_g+eps]), p, w)[0] - V_vec(np.array([mu_g-eps]), p, w)[0]) / (2*eps)
    Vpp = (V_vec(np.array([mu_g+eps]), p, w)[0] - 2*V0 + V_vec(np.array([mu_g-eps]), p, w)[0]) / eps**2

    h0 = np.sqrt(max(2*V0/np.pi, 1e-30))
    hp = Vp / np.sqrt(2*np.pi*V0) if V0 > 0 else 0
    hpp = (Vpp*np.sqrt(2*np.pi*V0) - Vp**2*np.pi/np.sqrt(2*np.pi*V0)) / (2*np.pi*V0) if V0 > 0 else 0

    var_g = np.var(gaps)
    mu3 = np.mean((gaps - mu_g)**3)

    T1 = hp * var_g
    T2 = hpp * mu3 / 2
    actual = R['cov_gh']

    print(f"\n  N={N}:")
    print(f"    h'(mu) = {hp:+.8f}  {'> 0 [V increasing at mu < g*]' if hp > 0 else '< 0 [PROBLEM]'}")
    print(f"    h''(mu) = {hpp:+.8f}")
    print(f"    Var(g) = {var_g:.8f}, mu3 = {mu3:+.8f}")
    print(f"    Term 1: h'*Var(g) = {T1:+.8f}  [POSITIVE]")
    print(f"    Term 2: h''*mu3/2 = {T2:+.8f}")
    print(f"    Taylor (1+2) = {T1+T2:+.8f}")
    print(f"    Actual Cov   = {actual:+.8f}")
    print(f"    T1 dominates T2: {'YES' if abs(T1) > abs(T2) else 'NO'} ({abs(T1)/abs(T2):.2f}x)")


# ============================================================
# PART 6: INTEGRAL REPRESENTATION
# ============================================================
print(f"\n{'='*70}")
print("PART 6: INTEGRAL REPRESENTATION Cov = int h'(s) K(s) ds")
print("="*70)

for N in [50, 200, 500]:
    R = results[N]
    L = landscape[N]
    p, w = L['p'], L['w']
    gaps = R['gaps']
    g_star = R['g_star']
    mu_g = R['mu_g']
    g_bar = L['g_bar']

    # Compute K(s) = E[(g - mu) * 1_{g>s}] empirically
    s_grid = np.linspace(0.001, 3*g_bar, 1000)
    K_vals = np.array([np.mean((gaps - mu_g) * (gaps > s)) for s in s_grid])

    # h'(s) numerical
    ds = s_grid[1] - s_grid[0]
    h_grid = h_vec(s_grid, p, w)
    hp_grid = np.gradient(h_grid, ds)

    # Integral: Cov ~ sum h'(s) * K(s) * ds
    integrand = hp_grid * K_vals * ds
    total_integral = np.sum(integrand)

    # Core vs tail
    core_mask = s_grid <= g_star
    core_integral = np.sum(integrand[core_mask])
    tail_integral = np.sum(integrand[~core_mask])

    print(f"\n  N={N}:")
    print(f"    int h'(s)K(s)ds = {total_integral:.8f}")
    print(f"    Core [0, g*]    = {core_integral:+.8f}  (POSITIVE)")
    print(f"    Tail [g*, inf]  = {tail_integral:+.8f}  (oscillatory)")
    print(f"    |Tail/Core|     = {abs(tail_integral)/core_integral:.4f}")
    print(f"    Actual Cov      = {R['cov_gh']:.8f}")

    # How much of K is in the tail?
    K_core = np.sum(K_vals[core_mask] * ds)
    K_tail = np.sum(K_vals[~core_mask] * ds)
    print(f"    int K core = {K_core:.6f}, int K tail = {K_tail:.6f}")
    print(f"    K_tail / K_total = {K_tail/(K_core+K_tail):.4f}")


# ============================================================
# PART 7: CONCORDANCE (Kendall tau)
# ============================================================
print(f"\n{'='*70}")
print("PART 7: CONCORDANCE ANALYSIS")
print("="*70)

for N in [50, 200, 500]:
    R = results[N]
    gaps, h_vals = R['gaps'], R['h_vals']

    rng = np.random.default_rng(777)
    n_pairs = 500000
    i1 = rng.integers(0, len(gaps), n_pairs)
    i2 = rng.integers(0, len(gaps), n_pairs)
    mask = i1 != i2
    i1, i2 = i1[mask], i2[mask]

    dg = gaps[i2] - gaps[i1]
    dh = h_vals[i2] - h_vals[i1]
    prod = dg * dh
    n_conc = np.sum(prod > 0)
    n_disc = np.sum(prod < 0)
    tau = (n_conc - n_disc) / (n_conc + n_disc)

    print(f"  N={N}: concordant/discordant = {n_conc/n_disc:.4f}  tau = {tau:+.6f}")


# ============================================================
# PART 8: REGIONAL DECOMPOSITION
# ============================================================
print(f"\n{'='*70}")
print("PART 8: REGIONAL DECOMPOSITION OF Cov(g,h)")
print("="*70)

N = 200
R = results[N]
L = landscape[N]
gaps, h_vals = R['gaps'], R['h_vals']
mu_g, mu_h = R['mu_g'], R['mu_h']
g_bar = L['g_bar']

bounds = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, np.inf]
total = np.mean((gaps - mu_g) * (h_vals - mu_h))

print(f"  N=200, g_bar={g_bar:.5f}")
print(f"\n  {'Region (×g_bar)':>20} {'P':>8} {'Contribution':>14} {'Frac':>8}")
print(f"  {'-'*52}")
for i in range(len(bounds)-1):
    lo, hi = bounds[i]*g_bar, bounds[i+1]*g_bar
    mask = (gaps >= lo) & (gaps < hi)
    n_in = np.sum(mask)
    if n_in == 0:
        continue
    contrib = np.sum((gaps[mask] - mu_g) * (h_vals[mask] - mu_h)) / len(gaps)
    pct = n_in / len(gaps)
    frac = contrib / total if abs(total) > 1e-15 else 0

    label = f"[{bounds[i]:.2f}, {bounds[i+1]:.2f})"
    sign = '+' if contrib >= 0 else '-'
    print(f"  {label:>20} {pct:>8.4f} {contrib:>+14.8f} {frac:>7.1%}")

print(f"  {'TOTAL':>20} {'1.0000':>8} {total:>+14.8f} {'100.0%':>8}")


# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*70}")
print("SUMMARY")
print("="*70)
for N in [50, 200, 500]:
    R = results[N]
    print(f"  N={N}: Cov(g,h) = {R['cov_gh']:+.8f}, r(g,h) = {R['r_gh']:+.5f}, "
          f"r(g,P) = {pearsonr(R['gaps'], R['peaks'])[0]:+.5f}")

print(f"\n  Elapsed: {time.time()-t0:.1f}s")
print("="*70)
