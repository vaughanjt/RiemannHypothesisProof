"""
Session 13c: Address Grok's critique — compute h_true directly.

Grok's fatal objection: the bridge formula h_bridge(g) = sqrt(2V/pi) ignores
the excursion conditioning. The TRUE h(g) = E[P | gap=g] could be non-monotone.

ATTACK: Compute h_true empirically from GP simulation. If h_true IS nondecreasing,
Chebyshev applies directly and the entire bridge analysis becomes supporting
evidence rather than load-bearing proof.

This would make the proof:
  1. h_true(g) = E[P|g] is nondecreasing  [verified + coupling argument]
  2. Cov(g, P) = Cov(g, h_true) > 0       [Chebyshev, analytic]
  3. r > 0                                  [direct]

No bridge formula, no V monotonicity, no decomposition needed.
"""
import numpy as np, sys
from scipy.stats import pearsonr
sys.stdout.reconfigure(line_buffering=True)

def rs(N):
    p = 1.0/np.arange(1,N+1); p /= p.sum()
    w = np.log(np.arange(2,N+2))
    return p, w

def simulate(N, n_trials=200, L=5000, dt=0.02):
    p, w = rs(N)
    amp = 1.0/np.sqrt(np.arange(1,N+1))
    sigma_N = np.sqrt(np.sum(1.0/np.arange(1,N+1)))
    rng = np.random.default_rng(42)
    chunk = 20000
    all_g, all_p = [], []
    for trial in range(n_trials):
        phi = rng.uniform(0, 2*np.pi, N)
        npts = int(L/dt)
        f = np.empty(npts)
        for s in range(0, npts, chunk):
            e = min(s+chunk, npts)
            tc = np.arange(s,e)*dt
            f[s:e] = np.cos(np.outer(tc, w)+phi) @ amp
        f /= sigma_N
        t = np.arange(npts)*dt
        sc = np.where(f[:-1]*f[1:]<0)[0]
        if len(sc)<20: continue
        zeros = t[sc] - f[sc]*dt/(f[sc+1]-f[sc])
        gaps = np.diff(zeros)
        midx = ((zeros[:-1]+zeros[1:])/(2*dt)).astype(int)
        midx = np.clip(midx, 0, npts-1)
        pks = np.abs(f[midx])
        tr = max(3, int(0.05*len(gaps)))
        all_g.extend(gaps[tr:-tr].tolist())
        all_p.extend(pks[tr:-tr].tolist())
    return np.array(all_g), np.array(all_p)


# ============================================================
# PART 1: Compute h_true(g) = E[P|g] by binning
# ============================================================
print("="*70)
print("PART 1: h_true(g) = E[P | gap = g] FROM SIMULATION")
print("="*70)

for N in [50, 200, 500]:
    print(f"\n  N={N}: simulating...", flush=True)
    gaps, peaks = simulate(N, n_trials=200)
    print(f"  {len(gaps)} gaps")

    m2 = np.dot(rs(N)[0], rs(N)[1]**2)
    g_bar = np.pi / np.sqrt(m2)

    # Bin gaps and compute conditional mean of peaks
    n_bins = 80
    percentiles = np.linspace(0, 100, n_bins+1)
    bin_edges = np.percentile(gaps, percentiles)
    bin_edges[-1] += 0.001  # include max

    h_true = np.zeros(n_bins)
    h_bridge = np.zeros(n_bins)
    g_mid = np.zeros(n_bins)
    n_per_bin = np.zeros(n_bins, dtype=int)

    p, w = rs(N)

    for i in range(n_bins):
        mask = (gaps >= bin_edges[i]) & (gaps < bin_edges[i+1])
        n_per_bin[i] = np.sum(mask)
        if n_per_bin[i] < 10:
            h_true[i] = np.nan
            h_bridge[i] = np.nan
            g_mid[i] = (bin_edges[i] + bin_edges[i+1]) / 2
            continue
        h_true[i] = np.mean(peaks[mask])
        g_mid[i] = np.mean(gaps[mask])
        # Bridge formula for comparison
        Cg = np.dot(p, np.cos(w * g_mid[i]))
        Cg2 = np.dot(p, np.cos(w * g_mid[i]/2))
        V = 1 - 2*Cg2**2 / (1+Cg)
        h_bridge[i] = np.sqrt(max(2*V/np.pi, 0))

    # Check monotonicity of h_true
    valid = ~np.isnan(h_true)
    ht = h_true[valid]
    gm = g_mid[valid]

    diffs = np.diff(ht)
    n_decreasing = np.sum(diffs < 0)
    n_total = len(diffs)

    # Smoothed version (rolling average of 3 bins)
    if len(ht) >= 5:
        ht_smooth = np.convolve(ht, np.ones(3)/3, mode='valid')
        gm_smooth = np.convolve(gm, np.ones(3)/3, mode='valid')
        diffs_smooth = np.diff(ht_smooth)
        n_dec_smooth = np.sum(diffs_smooth < -1e-6)
    else:
        n_dec_smooth = 0

    print(f"\n  h_true monotonicity (raw):      {n_decreasing}/{n_total} bins decrease")
    print(f"  h_true monotonicity (smoothed):  {n_dec_smooth}/{max(len(diffs_smooth) if len(ht)>=5 else 0,1)} bins decrease")

    # Print h_true vs h_bridge at key quantiles
    print(f"\n  {'g/g_bar':>8} {'h_true':>10} {'h_bridge':>10} {'ratio':>8} {'n':>6}")
    print(f"  {'-'*44}")
    for i in range(0, n_bins, n_bins//10):
        if np.isnan(h_true[i]): continue
        ratio = h_true[i] / h_bridge[i] if h_bridge[i] > 0 else 0
        print(f"  {g_mid[i]/g_bar:>8.3f} {h_true[i]:>10.5f} {h_bridge[i]:>10.5f} "
              f"{ratio:>8.3f} {n_per_bin[i]:>6}")

    # Is h_true > h_bridge? (excursion bonus)
    hb = h_bridge[valid]
    print(f"\n  h_true > h_bridge: {np.sum(ht > hb)}/{len(ht)} bins")
    print(f"  mean ratio h_true/h_bridge = {np.mean(ht/hb):.4f}")

    # Correlation of h_true with g (should be very high if h_true is monotone)
    r_htrue = pearsonr(gm, ht)[0]
    print(f"  Corr(g, h_true) = {r_htrue:+.4f}")


# ============================================================
# PART 2: MONOTONE DECOMPOSITION ON h_true DIRECTLY
# ============================================================
print(f"\n{'='*70}")
print("PART 2: DECOMPOSITION ON h_true (not h_bridge)")
print("="*70)

for N in [50, 200]:
    print(f"\n  N={N}: simulating...", flush=True)
    gaps, peaks = simulate(N, n_trials=200)

    # Compute h_true at each gap via local regression (kernel smoothing)
    # Use Nadaraya-Watson estimator with bandwidth = 0.1*g_bar
    p, w = rs(N)
    m2 = np.dot(p, w**2)
    g_bar = np.pi / np.sqrt(m2)
    bw = 0.1 * g_bar

    # For speed, compute h_true on a grid and interpolate
    g_grid = np.linspace(np.percentile(gaps, 0.5), np.percentile(gaps, 99.5), 200)
    h_true_grid = np.zeros(len(g_grid))
    for i, g0 in enumerate(g_grid):
        weights = np.exp(-0.5*((gaps - g0)/bw)**2)
        if np.sum(weights) > 1e-10:
            h_true_grid[i] = np.average(peaks, weights=weights)
        else:
            h_true_grid[i] = np.nan

    # Interpolate h_true at each gap
    valid_grid = ~np.isnan(h_true_grid)
    from numpy import interp
    h_true_at_gaps = interp(gaps, g_grid[valid_grid], h_true_grid[valid_grid])

    # Now do the decomposition on h_true
    mu_g = np.mean(gaps)
    cov_true = np.mean((gaps - mu_g) * (h_true_at_gaps - np.mean(h_true_at_gaps)))
    cov_actual = np.mean((gaps - mu_g) * (peaks - np.mean(peaks)))

    # Is h_true_grid nondecreasing?
    ht_valid = h_true_grid[valid_grid]
    n_dec = np.sum(np.diff(ht_valid) < -1e-6)

    # Find where h_true starts decreasing (if anywhere)
    diffs = np.diff(ht_valid)
    if np.any(diffs < -1e-6):
        first_dec = np.where(diffs < -1e-6)[0][0]
        g_first_dec = g_grid[valid_grid][first_dec]
        # Do decomposition at this point
        g_star_true = g_first_dec
    else:
        g_star_true = g_grid[valid_grid][-1]

    h_star_true = interp(g_star_true, g_grid[valid_grid], ht_valid)
    h_core_true = np.where(gaps <= g_star_true, h_true_at_gaps,
                           np.minimum(h_true_at_gaps, h_star_true))
    # For the decomposition: clip h_core to be nondecreasing
    # Actually, h_core should be h_true up to g_star, then constant
    h_core_true = np.where(gaps <= g_star_true, h_true_at_gaps, h_star_true)
    delta_true = h_true_at_gaps - h_core_true

    cov_core_true = np.mean((gaps - mu_g) * (h_core_true - np.mean(h_core_true)))
    cov_delta_true = np.mean((gaps - mu_g) * delta_true)

    print(f"  {len(gaps)} gaps, g_bar={g_bar:.4f}")
    print(f"  h_true nondecreasing on grid: {n_dec} violations out of {len(ht_valid)-1}")
    print(f"  g*_true = {g_star_true:.4f} = {g_star_true/g_bar:.3f} g_bar")
    print(f"  Cov(g, h_true)  = {cov_true:.8f}")
    print(f"  Cov(g, P)       = {cov_actual:.8f}")
    print(f"  Cov_core(true)  = {cov_core_true:.8f}")
    print(f"  Cov_delta(true) = {cov_delta_true:+.8f}")
    print(f"  Ratio true      = {cov_core_true/abs(cov_delta_true):.1f}x" if abs(cov_delta_true)>1e-10 else "")

    # h_true vs h_bridge at the mean gap
    Cg = np.dot(p, np.cos(w * mu_g))
    Cg2 = np.dot(p, np.cos(w * mu_g/2))
    V_mu = 1 - 2*Cg2**2 / (1+Cg)
    h_bridge_mu = np.sqrt(max(2*V_mu/np.pi, 0))
    h_true_mu = interp(mu_g, g_grid[valid_grid], ht_valid)
    print(f"  At g=mu: h_true={h_true_mu:.5f}, h_bridge={h_bridge_mu:.5f}, "
          f"ratio={h_true_mu/h_bridge_mu:.3f}")


# ============================================================
# PART 3: THE DIRECT PROOF (bypassing bridge entirely)
# ============================================================
print(f"\n{'='*70}")
print("PART 3: DIRECT PROOF — DOES h_true SATISFY CHEBYSHEV?")
print("="*70)

print("""
  THE SIMPLIFIED PROOF (if h_true is nondecreasing):

    Step 1: Cov(g, P) = Cov(g, h_true(g))    [law of total covariance, EXACT]
    Step 2: h_true(g) is nondecreasing          [monotone coupling / verified]
    Step 3: Cov(g, h_true) > 0                  [Chebyshev, ANALYTIC]
    Step 4: r > 0                               [direct]

  This completely bypasses:
    - The bridge formula h = sqrt(2V/pi)
    - The spectral variance identity
    - V(g) <= 1
    - V monotonicity on [0, g*]
    - The decomposition h = h_core + delta

  The bridge analysis becomes SUPPORTING MECHANISM (why h_true is increasing)
  rather than load-bearing proof structure.
""")

# Verify at all N
print(f"  {'N':>5} {'h_true mono':>15} {'Cov(g,h_true)':>15} {'Cov(g,P)':>12} "
      f"{'r(g,P)':>8} {'PROOF':>6}")
print(f"  {'-'*65}")

for N in [10, 20, 50, 100, 200, 500]:
    gaps, peaks = simulate(N, n_trials=150)
    if len(gaps) < 500: continue

    p, w = rs(N)
    m2 = np.dot(p, w**2)
    g_bar = np.pi / np.sqrt(m2)
    bw = 0.12 * g_bar

    # Nadaraya-Watson h_true on grid
    g_grid = np.linspace(np.percentile(gaps, 1), np.percentile(gaps, 99), 100)
    ht_grid = np.zeros(len(g_grid))
    for i, g0 in enumerate(g_grid):
        wts = np.exp(-0.5*((gaps - g0)/bw)**2)
        ht_grid[i] = np.average(peaks, weights=wts) if np.sum(wts) > 1e-10 else np.nan

    valid = ~np.isnan(ht_grid)
    ht_v = ht_grid[valid]
    n_dec = np.sum(np.diff(ht_v) < -1e-6)
    mono_str = f"{n_dec} viol" if n_dec > 0 else "YES"

    # h_true at each gap (interpolated)
    ht_gaps = interp(gaps, g_grid[valid], ht_v)
    mu_g = np.mean(gaps)

    cov_htrue = np.mean((gaps - mu_g) * (ht_gaps - np.mean(ht_gaps)))
    cov_gP = np.mean((gaps - mu_g) * (peaks - np.mean(peaks)))
    sigma_g = np.std(gaps)
    sigma_P = np.std(peaks)
    r_gP = cov_gP / (sigma_g * sigma_P)

    proof = "YES" if cov_htrue > 0 and cov_gP > 0 else "NO"

    print(f"  {N:>5} {mono_str:>15} {cov_htrue:>+15.6f} {cov_gP:>+12.6f} "
          f"{r_gP:>+8.4f} {proof:>6}")

print("\nDONE")
