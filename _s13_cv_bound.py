"""
Session 13h: PROVE THE CONDITIONAL CV BOUND
=============================================

Target: show CV(Q|g) = CV(P/g | g) >= c for some universal c > 0.

Q|g = P/g | g = |f(mid)|/g | gap = g

For the bridge (no excursion conditioning):
  f(mid) | f(0)=f(g)=0 ~ N(0, V(g))
  P = |f(mid)| ~ half-normal with scale sqrt(V(g))
  E[P|g] = sqrt(2V(g)/pi)
  Var(P|g) = V(g)(1 - 2/pi)
  CV(P|g) = sqrt(pi/2 - 1) = 0.7555  (UNIVERSAL, independent of g!)
  CV(Q|g) = CV(P|g) = 0.7555  (dividing by g doesn't change CV)

For the excursion (conditioning on no interior zeros):
  The distribution of f(mid) is NO LONGER half-normal.
  It's the bridge conditioned on f > 0 throughout (0,g).
  This removes low-|f(mid)| paths, reducing variance more than mean.
  So CV(Q|g, excursion) < 0.7555.

But the simulation shows CV(Q|g, excursion) ~ 0.40-0.60.

KEY QUESTION: Can we prove CV(Q|g, excursion) >= 0.35 for all g?

If yes: noise fraction >= c^2/(1+c^2) per gap, giving a universal
lower bound on the total noise fraction.

APPROACH: The excursion midpoint value comes from a TRUNCATED Gaussian.
The bridge gives f(mid) ~ N(0, V(g)), and the excursion conditions
on f staying positive on (0,g). If we can bound the probability of
"no interior zero" given f(mid) = x, we can bound the excursion
distribution of f(mid) and hence its CV.
"""
import numpy as np, sys
from scipy.stats import pearsonr, norm
sys.stdout.reconfigure(line_buffering=True)

def rs(N):
    p = 1.0/np.arange(1,N+1); p /= p.sum()
    w = np.log(np.arange(2,N+2))
    return p, w

def simulate_full(N, n_trials=200, L=5000, dt=0.01):
    p, w = rs(N)
    amp = 1.0/np.sqrt(np.arange(1,N+1))
    sigma_N = np.sqrt(np.sum(1.0/np.arange(1,N+1)))
    rng = np.random.default_rng(42)
    chunk = 40000
    all_g, all_P = [], []
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
        all_P.extend(pks[tr:-tr].tolist())
    return np.array(all_g), np.array(all_P)


# ============================================================
# PART 1: CV(Q|g) PROFILE — BRIDGE vs EXCURSION
# ============================================================
print("="*70)
print("PART 1: CV(Q|g) = CV(P/g | g) PROFILE")
print("="*70)

print(f"\n  Bridge prediction: CV = sqrt(pi/2 - 1) = {np.sqrt(np.pi/2 - 1):.4f} (universal)")

for N in [50, 200]:
    print(f"\n  N={N}: simulating...", flush=True)
    gaps, peaks = simulate_full(N)
    Q = peaks / gaps
    print(f"  {len(gaps)} observations")

    p_rs, w_rs = rs(N)
    g_bar = np.pi / np.sqrt(np.dot(p_rs, w_rs**2))

    # CV(Q|g) by fine bins
    n_bins = 40
    edges = np.percentile(gaps, np.linspace(0, 100, n_bins+1))
    edges[-1] += 0.001

    print(f"\n  {'g/g_bar':>8} {'E[Q|g]':>10} {'std(Q|g)':>10} {'CV(Q|g)':>10} "
          f"{'CV_bridge':>10} {'ratio':>8} {'n':>6}")
    print(f"  {'-'*62}")

    cv_exc_all = []
    for i in range(n_bins):
        mask = (gaps >= edges[i]) & (gaps < edges[i+1])
        n = np.sum(mask)
        if n < 100: continue
        g_mid = np.mean(gaps[mask])
        q_mean = np.mean(Q[mask])
        q_std = np.std(Q[mask])
        cv_exc = q_std / q_mean if q_mean > 0.01 else np.nan
        cv_bridge = np.sqrt(np.pi/2 - 1)  # 0.7555
        cv_exc_all.append(cv_exc)

        if i % 4 == 0:
            print(f"  {g_mid/g_bar:>8.3f} {q_mean:>10.4f} {q_std:>10.4f} {cv_exc:>10.4f} "
                  f"{cv_bridge:>10.4f} {cv_exc/cv_bridge:>8.3f} {n:>6}")

    cv_min = np.nanmin(cv_exc_all)
    cv_mean = np.nanmean(cv_exc_all)
    print(f"\n  CV(Q|g) range: [{cv_min:.4f}, {np.nanmax(cv_exc_all):.4f}]")
    print(f"  CV(Q|g) mean:  {cv_mean:.4f}")
    print(f"  CV(Q|g) min:   {cv_min:.4f} (this is the bottleneck)")
    print(f"  Bridge CV:     0.7555")
    print(f"  Excursion reduces CV by factor: {cv_min/0.7555:.3f} at minimum")


# ============================================================
# PART 2: WHAT'S THE MINIMUM CV ACROSS ALL g AND ALL N?
# ============================================================
print(f"\n{'='*70}")
print("PART 2: MINIMUM CV(Q|g) ACROSS ALL N")
print("="*70)

print(f"{'N':>5} {'min CV':>10} {'at g/gbar':>10} {'mean CV':>10}")
print("-"*38)

for N in [10, 20, 50, 100, 200]:
    gaps, peaks = simulate_full(N, n_trials=150)
    Q = peaks/gaps
    p_rs, w_rs = rs(N)
    g_bar = np.pi / np.sqrt(np.dot(p_rs, w_rs**2))
    # actually let me fix that
    p_rs, w_rs = rs(N)
    g_bar = np.pi / np.sqrt(np.dot(p_rs, w_rs**2))

    edges = np.percentile(gaps, np.linspace(0, 100, 41))
    edges[-1] += 0.001
    min_cv = 999; min_g = 0; cvs = []
    for i in range(40):
        mask = (gaps >= edges[i]) & (gaps < edges[i+1])
        n = np.sum(mask)
        if n < 50: continue
        q_mean = np.mean(Q[mask])
        q_std = np.std(Q[mask])
        cv = q_std/q_mean if q_mean > 0.01 else 999
        cvs.append(cv)
        if cv < min_cv:
            min_cv = cv; min_g = np.mean(gaps[mask])

    print(f"{N:>5} {min_cv:>10.4f} {min_g/g_bar:>10.3f} {np.mean(cvs):>10.4f}")


# ============================================================
# PART 3: ANALYTICAL LOWER BOUND ON CV
# ============================================================
print(f"\n{'='*70}")
print("PART 3: ANALYTICAL ARGUMENT FOR CV LOWER BOUND")
print("="*70)

print("""
  For the BRIDGE: f(mid) | bridge ~ N(0, V(g)), so P ~ |N(0,V)|.
  CV(P|g) = sqrt(pi/2 - 1) = 0.7555 (EXACT, universal).

  For the EXCURSION: f(mid) | excursion is NOT half-normal.
  It's the bridge conditioned on "f > 0 on (0,g)" (for positive excursion).

  The excursion conditioning is EQUIVALENT to:
    - Reweight the bridge distribution of f(mid) by the probability
      P(no zero in (0,g) | f(0)=0, f(g)=0, f(mid)=x)
    - This reweighting favors large x (larger midpoint → harder to cross zero)

  Let w(x) = P(no zero | f(mid) = x) / P(no zero) be the likelihood ratio.
  Then the excursion distribution of f(mid) is:
    rho_exc(x) = rho_bridge(x) * w(x) / Z

  where Z = E_bridge[w(f(mid))].

  KEY INSIGHT: w(x) is bounded above and below.
  w(x) <= 1/P(no zero) (trivially, since w is a probability ratio)
  w(x) >= something > 0 (the excursion ALWAYS has positive probability
  of no zero, even for small x, because the two halves can still avoid zero)

  If w(x) is not too concentrated (i.e., the reweighting is not too extreme),
  then the CV of the excursion distribution is not too far from 0.7555.

  QUESTION: Is the reweighting "gentle" enough to keep CV >= 0.35?
""")

# Compute the reweighting empirically
# For each (gap, peak) observation, the "bridge probability" is prop to
# the half-normal density at peak with scale sqrt(V(g)).
# The excursion distribution is the empirical distribution.
# The ratio gives us w(x).

N = 50
print(f"  Computing reweighting w(x) for N={N}...", flush=True)
gaps, peaks = simulate_full(N, n_trials=200)
Q = peaks/gaps
p_rs, w_rs = rs(N)
g_bar = np.pi / np.sqrt(np.dot(p_rs, w_rs**2))

# At the median gap
g_med = np.median(gaps)
bw_g = 0.05 * g_bar
mask = np.abs(gaps - g_med) < bw_g
Q_slice = Q[mask]
P_slice = peaks[mask]
g_slice = gaps[mask]

print(f"  Gap ~ {g_med/g_bar:.2f} g_bar, {np.sum(mask)} observations")

# Bridge V(g) at median gap
Cg = np.dot(p_rs, np.cos(w_rs * g_med))
Cg2 = np.dot(p_rs, np.cos(w_rs * g_med/2))
V_med = 1 - 2*Cg2**2/(1+Cg)

# Bridge distribution: P ~ half-normal(sqrt(V))
# p_bridge(x) = 2/sqrt(2*pi*V) * exp(-x^2/(2V)) for x > 0
# Excursion distribution: empirical histogram

import matplotlib
# No plotting, just compute stats

# Bridge prediction
E_bridge = np.sqrt(2*V_med/np.pi)
Var_bridge = V_med * (1 - 2/np.pi)
CV_bridge = np.sqrt(Var_bridge) / E_bridge

E_exc = np.mean(P_slice)
Var_exc = np.var(P_slice)
CV_exc = np.sqrt(Var_exc) / E_exc

print(f"  Bridge: E[P]={E_bridge:.4f}, Var={Var_bridge:.4f}, CV={CV_bridge:.4f}")
print(f"  Excursion: E[P]={E_exc:.4f}, Var={Var_exc:.4f}, CV={CV_exc:.4f}")
print(f"  CV ratio: {CV_exc/CV_bridge:.4f}")

# The excursion mean is higher (1.5x) but variance is also affected
# Net CV reduction factor ~ 0.55

# How does the reweighting look?
# For bins of P, compute the ratio of excursion density to bridge density
P_bins = np.linspace(0, np.percentile(P_slice, 99), 20)
print(f"\n  {'P range':>15} {'frac(exc)':>10} {'frac(bridge)':>12} {'w ratio':>10}")
for i in range(len(P_bins)-1):
    in_bin = (P_slice >= P_bins[i]) & (P_slice < P_bins[i+1])
    frac_exc = np.mean(in_bin)
    # Bridge fraction: half-normal CDF difference
    x_lo, x_hi = P_bins[i], P_bins[i+1]
    from scipy.stats import halfnorm
    frac_bridge = halfnorm.cdf(x_hi, scale=np.sqrt(V_med)) - halfnorm.cdf(x_lo, scale=np.sqrt(V_med))
    w_ratio = frac_exc / frac_bridge if frac_bridge > 1e-6 else np.nan
    if i % 2 == 0:
        print(f"  [{P_bins[i]:.2f},{P_bins[i+1]:.2f}) {frac_exc:>10.4f} {frac_bridge:>12.4f} {w_ratio:>10.3f}")


# ============================================================
# PART 4: THE PROOF PATH — WHAT BOUNDS SUFFICE?
# ============================================================
print(f"\n{'='*70}")
print("PART 4: WHAT BOUNDS CLOSE THE PROOF?")
print("="*70)

# We need: |Corr(q,W)| * sqrt(R) < 0.497
# R = Var(q)/Var(Q) = 1 - E[Var(Q|g)]/Var(Q)
#
# E[Var(Q|g)] = E[(CV(Q|g) * E[Q|g])^2]  ... not quite
# Var(Q) = Var(q(g)) + E[Var(Q|g)]
# E[Var(Q|g)] = E[CV(Q|g)^2 * q(g)^2]
#
# If CV(Q|g) >= c for all g:
# E[Var(Q|g)] >= c^2 * E[q(g)^2]
# And: Var(Q) = Var(q) + E[Var(Q|g)]
# So: R = Var(q)/(Var(q) + E[Var(Q|g)]) <= Var(q)/(Var(q) + c^2*E[q^2])
#
# Since E[q^2] = Var(q) + E[q]^2:
# R <= Var(q) / (Var(q) + c^2*(Var(q) + E[q]^2))
# = 1 / (1 + c^2*(1 + E[q]^2/Var(q)))
# = 1 / (1 + c^2*(1 + 1/CV(q)^2))
# = 1 / (1 + c^2/CV(q)^2 + c^2)
#   ... hmm, let me simplify
# R <= 1 / (1 + c^2 * E[q^2]/Var(q))
# = 1 / (1 + c^2 * (1 + mu_q^2/Var(q)))
# = 1 / (1 + c^2 * (1 + 1/CV_q^2))

# From data: CV_q = sigma_q / mu_q ~ 0.42 (at N=50)
# c = min CV(Q|g) ~ 0.40

# Let me compute this bound
for N in [50, 200]:
    gaps, peaks = simulate_full(N, n_trials=150)
    Q = peaks/gaps; mu = np.mean(gaps)
    W = gaps*(gaps-mu)
    p_rs, w_rs = rs(N)
    g_bar = np.pi / np.sqrt(np.dot(p_rs, w_rs**2))
    bw = 0.10*g_bar

    # Compute q(g) and CV(Q|g)
    gg = np.linspace(np.percentile(gaps,1), np.percentile(gaps,99), 100)
    qq = np.zeros(len(gg))
    vqg = np.zeros(len(gg))
    for i, g0 in enumerate(gg):
        wts = np.exp(-0.5*((gaps-g0)/bw)**2)
        if np.sum(wts) > 50:
            qq[i] = np.average(Q, weights=wts)
            vqg[i] = np.average((Q-qq[i])**2, weights=wts)
        else:
            qq[i] = np.nan; vqg[i] = np.nan

    v = ~np.isnan(qq)
    q_at = np.interp(gaps, gg[v], qq[v])

    Var_Q = np.var(Q)
    Var_q = np.var(q_at)
    R_actual = Var_q / Var_Q
    corr_qW = pearsonr(q_at, W)[0]

    # Measured quantities
    min_cv = np.sqrt(np.nanmin(vqg[v]/qq[v]**2))
    mu_q = np.mean(q_at)
    CV_q = np.std(q_at)/mu_q

    # Analytical bound on R given min CV(Q|g) = c
    c = min_cv
    R_bound = 1.0 / (1.0 + c**2 * (1.0 + 1.0/CV_q**2))

    # Bound on the full product
    product_bound = abs(corr_qW) * np.sqrt(R_bound)
    product_actual = abs(corr_qW) * np.sqrt(R_actual)

    print(f"\n  N={N}:")
    print(f"    min CV(Q|g) = c = {c:.4f}")
    print(f"    CV(q(g)) = {CV_q:.4f}")
    print(f"    R_actual = {R_actual:.4f}")
    print(f"    R_bound (from c) = {R_bound:.4f}")
    print(f"    |Corr(q,W)| = {abs(corr_qW):.4f}")
    print(f"    Product (actual) = {product_actual:.4f}")
    print(f"    Product (bound) = {product_bound:.4f}")
    print(f"    Threshold = 0.497")
    print(f"    CLOSED (actual): {'YES' if product_actual < 0.497 else 'NO'}")
    print(f"    CLOSED (bound):  {'YES' if product_bound < 0.497 else 'NO'}")

    # What min c would close the proof?
    # Need: |corr_qW| * sqrt(R_bound(c)) < 0.497
    # R_bound(c) = 1/(1 + c^2*(1+1/CV_q^2))
    # |corr|^2 * R_bound < 0.497^2
    # 1/(1 + c^2*(1+1/CV_q^2)) < 0.497^2/|corr|^2
    # 1 + c^2*(1+1/CV_q^2) > |corr|^2/0.497^2
    # c^2 > (|corr|^2/0.497^2 - 1) / (1+1/CV_q^2)
    corr_sq = corr_qW**2
    needed_c2 = (corr_sq/0.497**2 - 1) / (1 + 1/CV_q**2)
    needed_c = np.sqrt(max(needed_c2, 0))
    print(f"    Min c needed to close: {needed_c:.4f}")
    print(f"    Actual min c: {c:.4f}")
    print(f"    Headroom: {c/needed_c:.2f}x" if needed_c > 0 else "")


print(f"\n{'='*70}")
print("DONE")
print("="*70)
