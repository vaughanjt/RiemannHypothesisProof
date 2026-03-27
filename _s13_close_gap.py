"""
Session 13g: CLOSE THE GAP
============================

Target: prove |Corr(Q, W)| < 0.497 where Q = P/g, W = g(g-mu).

KEY IDENTITY (noise dilution):
  |Corr(Q, W)| = |Corr(q(g), W)| * sigma_q / sigma_Q

  where q(g) = E[Q|g] = h(g)/g, and sigma_q/sigma_Q < 1 because
  Q has stochastic noise beyond the deterministic function q(g).

  Specifically: sigma_q/sigma_Q = sqrt(R) where
  R = Var(q(g)) / Var(Q) = 1 - E[Var(Q|g)] / Var(Q)

  If we can show |Corr(q(g), W)| * sqrt(R) < 0.497, PROOF IS DONE.

QUESTION: Is the noise dilution enough?
"""
import numpy as np, sys
from scipy.stats import pearsonr
sys.stdout.reconfigure(line_buffering=True)

def rs(N):
    p = 1.0/np.arange(1,N+1); p /= p.sum()
    w = np.log(np.arange(2,N+2))
    return p, w

def simulate_full(N, n_trials=150, L=5000, dt=0.01):
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


print("="*70)
print("NOISE DILUTION ATTACK ON THE MOMENT INEQUALITY")
print("="*70)

for N in [50, 200]:
    print(f"\n{'='*70}")
    print(f"N = {N}")
    print(f"{'='*70}")
    gaps, peaks = simulate_full(N, n_trials=200)
    print(f"{len(gaps)} observations")

    Q = peaks / gaps
    mu_g = np.mean(gaps)
    W = gaps * (gaps - mu_g)

    # Step 1: Compute q(g) = E[Q|g] via Nadaraya-Watson
    p_rs, w_rs = rs(N)
    m2 = np.dot(p_rs, w_rs**2)
    g_bar = np.pi / np.sqrt(m2)
    bw = 0.10 * g_bar

    g_grid = np.linspace(np.percentile(gaps, 0.5), np.percentile(gaps, 99.5), 200)
    q_grid = np.zeros(len(g_grid))
    var_Q_given_g = np.zeros(len(g_grid))  # Var(Q|g) at each grid point

    for i, g0 in enumerate(g_grid):
        wts = np.exp(-0.5*((gaps - g0)/bw)**2)
        if np.sum(wts) > 100:
            q_grid[i] = np.average(Q, weights=wts)
            # Conditional variance
            var_Q_given_g[i] = np.average((Q - q_grid[i])**2, weights=wts)
        else:
            q_grid[i] = np.nan
            var_Q_given_g[i] = np.nan

    # Interpolate q(g) at each gap
    valid = ~np.isnan(q_grid)
    q_at_gaps = np.interp(gaps, g_grid[valid], q_grid[valid])

    # Step 2: Compute the three key quantities
    Var_Q = np.var(Q)
    Var_q = np.var(q_at_gaps)  # Var(q(g)) = Var(E[Q|g])
    E_VarQg = Var_Q - Var_q    # E[Var(Q|g)] by law of total variance
    R = Var_q / Var_Q           # signal variance ratio

    # Corr(q(g), W) — deterministic correlation
    corr_qW = pearsonr(q_at_gaps, W)[0]

    # Corr(Q, W) — actual
    corr_QW = pearsonr(Q, W)[0]

    # The bound: |Corr(Q, W)| = |Corr(q, W)| * sqrt(R)
    bound = abs(corr_qW) * np.sqrt(R)

    print(f"\n  NOISE DILUTION DECOMPOSITION:")
    print(f"  Var(Q)     = {Var_Q:.6f}")
    print(f"  Var(q(g))  = {Var_q:.6f}  (signal)")
    print(f"  E[Var(Q|g)]= {E_VarQg:.6f}  (noise)")
    print(f"  R = signal/total = {R:.4f}")
    print(f"  sqrt(R)    = {np.sqrt(R):.4f}")
    print(f"")
    print(f"  |Corr(q(g), W)| = {abs(corr_qW):.4f}  (deterministic)")
    print(f"  |Corr(Q, W)|    = {abs(corr_QW):.4f}  (actual)")
    print(f"  Predicted bound  = {bound:.4f}")
    print(f"  Threshold        = 0.4970")
    print(f"")
    print(f"  BOUND < THRESHOLD: {'YES' if bound < 0.497 else 'NO'}")
    if bound < 0.497:
        print(f"  *** PROOF WORKS via noise dilution! ***")
        print(f"  Gap: threshold - bound = {0.497 - bound:.4f}")
    else:
        print(f"  Need: |Corr(q,W)| * sqrt(R) < 0.497")
        print(f"  Have: {abs(corr_qW):.4f} * {np.sqrt(R):.4f} = {bound:.4f}")
        print(f"  Deficit: {bound - 0.497:.4f}")

    # Step 3: What if we could prove |Corr(q, W)| < some constant?
    print(f"\n  ANALYSIS OF Corr(q(g), W):")
    print(f"  q(g) = h(g)/g shape: peaks at {g_grid[valid][np.argmax(q_grid[valid])]/g_bar:.2f} g_bar")
    print(f"  W = g(g-mu) shape: zero at g=mu = {mu_g:.4f} = {mu_g/g_bar:.2f} g_bar")

    # Decompose the correlation region by region
    below = gaps < mu_g
    above = gaps >= mu_g
    contrib_below = np.sum((q_at_gaps[below]-np.mean(q_at_gaps))*(W[below]-np.mean(W)))/len(gaps)
    contrib_above = np.sum((q_at_gaps[above]-np.mean(q_at_gaps))*(W[above]-np.mean(W)))/len(gaps)
    print(f"  Cov(q,W) contribution from g < mu: {contrib_below:+.6f}")
    print(f"  Cov(q,W) contribution from g >= mu: {contrib_above:+.6f}")

    # Step 4: What fraction of Q's variance is noise?
    print(f"\n  NOISE ANALYSIS:")
    print(f"  Signal fraction R = {R:.4f} ({R*100:.1f}%)")
    print(f"  Noise fraction 1-R = {1-R:.4f} ({(1-R)*100:.1f}%)")

    # Conditional variance by gap quantile
    print(f"\n  Var(Q|g) by gap quantile:")
    for pct in [10, 25, 50, 75, 90]:
        g_val = np.percentile(gaps, pct)
        idx = np.argmin(np.abs(g_grid[valid] - g_val))
        vqg = var_Q_given_g[valid][idx]
        qg = q_grid[valid][idx]
        print(f"    g = {pct}th pctile ({g_val/g_bar:.2f} g_bar): "
              f"E[Q|g]={qg:.4f}, Var(Q|g)={vqg:.4f}, "
              f"CV(Q|g)={np.sqrt(vqg)/qg:.3f}" if qg > 0.01 else "")


# ============================================================
# PART 2: SWEEP ACROSS N — does noise dilution close the gap?
# ============================================================
print(f"\n{'='*70}")
print("SWEEP: NOISE DILUTION ACROSS ALL N")
print("="*70)

print(f"{'N':>5} {'|Corr(q,W)|':>12} {'sqrt(R)':>10} {'Bound':>10} "
      f"{'Threshold':>10} {'CLOSED?':>8}")
print("-"*58)

for N in [10, 20, 50, 100, 200]:
    gaps, peaks = simulate_full(N, n_trials=150)
    if len(gaps) < 500: continue

    Q = peaks/gaps; mu = np.mean(gaps); W = gaps*(gaps-mu)
    p_rs, w_rs = rs(N)
    g_bar = np.pi / np.sqrt(np.dot(p_rs, w_rs**2))
    bw = 0.10 * g_bar

    # q(g) via NW
    gg = np.linspace(np.percentile(gaps,1), np.percentile(gaps,99), 150)
    qq = np.zeros(len(gg))
    for i, g0 in enumerate(gg):
        wts = np.exp(-0.5*((gaps-g0)/bw)**2)
        qq[i] = np.average(Q, weights=wts) if np.sum(wts)>50 else np.nan
    v = ~np.isnan(qq)
    q_at = np.interp(gaps, gg[v], qq[v])

    Var_Q = np.var(Q); Var_q = np.var(q_at)
    R = Var_q / Var_Q
    corr_qW = pearsonr(q_at, W)[0]
    bound = abs(corr_qW) * np.sqrt(R)
    closed = bound < 0.497

    print(f"{N:>5} {abs(corr_qW):>12.4f} {np.sqrt(R):>10.4f} {bound:>10.4f} "
          f"{'0.4970':>10} {'YES' if closed else 'NO':>8}")


print(f"\n{'='*70}")
print("DONE")
print("="*70)
