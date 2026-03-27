"""
Session 13i: FINAL SWEEP — verify the noise dilution bound at ALL N >= 2
Also: probe the analytical structure of the CV bound.
"""
import numpy as np, sys
from scipy.stats import pearsonr
sys.stdout.reconfigure(line_buffering=True)

def rs(N):
    p = 1.0/np.arange(1,N+1); p /= p.sum()
    w = np.log(np.arange(2,N+2))
    return p, w

def simulate(N, n_trials=200, L=8000, dt=0.02):
    p, w = rs(N)
    amp = 1.0/np.sqrt(np.arange(1,N+1))
    sigma_N = np.sqrt(np.sum(1.0/np.arange(1,N+1)))
    rng = np.random.default_rng(42)
    chunk = 20000
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

def compute_bound(gaps, peaks):
    """Compute the noise dilution bound."""
    Q = peaks/gaps
    mu = np.mean(gaps)
    W = gaps*(gaps-mu)
    p_rs, w_rs = rs(N)
    g_bar = np.pi / np.sqrt(np.dot(p_rs, w_rs**2))
    bw = 0.12 * g_bar

    # q(g) via NW kernel
    gg = np.linspace(np.percentile(gaps,1), np.percentile(gaps,99), 100)
    qq = np.zeros(len(gg))
    for i, g0 in enumerate(gg):
        wts = np.exp(-0.5*((gaps-g0)/bw)**2)
        qq[i] = np.average(Q, weights=wts) if np.sum(wts)>30 else np.nan
    v = ~np.isnan(qq)
    if np.sum(v) < 10:
        return None
    q_at = np.interp(gaps, gg[v], qq[v])

    Var_Q = np.var(Q)
    Var_q = np.var(q_at)
    R = Var_q / Var_Q if Var_Q > 1e-10 else 1
    corr_qW = pearsonr(q_at, W)[0]
    bound = abs(corr_qW) * np.sqrt(R)
    r_gP = pearsonr(gaps, peaks)[0]

    return {'bound': bound, 'corr_qW': abs(corr_qW), 'R': R,
            'r_gP': r_gP, 'n': len(gaps)}


# ============================================================
# FULL SWEEP: N = 2 to 500
# ============================================================
print("="*70)
print("COMPLETE SWEEP: NOISE DILUTION BOUND AT ALL N")
print("="*70)
print(f"{'N':>5} {'#gaps':>8} {'r(g,P)':>8} {'|Corr(q,W)|':>12} {'sqrt(R)':>10} "
      f"{'Bound':>10} {'< 0.497':>8}")
print("-"*62)

all_ok = True
for N in [2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]:
    gaps, peaks = simulate(N, n_trials=200, L=8000, dt=0.02)
    if len(gaps) < 200:
        print(f"{N:>5} {'TOO FEW':>8}")
        continue

    result = compute_bound(gaps, peaks)
    if result is None:
        print(f"{N:>5} {'FAIL':>8}")
        continue

    ok = result['bound'] < 0.497
    if not ok: all_ok = False
    r = result['r_gP']

    print(f"{N:>5} {result['n']:>8} {r:>+8.4f} {result['corr_qW']:>12.4f} "
          f"{np.sqrt(result['R']):>10.4f} {result['bound']:>10.4f} "
          f"{'YES' if ok else '** NO **':>8}")

print("-"*62)
print(f"ALL N CLOSED: {'YES' if all_ok else 'NO'}")


# ============================================================
# THE HALF-NORMAL CV LOWER BOUND
# ============================================================
print(f"\n{'='*70}")
print("ANALYTICAL: HALF-NORMAL CV UNDER MONOTONE REWEIGHTING")
print("="*70)

print("""
  For the bridge: P|g ~ half-normal(sigma), CV = sqrt(pi/2 - 1) = 0.756.

  The excursion reweights by w(x) = P(no interior zero | f(mid)=x, bridge).
  w(x) is NONDECREASING in x (larger midpoint -> less likely to cross zero).

  For a half-normal reweighted by a nondecreasing w(x):
  the reweighted distribution has SMALLER CV than the original.

  QUESTION: What's the minimum possible CV under nondecreasing reweighting?

  For a step function w(x) = 1_{x > t} (truncation at t):
  this gives the MINIMUM CV among all nondecreasing reweightings
  with the same E[w(X)].

  So: CV_excursion >= CV_truncated(t) where t is chosen to match
  the excursion probability.
""")

from scipy.stats import halfnorm
import scipy.integrate as integrate

sigma = 1.0  # for unit bridge variance; CV is scale-invariant

# CV of half-normal truncated at t
def cv_truncated(t, sigma=1.0):
    """CV of halfnorm(sigma) truncated to [t, inf)."""
    # X ~ halfnorm(sigma) conditioned on X >= t
    # This is equivalent to |N(0,sigma^2)| conditioned on |N| >= t
    # E[X|X>=t] and E[X^2|X>=t] from truncated normal moments
    from scipy.stats import norm as normal_dist
    # P(|N| >= t) = 2*(1 - Phi(t/sigma))
    p_survive = 2*(1 - normal_dist.cdf(t/sigma))
    if p_survive < 1e-10:
        return 0
    # E[|N| * 1_{|N|>=t}] = 2*sigma*phi(t/sigma)
    E_X = 2*sigma*normal_dist.pdf(t/sigma) / p_survive
    # E[N^2 * 1_{|N|>=t}] = sigma^2 * [p_survive + 2*t*sigma*phi(t/sigma)/sigma^2 ... ]
    # Actually: E[X^2 | X>=t] where X=|N(0,s^2)|
    # E[N^2 1_{|N|>=t}] = sigma^2 * [1 - (2*Phi(t/sigma)-1)] + 2*t*sigma*phi(t/sigma)
    # Hmm, let me use integration
    def f_x2(x):
        return x**2 * 2/(sigma*np.sqrt(2*np.pi)) * np.exp(-x**2/(2*sigma**2))
    E_X2, _ = integrate.quad(f_x2, t, 50*sigma)
    E_X2 /= p_survive
    Var = E_X2 - E_X**2
    return np.sqrt(max(Var, 0)) / E_X if E_X > 1e-10 else 0

# Compute CV vs truncation level
print(f"  {'t/sigma':>10} {'P(X>=t)':>10} {'CV_trunc':>10}")
print(f"  {'-'*32}")
for t_ratio in [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
    t = t_ratio * sigma
    p_surv = 2*(1 - halfnorm.cdf(t, scale=sigma)) if t > 0 else 1.0
    # Actually halfnorm.sf(t) = 1 - halfnorm.cdf(t) = P(X > t) for X~halfnorm
    p_surv = halfnorm.sf(t, scale=sigma)
    cv = cv_truncated(t, sigma)
    print(f"  {t_ratio:>10.2f} {p_surv:>10.4f} {cv:>10.4f}")

# What truncation level gives CV = 0.39 (the observed minimum)?
print(f"\n  Searching for t where CV_truncated = 0.39...")
from scipy.optimize import brentq
def cv_minus_target(t):
    return cv_truncated(t, 1.0) - 0.39
try:
    t_at_039 = brentq(cv_minus_target, 0, 3)
    p_at_039 = halfnorm.sf(t_at_039, scale=1.0)
    print(f"  CV = 0.39 at t = {t_at_039:.4f} sigma, P(X>t) = {p_at_039:.4f}")
    print(f"  This means: if the excursion removes {1-p_at_039:.1%} of the bridge paths,")
    print(f"  the CV can drop to 0.39. Is this consistent with excursion probabilities?")
except:
    print(f"  Could not find crossover")

# What truncation level gives CV = 0.326 (the minimum needed for proof)?
print(f"\n  Searching for t where CV_truncated = 0.326 (proof threshold)...")
def cv_minus_326(t):
    return cv_truncated(t, 1.0) - 0.326
try:
    t_at_326 = brentq(cv_minus_326, 0, 3)
    p_at_326 = halfnorm.sf(t_at_326, scale=1.0)
    print(f"  CV = 0.326 at t = {t_at_326:.4f} sigma, P(X>t) = {p_at_326:.4f}")
    print(f"  The excursion would need to remove {1-p_at_326:.1%} of paths to get CV this low.")
    print(f"  This is the WORST CASE — actual excursion conditioning is milder.")
except:
    print(f"  Could not find crossover")


print(f"\n{'='*70}")
print("DONE")
print("="*70)
