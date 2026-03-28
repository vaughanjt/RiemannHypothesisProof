"""
Session 13N: TWO-COMPONENT CV BOUND — THE CLOSING CALCULATION
==============================================================

CV(P|g)^2 = Var(P|g) / E[P|g]^2

Var(P|g) >= Var(E[P|g, f'(0)]) = a(g)^2 * Var(|f'(0)| | gap=g)

(Law of total variance: total variance >= variance of conditional mean)

This gives a LOWER BOUND on CV that uses only the regression component.
The Slepian residual V_4 ADDS to this, making the actual CV even larger.

KEY INSIGHT: even ignoring V_4, the regression variance alone should
give CV >= 0.326 because:
  E[P | g, f'(0)=y] ~ a(g)*y  (regression)
  P = a(g)*|f'(0)| + noise
  CV(P|g) >= CV(a(g)*|f'(0)| | g) = CV(|f'(0)| | g)

If |f'(0)| given gap g has CV >= 0.326, we're done WITHOUT needing V_4!

The CV of |f'(0)| given gap g is the CV of the derivative at the zero,
conditional on the gap length. For a half-normal, CV = 0.756.
For the excursion, it's reduced but should be >= 0.33.
"""
import numpy as np, sys
from scipy.stats import pearsonr, norm
sys.stdout.reconfigure(line_buffering=True)

def rs(N):
    p = 1.0/np.arange(1,N+1); p /= p.sum()
    w = np.log(np.arange(2,N+2))
    return p, w

def simulate_with_deriv(N, n_trials=200, L=5000, dt=0.01):
    p, w = rs(N)
    amp = 1.0/np.sqrt(np.arange(1,N+1))
    sigma_N = np.sqrt(np.sum(1.0/np.arange(1,N+1)))
    rng = np.random.default_rng(42)
    chunk = 40000
    all_g, all_P, all_fp = [], [], []
    for trial in range(n_trials):
        phi = rng.uniform(0, 2*np.pi, N)
        npts = int(L/dt)
        f = np.empty(npts); fp = np.empty(npts)
        for s in range(0, npts, chunk):
            e = min(s+chunk, npts)
            tc = np.arange(s,e)*dt
            cv = np.cos(np.outer(tc, w)+phi)
            sv = np.sin(np.outer(tc, w)+phi)
            f[s:e] = cv @ amp; fp[s:e] = -(sv @ (amp*w))
        f /= sigma_N; fp /= sigma_N
        t = np.arange(npts)*dt
        sc = np.where(f[:-1]*f[1:]<0)[0]
        if len(sc)<20: continue
        zeros = t[sc] - f[sc]*dt/(f[sc+1]-f[sc])
        gaps = np.diff(zeros)
        fp_at_zeros = np.abs(fp[sc])
        midx = ((zeros[:-1]+zeros[1:])/(2*dt)).astype(int)
        midx = np.clip(midx, 0, npts-1)
        pks = np.abs(f[midx])
        fp_left = fp_at_zeros[:-1]
        tr = max(3, int(0.05*len(gaps)))
        all_g.extend(gaps[tr:-tr].tolist())
        all_P.extend(pks[tr:-tr].tolist())
        all_fp.extend(fp_left[tr:-tr].tolist())
    return np.array(all_g), np.array(all_P), np.array(all_fp)


print("="*70)
print("TWO-COMPONENT CV BOUND")
print("="*70)

for N in [50, 200]:
    print(f"\n{'='*70}")
    print(f"N = {N}")
    print(f"{'='*70}")

    gaps, peaks, fp0 = simulate_with_deriv(N)
    print(f"{len(gaps)} observations")

    Q = peaks/gaps
    p_rs, w_rs = rs(N)
    g_bar = np.pi / np.sqrt(np.dot(p_rs, w_rs**2))

    # Compute CV(Q|g) and CV(|f'(0)||g) by gap bins
    n_bins = 30
    edges = np.percentile(gaps, np.linspace(0, 100, n_bins+1))
    edges[-1] += 0.001

    print(f"\n  {'g/gbar':>8} {'CV(Q|g)':>10} {'CV(fp|g)':>10} {'CV_ratio':>10} "
          f"{'E[Q|g]':>10} {'E[fp|g]':>10} {'n':>6}")
    print(f"  {'-'*60}")

    cv_Q_list = []
    cv_fp_list = []
    cv_bound_holds = True

    for i in range(n_bins):
        mask = (gaps >= edges[i]) & (gaps < edges[i+1])
        n = np.sum(mask)
        if n < 100: continue

        g_mid = np.mean(gaps[mask])
        q_mean = np.mean(Q[mask])
        q_std = np.std(Q[mask])
        fp_mean = np.mean(fp0[mask])
        fp_std = np.std(fp0[mask])

        cv_Q = q_std / q_mean if q_mean > 0.01 else np.nan
        cv_fp = fp_std / fp_mean if fp_mean > 0.01 else np.nan

        cv_Q_list.append(cv_Q)
        cv_fp_list.append(cv_fp)

        if cv_fp < 0.326 and not np.isnan(cv_fp):
            cv_bound_holds = False

        if i % 3 == 0:
            ratio = cv_Q / cv_fp if cv_fp > 0 and not np.isnan(cv_fp) else np.nan
            print(f"  {g_mid/g_bar:>8.3f} {cv_Q:>10.4f} {cv_fp:>10.4f} "
                  f"{ratio:>10.4f} {q_mean:>10.4f} {fp_mean:>10.4f} {n:>6}")

    min_cv_Q = np.nanmin(cv_Q_list)
    min_cv_fp = np.nanmin(cv_fp_list)
    print(f"\n  min CV(Q|g) = {min_cv_Q:.4f}")
    print(f"  min CV(|f'(0)||g) = {min_cv_fp:.4f}")
    print(f"  CV(|f'(0)||g) >= 0.326 everywhere: {'YES' if cv_bound_holds else 'NO'}")

    # THE KEY: if CV(|f'(0)||g) >= 0.326, then by law of total variance:
    # Var(P|g) >= Var(E[P|g, f'(0)]) = a(g)^2 * Var(|f'(0)||g) >= 0.326^2 * a(g)^2 * E[|f'|g]^2
    # And E[P|g] <= a(g)*E[|f'|g] + sqrt(V_4)*sqrt(2/pi)  (mean of regression + mean of residual)
    # Hmm, this bound is not tight enough because E[P|g] != a(g)*E[|f'|g]

    # Better: use CV(P|g) >= CV(|f'(0)||g) * (regression fraction)
    # Since P = a*|f'| + residual(indep of f'), and CV(a*|f'| + noise) depends on signal-to-noise

    # Actually, the SIMPLEST argument:
    # Var(P|g) >= Var(E[P|g, f'(0)]) (law of total variance)
    # E[P|g, f'(0)=y] is a function of y (through the Slepian regression + truncation)
    # If this function has enough variation, Var is bounded below.

    # For small g: P ~ a*|f'(0)|, so Var(E[P|g,f']=a*f') = a^2*Var(f'|g)
    # And E[P|g]^2 ~ a^2*E[f'|g]^2
    # So CV >= CV(f'|g)

    # For larger g: the residual contributes, making CV larger
    # So the minimum CV is CV(|f'(0)||g) at the gap where the residual is smallest

    print(f"\n  CONCLUSION:")
    if min_cv_fp >= 0.326:
        print(f"  CV(|f'(0)| | g) >= {min_cv_fp:.4f} >= 0.326 at ALL gap quantiles")
        print(f"  => By law of total variance: CV(P|g) >= CV(|f'(0)||g) * (signal fraction)")
        print(f"  => This provides the noise fraction bound needed for the proof")
    else:
        print(f"  CV(|f'(0)| | g) drops to {min_cv_fp:.4f} at some gaps")
        print(f"  Need to include V_4 residual to reach 0.326")


# ============================================================
# PART 2: WHY CV(|f'(0)||g) >= 0.33 — THE ANALYTICAL ARGUMENT
# ============================================================
print(f"\n{'='*70}")
print("PART 2: WHY IS CV(|f'(0)| | gap = g) BOUNDED BELOW?")
print("="*70)

print("""
  For the GP excursion, |f'(0)| at the zero has distribution determined
  by Rice's formula. The gap density is:

    rho(g) ~ E[|f'(0)| * |f'(g)| * 1_excursion | bridge]

  This REWEIGHTS the bridge distribution of |f'(0)| by |f'(0)| * (stuff).

  For the bridge: f'(0) ~ N(0, m2_cond) where m2_cond = Var(f'(0)|bridge).
  So |f'(0)| ~ half-normal with CV = sqrt(pi/2 - 1) = 0.756.

  The Rice weighting by |f'(0)| changes half-normal to RAYLEIGH:
  f_Rice(y) ~ y * (y/sigma^2) * exp(-y^2/(2*sigma^2)) ~ y^2 * exp(...)

  Wait: Rice weights by |f'(0)|, so the distribution of |f'(0)| given
  that we're at a zero crossing becomes:

  p(y | zero) ~ y * half_normal(y) ~ y^2 * exp(-y^2/(2*sigma^2))

  This is a chi distribution with 3 degrees of freedom (or Maxwell).
  CV(Maxwell) = sqrt(3*pi/(8-3*pi+pi) - 1)... let me just compute it.
""")

# For half-normal weighted by y (Rice weighting):
# p(y) ~ y * (2/sqrt(2*pi*s^2)) * exp(-y^2/(2*s^2)) for y > 0
# = (2*y / (s^2 * sqrt(2*pi))) * exp(-y^2/(2*s^2))... this is a Rayleigh distribution!
# f_Rayleigh(y; sigma) = (y/sigma^2) * exp(-y^2/(2*sigma^2))

# CV of Rayleigh = sqrt(4/pi - 1) = sqrt(0.2732) = 0.5227

from scipy.stats import rayleigh as rayleigh_dist
# For Rayleigh: E[X] = sigma*sqrt(pi/2), Var[X] = sigma^2*(4-pi)/2
# CV = sqrt((4-pi)/2) / sqrt(pi/2) = sqrt((4-pi)/pi) = sqrt(4/pi - 1)
cv_rayleigh = np.sqrt(4/np.pi - 1)
print(f"  CV of Rayleigh = sqrt(4/pi - 1) = {cv_rayleigh:.4f}")
print(f"  This is the CV of |f'(0)| at a zero crossing (Rice weighting)")
print(f"  This is BEFORE conditioning on gap = g")

# Does conditioning on gap = g reduce the CV below 0.326?
# The gap g and |f'(0)| are weakly correlated (Corr ~ +0.05).
# Conditioning on g slightly narrows the distribution of |f'(0)|.
# But the reduction should be small.

# Compute: what's the maximum reduction in CV from conditioning?
# If Corr(|f'|, g) = rho, then:
# Var(|f'| | g) ~ Var(|f'|) * (1 - rho^2) (for Gaussian, exact; approximate for Rayleigh)
# CV(|f'| | g) ~ CV(|f'|) * sqrt(1 - rho^2) (approximate)

N = 50
gaps, peaks, fp0 = simulate_with_deriv(N, n_trials=150)
corr_fp_g = pearsonr(fp0, gaps)[0]
print(f"\n  N={N}: Corr(|f'(0)|, g) = {corr_fp_g:+.4f}")
print(f"  If Gaussian: CV reduction factor = sqrt(1-rho^2) = {np.sqrt(1-corr_fp_g**2):.4f}")
print(f"  Reduced CV = {cv_rayleigh * np.sqrt(1-corr_fp_g**2):.4f}")
print(f"  This is >> 0.326")

# But the actual conditional CV might differ from the Gaussian approximation.
# Let's check the actual conditional CV at the WORST gap quantile.

# Bin by gap and compute CV of |f'(0)|
edges = np.percentile(gaps, np.linspace(0, 100, 41))
edges[-1] += 0.001
min_cv = 999
for i in range(40):
    mask = (gaps >= edges[i]) & (gaps < edges[i+1])
    if np.sum(mask) < 50: continue
    cv = np.std(fp0[mask]) / np.mean(fp0[mask])
    if cv < min_cv: min_cv = cv

print(f"\n  Actual min CV(|f'(0)| | g) across all gap bins = {min_cv:.4f}")
print(f"  >= 0.326: {'YES' if min_cv >= 0.326 else 'NO'}")
print(f"  Rayleigh CV = {cv_rayleigh:.4f}")
print(f"  Ratio: {min_cv/cv_rayleigh:.4f}")


# ============================================================
# PART 3: SWEEP ACROSS ALL N
# ============================================================
print(f"\n{'='*70}")
print("PART 3: min CV(|f'(0)| | g) ACROSS ALL N")
print("="*70)

print(f"{'N':>5} {'Corr(fp,g)':>12} {'min CV(fp|g)':>14} {'Rayleigh':>10} "
      f"{'>=0.326':>8}")
print("-"*50)

for N in [5, 10, 20, 50, 100, 200]:
    gaps, peaks, fp0 = simulate_with_deriv(N, n_trials=150)
    if len(gaps) < 500: continue
    corr = pearsonr(fp0, gaps)[0]
    edges = np.percentile(gaps, np.linspace(0, 100, 31))
    edges[-1] += 0.001
    min_cv = 999
    for i in range(30):
        mask = (gaps >= edges[i]) & (gaps < edges[i+1])
        if np.sum(mask) < 50: continue
        m = np.mean(fp0[mask])
        if m > 0.01:
            cv = np.std(fp0[mask]) / m
            if cv < min_cv: min_cv = cv
    ok = min_cv >= 0.326
    print(f"{N:>5} {corr:>+12.4f} {min_cv:>14.4f} {cv_rayleigh:>10.4f} "
          f"{'YES' if ok else 'NO':>8}")


# ============================================================
# PART 4: THE COMPLETE PROOF
# ============================================================
print(f"\n{'='*70}")
print("THE PROOF (if min CV(|f'(0)||g) >= 0.326 holds)")
print("="*70)

print("""
  GIVEN: CV(|f'(0)| | gap = g) >= c = 0.326 for all g.

  BY LAW OF TOTAL VARIANCE:
    Var(Q|g) >= Var(E[Q|g, f'(0)])

  Since Q = P/g and the Slepian model gives P ~ a(g)*|f'(0)| + residual:
    E[Q|g, f'(0)=y] ~ a(g)*y/g = (a(g)/g) * y

  So: Var(E[Q|g, f'(0)]) = (a(g)/g)^2 * Var(|f'(0)||g)
                          >= (a(g)/g)^2 * c^2 * E[|f'(0)||g]^2

  And: E[Q|g]^2 = ((a(g)/g)*E[|f'(0)||g] + residual_mean/g)^2
                 <= 2*(a(g)/g)^2*E[|f'(0)||g]^2 + 2*(residual_mean/g)^2
                 (by (x+y)^2 <= 2x^2 + 2y^2)

  Therefore: CV(Q|g)^2 = Var(Q|g)/E[Q|g]^2
                        >= c^2 * (a(g)/g)^2 * E[|f'|]^2 / [2*(a(g)/g)^2*E[|f'|]^2 + 2*res^2]
                        = c^2 / [2 + 2*(res/(a*E[|f'|]))^2]

  If the residual is small compared to the regression (which it IS for
  small-to-medium gaps where the Slepian regression dominates):
    CV(Q|g) >= c / sqrt(2) = 0.326/1.414 = 0.231

  Hmm, this loses a factor of sqrt(2). Not tight enough.

  BETTER: Don't use (x+y)^2 <= 2x^2+2y^2. Use:
    E[Q|g] = (a/g)*E[|f'||g] + residual_contribution
           <= (a/g)*E[|f'||g] * (1 + delta)
  where delta = residual / (a*E[|f'|]) is the relative residual.

  Then: CV(Q|g) >= c / (1+delta)

  For delta < 1: CV >= c/2 = 0.163. Still not enough.

  The issue: the (1+delta)^2 in the denominator loses too much.

  ALTERNATIVE: Directly use the noise dilution with the regression CV.

  R = Var(q)/Var(Q) where q(g) = E[Q|g].
  Var(Q) = Var(q) + E[Var(Q|g)]
  E[Var(Q|g)] >= E[(a/g)^2 * c^2 * E[|f'||g]^2] = c^2 * E[q(g)^2 * adjust]

  This is getting complicated. The direct simulation shows CV >= 0.39.
  The analytical bound via the f'(0) CV alone gives a weaker result.
""")


print(f"\n{'='*70}")
print("DONE")
print("="*70)
