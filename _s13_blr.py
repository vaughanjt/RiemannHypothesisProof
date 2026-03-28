"""
Session 13P: BOUNDED LIKELIHOOD RATIO — THE KILL SHOT
=====================================================

GOAL: Prove CV(|f'(0)| | gap = g) >= 0.326

APPROACH: The conditional distribution of |f'(0)| given gap g is a
REWEIGHTED Rayleigh. The reweighting is:

  p(y | gap=g) = p(y) * w(y,g) / E[w(Y,g)]

where p(y) is the Rayleigh density (Rice marginal) and
w(y,g) = p(gap=g | |f'(0)|=y) / p(gap=g) is the likelihood ratio.

If w(y,g) is bounded: w_min <= w(y,g) <= w_max, then the reweighted
CV is bounded below by a function of CV_Rayleigh and w_max/w_min.

THEOREM (Bounded LR => CV bound):
For a random variable X with CV = c, reweighted by w(x) with
w_min/w_max >= eta > 0:

  CV(X; w) >= c * sqrt(eta) / (1 + c^2*(1-eta))^{1/2}    [approximate]

Actually, the exact bound comes from:
  Var_w(X) >= (w_min/w_max) * Var(X) - (w_max-w_min)^2/(4*w_max^2) * E[X]^2

This is NOT standard — let me derive the correct bound.

ALTERNATIVE (simpler): Use the Efron-Stein or Poincare inequality.
Or: just bound the conditional variance directly from the correlation.
"""
import numpy as np, sys
from scipy.stats import pearsonr, rayleigh as rayleigh_dist
sys.stdout.reconfigure(line_buffering=True)

def rs(N):
    p = 1.0/np.arange(1,N+1); p /= p.sum()
    w = np.log(np.arange(2,N+2))
    return p, w

def simulate_with_deriv(N, n_trials=250, L=8000, dt=0.01):
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
        fp_left = np.abs(fp[sc[:-1]])
        midx = ((zeros[:-1]+zeros[1:])/(2*dt)).astype(int)
        midx = np.clip(midx, 0, npts-1)
        pks = np.abs(f[midx])
        tr = max(3, int(0.05*len(gaps)))
        all_g.extend(gaps[tr:-tr].tolist())
        all_P.extend(pks[tr:-tr].tolist())
        all_fp.extend(fp_left[tr:-tr].tolist())
    return np.array(all_g), np.array(all_P), np.array(all_fp)


# ============================================================
# PART 1: MEASURE THE LIKELIHOOD RATIO w(y,g)
# ============================================================
print("="*70)
print("PART 1: LIKELIHOOD RATIO w(y,g) = p(|f'|=y | gap=g) / p(|f'|=y)")
print("="*70)

N = 5  # The bottleneck case
print(f"N={N} (the hardest case)", flush=True)
gaps, peaks, fp0 = simulate_with_deriv(N, n_trials=300, L=10000)
print(f"{len(gaps)} observations")

# Compute the likelihood ratio by binning
# p(y | g in bin_j) / p(y)  for each (y, g) bin

n_y_bins = 20
n_g_bins = 10
y_edges = np.percentile(fp0, np.linspace(0, 100, n_y_bins+1))
g_edges = np.percentile(gaps, np.linspace(0, 100, n_g_bins+1))
y_edges[-1] += 0.001; g_edges[-1] += 0.001

# p(y) = marginal density of |f'(0)| (proportional to bin counts)
# p(y|g) = density in (y, g) bin / density in g bin

print(f"\n  Likelihood ratio w(y, g) = p(y|g)/p(y):")
print(f"  (rows = |f'(0)| quantiles, columns = gap quantiles)")
print(f"  Values near 1 = weak dependence = good for us")

w_matrix = np.zeros((n_y_bins, n_g_bins))
for i in range(n_y_bins):
    y_mask = (fp0 >= y_edges[i]) & (fp0 < y_edges[i+1])
    p_y = np.mean(y_mask)
    for j in range(n_g_bins):
        g_mask = (gaps >= g_edges[j]) & (gaps < g_edges[j+1])
        p_g = np.mean(g_mask)
        p_yg = np.mean(y_mask & g_mask)
        w_matrix[i,j] = (p_yg / (p_y * p_g)) if (p_y > 0 and p_g > 0) else 1

# Print summary: min and max of w across y for each g bin
print(f"\n  {'g quantile':>12} {'w_min':>8} {'w_max':>8} {'w_max/w_min':>12} {'eta':>8}")
print(f"  {'-'*50}")
for j in range(n_g_bins):
    w_col = w_matrix[:, j]
    w_min = np.min(w_col)
    w_max = np.max(w_col)
    eta = w_min / w_max if w_max > 0 else 0
    g_lo = np.percentile(gaps, j*10)
    g_hi = np.percentile(gaps, (j+1)*10)
    p_rs, w_rs = rs(N)
    g_bar = np.pi / np.sqrt(np.dot(p_rs, w_rs**2))
    print(f"  [{g_lo/g_bar:.2f},{g_hi/g_bar:.2f})gbar {w_min:>8.3f} {w_max:>8.3f} "
          f"{w_max/w_min:>12.3f} {eta:>8.3f}")


# ============================================================
# PART 2: ANALYTICAL BOUND — CV OF REWEIGHTED RAYLEIGH
# ============================================================
print(f"\n{'='*70}")
print("PART 2: CV OF REWEIGHTED RAYLEIGH")
print("="*70)

print("""
  For X ~ Rayleigh(sigma), reweighted by w(x) with w_min <= w(x) <= w_max:

  E_w[X^k] = E[X^k * w(X)] / E[w(X)]

  Var_w = E_w[X^2] - E_w[X]^2

  LOWER BOUND on Var_w:
  Let eta = w_min/w_max. By rewriting w = w_min + (w - w_min):

  E_w[X] = [w_min * E[X] + E[(w-w_min)*X]] / E[w]
         <= [w_min * E[X] + w_max * E[X]] / (2*w_min)  ... not useful

  Better approach: use the CONDITIONAL VARIANCE formula.
  If X and Y are jointly distributed with |Corr(X,Y)| = rho:
  Var(X|Y) >= Var(X) * (1 - rho^2)  [exact for Gaussian, approximate otherwise]

  For our case: X = |f'(0)|, Y = g, rho = Corr(|f'|, g).
  Then: Var(|f'| | g) >= Var(|f'|) * (1 - rho^2)  [approximate lower bound]
  And: CV(|f'| | g) >= CV(|f'|) * sqrt(1 - rho^2)

  This gives: CV >= 0.5227 * sqrt(1 - 0.125^2) = 0.5227 * 0.9922 = 0.519

  But the ACTUAL conditional Var can be much lower than Var*(1-rho^2)
  because the relationship is nonlinear. The Gaussian formula gives
  an optimistic bound.

  Let me check: does Var(X|Y) >= Var(X)*(1-rho^2) hold generally?
""")

# Check: is Var(|f'| | g) >= Var(|f'|) * (1-rho^2) at all gap bins?
rho = pearsonr(fp0, gaps)[0]
var_fp = np.var(fp0)
gaussian_lower = var_fp * (1 - rho**2)

print(f"  N={N}: rho = {rho:.4f}")
print(f"  Var(|f'|) = {var_fp:.4f}")
print(f"  Gaussian lower bound: Var*(1-rho^2) = {gaussian_lower:.4f}")

edges = np.percentile(gaps, np.linspace(0, 100, 21))
edges[-1] += 0.001
print(f"\n  {'g pctile':>10} {'Var(fp|g)':>12} {'Gauss lower':>12} {'holds?':>8}")
for i in range(20):
    mask = (gaps >= edges[i]) & (gaps < edges[i+1])
    if np.sum(mask) < 50: continue
    v = np.var(fp0[mask])
    ok = v >= gaussian_lower * 0.95  # allow 5% for estimation error
    if i % 2 == 0:
        print(f"  {i*5}-{(i+1)*5}% {v:>12.4f} {gaussian_lower:>12.4f} "
              f"{'YES' if ok else 'NO':>8}")


# ============================================================
# PART 3: THE SHARPER BOUND — USE R^2 DIRECTLY
# ============================================================
print(f"\n{'='*70}")
print("PART 3: DIRECT R^2 BOUND ON CONDITIONAL VARIANCE")
print("="*70)

print("""
  The correct general bound (no Gaussianity needed):

  E[Var(X|Y)] = Var(X) - Var(E[X|Y])
              >= Var(X) - Var(X)  (trivially)
              = 0

  But we can do better: Var(E[X|Y]) <= Var(X) * R^2_max

  where R^2_max is the maximum R^2 from regressing X on ANY function of Y.

  For the linear case: Var(E[X|Y]) = Var(X) * rho^2 only if E[X|Y] is linear.
  In general: Var(E[X|Y]) can be up to Var(X) (if X is a function of Y).

  But the KEY POINT: we can COMPUTE Var(E[|f'||g]) directly from
  the simulation, and it gives us the exact R^2 for the nonlinear
  regression.

  R^2_nonlinear = Var(E[|f'||g]) / Var(|f'|)

  If R^2_nonlinear < 1 - 0.326^2/0.5227^2 = 1 - 0.390 = 0.610:
  Then CV(|f'||g) >= 0.326.

  Actually: CV(|f'||g)^2 >= CV(|f'|)^2 * (1 - R^2_nl) / (1 + something)
  ... this is getting circular again.

  Let me just compute the NONLINEAR R^2 of |f'| on g.
""")

# Compute E[|f'| | g] by kernel smoothing and get Var(E[|f'||g])
p_rs, w_rs = rs(N)
g_bar = np.pi / np.sqrt(np.dot(p_rs, w_rs**2))
bw = 0.12 * g_bar

gg = np.linspace(np.percentile(gaps, 1), np.percentile(gaps, 99), 100)
fp_cond = np.array([np.average(fp0, weights=np.exp(-0.5*((gaps-g0)/bw)**2))
                     if np.sum(np.exp(-0.5*((gaps-g0)/bw)**2)) > 20 else np.nan
                     for g0 in gg])
v = ~np.isnan(fp_cond)
fp_at_gaps = np.interp(gaps, gg[v], fp_cond[v])

R2_nonlinear = np.var(fp_at_gaps) / np.var(fp0)
rho_linear = pearsonr(fp0, gaps)[0]
R2_linear = rho_linear**2

print(f"  R^2 (linear) = {R2_linear:.6f}")
print(f"  R^2 (nonlinear) = {R2_nonlinear:.6f}")
print(f"  Ratio: {R2_nonlinear/R2_linear:.2f}x")
print(f"  Fraction of Var(|f'|) explained by g: {R2_nonlinear:.4f} = {R2_nonlinear*100:.1f}%")
print(f"  Fraction UNEXPLAINED: {1-R2_nonlinear:.4f} = {(1-R2_nonlinear)*100:.1f}%")

# Now: the CONDITIONAL CV
# E[Var(|f'||g)] = Var(|f'|) * (1 - R^2_nl)
# E[|f'||g]^2 has average = Var(E[|f'||g]) + E[|f'|]^2 = R2*Var + E^2
# CV(|f'||g)^2 at each g = Var(|f'||g) / E[|f'||g]^2
# Average CV^2 = E[Var/E^2] which is NOT simply (1-R2)*Var/E^2

# But a LOWER BOUND on the minimum CV:
# min_g CV(|f'||g) >= ?

# From Var(|f'||g) = Var(|f'|) - (E[|f'||g] - E[|f'|])^2 - correction
# This isn't quite right either.

# The cleanest approach: for each g bin, Var(|f'||g) / E[|f'||g]^2 is computed.
# The minimum over bins is the answer.
# Can we bound this from the R^2?

# If E[|f'||g] is bounded in [a, b] and Var(|f'||g) >= c:
# CV(|f'||g)^2 >= c / b^2

# With c = Var(|f'|)*(1-R2) = average unexplained variance
# And b = max E[|f'||g] = maximum conditional mean

max_cond_mean = np.max(fp_at_gaps)
avg_cond_var = np.var(fp0) * (1 - R2_nonlinear)
cv_lower = np.sqrt(avg_cond_var) / max_cond_mean

print(f"\n  max E[|f'||g] = {max_cond_mean:.4f}")
print(f"  avg Var(|f'||g) = {avg_cond_var:.4f}")
print(f"  CV lower bound = sqrt(avg_var) / max_mean = {cv_lower:.4f}")
print(f"  >= 0.326: {'YES' if cv_lower >= 0.326 else 'NO'}")

# Hmm, this bound is too loose because it uses max of mean and average of variance.
# The worst case is at the g where E[|f'||g] is maximized and Var is minimized.

# Better: use the SPECIFIC conditional moments at each g.

print(f"\n  AT THE WORST GAP (where CV is minimized):")
edges = np.percentile(gaps, np.linspace(0, 100, 31))
edges[-1] += 0.001
worst_cv = 999; worst_g_ratio = 0
worst_mean = 0; worst_var = 0

for i in range(30):
    mask = (gaps >= edges[i]) & (gaps < edges[i+1])
    if np.sum(mask) < 100: continue
    m = np.mean(fp0[mask])
    v = np.var(fp0[mask])
    cv = np.sqrt(v)/m if m > 0.01 else 999
    if cv < worst_cv:
        worst_cv = cv
        worst_g_ratio = np.mean(gaps[mask])/g_bar
        worst_mean = m
        worst_var = v

print(f"  g/g_bar = {worst_g_ratio:.3f}")
print(f"  E[|f'||g] = {worst_mean:.4f}")
print(f"  Var(|f'||g) = {worst_var:.4f}")
print(f"  CV(|f'||g) = {worst_cv:.4f}")
print(f"  Rayleigh CV = 0.5227")
print(f"  Reduction factor = {worst_cv/0.5227:.4f}")

# What's the theoretical minimum CV for a Rayleigh with this R^2?
# If R^2 = 0.016, then 1.6% of variance is explained.
# The conditional distribution is "almost Rayleigh" with a slight shape change.
# The CV can't drop much from 0.5227.

# For Gaussian: CV_cond = CV * sqrt(1-R^2) = 0.5227 * sqrt(0.984) = 0.519
# The actual worst CV is 0.327 — much lower! WHY?

# Because the NONLINEAR regression captures more than R^2 suggests at specific g.
# At the worst g, the conditional mean is high (E[|f'||g] is above average),
# making the denominator of CV large, reducing CV even though Var doesn't drop much.

# So: CV drops not because Var drops, but because E INCREASES at specific g values.

print(f"\n  WHY CV drops: it's the MEAN, not the VARIANCE")
print(f"  Overall E[|f'|] = {np.mean(fp0):.4f}")
print(f"  At worst g: E[|f'||g] = {worst_mean:.4f}")
print(f"  Ratio: {worst_mean/np.mean(fp0):.3f}x")
print(f"  Overall Var = {np.var(fp0):.4f}")
print(f"  At worst g: Var = {worst_var:.4f}")
print(f"  Ratio: {worst_var/np.var(fp0):.3f}x")

# The variance barely changes (ratio close to 1) but the mean increases
# by a factor of ~1.5 at the worst gap, reducing CV.

# For a BOUND: CV >= sqrt(Var) / max(E)
# Var(|f'||g) >= Var(|f'|) * (1 - R^2_at_worst_g)
# E[|f'||g] <= max_mean

# If we can bound max_mean / sqrt(overall Var), we get the CV bound.

print(f"\n  CONCLUSION:")
print(f"  The CV minimum comes from E[|f'||g] peaking at g ~ 0.5 g_bar,")
print(f"  NOT from Var(|f'||g) dropping. The variance stays near Rayleigh.")
print(f"  To prove CV >= 0.326, we need:")
print(f"  max_g E[|f'||g] / sqrt(Var(|f'||g)) <= 0.5227/0.326 * sqrt(Var/Var) ~ 1.60")
print(f"  Actual: {worst_mean/np.sqrt(worst_var):.3f}")
print(f"  Needed: <= {0.5227/0.326 * np.sqrt(np.var(fp0)/worst_var):.3f}")


# ============================================================
# PART 4: SWEEP ACROSS N — THE FULL PICTURE
# ============================================================
print(f"\n{'='*70}")
print("PART 4: THE BOTTLENECK AT N=5")
print("="*70)

for N in [5, 10, 50]:
    gaps, peaks, fp0 = simulate_with_deriv(N, n_trials=200)
    if len(gaps) < 500: continue

    # Find worst gap bin
    edges = np.percentile(gaps, np.linspace(0, 100, 31))
    edges[-1] += 0.001
    worst_cv = 999; worst_mean = 0; worst_var = 0; worst_gr = 0

    for i in range(30):
        mask = (gaps >= edges[i]) & (gaps < edges[i+1])
        if np.sum(mask) < 100: continue
        m = np.mean(fp0[mask]); v = np.var(fp0[mask])
        cv = np.sqrt(v)/m if m > 0.01 else 999
        if cv < worst_cv:
            worst_cv = cv; worst_mean = m; worst_var = v
            p_r, w_r = rs(N)
            worst_gr = np.mean(gaps[mask]) / (np.pi/np.sqrt(np.dot(p_r, w_r**2)))

    var_ratio = worst_var / np.var(fp0)
    mean_ratio = worst_mean / np.mean(fp0)

    print(f"\n  N={N}: worst CV(|f'||g) = {worst_cv:.4f} at g = {worst_gr:.2f} g_bar")
    print(f"    Var ratio (conditional/marginal) = {var_ratio:.3f}")
    print(f"    Mean ratio (conditional/marginal) = {mean_ratio:.3f}")
    print(f"    CV drops because mean increases by {mean_ratio:.2f}x, var changes by {var_ratio:.2f}x")
    print(f"    => The issue is: can the conditional mean peak high enough to push CV below 0.326?")


print(f"\n{'='*70}")
print("DONE")
print("="*70)
