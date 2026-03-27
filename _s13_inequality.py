"""
Session 13f: RALPH MODE — Attack the moment inequality
======================================================

TARGET: E[|f'_avg|] * Var(g) > |Cov(|f'_avg|, g(g-mu))|

STRATEGY: Understand the joint distribution of (g, |f'_avg|) well enough
to bound the ratio. Several angles:

1. DECOMPOSE |f'_avg| into components with known correlation structure
   |f'_avg| = (2/g) * int_0^{g/2} f'(gamma+s) ds
   The integral averages f' over a half-excursion. f' at different points
   has known covariance (from spectral density). Can we express
   E[|f'_avg| | g] in terms of spectral quantities?

2. CONDITION ON f'(gamma_k)
   At the zero, f'(gamma_k) is known (from Rice's formula conditioning).
   |f'_avg| depends on f' over [gamma_k, midpoint].
   The conditional distribution of f'(t) given f(gamma_k)=0 is Gaussian
   with known covariance.

3. THE "INDEPENDENCE LIMIT"
   If |f'_avg| were independent of g, the inequality holds trivially
   (Term 2 = 0). The question is: how much does the dependence cost?
   Can we bound the correlation?

4. REWRITE AS RATIO
   The inequality is equivalent to:
   |Corr(|f'_avg|, g(g-mu))| < E[|f'_avg|] * sigma(g)^2 / (sigma(|f'_avg|) * sigma(g(g-mu)))
   = E[|f'_avg|] / sigma(|f'_avg|) * sigma(g)^2 / sigma(g(g-mu))
   = (1/CV(|f'_avg|)) * sigma(g)^2 / sigma(g(g-mu))

   So we need: |Corr| * CV(|f'_avg|) * sigma(g(g-mu)) / Var(g) < 1
"""
import numpy as np, sys
from scipy.stats import pearsonr
sys.stdout.reconfigure(line_buffering=True)

def rs(N):
    p = 1.0/np.arange(1,N+1); p /= p.sum()
    w = np.log(np.arange(2,N+2))
    return p, w

def simulate_full(N, n_trials=150, L=5000, dt=0.01):
    """Simulate with gap, peak, f'_avg, f'(0) tracking."""
    p, w = rs(N)
    amp = 1.0/np.sqrt(np.arange(1,N+1))
    sigma_N = np.sqrt(np.sum(1.0/np.arange(1,N+1)))
    rng = np.random.default_rng(42)
    chunk = 40000
    all_g, all_P, all_fa, all_fp0 = [], [], [], []

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
        for k in range(len(zeros)-1):
            g = zeros[k+1]-zeros[k]
            li, ri = sc[k], sc[k+1]
            mi = (li+ri)//2
            if mi <= li: continue
            P = abs(f[mi])
            # f'_avg via trapezoid
            integral = np.sum(fp[li:mi+1])*dt - 0.5*(fp[li]+fp[mi])*dt
            fa = abs(2.0/g * integral) if g > 0 else 0
            all_g.append(g); all_P.append(P); all_fa.append(fa); all_fp0.append(abs(fp[li]))
    return np.array(all_g), np.array(all_P), np.array(all_fa), np.array(all_fp0)


# ============================================================
# PART 1: CHARACTERIZE THE JOINT DISTRIBUTION (g, |f'_avg|)
# ============================================================
print("="*70)
print("PART 1: JOINT DISTRIBUTION OF (g, |f'_avg|)")
print("="*70)

N = 50
print(f"N={N}, simulating...", flush=True)
gaps, peaks, fa, fp0 = simulate_full(N)
print(f"{len(gaps)} observations")

mu_g = np.mean(gaps)
mu_fa = np.mean(fa)
sig_g = np.std(gaps)
sig_fa = np.std(fa)
CV_fa = sig_fa / mu_fa

# The key quantities for the inequality
Var_g = np.var(gaps)
g_gmu = gaps * (gaps - mu_g)
sig_ggmu = np.std(g_gmu)
corr_fa_ggmu = pearsonr(fa, g_gmu)[0]

# The inequality: E[fa]*Var(g) > |Cov(fa, g(g-mu))|
# Rewritten:  1 > |Corr(fa, g(g-mu))| * CV(fa) * sig(g(g-mu)) / Var(g)
LHS = 1.0
RHS = abs(corr_fa_ggmu) * CV_fa * sig_ggmu / Var_g

print(f"\nKey quantities:")
print(f"  E[|f'_avg|]  = {mu_fa:.4f}")
print(f"  sigma(|f'_avg|) = {sig_fa:.4f}")
print(f"  CV(|f'_avg|) = {CV_fa:.4f}")
print(f"  Var(g)       = {Var_g:.6f}")
print(f"  sigma(g(g-mu)) = {sig_ggmu:.6f}")
print(f"  Corr(|f'_avg|, g(g-mu)) = {corr_fa_ggmu:+.4f}")
print(f"\nINEQUALITY CHECK:")
print(f"  Need: 1 > |Corr| * CV * sig(ggmu)/Var(g)")
print(f"  RHS = {abs(corr_fa_ggmu):.4f} * {CV_fa:.4f} * {sig_ggmu:.4f} / {Var_g:.4f}")
print(f"  RHS = {RHS:.4f}")
print(f"  1 > {RHS:.4f}: {'YES' if RHS < 1 else 'NO'}")
print(f"  Margin: {1/RHS:.2f}x")


# ============================================================
# PART 2: CAN WE DECOMPOSE |f'_avg| INTO INDEPENDENT PARTS?
# ============================================================
print(f"\n{'='*70}")
print("PART 2: DECOMPOSE |f'_avg| = component_indep + component_correlated")
print("="*70)

# |f'_avg| can be written as:
# |f'_avg| = |(2/g) int f'(gamma+s) ds|
#
# f'(gamma+s) for fixed gamma is a GP in s with known covariance
# Cov(f'(s), f'(t)) = -C''(s-t)
#
# The integral (2/g) int_0^{g/2} f'(gamma+s) ds depends on g.
# For FIXED g, this integral has a known Gaussian distribution.
# The dependence on g comes from:
# (a) the integration limit g/2
# (b) the normalization 2/g
# (c) the gap distribution (Rice formula)
#
# Key insight: f'_avg = (2/g) * [f(midpoint) - f(gamma)] = (2/g) * f(midpoint)
# since f(gamma) = 0.
# So |f'_avg| = (2/g) * P = (2/g) * |f(midpoint)|
#
# Therefore: P = (g/2)|f'_avg| <=> |f'_avg| = 2P/g
#
# This means |f'_avg| = 2P/g. The correlation structure of |f'_avg| with g
# is determined by the correlation structure of P/g with g.

# Verify: |f'_avg| = 2P/g
check = 2*peaks/gaps
print(f"  Verify |f'_avg| = 2P/g: Corr = {pearsonr(fa, check)[0]:.6f}")
print(f"  Mean |f'_avg| = {mu_fa:.4f}, Mean 2P/g = {np.mean(check):.4f}")

# So the inequality becomes:
# E[2P/g] * Var(g) > |Cov(2P/g, g(g-mu))|
# i.e., E[P/g] * Var(g) > |Cov(P/g, g(g-mu))|
#
# Now P/g = peak-per-unit-gap. Let's call it Q = P/g.
# The inequality is: E[Q] * Var(g) > |Cov(Q, g(g-mu))|

Q = peaks / gaps
mu_Q = np.mean(Q)
sig_Q = np.std(Q)
CV_Q = sig_Q / mu_Q
corr_Q_ggmu = pearsonr(Q, g_gmu)[0]

print(f"\n  Q = P/g (peak per unit gap):")
print(f"  E[Q] = {mu_Q:.4f}")
print(f"  CV(Q) = {CV_Q:.4f}")
print(f"  Corr(Q, g) = {pearsonr(Q, gaps)[0]:+.4f}")
print(f"  Corr(Q, g(g-mu)) = {corr_Q_ggmu:+.4f}")

# The inequality: E[Q]*Var(g) > |Cov(Q, g(g-mu))|
term1_Q = mu_Q * Var_g
term2_Q = np.cov(Q, g_gmu)[0,1]
print(f"\n  Term 1: E[Q]*Var(g) = {term1_Q:+.6f}")
print(f"  Term 2: Cov(Q, g(g-mu)) = {term2_Q:+.6f}")
print(f"  Margin: {term1_Q/abs(term2_Q):.2f}x")


# ============================================================
# PART 3: DECOMPOSE Q = P/g INTO BRIDGE + EXCURSION
# ============================================================
print(f"\n{'='*70}")
print("PART 3: WHAT DETERMINES Q = P/g?")
print("="*70)

# Q = P/g = |f(midpoint)|/g
# For the bridge: E[|f(mid)| | f(0)=f(g)=0] = sqrt(2V(g)/pi)
# So E[Q | g] = h(g)/g
# h(g)/g = expected peak per unit gap
# This should peak at small g (where f is approximately linear, Q ~ |f'|/2)
# and decrease for large g

p_rs, w_rs = rs(N)

# Compute h_bridge(g)/g and h_true(g)/g from simulation
n_bins = 30
edges = np.percentile(gaps, np.linspace(0, 100, n_bins+1))
edges[-1] += 0.001

print(f"\n  {'g/g_bar':>8} {'E[Q|g]':>10} {'h_br/g':>10} {'n':>6}")
print(f"  {'-'*38}")
g_bar = np.pi / np.sqrt(np.dot(p_rs, w_rs**2))

for i in range(0, n_bins, 3):
    mask = (gaps >= edges[i]) & (gaps < edges[i+1])
    if np.sum(mask) < 50: continue
    g_mid = np.mean(gaps[mask])
    Q_mean = np.mean(Q[mask])
    # Bridge prediction
    Cg = np.dot(p_rs, np.cos(w_rs*g_mid))
    Cg2 = np.dot(p_rs, np.cos(w_rs*g_mid/2))
    V = 1 - 2*Cg2**2/(1+Cg)
    h_br = np.sqrt(max(2*V/np.pi, 0))
    print(f"  {g_mid/g_bar:>8.3f} {Q_mean:>10.4f} {h_br/g_mid:>10.4f} {np.sum(mask):>6}")


# ============================================================
# PART 4: THE CAUCHY-SCHWARZ BOUND — HOW TIGHT IS IT?
# ============================================================
print(f"\n{'='*70}")
print("PART 4: CAUCHY-SCHWARZ ANALYSIS")
print("="*70)

# |Cov(Q, g(g-mu))| <= sigma(Q) * sigma(g(g-mu))
# We need: E[Q] * Var(g) > sigma(Q) * sigma(g(g-mu))
# i.e., E[Q]/sigma(Q) > sigma(g(g-mu))/Var(g)
# i.e., 1/CV(Q) > sigma(g(g-mu))/Var(g)

sig_ggmu = np.std(g_gmu)
ratio_LHS = 1/CV_Q
ratio_RHS = sig_ggmu / Var_g

print(f"  Cauchy-Schwarz sufficient condition:")
print(f"  1/CV(Q) > sigma(g(g-mu))/Var(g)")
print(f"  LHS = 1/{CV_Q:.4f} = {ratio_LHS:.4f}")
print(f"  RHS = {sig_ggmu:.4f} / {Var_g:.4f} = {ratio_RHS:.4f}")
print(f"  CS holds: {'YES' if ratio_LHS > ratio_RHS else 'NO'}")
if ratio_LHS > ratio_RHS:
    print(f"  CS margin: {ratio_LHS/ratio_RHS:.2f}x")
else:
    print(f"  CS fails by: {ratio_RHS/ratio_LHS:.2f}x")
    print(f"  => Need tighter bound than Cauchy-Schwarz")

# What about bounding Corr(Q, g(g-mu)) directly?
print(f"\n  Actual |Corr(Q, g(g-mu))| = {abs(corr_Q_ggmu):.4f}")
print(f"  CS bound: |Corr| <= 1")
print(f"  If we could prove |Corr(Q, g(g-mu))| < {1/(CV_Q * sig_ggmu/Var_g):.4f}")
print(f"  that would suffice.")


# ============================================================
# PART 5: SWEEP ACROSS N — IS THE MARGIN STABLE?
# ============================================================
print(f"\n{'='*70}")
print("PART 5: MARGIN ACROSS ALL N")
print("="*70)

print(f"{'N':>5} {'E[Q]Var(g)':>12} {'|Cov(Q,gg)|':>12} {'Margin':>8} "
      f"{'CV(Q)':>8} {'|Corr|':>8} {'CS?':>5}")
print("-"*60)

for N in [10, 20, 50, 100, 200]:
    gaps, peaks, fa, fp0 = simulate_full(N, n_trials=120, dt=0.01)
    if len(gaps) < 500: continue
    Q = peaks/gaps
    mu_Q = np.mean(Q); Vg = np.var(gaps)
    ggmu = gaps*(gaps-np.mean(gaps))
    t1 = mu_Q * Vg
    t2 = np.cov(Q, ggmu)[0,1]
    margin = t1/abs(t2) if abs(t2)>1e-10 else 9999
    cvQ = np.std(Q)/mu_Q
    corr = pearsonr(Q, ggmu)[0]
    # CS check
    cs_ok = 1/cvQ > np.std(ggmu)/Vg

    print(f"{N:>5} {t1:>+12.6f} {abs(t2):>12.6f} {margin:>7.2f}x "
          f"{cvQ:>8.4f} {abs(corr):>8.4f} {'YES' if cs_ok else 'NO':>5}")


# ============================================================
# PART 6: WHAT WOULD CLOSE THE PROOF?
# ============================================================
print(f"\n{'='*70}")
print("PART 6: WHAT WOULD CLOSE THE PROOF?")
print("="*70)

# Recompute for N=50
gaps, peaks, fa, fp0 = simulate_full(50, n_trials=150, dt=0.01)
Q = peaks/gaps
mu_Q = np.mean(Q); Vg = np.var(gaps); mu_g = np.mean(gaps)
ggmu = gaps*(gaps-mu_g)
corr = pearsonr(Q, ggmu)[0]
cvQ = np.std(Q)/mu_Q

print(f"""
  The inequality E[Q]*Var(g) > |Cov(Q, g(g-mu))| is equivalent to:

    |Corr(Q, g(g-mu))| < E[Q]*Var(g) / (sigma(Q)*sigma(g(g-mu)))
                        = 1/CV(Q) * Var(g)/sigma(g(g-mu))

  At N=50:
    |Corr(Q, g(g-mu))| = {abs(corr):.4f}
    Threshold           = {1/cvQ * Vg/np.std(ggmu):.4f}
    Margin              = {(1/cvQ * Vg/np.std(ggmu))/abs(corr):.2f}x

  APPROACHES TO PROVE |Corr| < threshold:

  (A) DIRECT: Show that Q = P/g and g(g-mu) cannot be too correlated
      because P/g depends on the process VALUE while g(g-mu) depends
      on the process ZEROS. These are different functionals.

  (B) CONDITIONAL: E[Q|g] = h(g)/g peaks then falls. The product
      g(g-mu)*h(g)/g = (g-mu)*h(g) changes sign at g=mu.
      Cov(Q, g(g-mu)) = Cov(h(g)/g, g(g-mu)) + noise terms.

  (C) SPECTRAL: Express everything via Rice's formula as integrals
      over the spectral measure. The inequality might reduce to
      a positivity condition on a specific kernel.

  (D) SCALING: For N -> inf, the spectral shape converges (alpha^2 -> 4/5).
      All moments scale with g_bar = pi/sqrt(m2).
      The NORMALIZED inequality (in units of g_bar) approaches a
      universal constant. If that constant < 1, the proof works for
      all large N, and small N are checked computationally.
""")

# Check the scaling
print("  SCALING CHECK: does the margin converge?")
print(f"  {'N':>5} {'margin':>8} {'|Corr|':>8} {'1/CV':>8} {'Vg/sig_ggmu':>12}")
for N in [10, 20, 50, 100, 200]:
    g, p, fa, _ = simulate_full(N, n_trials=100, dt=0.01)
    if len(g)<500: continue
    Q = p/g; mu = np.mean(g); vg = np.var(g)
    gg = g*(g-mu)
    c = pearsonr(Q, gg)[0]
    cv = np.std(Q)/np.mean(Q)
    t = 1/cv * vg/np.std(gg)
    print(f"  {N:>5} {t/abs(c):>8.2f}x {abs(c):>8.4f} {1/cv:>8.4f} {vg/np.std(gg):>12.4f}")


print(f"\n{'='*70}")
print("DONE")
print("="*70)
