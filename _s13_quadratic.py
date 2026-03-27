"""
Session 13j: GROK'S SUGGESTION — Quadratic test function approach
=================================================================

IDEA: Approximate q(g) = E[|f'_avg| | g] by a simple analytic form.
Then compute Cov(q_approx, W) using gap distribution moments.
If the bound holds for the approximation, it holds for the true q
(with error control).

q(g) is bell-shaped: rises from 0, peaks at ~0.94*g_bar, falls to ~0.
A natural fit: q(g) ~ A * g * exp(-B * g^2)  (Rayleigh-like shape)
Or simpler: q(g) ~ a*g - b*g^2  (quadratic, valid near the peak)

The covariance of the quadratic with W = g(g-mu) involves only
moments of the gap distribution: E[g^k] for k = 1..4.

For a GP, these moments can be expressed via the spectral density
(Rice-type formulas).
"""
import numpy as np, sys
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
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
# PART 1: FIT q(g) WITH ANALYTIC FORMS
# ============================================================
print("="*70)
print("PART 1: ANALYTIC FITS TO q(g) = E[P/g | g]")
print("="*70)

N = 50
p_rs, w_rs = rs(N)
g_bar = np.pi / np.sqrt(np.dot(p_rs, w_rs**2))

print(f"N={N}, simulating...", flush=True)
gaps, peaks = simulate(N)
Q = peaks/gaps
mu = np.mean(gaps)
print(f"{len(gaps)} observations, g_bar={g_bar:.4f}, mu={mu:.4f}")

# Compute q(g) on a grid via kernel smoothing
bw = 0.10 * g_bar
gg = np.linspace(0.05*g_bar, 2.5*g_bar, 100)
qq = np.zeros(len(gg))
for i, g0 in enumerate(gg):
    wts = np.exp(-0.5*((gaps-g0)/bw)**2)
    qq[i] = np.average(Q, weights=wts) if np.sum(wts) > 30 else np.nan
valid = ~np.isnan(qq)
gg_v, qq_v = gg[valid], qq[valid]

# Fit 1: Rayleigh-like q(g) = A * g * exp(-B * g^2)
def rayleigh_model(g, A, B):
    return A * g * np.exp(-B * g**2)

try:
    popt_ray, _ = curve_fit(rayleigh_model, gg_v, qq_v, p0=[2.0, 0.3])
    A_ray, B_ray = popt_ray
    qq_ray = rayleigh_model(gg_v, A_ray, B_ray)
    resid_ray = np.sqrt(np.mean((qq_v - qq_ray)**2))
    print(f"\n  Rayleigh fit: q(g) = {A_ray:.4f} * g * exp(-{B_ray:.4f} * g^2)")
    print(f"  RMSE = {resid_ray:.6f}")
    print(f"  Peak at g = {1/np.sqrt(2*B_ray):.4f} = {1/(np.sqrt(2*B_ray)*g_bar):.3f} g_bar")
except Exception as e:
    print(f"  Rayleigh fit failed: {e}")
    A_ray, B_ray = 2, 0.3

# Fit 2: Quadratic q(g) = a*g - b*g^2 (valid for g in [0, 2*g_bar])
def quad_model(g, a, b):
    return np.maximum(a*g - b*g**2, 0)

try:
    popt_q, _ = curve_fit(quad_model, gg_v, qq_v, p0=[1.5, 0.5])
    a_q, b_q = popt_q
    qq_quad = quad_model(gg_v, a_q, b_q)
    resid_quad = np.sqrt(np.mean((qq_v - qq_quad)**2))
    print(f"\n  Quadratic fit: q(g) = {a_q:.4f}*g - {b_q:.4f}*g^2")
    print(f"  RMSE = {resid_quad:.6f}")
    print(f"  Peak at g = {a_q/(2*b_q):.4f} = {a_q/(2*b_q*g_bar):.3f} g_bar")
except Exception as e:
    print(f"  Quadratic fit failed: {e}")
    a_q, b_q = 1.5, 0.5


# ============================================================
# PART 2: COMPUTE Cov(q_approx, W) FROM MOMENTS
# ============================================================
print(f"\n{'='*70}")
print("PART 2: COVARIANCE FROM GAP MOMENTS")
print("="*70)

# W = g(g-mu) = g^2 - mu*g
# For q_quad = a*g - b*g^2:
# Cov(q_quad, W) = Cov(a*g - b*g^2, g^2 - mu*g)
#                = a*Cov(g, g^2 - mu*g) - b*Cov(g^2, g^2 - mu*g)
# Cov(g, g^2 - mu*g) = Cov(g, g^2) - mu*Var(g) = E[g^3] - mu*E[g^2] - mu*Var(g)
#                     = E[g^3] - mu*E[g^2] - mu*(E[g^2]-mu^2)
#                     = E[g^3] - 2*mu*E[g^2] + mu^3 = mu_3 (third central moment)
# Cov(g^2, g^2 - mu*g) = Cov(g^2, g^2) - mu*Cov(g^2, g) = Var(g^2) - mu*(E[g^3]-mu*E[g^2])
#                       = E[g^4]-E[g^2]^2 - mu*(E[g^3]-mu*E[g^2])

# Gap moments from simulation
E_g = np.mean(gaps)
E_g2 = np.mean(gaps**2)
E_g3 = np.mean(gaps**3)
E_g4 = np.mean(gaps**4)
Var_g = E_g2 - E_g**2
mu_3 = np.mean((gaps-E_g)**3)  # third central moment
mu_4 = np.mean((gaps-E_g)**4)  # fourth central moment

print(f"  Gap moments:")
print(f"    E[g] = {E_g:.6f}")
print(f"    E[g^2] = {E_g2:.6f},  Var(g) = {Var_g:.6f}")
print(f"    E[g^3] = {E_g3:.6f},  mu_3 = {mu_3:.6f}")
print(f"    E[g^4] = {E_g4:.6f},  mu_4 = {mu_4:.6f}")
print(f"    Skewness = {mu_3/Var_g**1.5:.4f}")
print(f"    Kurtosis = {mu_4/Var_g**2:.4f}")

# For q_quad = a*g - b*g^2:
# E[q_quad] = a*E_g - b*E_g2
E_q_quad = a_q*E_g - b_q*E_g2

# Cov(q_quad, W) where W = g^2 - mu*g:
# = a * (E[g^3] - mu*E[g^2] - mu*Var_g) - b * (E[g^4] - E[g^2]^2 - mu*(E[g^3]-mu*E[g^2]))
# Simplify: Cov(g, W) = E[g*W] - E[g]*E[W] = E[g^3-mu*g^2] - E_g*Var_g
#                      = E[g^3] - mu*E[g^2] - E_g*Var_g = mu_3
# Cov(g^2, W) = E[g^2*W] - E[g^2]*E[W] = E[g^4-mu*g^3] - E_g2*Var_g
#             = E[g^4] - mu*E[g^3] - E_g2*Var_g

Cov_g_W = mu_3
Cov_g2_W = E_g4 - E_g*E_g3 - E_g2*Var_g

Cov_qq_W = a_q * Cov_g_W - b_q * Cov_g2_W

# Term 1 for q_quad: E[q_quad] * Var(g)
Term1_qq = E_q_quad * Var_g

print(f"\n  Quadratic fit: a={a_q:.4f}, b={b_q:.4f}")
print(f"  E[q_quad] = {E_q_quad:.6f}")
print(f"  Cov(g, W) = mu_3 = {Cov_g_W:.6f}")
print(f"  Cov(g^2, W) = {Cov_g2_W:.6f}")
print(f"  Cov(q_quad, W) = {Cov_qq_W:.6f}")
print(f"  Term1_quad = E[q_quad]*Var(g) = {Term1_qq:.6f}")
print(f"  Margin (quad): {Term1_qq/abs(Cov_qq_W):.2f}x" if abs(Cov_qq_W)>1e-8 else "")

# Compare with actual
Cov_gP_actual = np.cov(gaps, peaks)[0,1]
q_at_gaps = np.interp(gaps, gg_v, qq_v)
Cov_q_W_actual = np.cov(q_at_gaps, gaps*(gaps-mu))[0,1]

print(f"\n  Actual Cov(q, W) = {Cov_q_W_actual:.6f}")
print(f"  Quad approx      = {Cov_qq_W:.6f}")
print(f"  Actual Cov(g, P) = {Cov_gP_actual:.6f}")


# ============================================================
# PART 3: THE KEY — CAN WE BOUND mu_3 AND Cov(g^2, W)?
# ============================================================
print(f"\n{'='*70}")
print("PART 3: MOMENT STRUCTURE")
print("="*70)

print("""
  For q_quad = a*g - b*g^2:
    Cov(q, W) = a*mu_3 - b*(E[g^4] - mu*E[g^3] - E[g^2]*Var)
    Term1     = (a*mu - b*(Var + mu^2)) * Var

  The inequality Term1 > |Cov(q, W)| becomes a condition on moments.

  For a Gaussian process, the gap moments E[g^k] are determined by
  the spectral moments m_k. By Rice's formula:
    E_gap[g^k] = integral g^k * rho(g) dg
  where rho(g) is the gap density.

  KEY QUESTION: Can we express mu_3, E[g^4] etc. in terms of m_0, m_2, m_4?

  For the NORMALIZED gap g/g_bar, the moments converge as N -> inf
  (since alpha^2 -> 4/5). In the limit, all moments are universal.
""")

# Compute normalized moments
g_norm = gaps / g_bar
print(f"  Normalized gap moments (g/g_bar):")
for k in range(1, 6):
    mk = np.mean(g_norm**k)
    print(f"    E[(g/g_bar)^{k}] = {mk:.6f}")

# The normalized skewness and kurtosis should converge
print(f"  Normalized skewness = {np.mean((g_norm-np.mean(g_norm))**3)/np.std(g_norm)**3:.4f}")
print(f"  Normalized kurtosis = {np.mean((g_norm-np.mean(g_norm))**4)/np.std(g_norm)**4:.4f}")

# Check: does mu_3 > 0? (positive skewness)
print(f"\n  mu_3 (third central moment) = {mu_3:+.6f}")
print(f"  This is {'POSITIVE' if mu_3 > 0 else 'NEGATIVE'} (gap dist is right-skewed)")

# The covariance decomposition in terms of moments:
# Cov(g, P) = (1/2) * Cov(g, g*|f'_avg|)
# For the quadratic approximation:
# Cov(g, g*q_quad(g)) = Cov(g, a*g^2 - b*g^3)
#                     = a*Cov(g, g^2) - b*Cov(g, g^3)
#                     = a*(E[g^3] - mu*E[g^2]) - b*(E[g^4] - mu*E[g^3])

Cov_g_g2 = E_g3 - E_g*E_g2
Cov_g_g3 = E_g4 - E_g*E_g3

# Cov(g, P) via quadratic ≈ (1/2)*(a*Cov(g,g^2) - b*Cov(g,g^3))
Cov_gP_quad = 0.5*(a_q*Cov_g_g2 - b_q*Cov_g_g3)

print(f"\n  Cov(g, g^2) = {Cov_g_g2:.6f}")
print(f"  Cov(g, g^3) = {Cov_g_g3:.6f}")
print(f"  Cov(g, P) via quadratic = (1/2)({a_q:.4f}*{Cov_g_g2:.4f} - {b_q:.4f}*{Cov_g_g3:.4f})")
print(f"                          = {Cov_gP_quad:.6f}")
print(f"  Actual Cov(g, P) = {Cov_gP_actual:.6f}")

# IS Cov(g, P) via quadratic POSITIVE?
print(f"\n  QUADRATIC GIVES Cov(g,P) > 0: {'YES' if Cov_gP_quad > 0 else 'NO'}")
print(f"  Because: a*Cov(g,g^2) = {a_q*Cov_g_g2:+.6f}")
print(f"           b*Cov(g,g^3) = {b_q*Cov_g_g3:+.6f}")
print(f"  a*Cov(g,g^2) > b*Cov(g,g^3): {'YES' if a_q*Cov_g_g2 > b_q*Cov_g_g3 else 'NO'}")
print(f"  Margin: {a_q*Cov_g_g2/(b_q*Cov_g_g3):.2f}x")


# ============================================================
# PART 4: WHAT IF WE USE ONLY THE LINEAR PART?
# ============================================================
print(f"\n{'='*70}")
print("PART 4: LINEAR APPROXIMATION q(g) ~ a*g")
print("="*70)

# If q(g) ~ a*g (linear, valid for small g):
# P ~ q*g = a*g^2 (quadratic in g)
# Cov(g, P) ~ (1/2)*a*Cov(g, g^2) = (1/2)*a*(E[g^3]-mu*E[g^2])

# The slope a = q'(0) = lim_{g->0} h(g)/g^2 * g = derivative of q at 0
# From the fit: a = a_q (the linear coefficient)

# Cov(g, a*g^2/2) = (a/2)*Cov(g, g^2) = (a/2)*(E[g^3]-mu*E[g^2])
# This is positive iff E[g^3] > mu*E[g^2]
# i.e., E[g^3]/E[g^2] > mu = E[g]

# By Jensen (x^3/x^2 = x is linear): E[g^3]/E[g^2] = E[g^3/g^2] if ... no.
# Actually: E[g^3] > E[g]*E[g^2] iff Cov(g, g^2) > 0 iff g and g^2 are positively correlated.
# Since g^2 is an increasing function of g (for g>0), Cov(g, g^2) > 0 ALWAYS (Chebyshev).

print(f"  For q(g) ~ a*g (a = {a_q:.4f}):")
print(f"  P ~ (a/2)*g^2")
print(f"  Cov(g, P) ~ (a/2)*Cov(g, g^2)")
print(f"  Cov(g, g^2) = {Cov_g_g2:.6f}")
print(f"  This is ALWAYS POSITIVE by Chebyshev (g and g^2 are comonotone).")
print(f"\n  Linear Cov(g,P) = {0.5*a_q*Cov_g_g2:.6f}")
print(f"  Actual Cov(g,P) = {Cov_gP_actual:.6f}")
print(f"  Linear captures {0.5*a_q*Cov_g_g2/Cov_gP_actual*100:.0f}% of actual")

# The quadratic correction: -(b/2)*Cov(g, g^3)
# Cov(g, g^3) > 0 (also by Chebyshev)
# So the correction is NEGATIVE
print(f"\n  Quadratic correction: -(b/2)*Cov(g, g^3) = {-0.5*b_q*Cov_g_g3:.6f}")
print(f"  This is NEGATIVE (reduces the covariance)")
print(f"  Net: linear + correction = {Cov_gP_quad:.6f}")

# THE KEY INEQUALITY:
# Cov(g,P) > 0 iff a*Cov(g,g^2) > b*Cov(g,g^3)
# i.e., a/b > Cov(g,g^3)/Cov(g,g^2)
ratio_ab = a_q/b_q
ratio_cov = Cov_g_g3/Cov_g_g2
print(f"\n  THE KEY INEQUALITY (quadratic):")
print(f"  a/b = {ratio_ab:.4f}")
print(f"  Cov(g,g^3)/Cov(g,g^2) = {ratio_cov:.4f}")
print(f"  a/b > ratio: {'YES' if ratio_ab > ratio_cov else 'NO'}")
print(f"  Margin: {ratio_ab/ratio_cov:.2f}x")

# Note: a/b = 2*g_peak (the peak of the quadratic q = a*g - b*g^2)
# So the inequality is: 2*g_peak > Cov(g,g^3)/Cov(g,g^2)
# i.e., the peak of q must be "far enough to the right" relative to the
# moment ratio of the gap distribution.
print(f"\n  Interpretation: 2*g_peak = a/b = {ratio_ab:.4f}")
print(f"  g_peak = {ratio_ab/2:.4f} = {ratio_ab/(2*g_bar):.3f} g_bar")
print(f"  The peak of q must exceed Cov(g,g^3)/Cov(g,g^2) = {ratio_cov:.4f} = {ratio_cov/g_bar:.3f} g_bar")


# ============================================================
# PART 5: VERIFY ACROSS N
# ============================================================
print(f"\n{'='*70}")
print("PART 5: QUADRATIC INEQUALITY ACROSS N")
print("="*70)

print(f"{'N':>5} {'a':>8} {'b':>8} {'a/b':>8} {'Cov_ratio':>10} {'Margin':>8} {'Quad>0':>8}")
print("-"*55)

for N in [10, 20, 50, 100, 200]:
    gaps, peaks = simulate(N, n_trials=150)
    if len(gaps) < 500: continue
    Q = peaks/gaps
    p_rs, w_rs = rs(N)
    g_bar = np.pi / np.sqrt(np.dot(p_rs, w_rs**2))
    bw = 0.10*g_bar

    # Compute q on grid
    gg = np.linspace(0.05*g_bar, 2.5*g_bar, 80)
    qq = np.array([np.average(Q, weights=np.exp(-0.5*((gaps-g0)/bw)**2))
                    if np.sum(np.exp(-0.5*((gaps-g0)/bw)**2)) > 30 else np.nan
                    for g0 in gg])
    v = ~np.isnan(qq)

    # Fit quadratic
    try:
        popt, _ = curve_fit(quad_model, gg[v], qq[v], p0=[1.5, 0.5])
        a, b = popt
    except:
        continue

    E_g3 = np.mean(gaps**3); E_g2 = np.mean(gaps**2); E_g4 = np.mean(gaps**4)
    mu = np.mean(gaps)
    cov_g_g2 = E_g3 - mu*E_g2
    cov_g_g3 = E_g4 - mu*E_g3
    ratio = cov_g_g3 / cov_g_g2 if cov_g_g2 > 1e-10 else 999
    margin = (a/b) / ratio if ratio > 1e-10 else 999
    pos = a*cov_g_g2 > b*cov_g_g3

    print(f"{N:>5} {a:>8.4f} {b:>8.4f} {a/b:>8.4f} {ratio:>10.4f} {margin:>7.2f}x "
          f"{'YES' if pos else 'NO':>8}")


print(f"\n{'='*70}")
print("DONE")
print("="*70)
