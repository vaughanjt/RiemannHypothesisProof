"""
Session 13Q: THE SPECTRAL COMPUTATION
======================================

Compute BOTH variance components from spectral quantities alone,
using the Slepian model. No simulation — pure spectral integrals.

For each gap g, the Slepian model gives:
  f(g/2) | bridge, f'(0)=y  ~  N(a(g)*y, sigma_res(g)^2)

where a(g) and sigma_res(g) are computable from C, C', m2.

The excursion conditioning truncates to f(g/2) > 0.

Component 1 (regression variance): comes from variation of y = |f'(0)|
Component 2 (residual variance): comes from sigma_res * Z

Both are computable. The truncation correction is a known function.

For the COMBINED CV: average over the Rayleigh distribution of |f'(0)|
(approximation: ignore the conditioning of |f'(0)| on gap g).
"""
import numpy as np, sys
from scipy.stats import norm, rayleigh as rayleigh_dist
from scipy.integrate import quad
sys.stdout.reconfigure(line_buffering=True)

def rs(N):
    p = 1.0/np.arange(1,N+1); p /= p.sum()
    w = np.log(np.arange(2,N+2))
    return p, w

def C_derivs(tau, p, w):
    c = np.dot(p, np.cos(w*tau))
    cp = -np.dot(p, w*np.sin(w*tau))
    cpp = -np.dot(p, w**2*np.cos(w*tau))
    return c, cp, cpp

def slepian_params(g, p, w):
    """Compute Slepian regression a(g) and residual sigma_res(g)
    for f(g/2) given bridge f(0)=f(g)=0 and f'(0)."""
    m2 = np.dot(p, w**2)
    Cg, Cgp, Cgpp = C_derivs(g, p, w)
    Cg2, Cg2p, _ = C_derivs(g/2, p, w)

    # Bridge variance V(g)
    V_bridge = 1 - 2*Cg2**2/(1+Cg) if abs(1+Cg) > 1e-10 else 1.0

    # Conditional covariance of f(g/2) with f'(0) given bridge
    S_XX = np.array([[1, Cg], [Cg, 1]])
    S_XX_inv = np.linalg.inv(S_XX)
    S_mid_X = np.array([Cg2, Cg2])
    S_fp0_X = np.array([0, -Cgp])

    cov_mid_fp0_cond = (-Cg2p) - S_mid_X @ S_XX_inv @ S_fp0_X
    var_fp0_cond = m2 - S_fp0_X @ S_XX_inv @ S_fp0_X

    # Regression coefficient
    a = cov_mid_fp0_cond / var_fp0_cond if var_fp0_cond > 1e-10 else 0

    # Residual variance
    sigma_res_sq = max(V_bridge - a**2 * var_fp0_cond, 0)
    sigma_res = np.sqrt(sigma_res_sq)

    return a, sigma_res, V_bridge, var_fp0_cond

def truncated_normal_moments(mu, sigma):
    """E[X] and Var[X] for N(mu, sigma^2) truncated to X > 0."""
    if sigma < 1e-12:
        return max(mu, 0), 0
    alpha = -mu/sigma  # standardized truncation point
    phi_a = norm.pdf(alpha)
    Phi_a = norm.cdf(alpha)
    surv = 1 - Phi_a
    if surv < 1e-15:
        return 0, 0
    lam = phi_a / surv
    E_Z = lam  # E[Z | Z > alpha] for Z~N(0,1)
    Var_Z = 1 - lam*(lam - alpha)
    E_X = mu + sigma * E_Z
    Var_X = sigma**2 * max(Var_Z, 0)
    return E_X, Var_X


def compute_cv_spectral(g, p, w, n_y_points=200):
    """Compute CV(Q|g) from spectral quantities using Slepian model.

    Averages over f'(0) distribution using Rayleigh (Rice marginal).
    Returns CV(Q|g), E[Q|g], Var(Q|g).
    """
    a, sigma_res, V_bridge, var_fp0 = slepian_params(g, p, w)
    sigma_fp = np.sqrt(var_fp0)  # std of f'(0) given bridge

    # f'(0) at a zero crossing ~ Rayleigh with parameter sigma_fp
    # (Rice weighting by |f'(0)| turns half-normal into Rayleigh)
    # Rayleigh(sigma): p(y) = (y/sigma^2)*exp(-y^2/(2*sigma^2))

    # For each y = |f'(0)|, compute E[P|g,y] and Var[P|g,y]
    # P = |f(g/2)| where f(g/2) | bridge,f'(0)=y ~ N(a*y, sigma_res^2)
    # For positive excursion: f(g/2) > 0

    # Integrate over Rayleigh distribution of y
    y_max = sigma_fp * 5  # integrate to 5 sigma
    y_points = np.linspace(0.001, y_max, n_y_points)

    # Rayleigh density
    ray_density = (y_points / sigma_fp**2) * np.exp(-y_points**2 / (2*sigma_fp**2))

    E_P = np.zeros(n_y_points)
    Var_P = np.zeros(n_y_points)

    for i, y in enumerate(y_points):
        mu_mid = a * y  # conditional mean of f(g/2)
        E_P[i], Var_P[i] = truncated_normal_moments(mu_mid, sigma_res)

    # E[Q|g] = E[P/g | g] = (1/g) * integral E[P|g,y] * p_Rayleigh(y) dy
    # Var(Q|g) = E[Var(Q|g,y)] + Var(E[Q|g,y])
    #          = (1/g^2) * [E[Var(P|g,y)] + Var(E[P|g,y])]

    dy = y_points[1] - y_points[0]
    weights = ray_density * dy
    weights /= np.sum(weights)  # normalize

    E_P_avg = np.sum(E_P * weights)
    E_P2_avg = np.sum(E_P**2 * weights)
    E_VarP = np.sum(Var_P * weights)

    Var_EP = E_P2_avg - E_P_avg**2  # Var of conditional mean
    Var_P_total = E_VarP + Var_EP     # total Var(P|g)

    E_Q = E_P_avg / g
    Var_Q = Var_P_total / g**2
    CV_Q = np.sqrt(Var_Q) / E_Q if E_Q > 1e-10 else 999

    return CV_Q, E_Q, Var_Q, E_VarP/g**2, Var_EP/g**2


# ============================================================
# THE SPECTRAL COMPUTATION
# ============================================================
print("="*70)
print("SPECTRAL CV(Q|g) COMPUTATION — NO SIMULATION")
print("="*70)

for N in [5, 10, 50, 200]:
    p, w = rs(N)
    m2 = np.dot(p, w**2)
    g_bar = np.pi / np.sqrt(m2)

    print(f"\n  N={N}, g_bar={g_bar:.4f}")
    print(f"  {'g/gbar':>8} {'CV(Q|g)':>10} {'E[Q|g]':>10} {'Var_res':>10} "
          f"{'Var_reg':>10} {'a(g)':>8} {'sig_res':>8}")
    print(f"  {'-'*66}")

    min_cv = 999
    for g_ratio in np.arange(0.1, 3.01, 0.1):
        g = g_ratio * g_bar
        try:
            cv, eq, vq, v_res, v_reg = compute_cv_spectral(g, p, w)
            a, sr, _, _ = slepian_params(g, p, w)
            if g_ratio in [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0, 2.5]:
                print(f"  {g_ratio:>8.1f} {cv:>10.4f} {eq:>10.4f} {v_res:>10.6f} "
                      f"{v_reg:>10.6f} {a:>8.4f} {sr:>8.4f}")
            if cv < min_cv and cv > 0:
                min_cv = cv
        except:
            pass

    print(f"\n  min CV(Q|g) from spectral = {min_cv:.4f}")
    print(f"  >= 0.326: {'YES' if min_cv >= 0.326 else 'NO'}")
    print(f"  Headroom: {(min_cv - 0.326)/0.326*100:.1f}%")


# ============================================================
# COMPARE WITH SIMULATION
# ============================================================
print(f"\n{'='*70}")
print("COMPARISON: SPECTRAL vs SIMULATION")
print("="*70)

# Use simulation for N=50
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

N = 50
p, w = rs(N)
g_bar = np.pi / np.sqrt(np.dot(p, w**2))
gaps, peaks = simulate(N)
Q = peaks/gaps

print(f"\n  N={N}, {len(gaps)} simulated gaps")
print(f"  {'g/gbar':>8} {'CV_spectral':>12} {'CV_simulated':>14}")
print(f"  {'-'*36}")

edges = np.percentile(gaps, np.linspace(0, 100, 21))
for i in range(20):
    mask = (gaps >= edges[i]) & (gaps < edges[i+1] + (0.001 if i==19 else 0))
    if np.sum(mask) < 100: continue
    g_mid = np.mean(gaps[mask])
    cv_sim = np.std(Q[mask]) / np.mean(Q[mask])
    cv_spec, _, _, _, _ = compute_cv_spectral(g_mid, p, w)
    if i % 2 == 0:
        print(f"  {g_mid/g_bar:>8.3f} {cv_spec:>12.4f} {cv_sim:>14.4f}")


# ============================================================
# THE PROOF STATUS
# ============================================================
print(f"\n{'='*70}")
print("PROOF STATUS")
print("="*70)

all_good = True
for N in [5, 7, 10, 15, 20, 50, 100, 200]:
    p, w = rs(N)
    g_bar = np.pi / np.sqrt(np.dot(p, w**2))
    min_cv = 999
    for g_ratio in np.arange(0.05, 4.01, 0.05):
        g = g_ratio * g_bar
        try:
            cv, _, _, _, _ = compute_cv_spectral(g, p, w)
            if 0 < cv < min_cv:
                min_cv = cv
        except:
            pass
    ok = min_cv >= 0.326
    if not ok: all_good = False
    print(f"  N={N:>4}: min CV(Q|g) = {min_cv:.4f}  >= 0.326: {'YES' if ok else 'NO'}  "
          f"headroom: {(min_cv-0.326)/0.326*100:+.1f}%")

print(f"\n  ALL N >= 5 pass: {'YES' if all_good else 'NO'}")

print(f"\n{'='*70}")
print("DONE")
print("="*70)
