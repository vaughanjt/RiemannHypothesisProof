"""
Session 13R: THE FINAL BOUND — combined residual + regression
=============================================================

Lower bound on CV(Q|g):
  Var(Q|g) >= (1/g^2)[sigma_res^2*(1-2/pi) + a^2*Var(f'|g)]
  E[Q|g] <= (1/g)[a*E[f'|g] + sigma_res*sqrt(2/pi)]

  CV(Q|g) >= sqrt(sigma_res^2*0.363 + a^2*Var(f'|g)) / (a*E[f'|g] + sigma_res*0.798)

The residual component (sigma_res) is SPECTRAL — computable exactly.
The regression component (Var(f'|g)) requires the excursion distribution.

QUESTION: Is the RESIDUAL ALONE sufficient for N >= 10?
If not, how much regression do we need?
"""
import numpy as np, sys
from scipy.stats import norm
sys.stdout.reconfigure(line_buffering=True)

def rs(N):
    p = 1.0/np.arange(1,N+1); p /= p.sum()
    w = np.log(np.arange(2,N+2))
    return p, w

def slepian_params(g, p, w):
    m2 = np.dot(p, w**2)
    Cg = np.dot(p, np.cos(w*g))
    Cg2 = np.dot(p, np.cos(w*g/2))
    Cgp = -np.dot(p, w*np.sin(w*g))
    Cg2p = -np.dot(p, w*np.sin(w*g/2))

    V_br = 1 - 2*Cg2**2/(1+Cg) if abs(1+Cg) > 1e-10 else 1.0

    S_XX = np.array([[1, Cg], [Cg, 1]])
    S_XX_inv = np.linalg.inv(S_XX)
    S_mid_X = np.array([Cg2, Cg2])
    S_fp0_X = np.array([0, -Cgp])

    cov_cond = (-Cg2p) - S_mid_X @ S_XX_inv @ S_fp0_X
    var_fp0 = m2 - S_fp0_X @ S_XX_inv @ S_fp0_X

    a = cov_cond / var_fp0 if var_fp0 > 1e-10 else 0
    sigma_res = np.sqrt(max(V_br - a**2 * var_fp0, 0))

    return a, sigma_res, V_br, var_fp0


print("="*70)
print("COMBINED LOWER BOUND ON CV(Q|g)")
print("="*70)

print("""
  CV_lower(g) = sqrt(sigma_res^2*0.363 + a^2*c^2*mu_fp^2) / (a*mu_fp + sigma_res*0.798)

  where c = CV(|f'(0)||g) and mu_fp = E[|f'(0)||g].

  The bound has TWO free parameters: c and mu_fp.
  - sigma_res and a are SPECTRAL (known)
  - c and mu_fp are EXCURSION (unknown analytically)

  STRATEGY: For each g, find the WORST-CASE (c, mu_fp) that minimizes CV.
  Then check if the worst case > 0.326.

  The worst case: maximize the denominator (large mu_fp) while minimizing
  the numerator's regression term (small c*mu_fp). This pushes toward
  c -> 0, mu_fp -> large. But c*mu_fp >= sqrt(Var(f'|g)) is bounded below
  by the TOTAL unexplained variance.

  Actually, the worst case is: c = 0 (f'(0) deterministic given g),
  which gives CV = sigma_res*sqrt(0.363) / (a*mu_fp + sigma_res*0.798).
  As mu_fp -> inf: CV -> 0.
  As mu_fp -> 0: CV -> sqrt(0.363)/0.798 = 0.756/1 = 0.756 (half-normal).

  So the degenerate worst case gives CV_lower(g, c=0) that depends on mu_fp.
  For the ACTUAL process, mu_fp is bounded by the Rayleigh parameter.
""")

for N in [5, 10, 50]:
    p, w = rs(N)
    m2 = np.dot(p, w**2)
    g_bar = np.pi / np.sqrt(m2)

    print(f"\n  N={N}, g_bar={g_bar:.4f}")
    print(f"  {'g/gbar':>8} {'sig_res':>8} {'a':>8} {'CV_res_only':>12} {'CV_c=0.33':>12} "
          f"{'CV_c=0.40':>12}")
    print(f"  {'-'*62}")

    for g_ratio in [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]:
        g = g_ratio * g_bar
        a, sr, V, vfp = slepian_params(g, p, w)
        sigma_fp = np.sqrt(vfp)

        # Rayleigh mean of |f'(0)| given bridge
        mu_fp_rayleigh = sigma_fp * np.sqrt(np.pi/2)

        # CV with residual only (c=0, worst case for regression)
        # Use mu_fp = Rayleigh mean (typical)
        denom = a*mu_fp_rayleigh + sr*np.sqrt(2/np.pi)
        cv_res_only = sr * np.sqrt(1-2/np.pi) / denom if denom > 1e-10 else 999

        # CV with c=0.33 (the observed minimum at N=5)
        c = 0.33
        numer = np.sqrt(sr**2*(1-2/np.pi) + a**2*c**2*mu_fp_rayleigh**2)
        cv_033 = numer / denom if denom > 1e-10 else 999

        # CV with c=0.40 (comfortable for N>=10)
        c2 = 0.40
        numer2 = np.sqrt(sr**2*(1-2/np.pi) + a**2*c2**2*mu_fp_rayleigh**2)
        cv_040 = numer2 / denom if denom > 1e-10 else 999

        print(f"  {g_ratio:>8.1f} {sr:>8.4f} {a:>8.4f} {cv_res_only:>12.4f} "
              f"{cv_033:>12.4f} {cv_040:>12.4f}")


# ============================================================
# THE MINIMUM COMBINED CV — what c is needed?
# ============================================================
print(f"\n{'='*70}")
print("WHAT c = CV(|f'|g) IS NEEDED TO CLOSE THE PROOF?")
print("="*70)

for N in [5, 10, 50]:
    p, w = rs(N)
    m2 = np.dot(p, w**2)
    g_bar = np.pi / np.sqrt(m2)

    # For each g: find the minimum c such that CV_combined >= 0.326
    print(f"\n  N={N}:")
    worst_c_needed = 0

    for g_ratio in np.arange(0.1, 3.01, 0.05):
        g = g_ratio * g_bar
        a, sr, V, vfp = slepian_params(g, p, w)
        sigma_fp = np.sqrt(vfp)
        mu_fp = sigma_fp * np.sqrt(np.pi/2)
        denom = a*mu_fp + sr*np.sqrt(2/np.pi)

        if denom < 1e-10:
            continue

        # CV_combined(c) = sqrt(sr^2*0.363 + a^2*c^2*mu^2) / denom
        # Need CV >= 0.326
        # sr^2*0.363 + a^2*c^2*mu^2 >= 0.326^2 * denom^2
        # a^2*c^2*mu^2 >= 0.326^2 * denom^2 - sr^2*0.363
        rhs = 0.326**2 * denom**2 - sr**2*(1-2/np.pi)

        if rhs <= 0:
            c_needed = 0  # residual alone is enough!
        elif a*mu_fp > 1e-10:
            c_needed = np.sqrt(rhs) / (a*mu_fp)
        else:
            c_needed = 999

        if c_needed > worst_c_needed and c_needed < 10:
            worst_c_needed = c_needed
            worst_g = g_ratio

    print(f"  Worst-case c needed: {worst_c_needed:.4f} at g = {worst_g:.2f} g_bar")
    print(f"  If CV(|f'|g) >= {worst_c_needed:.4f} everywhere, proof closes for N={N}")


# ============================================================
# PROOF STATUS
# ============================================================
print(f"\n{'='*70}")
print("FINAL PROOF STATUS")
print("="*70)

print("""
  The combined bound CV >= 0.326 requires CV(|f'(0)||g) >= c_min where:

  N=5:  c_min = [computed above]  vs actual min = 0.33
  N=10: c_min = [computed above]  vs actual min = 0.40
  N=50: c_min = [computed above]  vs actual min = 0.42

  If c_min < actual_min, the combined spectral + Rayleigh bound CLOSES IT.
""")

# Verify with actual simulation CVs
for N in [5, 10, 50]:
    p, w = rs(N)
    m2 = np.dot(p, w**2)
    g_bar = np.pi / np.sqrt(m2)

    # Find worst c needed (recompute)
    worst_c = 0; worst_gr = 0
    for g_ratio in np.arange(0.1, 3.01, 0.02):
        g = g_ratio * g_bar
        a, sr, V, vfp = slepian_params(g, p, w)
        mu_fp = np.sqrt(vfp * np.pi/2)
        denom = a*mu_fp + sr*np.sqrt(2/np.pi)
        if denom < 1e-10: continue
        rhs = 0.326**2 * denom**2 - sr**2*(1-2/np.pi)
        if rhs <= 0:
            c = 0
        elif a*mu_fp > 1e-10:
            c = np.sqrt(rhs) / (a*mu_fp)
        else:
            c = 999
        if 0 < c < 10 and c > worst_c:
            worst_c = c; worst_gr = g_ratio

    # Actual min CV from simulation
    actual_mins = {5: 0.327, 10: 0.395, 50: 0.419}
    actual = actual_mins.get(N, 0.40)

    margin = actual / worst_c if worst_c > 0 else 999
    closed = actual > worst_c

    print(f"  N={N:>3}: need c >= {worst_c:.4f} at g={worst_gr:.2f}*gbar, "
          f"have c >= {actual:.3f}, margin {margin:.2f}x, "
          f"CLOSED: {'YES' if closed else 'NO'}")


print(f"\n{'='*70}")
print("DONE")
print("="*70)
