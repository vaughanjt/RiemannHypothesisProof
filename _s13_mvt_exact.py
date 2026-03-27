"""
Session 13e: THE EXACT MVT IDENTITY
====================================

KEY INSIGHT: For any excursion on [gamma_k, gamma_{k+1}]:
  f(m) = integral_0^{g/2} f'(gamma_k + s) ds

  Therefore P = |f(m)| = (g/2) * |f'_avg|

  where f'_avg = (2/g) * integral_0^{g/2} f'(gamma_k + s) ds

This is EXACT. No approximation.

Then: Cov(g, P) = (1/2) * Cov(g, g * |f'_avg|)
               = (1/2) * [E[|f'_avg|]*Var(g) + Cov(|f'_avg|, g*(g-mu))]

The first term is ALWAYS POSITIVE (E[|f'_avg|] > 0, Var(g) > 0).
If the second term is small relative to the first, PROOF IS DONE.

The question: how does |f'_avg| depend on g?
"""
import numpy as np, sys
from scipy.stats import pearsonr
sys.stdout.reconfigure(line_buffering=True)

def rs(N):
    p = 1.0/np.arange(1,N+1); p /= p.sum()
    w = np.log(np.arange(2,N+2))
    return p, w

# High-resolution simulation with derivative tracking
def simulate_with_fpavg(N, n_trials=200, L=5000, dt=0.005):
    """Compute gaps, peaks, AND f'_avg for each gap."""
    p, w = rs(N)
    amp = 1.0/np.sqrt(np.arange(1,N+1))
    sigma_N = np.sqrt(np.sum(1.0/np.arange(1,N+1)))
    rng = np.random.default_rng(42)
    chunk = 40000

    all_g, all_P, all_fpavg, all_fp0, all_Pmodel = [], [], [], [], []

    for trial in range(n_trials):
        phi = rng.uniform(0, 2*np.pi, N)
        npts = int(L/dt)
        f = np.empty(npts)
        fp = np.empty(npts)
        for s in range(0, npts, chunk):
            e = min(s+chunk, npts)
            tc = np.arange(s,e)*dt
            cos_v = np.cos(np.outer(tc, w)+phi)
            sin_v = np.sin(np.outer(tc, w)+phi)
            f[s:e] = cos_v @ amp
            fp[s:e] = -(sin_v @ (amp * w))
        f /= sigma_N
        fp /= sigma_N
        t = np.arange(npts)*dt

        sc = np.where(f[:-1]*f[1:]<0)[0]
        if len(sc)<20: continue
        zeros = t[sc] - f[sc]*dt/(f[sc+1]-f[sc])
        zero_idx = sc  # grid index of left side of zero crossing

        gaps = np.diff(zeros)
        for k in range(len(gaps)):
            g = gaps[k]
            left_idx = zero_idx[k]
            right_idx = zero_idx[k+1]
            mid_idx = (left_idx + right_idx) // 2

            # P = |f(midpoint)|
            P = abs(f[mid_idx])

            # f'_avg = (2/g) * integral of f' from gamma_k to midpoint
            # Approximate by trapezoid rule on grid
            half_indices = range(left_idx, mid_idx+1)
            if len(half_indices) < 2:
                continue
            fp_slice = fp[left_idx:mid_idx+1]
            integral_fp = np.trapezoid(fp_slice, dx=dt)
            fpavg = (2.0/g) * integral_fp if g > 0 else 0

            # P_model = (g/2) * |f'_avg| (should match P)
            P_model = (g/2) * abs(fpavg)

            # |f'(0)| at the zero
            fp0 = abs(fp[left_idx])

            all_g.append(g)
            all_P.append(P)
            all_fpavg.append(abs(fpavg))
            all_fp0.append(fp0)
            all_Pmodel.append(P_model)

    return (np.array(all_g), np.array(all_P), np.array(all_fpavg),
            np.array(all_fp0), np.array(all_Pmodel))


# ============================================================
# PART 1: VERIFY P = (g/2) * |f'_avg|
# ============================================================
print("="*70)
print("PART 1: VERIFY EXACT IDENTITY P = (g/2) * |f'_avg|")
print("="*70)

for N in [50, 200]:
    print(f"\n  N={N}: simulating (dt=0.005, high res)...", flush=True)
    gaps, peaks, fpavg, fp0, Pmodel = simulate_with_fpavg(N, n_trials=150, dt=0.005)
    print(f"  {len(gaps)} observations")

    # Check P ≈ P_model
    residual = peaks - Pmodel
    rel_error = np.abs(residual) / (peaks + 1e-10)
    print(f"  P vs (g/2)*|f'_avg|: mean rel error = {np.mean(rel_error):.4f}")
    print(f"  Corr(P, P_model) = {pearsonr(peaks, Pmodel)[0]:.6f}")
    print(f"  mean(P) = {np.mean(peaks):.5f}, mean(P_model) = {np.mean(Pmodel):.5f}")

    # ============================================================
    # PART 2: CORRELATION STRUCTURE OF |f'_avg| WITH g
    # ============================================================
    print(f"\n  CORRELATION STRUCTURE:")
    mu_g = np.mean(gaps)
    print(f"  Corr(g, |f'_avg|) = {pearsonr(gaps, fpavg)[0]:+.6f}")
    print(f"  Corr(g, |f'(0)|)  = {pearsonr(gaps, fp0)[0]:+.6f}")
    print(f"  Corr(g, P)        = {pearsonr(gaps, peaks)[0]:+.6f}")
    print(f"  Corr(|f'_avg|, P) = {pearsonr(fpavg, peaks)[0]:+.6f}")

    # ============================================================
    # PART 3: THE DECOMPOSITION Cov(g,P) = (1/2)*[E[|f'|]*Var(g) + correction]
    # ============================================================
    E_fpavg = np.mean(fpavg)
    Var_g = np.var(gaps)

    # Cov(g, g*|f'_avg|) = E[|f'_avg|]*Var(g) + Cov(|f'_avg|, g*(g-mu))
    term1 = E_fpavg * Var_g  # ALWAYS POSITIVE
    cov_g_gfp = np.cov(gaps, gaps * fpavg)[0,1]  # Cov(g, g*|f'_avg|)
    term2 = cov_g_gfp - term1  # = Cov(|f'_avg|, g*(g-mu))

    # Cov(g, P) = (1/2) * Cov(g, g*|f'_avg|)  [by the exact identity]
    cov_gP_from_identity = 0.5 * cov_g_gfp
    cov_gP_actual = np.cov(gaps, peaks)[0,1]

    print(f"\n  EXACT DECOMPOSITION:")
    print(f"  E[|f'_avg|]                = {E_fpavg:.6f}")
    print(f"  Var(g)                     = {Var_g:.6f}")
    print(f"  Term 1: E[|f'_avg|]*Var(g) = {term1:+.6f}  [ALWAYS > 0]")
    print(f"  Term 2: Cov(|f'_avg|, g(g-mu)) = {term2:+.6f}")
    print(f"  Sum: Cov(g, g*|f'_avg|)    = {cov_g_gfp:+.6f}")
    print(f"  (1/2) * Sum                = {cov_gP_from_identity:+.6f}")
    print(f"  Actual Cov(g, P)           = {cov_gP_actual:+.6f}")

    margin = term1 / abs(term2) if abs(term2) > 1e-10 else float('inf')
    print(f"\n  MARGIN: Term1/|Term2| = {margin:.1f}x")
    print(f"  PROOF WORKS: {'YES' if term1 + term2 > 0 else 'NO'}")

    # ============================================================
    # PART 4: WHY DOES |f'_avg| DEPEND ON g?
    # ============================================================
    print(f"\n  |f'_avg| BY GAP QUANTILE:")
    quantiles = [0, 10, 25, 50, 75, 90, 100]
    edges = np.percentile(gaps, quantiles)
    print(f"  {'quantile':>10} {'g range':>18} {'mean |f_avg|':>14} {'n':>6}")
    for i in range(len(edges)-1):
        mask = (gaps >= edges[i]) & (gaps < edges[i+1] + (0.001 if i==len(edges)-2 else 0))
        if np.sum(mask) > 0:
            label = f"{quantiles[i]}-{quantiles[i+1]}%"
            g_range = f"[{edges[i]:.3f}, {edges[i+1]:.3f})"
            print(f"  {label:>10} {g_range:>18} {np.mean(fpavg[mask]):>14.5f} {np.sum(mask):>6}")


# ============================================================
# PART 5: VERIFY ACROSS ALL N
# ============================================================
print(f"\n{'='*70}")
print("PART 5: MARGIN ACROSS ALL N")
print("="*70)

print(f"{'N':>5} {'E[|f_a|]*Var':>14} {'Correction':>12} {'Sum':>12} "
      f"{'Cov(g,P)':>10} {'Margin':>8} {'OK':>4}")
print("-"*68)

for N in [10, 20, 50, 100, 200, 500]:
    gaps, peaks, fpavg, fp0, Pmodel = simulate_with_fpavg(N, n_trials=120, dt=0.01)
    if len(gaps) < 500: continue

    E_fa = np.mean(fpavg)
    Vg = np.var(gaps)
    t1 = E_fa * Vg
    cov_ggfa = np.cov(gaps, gaps*fpavg)[0,1]
    t2 = cov_ggfa - t1
    cov_gP = np.cov(gaps, peaks)[0,1]
    margin = t1 / abs(t2) if abs(t2) > 1e-10 else 9999
    ok = (t1 + t2) > 0

    print(f"{N:>5} {t1:>+14.6f} {t2:>+12.6f} {cov_ggfa:>+12.6f} "
          f"{cov_gP:>+10.6f} {margin:>7.1f}x {'YES' if ok else 'NO':>4}")


print(f"\n{'='*70}")
print("SUMMARY")
print("="*70)
print("""
  EXACT IDENTITY: P = (g/2) * |f'_avg| where f'_avg = (2/g) int f' ds

  DECOMPOSITION:
    Cov(g, P) = (1/2) * [E[|f'_avg|]*Var(g) + Cov(|f'_avg|, g(g-mu))]
                         ^^^^^^^^^^^^^^^^^^^^^^^
                         ALWAYS POSITIVE (Term 1)

  If Term 1 > |Term 2|, then Cov(g, P) > 0. QED.

  For a RIGOROUS PROOF, we need to show:
    E[|f'_avg|]*Var(g) > |Cov(|f'_avg|, g(g-mu))|

  This is equivalent to:
    1 > |Corr(|f'_avg|, g(g-mu))| * std(g(g-mu)) / (E[|f'_avg|]*std(g)^2)
""")

print("DONE")
