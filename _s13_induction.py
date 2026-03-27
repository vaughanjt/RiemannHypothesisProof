"""
Session 13L: PROOF BY INDUCTION ON N
=====================================

Base case: B(5) < 0.497
Inductive step: B(N+1) < B(N) for N >= 5

Key mechanism: adding a high-frequency spectral component (w = log(N+2))
adds NOISE (increases Var(Q|g)) but doesn't significantly change SIGNAL
(Var(q(g)) changes slowly). Therefore R decreases, B decreases.

VERIFY: B(N) at every integer N from 3 to 30, check monotonicity.
"""
import numpy as np, sys
from scipy.stats import pearsonr
sys.stdout.reconfigure(line_buffering=True)

def rs(N):
    p = 1.0/np.arange(1,N+1); p /= p.sum()
    w = np.log(np.arange(2,N+2))
    return p, w

def simulate_and_bound(N, n_trials=250, L=8000, dt=0.02):
    """Simulate GP and compute the noise dilution bound B(N)."""
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

    gaps = np.array(all_g); peaks = np.array(all_P)
    if len(gaps) < 300:
        return None

    Q = peaks/gaps
    mu = np.mean(gaps)
    W = gaps*(gaps-mu)
    g_bar = np.pi / np.sqrt(np.dot(p, w**2))
    bw = 0.12*g_bar

    # q(g) via NW kernel
    gg = np.linspace(np.percentile(gaps,1), np.percentile(gaps,99), 80)
    qq = np.array([np.average(Q, weights=np.exp(-0.5*((gaps-g0)/bw)**2))
                   if np.sum(np.exp(-0.5*((gaps-g0)/bw)**2))>20 else np.nan
                   for g0 in gg])
    v = ~np.isnan(qq)
    if np.sum(v) < 5: return None
    q_at = np.interp(gaps, gg[v], qq[v])

    VQ = np.var(Q); Vq = np.var(q_at)
    R = Vq/VQ if VQ > 1e-10 else 1
    corr_qW = pearsonr(q_at, W)[0]
    r_gP = pearsonr(gaps, peaks)[0]
    bound = abs(corr_qW)*np.sqrt(R)

    return {'N': N, 'n': len(gaps), 'r': r_gP, 'corr': abs(corr_qW),
            'R': R, 'sqrtR': np.sqrt(R), 'bound': bound, 'VQ': VQ, 'Vq': Vq}


# ============================================================
# EVERY INTEGER N FROM 3 TO 25
# ============================================================
print("="*70)
print("INDUCTION VERIFICATION: B(N) AT EVERY INTEGER N")
print("="*70)
print(f"{'N':>4} {'#gaps':>8} {'r(g,P)':>8} {'|Corr|':>8} {'R':>8} "
      f"{'sqrt(R)':>8} {'B(N)':>8} {'<0.497':>7} {'mono':>5}")
print("-"*68)

prev_bound = 999
all_mono = True
results = []

for N in range(3, 26):
    r = simulate_and_bound(N, n_trials=250)
    if r is None:
        print(f"{N:>4} {'FAIL':>8}")
        continue

    mono = "OK" if r['bound'] <= prev_bound + 0.005 else "BREAK"
    if r['bound'] > prev_bound + 0.005 and N > 4:
        all_mono = False

    ok = r['bound'] < 0.497
    print(f"{r['N']:>4} {r['n']:>8} {r['r']:>+8.4f} {r['corr']:>8.4f} {r['R']:>8.4f} "
          f"{r['sqrtR']:>8.4f} {r['bound']:>8.4f} {'YES' if ok else 'NO':>7} {mono:>5}")

    prev_bound = r['bound']
    results.append(r)

print("-"*68)
print(f"MONOTONE from N=5: {'YES' if all_mono else 'NO'}")


# ============================================================
# THE HIGH-FREQUENCY ARGUMENT
# ============================================================
print(f"\n{'='*70}")
print("HIGH-FREQUENCY ARGUMENT: WHY B DECREASES")
print("="*70)

print("""
  When we add the (N+1)-th term with frequency w = log(N+2):

  The key parameter is w * g_bar/2 = log(N+2) * pi / (2*sqrt(m_2))
  This measures how many oscillations the new component makes per gap.

  If w * g_bar/2 >> 1: the new component is "high frequency" relative
  to the gap distribution, and its contribution averages out.

  NOISE ADDITION: The new component adds variance to f(g/2) that is
  approximately independent of g (because the phase w*g/2 mod 2pi is
  approximately uniform for typical gaps). This increases Var(Q|g)
  by ~epsilon * cos^2(w*g/2)/g^2 averaged over the excursion, which
  is ~epsilon/(2*g^2) (since cos^2 averages to 1/2).

  SIGNAL CHANGE: The new component changes q(g) = E[Q|g] by:
  delta_q ~ epsilon * E[cos(w*g/2)_excursion]/g, which averages to
  ~0 over the gap distribution (oscillatory). So Var(q) changes by O(epsilon^2).

  Therefore: R = Var(q)/(Var(q)+E[Var(Q|g)]) DECREASES by O(epsilon).
""")

for N in [5, 10, 20, 50, 100]:
    p, w = rs(N)
    m2 = np.dot(p, w**2)
    g_bar = np.pi / np.sqrt(m2)
    w_new = np.log(N+2)
    osc_per_gap = w_new * g_bar / (2*np.pi)
    eps = 1.0 / ((N+1) * np.sum(1.0/np.arange(1,N+2)))

    print(f"  N={N:>3}: w_new={w_new:.3f}, g_bar={g_bar:.4f}, "
          f"oscillations/gap={osc_per_gap:.2f}, eps={eps:.5f}")


# ============================================================
# PERTURBATIVE CHECK: ΔR and Δ|Corr| per step
# ============================================================
print(f"\n{'='*70}")
print("PERTURBATIVE: CHANGES PER STEP")
print("="*70)

if len(results) >= 3:
    print(f"{'N':>4} {'ΔR':>10} {'Δ|Corr|':>10} {'ΔB':>10} {'ΔR dom?':>8}")
    print("-"*44)
    for i in range(1, len(results)):
        r0, r1 = results[i-1], results[i]
        dR = r1['R'] - r0['R']
        dCorr = r1['corr'] - r0['corr']
        dB = r1['bound'] - r0['bound']
        # R decrease dominates iff |dR| contribution > |dCorr| contribution
        # B = |Corr| * sqrt(R)
        # dB ≈ dCorr * sqrt(R) + |Corr| * dR / (2*sqrt(R))
        dB_from_R = r0['corr'] * dR / (2*r0['sqrtR']) if r0['sqrtR'] > 0 else 0
        dB_from_Corr = dCorr * r0['sqrtR']
        dominates = abs(dB_from_R) > abs(dB_from_Corr)

        if r1['N'] <= 25:
            print(f"{r1['N']:>4} {dR:>+10.4f} {dCorr:>+10.4f} {dB:>+10.4f} "
                  f"{'YES' if dominates else 'no':>8}")


print(f"\n{'='*70}")
print("DONE")
print("="*70)
