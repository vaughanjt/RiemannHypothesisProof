"""Verify the decomposition proof at N=10,20,50,100,200,500 — all must have MARGIN > 0."""
import numpy as np, sys
from scipy.stats import pearsonr
sys.stdout.reconfigure(line_buffering=True)

def rs(N):
    p = 1.0/np.arange(1,N+1); p /= p.sum()
    w = np.log(np.arange(2,N+2))
    return p, w

def Cf(tau, p, w):
    return np.array([np.dot(p, np.cos(w*t)) for t in np.atleast_1d(tau)])

def Vf(g, p, w):
    g = np.atleast_1d(g)
    return 1.0 - 2.0*Cf(g/2,p,w)**2 / (1.0 + Cf(g,p,w))

def hf(g, p, w):
    return np.sqrt(np.maximum(2*Vf(g,p,w)/np.pi, 0))

def find_gstar(p, w, g_bar):
    tau = np.linspace(0.01, 3*g_bar, 80000)
    C = Cf(tau, p, w)
    sc = np.where(C[:-1]*C[1:] < 0)[0]
    return 2*tau[sc[0]] if len(sc) > 0 else 5*g_bar

def simulate(N, n_trials=120, L=5000, dt=0.02):
    p, w = rs(N)
    omega = w; amp = 1.0/np.sqrt(np.arange(1,N+1))
    sigma_N = np.sqrt(np.sum(1.0/np.arange(1,N+1)))
    rng = np.random.default_rng(42)
    chunk = 20000
    all_g, all_p = [], []
    for trial in range(n_trials):
        phi = rng.uniform(0, 2*np.pi, N)
        npts = int(L/dt)
        f = np.empty(npts)
        for s in range(0, npts, chunk):
            e = min(s+chunk, npts)
            tc = np.arange(s,e)*dt
            f[s:e] = np.cos(np.outer(tc, omega)+phi) @ amp
        f /= sigma_N
        t = np.arange(npts)*dt
        sc = np.where(f[:-1]*f[1:]<0)[0]
        if len(sc) < 20: continue
        zeros = t[sc] - f[sc]*dt/(f[sc+1]-f[sc])
        gaps = np.diff(zeros)
        midx = ((zeros[:-1]+zeros[1:])/(2*dt)).astype(int)
        midx = np.clip(midx, 0, npts-1)
        pks = np.abs(f[midx])
        tr = max(3, int(0.05*len(gaps)))
        all_g.extend(gaps[tr:-tr].tolist())
        all_p.extend(pks[tr:-tr].tolist())
    return np.array(all_g), np.array(all_p)

print("="*78)
print("DECOMPOSITION PROOF VERIFICATION ACROSS ALL N")
print("="*78)
print(f"{'N':>5} {'#gaps':>8} {'r(g,h)':>8} {'Cov_core':>12} {'|Cov_d|':>12} "
      f"{'Margin':>12} {'Ratio':>8} {'OK':>4}")
print("-"*78)

for N in [10, 20, 50, 100, 200, 500]:
    p, w = rs(N)
    m2 = np.dot(p, w**2)
    g_bar = np.pi / np.sqrt(m2)
    g_star = find_gstar(p, w, g_bar)

    # Tail eps
    gt = np.linspace(g_star, 6*g_bar, 20000)
    Vt = Vf(gt, p, w)
    V_min = np.min(Vt)
    eps_h = np.sqrt(2/np.pi) - np.sqrt(2*V_min/np.pi)

    # Check V monotone in core
    gc = np.linspace(1e-4, g_star, 30000)
    Vc = Vf(gc, p, w)
    mono_ok = np.all(np.diff(Vc) >= -1e-10)

    # Simulate
    gaps, peaks = simulate(N, n_trials=120)
    if len(gaps) < 100:
        print(f"{N:>5} {'TOO FEW':>8}")
        continue

    mu_g = np.mean(gaps)
    h_vals = hf(gaps, p, w)
    h_star = np.sqrt(2/np.pi)

    h_core = np.where(gaps <= g_star, h_vals, h_star)
    delta = h_vals - h_core

    cov_core = np.mean((gaps-mu_g)*(h_core-np.mean(h_core)))
    raw_tail = np.mean((gaps-mu_g)*delta)  # = Cov(g, delta), must be <= 0
    cov_total = cov_core + raw_tail

    sigma_g = np.std(gaps)
    sigma_h = np.std(h_vals)
    r_gh = cov_total / (sigma_g * sigma_h)

    # Margin: cov_core must exceed |raw_tail|
    margin = cov_core - abs(raw_tail)
    ratio = cov_core / abs(raw_tail) if abs(raw_tail) > 1e-12 else 9999

    ok = margin > 0 and mono_ok
    print(f"{N:>5} {len(gaps):>8} {r_gh:>+8.4f} {cov_core:>12.6f} {abs(raw_tail):>12.6f} "
          f"{margin:>+12.6f} {ratio:>7.1f}x {'YES' if ok else 'NO':>4}")

    if not mono_ok:
        print(f"  WARNING: V not monotone on [0,g*] for N={N}")

print("-"*78)
print("DONE")
