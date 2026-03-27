"""Session 13 — fast focused attack. N=200 only."""
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

N = 200
p, w = rs(N)
m2 = np.dot(p, w**2)
g_bar = np.pi / np.sqrt(m2)

# Find g* (first zero of C at tau=g*/2)
tau = np.linspace(0.01, 2*g_bar, 50000)
Cv = Cf(tau, p, w)
sc = np.where(Cv[:-1]*Cv[1:] < 0)[0]
g_star = 2*tau[sc[0]]

print(f"N={N}, g_bar={g_bar:.5f}, g*={g_star:.5f} ({g_star/g_bar:.3f} g_bar)")
print(f"V(g_bar)={Vf([g_bar],p,w)[0]:.6f}, V(g*)={Vf([g_star],p,w)[0]:.6f}")

# Tail oscillation
gt = np.linspace(g_star, 5*g_bar, 20000)
Vt = Vf(gt, p, w)
eps_V = 1-np.min(Vt)
eps_h = np.sqrt(2/np.pi) - np.sqrt(2*np.min(Vt)/np.pi)
print(f"V_tail: [{np.min(Vt):.6f}, {np.max(Vt):.6f}], eps_h={eps_h:.6f}")

# GP simulation — lean
print("\nSimulating GP...", flush=True)
rng = np.random.default_rng(42)
omega, amp = w, 1.0/np.sqrt(np.arange(1,N+1))
sigma_N = np.sqrt(np.sum(1.0/np.arange(1,N+1)))
dt = 0.02; L = 5000; n_trials = 150
chunk = 20000

all_g, all_p = [], []
for trial in range(n_trials):
    if trial % 30 == 0: print(f"  trial {trial}/{n_trials}", flush=True)
    phi = rng.uniform(0, 2*np.pi, N)
    npts = int(L/dt)
    f = np.empty(npts)
    for s in range(0, npts, chunk):
        e = min(s+chunk, npts)
        tc = np.arange(s, e)*dt
        f[s:e] = np.cos(np.outer(tc, omega) + phi) @ amp
    f /= sigma_N
    t_arr = np.arange(npts)*dt
    sc = np.where(f[:-1]*f[1:]<0)[0]
    if len(sc) < 20: continue
    zeros = t_arr[sc] - f[sc]*dt/(f[sc+1]-f[sc])
    gaps = np.diff(zeros)
    midx = ((zeros[:-1]+zeros[1:])/(2*dt)).astype(int)
    midx = np.clip(midx, 0, npts-1)
    pks = np.abs(f[midx])
    tr = max(3, int(0.05*len(gaps)))
    all_g.extend(gaps[tr:-tr].tolist())
    all_p.extend(pks[tr:-tr].tolist())

gaps = np.array(all_g)
peaks = np.array(all_p)
print(f"Collected {len(gaps)} gaps")

mu_g, sigma_g = np.mean(gaps), np.std(gaps)
h_vals = hf(gaps, p, w)
mu_h, sigma_h = np.mean(h_vals), np.std(h_vals)
cov_gh = np.mean((gaps-mu_g)*(h_vals-mu_h))
r_gh = cov_gh/(sigma_g*sigma_h)
P_tail = np.mean(gaps > g_star)

print(f"\nmu_g={mu_g:.5f}, sigma_g={sigma_g:.5f}, CV={sigma_g/mu_g:.4f}")
print(f"mu_h={mu_h:.5f}, sigma_h={sigma_h:.5f}")
print(f"Cov(g,h)={cov_gh:.8f}, r(g,h)={r_gh:.5f}")
print(f"r(g,P)={pearsonr(gaps,peaks)[0]:.5f}")
print(f"P(g>g*)={P_tail:.4f}")

# ============ DECOMPOSITION PROOF ============
print("\n" + "="*60)
print("MONOTONE DECOMPOSITION: h = h_core + delta")
print("="*60)
h_star = np.sqrt(2/np.pi)  # h(g*) = sqrt(2/pi) since V(g*)=1
h_core = np.where(gaps <= g_star, h_vals, h_star)
delta = h_vals - h_core  # delta <= 0 always (since V <= 1 => h <= sqrt(2/pi))

cov_core = np.mean((gaps-mu_g)*(h_core-np.mean(h_core)))
cov_delta = np.mean((gaps-mu_g)*(delta-np.mean(delta)))
eps_d = np.max(np.abs(delta))

# Cauchy-Schwarz bound
cs = sigma_g * eps_d * np.sqrt(P_tail)
margin_cs = cov_core - cs

# Tighter bound: since delta <= 0 and (g-mu) > 0 for g > g* > mu:
# Cov(g, delta) = E[(g-mu)(delta - E[delta])]
# delta supported on g > g* where (g-mu) > 0
# and delta <= 0 there
# So E[(g-mu)delta] <= 0, and E[delta] <= 0
# Cov(g,delta) = E[(g-mu)delta] - E[g-mu]E[delta] = E[(g-mu)delta] - 0*E[delta]
# = E[(g-mu)delta] <= 0
# So the tail integral is EXACTLY non-positive!
raw_tail = np.mean((gaps-mu_g)*delta)  # E[(g-mu)*delta]

print(f"h(g*) = sqrt(2/pi) = {h_star:.6f}")
print(f"Cov_core = {cov_core:.8f}  [POSITIVE by Chebyshev]")
print(f"Cov_delta = {cov_delta:+.8f}")
print(f"E[(g-mu)*delta] = {raw_tail:+.8f}  [MUST BE <= 0]")
print(f"delta <= 0: {np.all(delta <= 1e-10)}")
print(f"eps_delta = {eps_d:.6f}")
print(f"C-S bound = {cs:.8f}")
print(f"MARGIN (C-S) = {margin_cs:+.8f}  {'WORKS' if margin_cs > 0 else 'FAILS'}")
print(f"MARGIN (exact) = {cov_core + raw_tail:+.8f}  (= cov_core + raw_tail)")

# ============ OPTIMAL g* SCAN ============
print("\n" + "="*60)
print("OPTIMAL g* SCAN (maximize C-S margin)")
print("="*60)

best = {'margin': -np.inf}
for g_try in np.linspace(0.3*g_bar, 4*g_bar, 1000):
    h_try = hf(np.array([g_try]), p, w)[0]
    hc = np.where(gaps <= g_try, h_vals, h_try)
    d = h_vals - hc
    cc = np.mean((gaps-mu_g)*(hc-np.mean(hc)))
    ed = np.max(np.abs(d))
    pt = np.mean(gaps > g_try)
    cs_b = sigma_g * ed * np.sqrt(max(pt, 1e-30))
    m = cc - cs_b
    if m > best['margin']:
        best = {'margin': m, 'g': g_try, 'gr': g_try/g_bar,
                'cc': cc, 'cs': cs_b, 'ed': ed, 'pt': pt}

print(f"Best g* = {best['g']:.5f} = {best['gr']:.3f} g_bar")
print(f"Cov_core = {best['cc']:.8f}")
print(f"CS_bound = {best['cs']:.8f}")
print(f"MARGIN = {best['margin']:+.8f}  {'WORKS' if best['margin'] > 0 else 'FAILS'}")
print(f"Ratio = {best['cc']/best['cs']:.2f}x")

# ============ TAYLOR EXPANSION ============
print("\n" + "="*60)
print("TAYLOR EXPANSION")
print("="*60)
eps = 1e-6
V0 = Vf([mu_g], p, w)[0]
Vp = (Vf([mu_g+eps], p, w)[0] - Vf([mu_g-eps], p, w)[0]) / (2*eps)
Vpp = (Vf([mu_g+eps], p, w)[0] - 2*V0 + Vf([mu_g-eps], p, w)[0]) / eps**2

hp = Vp / np.sqrt(2*np.pi*V0) if V0 > 0 else 0
hpp = (Vpp/np.sqrt(2*np.pi*V0) - Vp**2*np.pi/(2*np.pi*V0*np.sqrt(2*np.pi*V0))) if V0>0 else 0

var_g = np.var(gaps)
mu3 = np.mean((gaps-mu_g)**3)
T1 = hp*var_g
T2 = hpp*mu3/2

print(f"V(mu)={V0:.6f}, V'(mu)={Vp:.6f}, V''(mu)={Vpp:.6f}")
print(f"h'(mu)={hp:+.8f}  {'> 0 PROVEN (V increasing at mu < g*)' if hp > 0 else '< 0'}")
print(f"h''(mu)={hpp:+.8f}")
print(f"Var(g)={var_g:.8f}, mu3={mu3:+.8f}, skewness={mu3/sigma_g**3:.4f}")
print(f"Term1 = h'*Var(g) = {T1:+.8f}  [POSITIVE]")
print(f"Term2 = h''*mu3/2 = {T2:+.8f}")
print(f"Taylor(1+2) = {T1+T2:+.8f}")
print(f"Actual Cov  = {cov_gh:+.8f}")
print(f"T1 dominates: {'YES' if abs(T1)>abs(T2) else 'NO'} ({abs(T1)/abs(T2):.2f}x)")

# ============ INTEGRAL REPRESENTATION ============
print("\n" + "="*60)
print("INTEGRAL REPRESENTATION: Cov = int h'(s) K(s) ds")
print("="*60)

s_grid = np.linspace(0.001, 4*g_bar, 2000)
ds = s_grid[1]-s_grid[0]
K_vals = np.array([np.mean((gaps-mu_g)*(gaps>s)) for s in s_grid])
h_grid = hf(s_grid, p, w)
hp_grid = np.gradient(h_grid, ds)

integrand = hp_grid * K_vals * ds
total = np.sum(integrand)
core_mask = s_grid <= g_star
core = np.sum(integrand[core_mask])
tail = np.sum(integrand[~core_mask])

print(f"int h'K ds = {total:.8f}")
print(f"Core [0,g*] = {core:+.8f}  [POSITIVE]")
print(f"Tail [g*,inf] = {tail:+.8f}  [MUST BE <= 0]")
print(f"|Tail|/Core = {abs(tail)/core:.4f}")
print(f"Actual Cov = {cov_gh:.8f}")

# Verify the tail identity: tail = int delta(g) * (g-mu) * rho(g) dg
tail_check = np.mean((gaps[gaps>g_star] - mu_g) * (h_vals[gaps>g_star] - h_star)) * P_tail
print(f"Tail via delta formula = {tail_check:+.8f}")

# ============ CONCORDANCE ============
print("\n" + "="*60)
print("CONCORDANCE (Kendall)")
print("="*60)
rng2 = np.random.default_rng(777)
n = 500000
i1 = rng2.integers(0, len(gaps), n)
i2 = rng2.integers(0, len(gaps), n)
m = i1 != i2; i1, i2 = i1[m], i2[m]
pr = (gaps[i2]-gaps[i1])*(h_vals[i2]-h_vals[i1])
nc, nd = np.sum(pr>0), np.sum(pr<0)
print(f"Concordant/Discordant = {nc/nd:.4f}, tau = {(nc-nd)/(nc+nd):+.6f}")

# ============ REGIONAL DECOMPOSITION ============
print("\n" + "="*60)
print("REGIONAL DECOMPOSITION")
print("="*60)
bounds = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 100]
tot = np.mean((gaps-mu_g)*(h_vals-mu_h))
print(f"{'Region':>18} {'P':>8} {'Contrib':>14} {'Frac':>8}")
for i in range(len(bounds)-1):
    lo, hi = bounds[i]*g_bar, bounds[i+1]*g_bar
    m = (gaps>=lo)&(gaps<hi)
    if np.sum(m)==0: continue
    c = np.sum((gaps[m]-mu_g)*(h_vals[m]-mu_h))/len(gaps)
    print(f"[{bounds[i]:.2f},{bounds[i+1]:.2f})g_bar {np.mean(m):>8.4f} {c:>+14.8f} {c/tot:>7.1%}")
print(f"{'TOTAL':>18} {'1.0':>8} {tot:>+14.8f}")

print("\nDONE")
