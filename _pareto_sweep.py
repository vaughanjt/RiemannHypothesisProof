"""Pareto frontier: r vs p(GUE) across weight functions and epsilon.

Don't reconstruct. Don't guess. Sweep the full space of:
  H = diag(log(k+1)) + eps * f(j,k,gcd) / (jk)^alpha
for different f, alpha, and epsilon. Find all (r, p_gue) Pareto-optimal points.
"""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from scipy.stats import pearsonr, kstest
from sympy import totient, factorint, divisor_sigma
from riemann.analysis.bost_connes_operator import polynomial_unfold

t0 = time.time()

N = 400

def wigner_cdf(s):
    return 1 - np.exp(-np.pi * s ** 2 / 4)

def measure(eigs_raw):
    eigs = np.sort(eigs_raw)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(sp) < 20: return 0, 0
    sp = sp / np.mean(sp)
    n_trim = int(0.1 * len(eigs))
    et = eigs[n_trim:-n_trim]
    lp, g = [], []
    for i in range(min(len(sp), len(et) - 1)):
        z = (et[i] + et[i+1]) / 2
        lp.append(np.sum(np.log(np.abs(z - eigs) + 1e-30)))
        g.append(sp[i])
    g, lp = np.array(g), np.array(lp)
    r, _ = pearsonr(g, lp) if len(g) > 10 else (0, 1)
    _, p = kstest(sp, wigner_cdf) if len(sp) > 10 else (1, 0)
    return r, p

# Build diagonal
D = np.diag(np.log(np.arange(1, N + 1, dtype=float) + 1))
D_norm = np.linalg.norm(D, 'fro')

# Precompute arithmetic functions
print('Precomputing arithmetic functions...')
# Von Mangoldt
def lambda_val(n):
    if n <= 1: return 0.0
    f = factorint(n)
    return np.log(list(f.keys())[0]) if len(f) == 1 else 0.0

Lambda_cache = {n: lambda_val(n) for n in range(1, N + 1)}
phi_cache = {n: float(totient(n)) for n in range(1, N + 1)}
sigma_cache = {n: float(divisor_sigma(n)) for n in range(1, N + 1)}

# Weight functions: g(gcd_val, j, k) -> weight
weight_funcs = {
    'log(g+1)': lambda g, j, k: np.log(g + 1),
    'log(g)': lambda g, j, k: np.log(max(g, 1)),
    'sqrt(g)': lambda g, j, k: np.sqrt(g),
    'g': lambda g, j, k: float(g),
    'Lambda(g)': lambda g, j, k: Lambda_cache.get(g, 0),
    'phi(g)': lambda g, j, k: phi_cache.get(max(g, 1), 1),
    'sigma(g)': lambda g, j, k: sigma_cache.get(max(g, 1), 1),
    'Lambda(g)*log(g+1)': lambda g, j, k: Lambda_cache.get(g, 0) * np.log(g + 1),
    'div_indicator': lambda g, j, k: 1.0 if j % k == 0 or k % j == 0 else 0.0,
    'log(g+1)*div': lambda g, j, k: np.log(g + 1) * (1.0 if j % k == 0 or k % j == 0 else 0.5),
}

# Denominator scalings
alpha_vals = [0.5]  # sqrt(jk) — the standard one

# Epsilon range
eps_vals = [0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]

# ============================================================
# SWEEP
# ============================================================
print(f'\nSweeping {len(weight_funcs)} weights x {len(eps_vals)} epsilons at N={N}...')
print(f'\n{"Weight":<25} {"eps":>6} {"r":>8} {"p(GUE)":>8} {"score":>8}')
print('-' * 60)

all_results = []

for wname, wfunc in weight_funcs.items():
    # Build weight matrix once
    W = np.zeros((N, N))
    for j in range(1, N + 1):
        for k in range(j + 1, N + 1):
            g = np.gcd(j, k)
            val = wfunc(g, j, k) / np.sqrt(j * k)
            W[j-1, k-1] = val
            W[k-1, j-1] = val

    W_norm = np.linalg.norm(W, 'fro')
    if W_norm < 1e-10:
        continue
    W_sc = W * (D_norm / W_norm)

    best_for_weight = None
    for eps in eps_vals:
        H = D + eps * W_sc
        eigs = np.linalg.eigvalsh(H)
        r, p_gue = measure(eigs)

        # Score: geometric mean of r and p_gue (both must be positive)
        score = np.sqrt(max(r, 0) * min(p_gue, 1)) if r > 0 and p_gue > 0.01 else 0
        all_results.append((wname, eps, r, p_gue, score))

        if best_for_weight is None or score > best_for_weight[4]:
            best_for_weight = (wname, eps, r, p_gue, score)

    if best_for_weight and best_for_weight[4] > 0.05:
        w, e, r, p, s = best_for_weight
        tag = ' ***' if r > 0.4 and p > 0.05 else (' **' if r > 0.3 and p > 0.05 else '')
        print(f'{w:<25} {e:>6.2f} {r:>+8.4f} {p:>8.4f} {s:>8.4f}{tag}')

# ============================================================
# PARETO FRONTIER
# ============================================================
print('\n' + '=' * 70)
print('PARETO FRONTIER: r vs p(GUE)')
print('=' * 70)

# Sort by score (geometric mean of r and p)
all_results.sort(key=lambda x: -x[4])

print(f'\n  Top 20 configurations:')
print(f'  {"Weight":<25} {"eps":>6} {"r":>8} {"p(GUE)":>8} {"score":>8}')
print(f'  {"-"*58}')
for wname, eps, r, p_gue, score in all_results[:20]:
    tag = ''
    if r > 0.5 and p_gue > 0.05:
        tag = ' *** BOTH'
    elif r > 0.4 and p_gue > 0.05:
        tag = ' ** BOTH'
    elif r > 0.3 and p_gue > 0.05:
        tag = ' * both'
    print(f'  {wname:<25} {eps:>6.2f} {r:>+8.4f} {p_gue:>8.4f} {score:>8.4f}{tag}')

# Find the Pareto-optimal points (no other point dominates in BOTH r and p)
pareto = []
for i, (w1, e1, r1, p1, s1) in enumerate(all_results):
    dominated = False
    for w2, e2, r2, p2, s2 in all_results:
        if r2 > r1 and p2 > p1:  # strictly dominates
            dominated = True
            break
    if not dominated and r1 > 0 and p1 > 0.01:
        pareto.append((w1, e1, r1, p1, s1))

print(f'\n  Pareto-optimal points ({len(pareto)}):')
print(f'  {"Weight":<25} {"eps":>6} {"r":>8} {"p(GUE)":>8}')
print(f'  {"-"*52}')
for w, e, r, p, s in sorted(pareto, key=lambda x: -x[2])[:15]:
    print(f'  {w:<25} {e:>6.2f} {r:>+8.4f} {p:>8.4f}')

# ============================================================
# VERDICT
# ============================================================
print('\n' + '=' * 70)
print('VERDICT')
print('=' * 70)

# Best "both" point
both = [(w, e, r, p, s) for w, e, r, p, s in all_results if r > 0.3 and p > 0.05]
if both:
    best_both = max(both, key=lambda x: x[4])
    print(f'\n  BEST JOINT (r > 0.3 AND p > 0.05):')
    print(f'    Weight: {best_both[0]}')
    print(f'    epsilon: {best_both[1]}')
    print(f'    r = {best_both[2]:+.4f}')
    print(f'    p(GUE) = {best_both[3]:.4f}')
    print(f'    score = {best_both[4]:.4f}')

    # Best high-r point
    high_r = max(all_results, key=lambda x: x[2])
    print(f'\n  HIGHEST r:')
    print(f'    {high_r[0]} at eps={high_r[1]}: r={high_r[2]:+.4f}, p={high_r[3]:.4f}')

    # Best GUE point with any r > 0
    gue_good = [x for x in all_results if x[3] > 0.1 and x[2] > 0]
    if gue_good:
        best_gue = max(gue_good, key=lambda x: x[2])
        print(f'\n  BEST r WITH GOOD GUE (p > 0.1):')
        print(f'    {best_gue[0]} at eps={best_gue[1]}: r={best_gue[2]:+.4f}, p={best_gue[3]:.4f}')

print(f'\nTotal time: {time.time() - t0:.1f}s')
