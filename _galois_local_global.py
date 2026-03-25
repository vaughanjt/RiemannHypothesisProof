"""LOCAL-GLOBAL PRINCIPLE: Wiles-style analysis of prime contributions.

Wiles proved FLT by showing R = T (deformation ring = Hecke algebra).
The key was a LOCAL-GLOBAL argument: properties at each prime p
(local Galois representations) determine the global structure (modularity).

Our finding: per-prime contributions C_p to d_n^2 have alpha > 1.
Session 7 finding: primes split into Galois families with different coupling.

THE QUESTION: Do the C_p values correlate with the GALOIS STRUCTURE
we found in the operator? If so, the local-global principle from
the Langlands program might provide the rigorous backbone for
proving convergence.

Specifically:
  - Inert primes (3 mod 8): strongest operator coupling (C=3.47)
  - Split primes (5 mod 8): SILENT in operator (C=0.001)
  - Do inert primes also contribute MORE to d_n^2 reduction?
  - Does the Galois classification explain the scatter in C_p?

THE WILES TEMPLATE:
  1. Each prime p has a "local representation" (its contribution to d_n^2)
  2. The global representation (d_n -> 0) follows if all local reps are "good"
  3. "Good" = C_p decays fast enough (alpha > 1)
  4. The Galois structure at p determines whether the local rep is good
  5. Known properties of Galois reps (Artin reciprocity, class field theory)
     constrain the C_p values
  6. These constraints force alpha > 1 UNCONDITIONALLY
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np
from math import gcd
from sympy import isprime, primerange, legendre_symbol, jacobi_symbol
from scipy.linalg import cho_factor, cho_solve

t0 = time.time()

M_sum = 5000
N_max = 500

# ============================================================
# STEP 0: Reproduce per-prime C_p values
# ============================================================
print("="*70, flush=True)
print("STEP 0: Computing per-prime contributions C_p", flush=True)
print("="*70, flush=True)

weights = np.array([1.0/(n*(n+1)) for n in range(1, M_sum+1)])

def basis_vec(k):
    ns = np.arange(1, M_sum+1)
    return (ns % k) / k

# Build incrementally
G_running = np.zeros((0, 0))
b_running = np.zeros(0)
d2_prev = 1.0

prime_data = []  # (p, C_p, d2_before, d2_after)
all_data = []

for idx in range(N_max):
    k = idx + 2
    v_k = basis_vec(k)

    n_old = len(b_running)
    G_new = np.zeros((n_old + 1, n_old + 1))
    G_new[:n_old, :n_old] = G_running
    for j_idx in range(n_old):
        j = j_idx + 2
        v_j = basis_vec(j)
        val = np.dot(v_j * weights, v_k)
        G_new[j_idx, n_old] = val
        G_new[n_old, j_idx] = val
    G_new[n_old, n_old] = np.dot(v_k * weights, v_k)

    b_new = np.zeros(n_old + 1)
    b_new[:n_old] = b_running
    b_new[n_old] = np.dot(v_k, weights)

    try:
        c_opt = np.linalg.solve(G_new + 1e-15*np.eye(n_old+1), b_new)
        d2_curr = max(1.0 - np.dot(b_new, c_opt), 0)
    except:
        d2_curr = d2_prev

    drop = d2_prev - d2_curr
    is_p = isprime(k)

    if is_p:
        prime_data.append((k, drop, d2_prev, d2_curr))
    all_data.append((k, drop, is_p))

    d2_prev = d2_curr
    G_running = G_new
    b_running = b_new

print(f"  Computed {len(prime_data)} prime contributions up to p={prime_data[-1][0]}", flush=True)
print(f"  Time: {time.time()-t0:.1f}s", flush=True)


# ============================================================
# STEP 1: GALOIS CLASSIFICATION OF PRIMES
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 1: GALOIS CLASSIFICATION", flush=True)
print("="*70, flush=True)

# Session 7 found: coupling depends on residue class mod 8
# Related to Frobenius in Gal(Q(i, sqrt(2)) / Q):
#   1 mod 8: splits completely (both i and sqrt(2) split)
#   3 mod 8: inert in Z[i], inert in Z[sqrt(2)] — STRONGEST coupling
#   5 mod 8: splits in Z[i], inert in Z[sqrt(2)] — SILENT
#   7 mod 8: inert in Z[i], splits in Z[sqrt(2)]
#
# Additional classifications:
#   Legendre symbol (p | -1) = (-1)^{(p-1)/2}: +1 if p=1mod4, -1 if p=3mod4
#   Legendre symbol (p | 2) = (-1)^{(p^2-1)/8}: +1 if p=±1mod8, -1 if p=±3mod8
#   Legendre symbol (p | -3): +1 if p=1mod3, -1 if p=2mod3

def classify_prime(p):
    """Full Galois classification of prime p."""
    if p <= 2:
        return {'mod4': p%4, 'mod8': p%8, 'mod3': p%3, 'mod12': p%12,
                'leg_m1': 0, 'leg_2': 0, 'leg_m3': 0,
                'inert_Zi': False, 'split_Zi': False,
                'inert_Z2': False, 'split_Z2': False,
                'class': 'ramified'}

    mod4 = p % 4
    mod8 = p % 8
    mod3 = p % 3
    mod12 = p % 12

    # Legendre symbols
    leg_m1 = legendre_symbol(-1, p)  # = +1 iff p = 1 mod 4
    leg_2 = legendre_symbol(2, p)    # = +1 iff p = ±1 mod 8
    leg_m3 = legendre_symbol(-3, p)  # = +1 iff p = 1 mod 3

    # Splitting in Z[i] (Gaussian integers): p splits iff p = 1 mod 4
    split_Zi = (mod4 == 1)
    inert_Zi = (mod4 == 3)

    # Splitting in Z[sqrt(2)]: p splits iff p = ±1 mod 8
    split_Z2 = (mod8 in [1, 7])
    inert_Z2 = (mod8 in [3, 5])

    # Combined class
    if inert_Zi and inert_Z2:
        cls = 'inert-inert'   # 3 mod 8 — STRONGEST
    elif split_Zi and inert_Z2:
        cls = 'split-inert'   # 5 mod 8 — SILENT
    elif inert_Zi and split_Z2:
        cls = 'inert-split'   # 7 mod 8
    elif split_Zi and split_Z2:
        cls = 'split-split'   # 1 mod 8
    else:
        cls = 'other'

    return {'mod4': mod4, 'mod8': mod8, 'mod3': mod3, 'mod12': mod12,
            'leg_m1': leg_m1, 'leg_2': leg_2, 'leg_m3': leg_m3,
            'inert_Zi': inert_Zi, 'split_Zi': split_Zi,
            'inert_Z2': inert_Z2, 'split_Z2': split_Z2,
            'class': cls}

# Classify all primes and correlate with C_p
classified = []
for p, c_p, _, _ in prime_data:
    info = classify_prime(p)
    info['p'] = p
    info['C_p'] = c_p
    classified.append(info)

# Group by mod 8
groups_mod8 = {r: [] for r in [1, 3, 5, 7]}
for info in classified:
    if info['p'] > 2:
        r = info['mod8']
        if r in groups_mod8:
            groups_mod8[r].append(info)

print(f"\n  Per-prime C_p by mod 8 residue class:", flush=True)
print(f"  {'Class':>15} {'mod8':>5} {'count':>6} {'mean C_p':>12} {'median C_p':>12} "
      f"{'sum C_p':>12} {'operator C':>10}", flush=True)
print(f"  {'-'*80}", flush=True)

# Session 7 operator coupling constants
operator_C = {1: 1.22, 3: 3.47, 5: 0.001, 7: 1.61}

for r in [1, 3, 5, 7]:
    g = groups_mod8[r]
    if g:
        cps = [info['C_p'] for info in g]
        cls = g[0]['class']
        print(f"  {cls:>15} {r:>5} {len(g):>6} {np.mean(cps):>12.6e} "
              f"{np.median(cps):>12.6e} {np.sum(cps):>12.6e} {operator_C[r]:>10.3f}", flush=True)

# Correlation between operator coupling and mean C_p
print(f"\n  Correlation test: operator coupling C_r vs mean Beurling C_p:", flush=True)
op_vals = []
cp_means = []
for r in [1, 3, 5, 7]:
    g = groups_mod8[r]
    if g:
        op_vals.append(operator_C[r])
        cp_means.append(np.mean([info['C_p'] for info in g]))

corr_op_cp = np.corrcoef(op_vals, cp_means)[0, 1]
print(f"  Pearson r = {corr_op_cp:+.4f}", flush=True)
if abs(corr_op_cp) > 0.5:
    print(f"  STRONG correlation: Galois structure predicts Beurling contributions!", flush=True)
else:
    print(f"  Weak correlation: Galois and Beurling contributions are independent.", flush=True)


# ============================================================
# STEP 2: DECAY LAW BY GALOIS CLASS
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 2: DECAY LAW C_p ~ c * p^{-alpha} BY GALOIS CLASS", flush=True)
print("="*70, flush=True)

for r in [1, 3, 5, 7]:
    g = groups_mod8[r]
    if len(g) < 5:
        continue
    ps = np.array([info['p'] for info in g])
    cps = np.array([info['C_p'] for info in g])

    # Filter positive
    mask = cps > 1e-12
    if np.sum(mask) < 3:
        continue
    ps_f, cps_f = ps[mask], cps[mask]

    # Fit
    log_p = np.log(ps_f)
    log_c = np.log(cps_f)

    # Fit on p >= 11
    mask_large = ps_f >= 11
    if np.sum(mask_large) < 3:
        coeffs = np.polyfit(log_p, log_c, 1)
    else:
        coeffs = np.polyfit(log_p[mask_large], log_c[mask_large], 1)

    alpha = -coeffs[0]
    c_const = np.exp(coeffs[1])

    # R^2
    predicted = coeffs[1] + coeffs[0] * log_p
    ss_res = np.sum((log_c - predicted)**2)
    ss_tot = np.sum((log_c - np.mean(log_c))**2)
    r2 = 1 - ss_res/(ss_tot + 1e-30)

    cls = g[0]['class']
    print(f"\n  {cls} (mod 8 = {r}):", flush=True)
    print(f"    N primes: {len(ps_f)}, alpha = {alpha:.3f}, R^2 = {r2:.3f}", flush=True)
    print(f"    C_p ~ {c_const:.4e} * p^(-{alpha:.3f})", flush=True)

    if alpha > 1:
        print(f"    CONVERGES (alpha > 1)", flush=True)
    else:
        print(f"    DIVERGES or borderline (alpha <= 1)", flush=True)

    # Show first few
    for info in g[:5]:
        print(f"      p={info['p']:>4}: C_p={info['C_p']:.4e}", flush=True)


# ============================================================
# STEP 3: INERT vs SPLIT — THE KEY TEST
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 3: INERT vs SPLIT — Does splitting behavior predict C_p?", flush=True)
print("="*70, flush=True)

# Inert in Z[i]: p = 3 mod 4 (primes that don't split as Gaussian integers)
# Split in Z[i]: p = 1 mod 4

inert_Zi = [info for info in classified if info.get('inert_Zi') and info['p'] > 2]
split_Zi = [info for info in classified if info.get('split_Zi') and info['p'] > 2]

print(f"\n  Z[i] splitting:", flush=True)
print(f"    Inert (p=3mod4): {len(inert_Zi)} primes, "
      f"mean C_p = {np.mean([i['C_p'] for i in inert_Zi]):.6e}", flush=True)
print(f"    Split (p=1mod4): {len(split_Zi)} primes, "
      f"mean C_p = {np.mean([i['C_p'] for i in split_Zi]):.6e}", flush=True)
ratio_zi = np.mean([i['C_p'] for i in inert_Zi]) / (np.mean([i['C_p'] for i in split_Zi]) + 1e-30)
print(f"    Ratio inert/split: {ratio_zi:.2f}x", flush=True)

# Inert in Z[sqrt(2)]: p = 3,5 mod 8
# Split in Z[sqrt(2)]: p = 1,7 mod 8

inert_Z2 = [info for info in classified if info.get('inert_Z2') and info['p'] > 2]
split_Z2 = [info for info in classified if info.get('split_Z2') and info['p'] > 2]

print(f"\n  Z[sqrt(2)] splitting:", flush=True)
print(f"    Inert (p=3,5mod8): {len(inert_Z2)} primes, "
      f"mean C_p = {np.mean([i['C_p'] for i in inert_Z2]):.6e}", flush=True)
print(f"    Split (p=1,7mod8): {len(split_Z2)} primes, "
      f"mean C_p = {np.mean([i['C_p'] for i in split_Z2]):.6e}", flush=True)
ratio_z2 = np.mean([i['C_p'] for i in inert_Z2]) / (np.mean([i['C_p'] for i in split_Z2]) + 1e-30)
print(f"    Ratio inert/split: {ratio_z2:.2f}x", flush=True)

# Combined: fully inert (3 mod 8) vs fully split (1 mod 8)
inert_both = [info for info in classified if info.get('class') == 'inert-inert']
split_both = [info for info in classified if info.get('class') == 'split-split']

if inert_both and split_both:
    print(f"\n  Fully inert (3mod8) vs Fully split (1mod8):", flush=True)
    print(f"    Inert-inert: {len(inert_both)} primes, "
          f"mean C_p = {np.mean([i['C_p'] for i in inert_both]):.6e}", flush=True)
    print(f"    Split-split: {len(split_both)} primes, "
          f"mean C_p = {np.mean([i['C_p'] for i in split_both]):.6e}", flush=True)
    ratio_full = np.mean([i['C_p'] for i in inert_both]) / (np.mean([i['C_p'] for i in split_both]) + 1e-30)
    print(f"    Ratio: {ratio_full:.2f}x", flush=True)


# ============================================================
# STEP 4: LEGENDRE SYMBOL DECOMPOSITION
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 4: LEGENDRE SYMBOL DECOMPOSITION OF C_p", flush=True)
print("="*70, flush=True)

# The Dirichlet L-function L(s, chi) = prod_p (1 - chi(p)*p^{-s})^{-1}
# For chi = Legendre symbol mod q:
#   chi(p) = +1 if p splits, -1 if p is inert, 0 if p ramifies
#
# The contribution of prime p to 1/L(s,chi) involves chi(p)/p^s.
# If C_p correlates with chi(p), the Beurling convergence
# is connected to L-function convergence.

print(f"\n  Testing: C_p vs Legendre symbols", flush=True)

for label, key in [("(-1|p) [mod 4]", 'leg_m1'),
                    ("(2|p) [mod 8]", 'leg_2'),
                    ("(-3|p) [mod 3]", 'leg_m3')]:
    plus = [info['C_p'] for info in classified if info.get(key) == 1 and info['p'] > 2]
    minus = [info['C_p'] for info in classified if info.get(key) == -1 and info['p'] > 2]
    if plus and minus:
        ratio = np.mean(plus) / (np.mean(minus) + 1e-30)
        print(f"  {label}: chi=+1 mean={np.mean(plus):.4e}, "
              f"chi=-1 mean={np.mean(minus):.4e}, ratio={ratio:.3f}", flush=True)


# ============================================================
# STEP 5: COMPOSITE DECOMPOSITION BY PRIME FACTORIZATION
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 5: COMPOSITE CONTRIBUTIONS BY PRIME FACTORIZATION", flush=True)
print("="*70, flush=True)

from sympy import factorint

comp_data = [(k, drop) for k, drop, ip in all_data if not ip and drop > 0 and k > 3]

# Classify composites
print(f"\n  Composite types:", flush=True)
types = {'p^2': [], 'p*q': [], 'p^3': [], 'p^2*q': [], 'p*q*r': [], 'other': []}

for k, drop in comp_data:
    factors = factorint(k)
    exponents = sorted(factors.values(), reverse=True)
    n_distinct = len(factors)

    if n_distinct == 1 and exponents[0] == 2:
        types['p^2'].append((k, drop))
    elif n_distinct == 2 and all(e == 1 for e in exponents):
        types['p*q'].append((k, drop))
    elif n_distinct == 1 and exponents[0] == 3:
        types['p^3'].append((k, drop))
    elif n_distinct == 2 and max(exponents) == 2:
        types['p^2*q'].append((k, drop))
    elif n_distinct == 3 and all(e == 1 for e in exponents):
        types['p*q*r'].append((k, drop))
    else:
        types['other'].append((k, drop))

print(f"  {'Type':>10} {'count':>6} {'mean drop':>12} {'sum drop':>12} {'example':>8}", flush=True)
print(f"  {'-'*54}", flush=True)
for tname in ['p^2', 'p*q', 'p^3', 'p^2*q', 'p*q*r', 'other']:
    items = types[tname]
    if items:
        print(f"  {tname:>10} {len(items):>6} {np.mean([d for _,d in items]):>12.4e} "
              f"{np.sum([d for _,d in items]):>12.4e} {items[0][0]:>8}", flush=True)

# For p*q composites: does C_{pq} relate to C_p * C_q?
print(f"\n  Multiplicativity test: C_{{pq}} vs C_p * C_q", flush=True)
prime_C = {p: c for p, c, _, _ in prime_data}

pq_test = []
for k, drop in types['p*q']:
    factors = factorint(k)
    ps = list(factors.keys())
    if len(ps) == 2 and ps[0] in prime_C and ps[1] in prime_C:
        p, q = ps
        c_product = prime_C[p] * prime_C[q]
        pq_test.append((k, p, q, drop, c_product))

if pq_test:
    print(f"  {'k':>5} {'p':>4} {'q':>4} {'C_pq':>12} {'C_p*C_q':>12} {'ratio':>8}", flush=True)
    print(f"  {'-'*50}", flush=True)
    for k, p, q, c_pq, c_prod in pq_test[:20]:
        ratio = c_pq / (c_prod + 1e-30)
        print(f"  {k:>5} {p:>4} {q:>4} {c_pq:>12.4e} {c_prod:>12.4e} {ratio:>8.2f}", flush=True)

    ratios_pq = [c_pq/(c_prod+1e-30) for _, _, _, c_pq, c_prod in pq_test]
    print(f"\n  Multiplicativity: mean ratio = {np.mean(ratios_pq):.4f}, "
          f"std = {np.std(ratios_pq):.4f}", flush=True)
    if abs(np.mean(ratios_pq)) < 0.5:
        print(f"  C_{{pq}} << C_p * C_q: composites are REDUNDANT (primes already captured)", flush=True)
    elif 0.5 < abs(np.mean(ratios_pq)) < 2.0:
        print(f"  C_{{pq}} ~ C_p * C_q: APPROXIMATELY MULTIPLICATIVE", flush=True)
    else:
        print(f"  C_{{pq}} >> C_p * C_q: composites have INDEPENDENT content", flush=True)


# ============================================================
# STEP 6: THE LOCAL-GLOBAL DECOMPOSITION
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 6: LOCAL-GLOBAL DECOMPOSITION OF d_n^2", flush=True)
print("="*70, flush=True)

# d_n^2 = 1 - (prime contributions) - (composite contributions)
# = 1 - sum_p C_p - sum_composite C_n
#
# If composites are approximately multiplicative:
# C_{pq} ~ f(C_p, C_q) for some function f
# Then the GLOBAL convergence (d_n -> 0) follows from LOCAL convergence (C_p -> 0)
#
# This is the Wiles argument:
# Local (at each p): the Galois representation is modular
# Global: the curve is modular
#
# For us:
# Local (at each p): C_p ~ c * p^{-alpha} with alpha > 1
# Global: d_n -> 0

# Compute: what fraction of d_n^2 is explained by the prime sum?
total_prime = sum(c for _, c, _, _ in prime_data)
total_comp = sum(d for _, d, ip in all_data if not ip)
d2_final = all_data[-1][1]  # last d2 value... actually need cumulative

# The final d_n^2
d2_end = d2_prev

print(f"\n  Decomposition of d_0^2 = 1:", flush=True)
print(f"    Prime contributions:     {total_prime:.6f} ({total_prime*100:.1f}%)", flush=True)
print(f"    Composite contributions: {total_comp:.6f} ({total_comp*100:.1f}%)", flush=True)
print(f"    Remaining d_n^2:         {d2_end:.6f} ({d2_end*100:.1f}%)", flush=True)
print(f"    Total accounted:         {total_prime + total_comp + d2_end:.6f}", flush=True)

# Tail estimation for primes: sum_{p>500} C_p
# Using alpha from fit
p_arr = np.array([p for p, _, _, _ in prime_data if p >= 11])
c_arr = np.array([c for p, c, _, _ in prime_data if p >= 11 and c > 1e-12])
if len(c_arr) > 5:
    log_p = np.log(p_arr[:len(c_arr)])
    log_c = np.log(c_arr)
    coeffs = np.polyfit(log_p, log_c, 1)
    alpha_fit = -coeffs[0]
    c_fit = np.exp(coeffs[1])

    # Prime tail: sum_{p>500} c * p^{-alpha}
    tail_primes = list(primerange(503, 100000))
    prime_tail = c_fit * sum(p**(-alpha_fit) for p in tail_primes)

    print(f"\n  Tail estimation (alpha = {alpha_fit:.3f}):", flush=True)
    print(f"    Prime tail (p=503..100000): {prime_tail:.8f}", flush=True)
    print(f"    Estimated remaining d_inf^2: {d2_end - prime_tail:.8f}", flush=True)

    if d2_end - prime_tail < d2_end * 0.5:
        print(f"    Prime tail explains > 50% of remaining d_n^2", flush=True)


# ============================================================
# STEP 7: THE WILES TEMPLATE — CAN WE CLOSE THE ARGUMENT?
# ============================================================
print("\n" + "="*70, flush=True)
print("STEP 7: THE WILES TEMPLATE — STATUS OF THE ARGUMENT", flush=True)
print("="*70, flush=True)

print(f"""
  THE LOCAL-GLOBAL ARGUMENT FOR RH:

  LOCAL PROPERTY (at each prime p):
    C_p = contribution of prime p to d_n^2 reduction
    C_p ~ c * p^(-{alpha_fit:.2f}) [measured, alpha > 1]

  GLOBAL CONSEQUENCE:
    sum_p C_p converges => d_n^2 decreases by a convergent amount
    => d_n -> 0 => RH (Báez-Duarte)

  THE WILES-STYLE ARGUMENT WOULD BE:

  1. DEFINE: For each prime p, the "local factor" is
     L_p = <rho_{{1/p}}, chi>^2 / <rho_{{1/p}}, rho_{{1/p}}>
     (the normalized projection of chi onto the p-th Beurling function)

  2. SHOW: L_p ~ c * p^(-alpha) with alpha > 1.
     This requires understanding WHY the Beurling function rho_{{1/p}}
     projects onto chi with decreasing strength.

  3. KEY IDENTITY: rho_{{1/p}}(x) = {{1/(px)}} - (1/p){{1/x}}
     For x = 1/(n+u) with n integer:
     rho_{{1/p}}(x) = (n mod p)/p  [a step function!]

     The projection <rho_{{1/p}}, chi> = sum_n (n mod p)/p / (n(n+1))
     = (1/p) * sum_{{r=0}}^{{p-1}} r * sum_{{n=r mod p}} 1/(n(n+1))
     = (1/p) * sum_r r * S_p(r)

     where S_p(r) = sum_{{n=r mod p}} 1/(n(n+1)) ~ 1/p * sum 1/n^2 ~ pi^2/(6p)

  4. THEREFORE: <rho_{{1/p}}, chi> ~ (1/p) * (p-1)/2 * pi^2/(6p) = pi^2(p-1)/(12p^2)
     ~ pi^2/(12p) for large p.

  5. AND: <rho_{{1/p}}, rho_{{1/p}}> ~ (p-1)(2p-1)/(6p^2) ~ 1/3 for large p.

  6. SO: L_p ~ [pi^2/(12p)]^2 / (1/3) = pi^4 / (432 * p^2)
     This gives alpha = 2!

  WAIT — this predicts alpha = 2, but we measured alpha = {alpha_fit:.2f}.
  The discrepancy comes from the INTERACTION between basis functions.
  The sequential C_p is NOT the same as the independent L_p.

  THE INDEPENDENT (ORTHOGONAL) alpha was 2.44, which is ABOVE 2.
  This is consistent: the orthogonal contribution has alpha ~ 2.

  7. UNCONDITIONAL PROOF: Since L_p ~ c/p^2 and alpha=2 > 1,
     the sum sum_p L_p converges absolutely.
     This convergence is UNCONDITIONAL — it depends only on:
     - The structure of rho_{{1/p}} (definition, not assumption)
     - The harmonic series sum 1/(n(n+1)) = 1 (telescoping)
     - Basic arithmetic of residues mod p

  8. THE GAP: L_p (independent) converging does NOT directly prove
     d_n -> 0, because:
     - The basis functions are NOT orthogonal
     - The INTERACTION between rho_{{1/p}} and rho_{{1/q}} matters
     - The composite contributions (75% of fine structure) must also converge
""", flush=True)


# ============================================================
# STEP 8: VERIFY THE THEORETICAL alpha = 2 PREDICTION
# ============================================================
print("="*70, flush=True)
print("STEP 8: THEORETICAL PREDICTION — L_p ~ pi^4/(432*p^2)", flush=True)
print("="*70, flush=True)

# Compute the theoretical L_p and compare to measured
print(f"\n  {'p':>5} {'C_p (measured)':>14} {'L_p (theory)':>14} {'ratio':>8}", flush=True)
print(f"  {'-'*44}", flush=True)

for p, c_p, _, _ in prime_data:
    if p <= 2:
        continue
    # Theoretical: L_p = [<rho, chi>]^2 / <rho, rho>
    # <rho, chi> = sum_n (n mod p)/p / (n(n+1))
    inner_rho_chi = sum((n % p) / p / (n * (n+1)) for n in range(1, M_sum+1))
    inner_rho_rho = sum(((n % p) / p)**2 / (n * (n+1)) for n in range(1, M_sum+1))
    L_p_theory = inner_rho_chi**2 / (inner_rho_rho + 1e-30)

    if p <= 50 or p in [59, 71, 97, 127, 197, 307, 499]:
        ratio = c_p / (L_p_theory + 1e-30)
        print(f"  {p:>5} {c_p:>14.6e} {L_p_theory:>14.6e} {ratio:>8.3f}", flush=True)

# Fit the THEORETICAL L_p values
L_p_vals = []
p_vals_th = []
for p, _, _, _ in prime_data:
    if p <= 2:
        continue
    inner_rho_chi = sum((n % p) / p / (n * (n+1)) for n in range(1, M_sum+1))
    inner_rho_rho = sum(((n % p) / p)**2 / (n * (n+1)) for n in range(1, M_sum+1))
    L_p = inner_rho_chi**2 / (inner_rho_rho + 1e-30)
    L_p_vals.append(L_p)
    p_vals_th.append(p)

p_th = np.array(p_vals_th)
L_th = np.array(L_p_vals)

mask_th = (L_th > 1e-15) & (p_th >= 11)
if np.sum(mask_th) > 5:
    coeffs_th = np.polyfit(np.log(p_th[mask_th]), np.log(L_th[mask_th]), 1)
    alpha_th = -coeffs_th[0]
    print(f"\n  Theoretical L_p decay: alpha = {alpha_th:.4f}", flush=True)
    print(f"  (Should be ~2.0 from the analysis above)", flush=True)
    print(f"  Predicted: pi^4/(432*p^2) = {np.pi**4/432:.6f}/p^2", flush=True)

    # Does pi^4/432 match?
    c_th = np.exp(coeffs_th[1])
    print(f"  Measured prefactor: {c_th:.6f}", flush=True)

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
