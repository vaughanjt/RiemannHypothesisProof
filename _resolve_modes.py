"""Resolve all Ulam spiral modes: systematic decomposition via Dirichlet characters.

The mod-8 decomposition found 4 modes with asymmetric couplings.
To resolve ALL modes, we use the natural mathematical framework:
DIRICHLET CHARACTERS, which are the Fourier transform on (Z/qZ)*.

Each Dirichlet character chi mod q defines a mode:
  K_chi(d, T) = sum_p chi(p) * log(p)/p^{1/2} * cos(2*pi*d*log(p)/log(T))

The L-functions L(s, chi) encode each character's contribution to the
zero distribution. The coupling constant C_chi should be related to
L(1, chi) — the value of the L-function at s=1.

PLAN:
1. All characters mod 3, 4, 5, 7, 8, 12 (covers ~15 distinct modes)
2. Also: mod 24 (the lcm of 3,8) gives 8 modes — this should resolve
   most Ulam spiral directions
3. Weight each character by |L(1, chi)| — first-principles coupling
4. Test: does L-function weighting match or beat optimized couplings?
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar, minimize as minimize_nd
import mpmath
mpmath.mp.dps = 20

t0 = time.time()

# ============================================================
# Setup (N=200 for speed)
# ============================================================
N = 200
print("Computing 200 zeros...", flush=True)
zeros = np.array([float(mpmath.im(mpmath.zetazero(k))) for k in range(1, N+1)])
from sympy import primerange
all_primes = list(primerange(2, 1000))
trim = int(0.1*N)
ms = np.mean(np.diff(zeros[trim:-trim]))

def N_deriv(T):
    return np.log(max(T,2)/(2*np.pi))/(2*np.pi) if T>2 else 0.001
def weyl_zero(n):
    t = 2*np.pi*n/np.log(max(n,2)+2)
    for _ in range(30):
        if t<1: t=10.
        Nt = t/(2*np.pi)*np.log(t/(2*np.pi)) - t/(2*np.pi) + 7/8
        dNt = N_deriv(t)
        if abs(dNt)<1e-30: break
        t -= (Nt-n)/dNt
    return t
def S_func(T, ps, mm=3):
    s=0.
    for p in ps:
        lp=np.log(p)
        for m in range(1,mm+1): s -= np.sin(2*m*T*lp)/(m*p**(m/2))
    return s/np.pi

print("Building alpha...", flush=True)
alpha = np.zeros(N)
for k in range(1,N+1):
    Tw=weyl_zero(k); dN=N_deriv(Tw)
    alpha[k-1] = Tw + S_func(Tw, all_primes[:168], 5)/dN
baseline = np.mean(np.abs(alpha-zeros)[trim:-trim])
print(f"  Baseline: {baseline:.4f}", flush=True)


# ============================================================
# Dirichlet characters
# ============================================================
def dirichlet_characters(q):
    """Generate all Dirichlet characters mod q.

    Returns list of (label, chi_dict) where chi_dict maps residues to values.
    Uses the structure of (Z/qZ)*.
    """
    from math import gcd
    # Generators of (Z/qZ)*
    units = [a for a in range(1, q) if gcd(a, q) == 1]
    phi_q = len(units)

    # For small q, enumerate characters directly
    # A character is determined by its values on generators
    # chi(a) is a phi(q)-th root of unity

    chars = []

    # Principal character
    chi0 = {a: 1.0 for a in units}
    chi0[0] = 0.0
    chars.append(("chi_0", chi0))

    # For q prime, characters are powers of a primitive root
    # For general q, use CRT decomposition
    # Simplification: enumerate all possible character tables

    # Use DFT on the group
    # Map units to indices
    unit_to_idx = {a: i for i, a in enumerate(units)}

    for j in range(1, phi_q):
        chi = {}
        for a in units:
            # Character j: chi_j(a) = exp(2*pi*i * j * discrete_log(a) / phi_q)
            # For simplicity, use: chi_j(a) = exp(2*pi*i*j*unit_to_idx[a]/phi_q)
            chi[a] = np.exp(2j * np.pi * j * unit_to_idx[a] / phi_q)
        chi[0] = 0.0
        # Check if this is actually a character (multiplicative)
        is_char = True
        for a in units[:3]:
            for b in units[:3]:
                ab = (a*b) % q
                if ab in chi and a in chi and b in chi:
                    if abs(chi[ab] - chi[a]*chi[b]) > 1e-6:
                        is_char = False
                        break
        if is_char:
            chars.append((f"chi_{j}", chi))

    return chars, units


def kernel_character(d, T, chi_dict, q, primes, max_m=2):
    """Kernel for a specific Dirichlet character."""
    logT = np.log(max(T,10)/(2*np.pi))
    if logT < 0.1: logT = 0.1
    val = 0.0
    for p in primes:
        r = p % q
        if r in chi_dict:
            chi_val = chi_dict[r]
            # Use real part of chi(p) * kernel
            lp = np.log(p)
            for m in range(1, max_m+1):
                amp = lp / (p**(m/2) * logT)
                val += np.real(chi_val) * amp * np.cos(2*np.pi*d*m*lp/logT)
    return val


def build_multichar_banded(alpha_vals, char_list, q, units, W, couplings, primes):
    """Build banded matrix with per-character couplings."""
    n = len(alpha_vals)
    H = np.diag(alpha_vals.copy())
    for k in range(n):
        Tk = alpha_vals[k]
        for d in range(1, W+1):
            if k+d < n:
                val = 0.0
                for i, (cname, chi) in enumerate(char_list):
                    if i < len(couplings):
                        val += couplings[i] * kernel_character(d, Tk, chi, q, primes)
                H[k, k+d] = val
                H[k+d, k] = val
    return H


def score(alpha_vals, char_list, q, units, W, couplings, primes, actual):
    H = build_multichar_banded(alpha_vals, char_list, q, units, W, couplings, primes)
    eigs = np.sort(np.linalg.eigvalsh(H))
    t = int(0.1*len(eigs))
    return np.mean(np.abs(eigs - actual[:len(eigs)])[t:-t])


# ============================================================
# TEST: Systematic moduli
# ============================================================
print("\n" + "="*70, flush=True)
print("DIRICHLET CHARACTER DECOMPOSITION", flush=True)
print("="*70, flush=True)

W = 3
primes_k = all_primes[:168]

results = {}

for q in [3, 4, 5, 7, 8, 11, 12, 24]:
    chars, units = dirichlet_characters(q)
    n_chars = len(chars)

    if n_chars < 2:
        continue

    # Optimize couplings (use Nelder-Mead)
    def obj(params, cl=chars, qq=q, uu=units):
        cs = np.exp(params)
        return score(alpha, cl, qq, uu, W, cs, primes_k, zeros)

    x0 = np.zeros(n_chars)
    res = minimize_nd(obj, x0, method='Nelder-Mead',
                      options={'maxiter': n_chars * 80, 'xatol': 0.01})
    err = res.fun
    c_opt = np.exp(res.x)
    improv = (1 - err/baseline)*100

    # Also compute L(1, chi) for each character
    L_values = []
    for cname, chi in chars:
        # L(1, chi) = sum_n chi(n)/n (truncated)
        L_val = sum(np.real(chi.get(n%q, 0))/n for n in range(1, 500))
        L_values.append(abs(L_val))

    results[q] = (err, improv, n_chars, c_opt, L_values, chars)

    print(f"\n  mod {q}: {n_chars} characters, err={err:.4f} ({improv:+.1f}%)", flush=True)
    for i, (cname, chi) in enumerate(chars):
        print(f"    {cname}: C={c_opt[i]:.4f}, |L(1,chi)|~{L_values[i]:.4f}", flush=True)

# ============================================================
# Compare all moduli
# ============================================================
print("\n" + "="*70, flush=True)
print("COMPARISON", flush=True)
print("="*70, flush=True)

print(f"\n  {'q':>4} {'n_chars':>8} {'error':>8} {'improv':>8} {'%<half':>8}", flush=True)
print(f"  {'-'*40}", flush=True)

for q in sorted(results.keys()):
    err, imp, nc, c_opt, Lv, chars = results[q]
    # Compute % < half gap
    H = build_multichar_banded(alpha, chars, q, [a for a in range(1,q) if np.gcd(a,q)==1],
                               W, c_opt, primes_k)
    eigs = np.sort(np.linalg.eigvalsh(H))
    errs = np.abs(eigs - zeros[:len(eigs)])[trim:-trim]
    pct = np.mean(errs < ms/2)*100
    print(f"  {q:>4} {nc:>8} {err:>8.4f} {imp:>+7.1f}% {pct:>7.1f}%", flush=True)

# ============================================================
# L-function weighting test
# ============================================================
print("\n" + "="*70, flush=True)
print("L-FUNCTION WEIGHTING: C_chi = |L(1,chi)| * global_scale", flush=True)
print("="*70, flush=True)

for q in [8, 12, 24]:
    if q not in results:
        continue
    err_opt, _, nc, c_opt, Lv, chars = results[q]
    units_q = [a for a in range(1,q) if np.gcd(a,q)==1]

    # L-function weighted: C_i = |L(1,chi_i)| * scale
    Lv_arr = np.array(Lv)
    if np.max(Lv_arr) > 0:
        Lv_norm = Lv_arr / np.max(Lv_arr)
    else:
        Lv_norm = np.ones(nc)

    def obj_L(log_scale):
        cs = Lv_norm * np.exp(log_scale)
        return score(alpha, chars, q, units_q, W, cs, primes_k, zeros)

    res_L = minimize_scalar(obj_L, bounds=(-5, 5), method='bounded')
    err_L = res_L.fun
    scale_L = np.exp(res_L.x)

    print(f"\n  mod {q}:", flush=True)
    print(f"    Optimized (free):      err = {err_opt:.4f}", flush=True)
    print(f"    L-function weighted:   err = {err_L:.4f}", flush=True)
    print(f"    Gap: {(err_L/err_opt - 1)*100:+.1f}% (positive = L-weight is worse)", flush=True)

    # Correlation between optimal C and |L(1,chi)|
    if nc > 2:
        from scipy.stats import pearsonr
        r_CL, p_CL = pearsonr(c_opt, Lv_arr)
        print(f"    r(C_opt, |L(1,chi)|) = {r_CL:+.4f} (p={p_CL:.4f})", flush=True)


# ============================================================
# Best overall
# ============================================================
print("\n" + "="*70, flush=True)
print("BEST RESULT", flush=True)
print("="*70, flush=True)

best_q = min(results, key=lambda q: results[q][0])
err, imp, nc, c_opt, Lv, chars = results[best_q]
units_q = [a for a in range(1,best_q) if np.gcd(a,best_q)==1]

H_best = build_multichar_banded(alpha, chars, best_q, units_q, W, c_opt, primes_k)
eigs_best = np.sort(np.linalg.eigvalsh(H_best))
errs_best = np.abs(eigs_best - zeros[:len(eigs_best)])[trim:-trim]

print(f"\n  Best modulus: q={best_q} ({nc} characters)", flush=True)
print(f"  Mean error: {np.mean(errs_best):.4f}", flush=True)
print(f"  Median error: {np.median(errs_best):.4f}", flush=True)
print(f"  % < half gap: {np.mean(errs_best < ms/2)*100:.1f}%", flush=True)
print(f"  % < full gap: {np.mean(errs_best < ms)*100:.1f}%", flush=True)
print(f"  % < 10% gap:  {np.mean(errs_best < ms*0.1)*100:.1f}%", flush=True)
print(f"  Improvement over diagonal: {imp:+.1f}%", flush=True)

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
