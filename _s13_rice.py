"""
Session 13d: RALPH MODE — Rice formula direct attack on Cov(g, P) > 0
======================================================================

GOAL: Express Cov(g, P) directly as an integral over the spectral density,
bypassing h(g) entirely. If the integral is manifestly positive, proof is done.

APPROACH: Use Rice's formula for zero crossings of Gaussian processes.

For a stationary GP with spectral density:
  - Joint density of consecutive zeros at (0, g): involves the "gap density"
    rho(g) which depends on the "no-crossing probability"
  - E[P | gap=g] = E[|f(g/2)| | f(0)=0, f(g)=0, no zero in (0,g)]
  - Cov(g, P) = int g * E[P|g] * rho(g) dg - mu_g * mu_P

KEY IDEA: Instead of decomposing h, express Cov(g, P) as a DOUBLE integral
over the joint density of (g, f(g/2), f'(0), f'(g)) and check positivity.

Rice's formula for the gap density:
  rho(g) = lambda_0^2 * E[|f'(0)| * |f'(g)| * 1_{no zero in (0,g)} | f(0)=f(g)=0]

where lambda_0 = sqrt(m_2)/pi is the zero intensity.

The midpoint value f(g/2) conditioned on (f(0)=0, f(g)=0, f'(0), f'(g))
has a known Gaussian distribution (from the 4-point conditioning).

Can we express Cov(g, P) directly from these formulas?
"""
import numpy as np, sys
from scipy.stats import pearsonr, norm
from scipy.integrate import quad
sys.stdout.reconfigure(line_buffering=True)

def rs(N):
    p = 1.0/np.arange(1,N+1); p /= p.sum()
    w = np.log(np.arange(2,N+2))
    return p, w

def C_and_derivs(tau, p, w):
    """Compute C(tau), C'(tau), C''(tau)"""
    c = np.dot(p, np.cos(w * tau))
    cp = -np.dot(p, w * np.sin(w * tau))
    cpp = -np.dot(p, w**2 * np.cos(w * tau))
    return c, cp, cpp

# ============================================================
# PART 1: STRUCTURE OF THE JOINT DENSITY
# ============================================================
print("="*70)
print("RICE FORMULA EXPLORATION")
print("="*70)

N = 200
p, w = rs(N)
m0 = np.dot(p, w**0)  # = 1
m2 = np.dot(p, w**2)
m4 = np.dot(p, w**4)

print(f"N={N}, m2={m2:.4f}, m4={m4:.4f}")
print(f"Zero intensity lambda_0 = sqrt(m2)/pi = {np.sqrt(m2)/np.pi:.4f}")
print(f"Mean gap g_bar = pi/sqrt(m2) = {np.pi/np.sqrt(m2):.5f}")

g_bar = np.pi / np.sqrt(m2)

print(f"""
RICE'S FORMULA FOR THE GAP DENSITY:

  For a unit-variance stationary GP with correlation C(tau):

  rho(g) = lambda_0^2 * E[|Y_1| * |Y_2| * 1_{{no zero in (0,g)}} | X_1=X_2=0]

  where (X_1, Y_1, X_2, Y_2) = (f(0), f'(0), f(g), f'(g))

  The conditioning X_1 = X_2 = 0 gives a Gaussian for (Y_1, Y_2)
  with covariance matrix:

  Sigma_YY|X = Sigma_YY - Sigma_YX Sigma_XX^{{-1}} Sigma_XY

THE KEY OBSERVATION:

  Cov(g, P) = int_0^inf g * E[|f(g/2)| * joint_stuff] dg
            - mu_g * mu_P

  If we can express this as a single integral that's manifestly positive,
  we're done.

  ALTERNATIVE: Use the identity Cov(g, P) = Cov(g, h(g)) and express
  h(g) directly from the excursion density, then show Cov > 0.
""")


# ============================================================
# PART 2: CONDITIONAL COVARIANCE STRUCTURE
# ============================================================
print("="*70)
print("PART 2: CONDITIONAL COVARIANCE MATRIX")
print("="*70)

# For the 4-point system (f(0), f(g/2), f(g), f'(0), f'(g)):
# We need the covariance matrix of f(g/2), f'(0), f'(g) given f(0)=f(g)=0

# The full 5x5 covariance matrix:
# Variables: X1=f(0), X2=f(g/2), X3=f(g), Y1=f'(0), Y2=f'(g)
# We condition on X1 = X3 = 0

for g_ratio in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
    g = g_ratio * g_bar

    C00, C00p, C00pp = C_and_derivs(0, p, w)
    Cg, Cgp, Cgpp = C_and_derivs(g, p, w)
    Cg2, Cg2p, Cg2pp = C_and_derivs(g/2, p, w)

    # C(0) = 1, C'(0) = 0, C''(0) = -m2
    # Covariance matrix of (f(0), f(g/2), f(g)):
    Sigma_X = np.array([
        [1,    Cg2,  Cg],
        [Cg2,  1,    Cg2],
        [Cg,   Cg2,  1]
    ])

    # Cov(f(0), f'(0)) = C'(0) = 0
    # Cov(f(g), f'(0)) = -C'(g)  [d/d(0) C(g-0) = -C'(g)]
    # Actually: Cov(f(s), f'(t)) = d/dt C(s-t) = -C'(s-t) * (-1) = C'(t-s)
    # Wait, let me be careful.
    # f(s) and f'(t): Cov = d/dt Cov(f(s), f(t)) = d/dt C(s-t) = -C'(s-t)
    # So Cov(f(0), f'(0)) = -C'(0) = 0  [since C'(0) = 0 for even C]
    # Cov(f(0), f'(g)) = -C'(0-g) = -C'(-g) = C'(g)  [C' odd]
    # Cov(f(g), f'(0)) = -C'(g-0) = -C'(g)
    # Cov(f(g), f'(g)) = -C'(0) = 0
    # Cov(f(g/2), f'(0)) = -C'(g/2)
    # Cov(f(g/2), f'(g)) = -C'(g/2 - g) = -C'(-g/2) = C'(g/2) = -Cg2p  ???

    # Let me recompute. C'(tau) = -sum p_n w_n sin(w_n tau).
    # C'(g/2) = -sum p_n w_n sin(w_n g/2)
    # Cov(f(s), f'(t)) = d/dt C(s-t) = -C'(s-t)
    # Cov(f(g/2), f'(g)) = -C'(g/2 - g) = -C'(-g/2) = C'(g/2)
    #   since C' is odd: C'(-x) = -C'(x)

    Cg2_prime = -np.dot(p, w * np.sin(w * g/2))  # C'(g/2)
    Cg_prime = -np.dot(p, w * np.sin(w * g))      # C'(g)

    # Bridge variance V(g) = 1 - 2*Cg2^2/(1+Cg)
    V = 1 - 2*Cg2**2 / (1 + Cg)

    # Conditional variance of f(g/2) given f(0)=f(g)=0
    # This is V(g) from the bridge formula
    # Conditional mean is 0 (by symmetry)

    # Now: what about including f'(0) and f'(g) in the conditioning?
    # The excursion conditioning effectively conditions on the SIGNS of f'(0) and f'(g)
    # and on "no zero in (0,g)"

    # For a positive excursion (f > 0 on (0,g)):
    # f'(0) > 0 (function goes positive) and f'(g) < 0 (function comes back to zero)

    # The conditional distribution of f(g/2) given f(0)=f(g)=0, f'(0)>0, f'(g)<0
    # is different from the bridge distribution

    # Covariance of f(g/2) with f'(0) given f(0)=f(g)=0:
    # Cov(f(g/2), f'(0) | f(0)=0, f(g)=0) = ...
    # Using Schur complement on the joint (f(g/2), f'(0)) given (f(0), f(g)) = (0,0)

    # Full cov of (f(g/2), f'(0)):
    # Var(f(g/2)) = 1
    # Var(f'(0)) = m2
    # Cov(f(g/2), f'(0)) = -C'(g/2) = -Cg2_prime

    # Regression of (f(g/2), f'(0)) on (f(0), f(g)):
    # [f(g/2)]     [Cg2  Cg2]   [1  Cg]^{-1}   [f(0)]
    # [f'(0) ] = [  0  -Cg' ] * [Cg  1]       * [f(g) ]

    # For f(0)=f(g)=0, the conditional mean is zero for both.
    # Conditional covariance:

    S_XX = np.array([[1, Cg], [Cg, 1]])  # cov of f(0), f(g)
    S_XX_inv = np.linalg.inv(S_XX)

    # Cross-covariance of (f(g/2), f'(0)) with (f(0), f(g)):
    S_ZX = np.array([
        [Cg2, Cg2],     # f(g/2) with f(0), f(g)
        [0, -Cg_prime]   # f'(0) with f(0), f(g)
    ])

    # Covariance of (f(g/2), f'(0)):
    S_ZZ = np.array([
        [1, -Cg2_prime],
        [-Cg2_prime, m2]
    ])

    # Conditional covariance
    S_cond = S_ZZ - S_ZX @ S_XX_inv @ S_ZX.T

    cond_var_mid = S_cond[0,0]  # should match V(g)
    cond_cov_mid_fp = S_cond[0,1]
    cond_var_fp = S_cond[1,1]

    print(f"\n  g/g_bar = {g_ratio:.1f}:")
    print(f"    V(g) from bridge = {V:.6f}")
    print(f"    V(g) from Schur  = {cond_var_mid:.6f}  {'MATCH' if abs(V-cond_var_mid)<1e-6 else 'MISMATCH'}")
    print(f"    Cov(f(mid), f'(0) | bridge) = {cond_cov_mid_fp:+.6f}")
    print(f"    Var(f'(0) | bridge) = {cond_var_fp:.6f}")

    # The key: if Cov(f(mid), f'(0) | bridge) > 0, then conditioning on f'(0) > 0
    # (positive excursion start) INCREASES the conditional mean of f(mid)
    if cond_cov_mid_fp > 0:
        direction = "f'(0)>0 INCREASES E[f(mid)]"
    else:
        direction = "f'(0)>0 DECREASES E[f(mid)]"
    print(f"    => {direction}")

    # What about the effect of conditioning on |f'(0)| (derivative magnitude)?
    # Larger |f'(0)| -> steeper zero crossing -> larger initial slope
    # Does this correlate with f(mid)?
    # Sign of cond_cov tells us:
    # If positive: steeper upward crossing -> larger midpoint value
    # If negative: steeper crossing -> SMALLER midpoint (overshoots and comes back?)

    # For the excursion: f'(0) > 0, so E[f(mid) | f'(0) = y, bridge] = cond_cov * y / cond_var_fp
    # For large y: E[f(mid)] is larger (if cond_cov > 0)
    # But large |f'(0)| might also correlate with SHORTER gaps...

    # The gap g and f'(0) are related: by Rice's formula,
    # the gap density involves |f'(0)| * |f'(g)|.
    # Gaps with larger |f'(0)| tend to be SHORTER (faster oscillation).

    # So there's a trade-off:
    # Small gap -> large |f'(0)| -> larger f(mid) (if cov > 0)
    # But small gap -> small bridge variance -> smaller f(mid)
    # These two effects compete!


# ============================================================
# PART 3: DECOMPOSE Cov(g, P) VIA DERIVATIVES
# ============================================================
print(f"\n{'='*70}")
print("PART 3: Cov(g, P) VIA DERIVATIVE CONDITIONING")
print("="*70)

print("""
  IDEA: Express Cov(g, P) as an integral over f'(0):

  Cov(g, P) = E[g * P] - E[g] * E[P]

  By Rice: E[g * |f(g/2)| * |f'(0)| * |f'(g)| * 1_{excursion}]
           is an integral over (g, f'(0), f'(g)) with known Gaussian weights.

  The key question: can we factor this integral into a manifestly positive form?

  If f'(0) and g are positively correlated in the excursion,
  AND f'(0) and P are positively correlated given the bridge,
  then Cov(g, P) > 0 by a transitive-correlation argument.

  But the sign of Corr(f'(0), g | excursion) is not obvious.
""")

# Simulate to check: is Corr(|f'(0)|, g) positive or negative?
print("  Simulating to check Corr(|f'(gamma)|, g)...", flush=True)

rng = np.random.default_rng(42)
omega = w
amp = 1.0/np.sqrt(np.arange(1,N+1))
sigma_N = np.sqrt(np.sum(1.0/np.arange(1,N+1)))
chunk = 20000

all_g, all_fp, all_pk = [], [], []
for trial in range(150):
    phi = rng.uniform(0, 2*np.pi, N)
    npts = int(5000/0.02)
    f = np.empty(npts)
    fp = np.empty(npts)  # derivative
    for s in range(0, npts, chunk):
        e = min(s+chunk, npts)
        tc = np.arange(s,e)*0.02
        cos_vals = np.cos(np.outer(tc, omega)+phi)
        sin_vals = np.sin(np.outer(tc, omega)+phi)
        f[s:e] = cos_vals @ amp
        fp[s:e] = -(sin_vals @ (amp * omega))
    f /= sigma_N
    fp /= sigma_N
    t = np.arange(npts)*0.02
    sc = np.where(f[:-1]*f[1:]<0)[0]
    if len(sc)<20: continue
    zeros = t[sc] - f[sc]*0.02/(f[sc+1]-f[sc])
    # Derivative at zeros (interpolated)
    fp_at_zeros = fp[sc]  # approximate
    gaps = np.diff(zeros)
    fp_left = np.abs(fp_at_zeros[:-1])
    midx = ((zeros[:-1]+zeros[1:])/(2*0.02)).astype(int)
    midx = np.clip(midx, 0, npts-1)
    pks = np.abs(f[midx])
    tr = max(3, int(0.05*len(gaps)))
    all_g.extend(gaps[tr:-tr].tolist())
    all_fp.extend(fp_left[tr:-tr].tolist())
    all_pk.extend(pks[tr:-tr].tolist())

gaps = np.array(all_g)
fp_vals = np.array(all_fp)
peaks = np.array(all_pk)

r_gP = pearsonr(gaps, peaks)[0]
r_gfp = pearsonr(gaps, fp_vals)[0]
r_fpP = pearsonr(fp_vals, peaks)[0]

print(f"\n  N={N}, {len(gaps)} observations:")
print(f"  Corr(g, P)     = {r_gP:+.4f}  [the target]")
print(f"  Corr(g, |f'|)  = {r_gfp:+.4f}  [gap vs derivative at zero]")
print(f"  Corr(|f'|, P)  = {r_fpP:+.4f}  [derivative vs peak]")

print(f"""
  INTERPRETATION:
    Corr(g, |f'|) = {r_gfp:+.4f}: {'POSITIVE' if r_gfp > 0 else 'NEGATIVE'}
      -> Larger gaps have {'larger' if r_gfp > 0 else 'smaller'} derivatives at zeros
    Corr(|f'|, P) = {r_fpP:+.4f}: {'POSITIVE' if r_fpP > 0 else 'NEGATIVE'}
      -> Larger derivatives {'produce' if r_fpP > 0 else 'do not produce'} larger peaks
""")

# The MVT model: P ~ |f'| * g / c
# So P/g ~ |f'| / c, meaning the "peak per unit gap" is proportional to |f'|
# Corr(g, P) = Corr(g, |f'|*g/c) > 0 if |f'| and g are not too negatively correlated

# Partial correlations
from numpy.linalg import solve
# Corr(g, P | |f'|) = partial correlation controlling for |f'|
X = np.column_stack([gaps, peaks, fp_vals])
C_mat = np.corrcoef(X, rowvar=False)
# Partial corr(0,1 | 2) = (C01 - C02*C12) / sqrt((1-C02^2)(1-C12^2))
C01, C02, C12 = C_mat[0,1], C_mat[0,2], C_mat[1,2]
r_gP_partial = (C01 - C02*C12) / np.sqrt((1-C02**2)*(1-C12**2))

print(f"  Partial Corr(g, P | |f'|) = {r_gP_partial:+.4f}")
print(f"    (The correlation of g and P AFTER removing |f'| effect)")

if r_gP_partial > 0:
    print(f"    => g and P are correlated BEYOND the |f'| mechanism")
    print(f"    => The bridge variance effect (V increasing) contributes")
else:
    print(f"    => The g-P correlation is ENTIRELY mediated by |f'|")

# Check: does the MVT model P = |f'|*g/c explain r?
model_P = fp_vals * gaps
r_model = pearsonr(gaps, model_P)[0]
print(f"\n  MVT model: Corr(g, |f'|*g) = {r_model:+.4f}")
print(f"  Actual:    Corr(g, P)       = {r_gP:+.4f}")
print(f"  => MVT model {'captures' if abs(r_model - r_gP) < 0.05 else 'partially explains'} the correlation")


# ============================================================
# PART 4: THE TRANSITIVE CORRELATION ARGUMENT
# ============================================================
print(f"\n{'='*70}")
print("PART 4: CAN WE PROVE Cov(g, P) > 0 VIA f'?")
print("="*70)

print(f"""
  THREE-VARIABLE SYSTEM: (g, |f'(gamma)|, P)

  Corr(g, |f'|) = {r_gfp:+.4f}
  Corr(|f'|, P) = {r_fpP:+.4f}
  Corr(g, P)    = {r_gP:+.4f}

  If |f'| mediates the g-P relationship, then:
    Cov(g, P) = Cov(g, E[P | |f'|, g]) + ...

  The MVT approximation: P ~ |f'| * g / c
  gives Cov(g, P) ~ Cov(g, |f'|*g/c) = (1/c) * Cov(g, |f'|*g)

  Cov(g, |f'|*g) = E[|f'|*g^2] - E[g]*E[|f'|*g]
                  = E[|f'|]*Var(g) + Cov(|f'|, g^2 - E[g]*g)

  The first term E[|f'|]*Var(g) is ALWAYS positive.
  The second term depends on Cov(|f'|, g*(g-E[g])).

  Since g*(g-E[g]) is increasing for g > E[g]/2 (most of the dist.),
  and Corr(|f'|, g) = {r_gfp:+.4f} {'> 0' if r_gfp > 0 else '< 0'}:
""")

# Compute the MVT bound
E_fp = np.mean(fp_vals)
Var_g = np.var(gaps)
term1 = E_fp * Var_g
cov_fp_gsq = np.mean(fp_vals * gaps * (gaps - np.mean(gaps))) - 0  # E[g-mu]=0
term2 = cov_fp_gsq

print(f"  E[|f'|] = {E_fp:.4f}")
print(f"  Var(g)  = {Var_g:.6f}")
print(f"  Term 1 (E[|f'|]*Var(g)) = {term1:+.6f}  [ALWAYS POSITIVE]")
print(f"  Term 2 (Cov(|f'|, g(g-mu))) = {term2:+.6f}")
print(f"  Sum = {term1+term2:+.6f}")
print(f"  Actual Cov(g, |f'|*g) = {np.cov(gaps, fp_vals*gaps)[0,1]:+.6f}")
print(f"  Actual Cov(g, P)      = {np.cov(gaps, peaks)[0,1]:+.6f}")

if term1 + term2 > 0:
    print(f"\n  THE MVT BOUND WORKS: Cov(g, |f'|*g) > 0")
    print(f"  Term 1 / |Term 2| = {abs(term1/term2):.2f}x margin")
else:
    print(f"\n  THE MVT BOUND FAILS for this correlation structure")


print(f"\n{'='*70}")
print("DONE")
print("="*70)
