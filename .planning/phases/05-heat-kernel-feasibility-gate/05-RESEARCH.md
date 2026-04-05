# Phase 5: Heat Kernel Feasibility Gate - Research

**Researched:** 2026-04-04
**Domain:** Computational number theory -- heat kernel trace on SL(2,Z)\H, Maass form spectral sums, Eisenstein continuous spectrum, parameter identification, dual-precision computation
**Confidence:** MEDIUM-HIGH

## Summary

Phase 5 is the numerical feasibility gate for the v2.0 proof strategy: express the Connes barrier B(L) as a heat kernel trace on the modular surface, then exploit structural positivity. The phase must answer three questions: (1) Does the heat kernel trace K(t(L)) match B(L) to proof-grade precision? (2) What is the precise parameter mapping t = t(L)? (3) How large is the Eisenstein continuous spectrum contribution compared to the 0.036 error budget? If the answer to (1) is no, or (3) is "too large to bound," the entire v2.0 approach is dead and we stop early -- that is the purpose of a feasibility gate.

The implementation builds one new module (`heat_kernel.py`) in `src/riemann/analysis/`, adds python-flint 0.8.0 as the sole new dependency, and extends the existing LMFDB client for Maass form data. The existing codebase provides everything else: 500 Maass spectral parameters already cached in `data/maass_forms.json`, the barrier computation from `session41g_uncapped_barrier.py` (to be promoted into the module system), the `validated_computation` P-vs-2P pattern, and the stress-test framework. The dual-precision pattern (mpmath for exploration + python-flint ball arithmetic for certification) is the primary new engineering concern.

**Primary recommendation:** Build the heat kernel trace computation with discrete Maass sum + continuous Eisenstein integral + constant term, run the barrier comparison at 100+ L values with always-dual mpmath/python-flint, and report the parameter mapping and Eisenstein magnitude as the go/no-go diagnostics. This is a numerical investigation, not a proof -- keep the code lean and diagnostic-focused.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Agreement threshold is proof-grade -- whatever is necessary for a rigorous proof, not an arbitrary digit count. The feasibility check must demonstrate that the heat kernel trace and barrier agree well enough to build a proof on.
- **D-02:** No artificial kill threshold on the Eisenstein continuous spectrum contribution. The magnitude is data -- Phase 7 (correction bounds) determines if it's fatal. Don't pre-judge.
- **D-03:** Test at 100+ L values spanning the full range (L ~ 1 to 50+), with density concentrated where the barrier margin is smallest.
- **D-04:** Diagnostic output is BOTH summary table (L, K(t), B(L), digits of agreement, verdict) AND convergence plots (spectral sum convergence, agreement heatmap).
- **D-05:** Pursue analytic derivation AND numerical fitting in parallel for t(L). Derive t(L) from the Lorentzian test function structure; simultaneously fit numerically by optimizing t for each L. Cross-validate: analytic formula must match numerical fit.
- **D-06:** Claude judges whether t(L) mapping complexity is acceptable for downstream proof assembly. Simple expressions preferred, but special functions acceptable if well-defined and computable.
- **D-09:** Always-dual computation: every computation runs in both mpmath and python-flint, flagging disagreement.
- **D-10:** Claude's discretion on the integration pattern for dual precision (wrapper functions vs backend flag vs other).

### Claude's Discretion
- Maass eigenvalue data source and count (D-07, D-08)
- Dual-precision integration pattern (D-10)
- t(L) mapping complexity acceptance (D-06)
- Heat kernel spectral sum truncation strategy
- Continuous spectrum numerical integration method
- Module organization within `src/riemann/analysis/`
- All performance optimization decisions

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| HEAT-01 | Compute heat kernel trace Tr(e^{-t*Delta}) on SL(2,Z)\H including discrete Maass sum and continuous Eisenstein spectrum | Standard spectral decomposition formula; 500 Maass eigenvalues already in data/maass_forms.json; Eisenstein integral via scattering matrix phi(s); mpmath.quad for integration |
| HEAT-02 | Identify parameter mapping t = t(L) validated by 6+ digits agreement | Four candidate mappings documented; parallel analytic + numerical fitting; barrier values from session41g code |
| HEAT-03 | Compute discrete spectral sum over Maass eigenvalues with configurable truncation and convergence diagnostics | LMFDB data (500 eigenvalues, r up to ~91); Weyl's law tail bound; validated_computation P-vs-2P pattern |
| HEAT-04 | Dual-precision: mpmath exploratory + python-flint ball arithmetic for rigorous bounds | python-flint 0.8.0 has Python 3.13 wheels on PyPI; acb type provides ball arithmetic; arb type for real bounds; not yet installed -- must add to pyproject.toml |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| mpmath | 1.3.0 (installed) | Arbitrary-precision heat kernel computation: besselk, quad, nsum, digamma, zeta, log | Already the project workhorse; handles all special functions needed for heat kernel orbital integrals and Eisenstein integral |
| python-flint | 0.8.0 (NEW) | Ball arithmetic for certified computation: arb for real intervals, acb for complex intervals | Wraps FLINT 3.3.1/Arb; returns machine-certified error bounds; Python 3.13 Windows x86-64 wheel on PyPI; required for D-09 dual precision |
| numpy | 2.4.3 (installed) | Vectorized barrier computation, parameter sweep arrays | Already used by session41g barrier code and throughout the project |
| scipy | 1.17.1 (installed) | scipy.special.kv for machine-precision Bessel K in bulk sweeps; scipy.optimize for parameter fitting | Fast C-level special functions for exploration before mpmath high-precision |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy.optimize | (in scipy) | minimize_scalar, curve_fit for numerical t(L) parameter fitting | D-05 numerical fitting track |
| requests | 2.32.0 (installed) | LMFDB API queries if more Maass eigenvalues needed | Only if 500 eigenvalues prove insufficient for convergence |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| python-flint ball arithmetic | mpmath.iv (interval arithmetic) | mpmath.iv exists but is less developed than Arb; python-flint is the gold standard for certified number theory |
| Manual Eisenstein integral | Precomputed tables | No tables exist for the specific scattering phase integral we need; must compute |
| LMFDB for all eigenvalues | Hejhal's algorithm | Hejhal's algorithm is complex to implement correctly; LMFDB already has 500+ eigenvalues at sufficient precision |

**Installation:**
```bash
pip install python-flint>=0.7.0
# Then add to pyproject.toml dependencies
```

**Version verification:** python-flint 0.8.0 confirmed available on PyPI with Python 3.13 wheels. mpmath 1.3.0 already installed. Python 3.13.7 is the runtime.

## Architecture Patterns

### Recommended Project Structure
```
src/riemann/analysis/
    heat_kernel.py          # NEW: Heat kernel trace on SL(2,Z)\H (HEAT-01, HEAT-03)
    (existing) modular_forms.py   # Eisenstein series E_k, q-series
    (existing) trace_formula.py   # Weil explicit formula
    (existing) lmfdb_client.py    # REST API + SQLite cache

src/riemann/engine/
    (existing) precision.py       # validated_computation, precision_scope
    (existing) validation.py      # stress_test framework
    dual_precision.py       # NEW: mpmath + python-flint dual backend (HEAT-04)

data/
    (existing) maass_forms.json   # 500 spectral parameters from LMFDB
```

### Pattern 1: Dual-Precision Backend
**What:** Every computation runs in both mpmath and python-flint, comparing results and flagging disagreement. This catches subtle precision bugs and provides certified bounds alongside exploratory values.
**When to use:** All Phase 5 computations (D-09 locked decision).
**Example:**
```python
from flint import arb, acb

def dual_compute(func_mpmath, func_flint, *, dps=50, label=""):
    """Run computation in both backends, compare, return both results.
    
    Args:
        func_mpmath: Callable returning mpmath result at current dps.
        func_flint: Callable returning flint arb/acb result at current prec.
        dps: Decimal precision for mpmath.
        label: Human-readable label for diagnostics.
    
    Returns:
        DualResult with mpmath_value, flint_value, agreement_digits, flag.
    """
    # mpmath computation at dps
    with mpmath.workdps(dps + 5):
        mp_val = func_mpmath()
    
    # python-flint computation at equivalent bit precision
    prec = int(dps * 3.32193) + 20  # dps -> bits + guard
    # flint precision set via ctx or workprec context
    fl_val = func_flint(prec)
    
    # Compare: extract midpoint from flint ball, compare to mpmath
    fl_mid = float(fl_val.mid()) if hasattr(fl_val, 'mid') else float(fl_val)
    mp_float = float(mp_val)
    
    if mp_float != 0:
        agreement = -math.log10(abs(fl_mid - mp_float) / abs(mp_float) + 1e-300)
    else:
        agreement = -math.log10(abs(fl_mid - mp_float) + 1e-300)
    
    return DualResult(
        mpmath_value=mp_val,
        flint_value=fl_val,
        agreement_digits=agreement,
        label=label,
    )
```

### Pattern 2: Validated Spectral Sum with Convergence Diagnostics
**What:** Heat kernel discrete sum with tail bound tracking and P-vs-2P validation.
**When to use:** All spectral sums (HEAT-03).
**Example:**
```python
def maass_spectral_sum(t, n_terms=None, *, dps=50):
    """Sum over Maass cusp form contributions to heat kernel trace.
    
    K_discrete(t) = sum_{j=1}^{N} exp(-(1/4 + r_j^2) * t)
    
    Returns value and convergence diagnostics.
    """
    spectral_params = load_maass_spectral_params()
    if n_terms is None:
        # Auto-select: include all terms where exp(-lambda_j * t) > 10^{-dps}
        lambda_threshold = dps * math.log(10) / t
        n_terms = sum(1 for r in spectral_params if 0.25 + r**2 < lambda_threshold)
        n_terms = max(n_terms, 10)  # minimum 10 terms
    
    n_terms = min(n_terms, len(spectral_params))
    
    def _compute():
        total = mpmath.mpf(0)
        terms = []
        for j in range(n_terms):
            r_j = mpmath.mpf(str(spectral_params[j]))
            lambda_j = mpmath.mpf('0.25') + r_j**2
            term = mpmath.exp(-lambda_j * mpmath.mpf(str(t)))
            total += term
            terms.append(float(term))
        
        tail_variation = abs(terms[-1]) if terms else 0
        return total, {
            'n_terms': n_terms,
            'tail_variation': tail_variation,
            'last_term_magnitude': float(abs(terms[-1])) if terms else 0,
            'convergence_rate': float(abs(terms[-1] / terms[-2])) if len(terms) >= 2 else 0,
        }
    
    return validated_computation(_compute, dps=dps, algorithm="maass_spectral_sum")
```

### Pattern 3: Barrier Comparison with Diagnostic Table
**What:** Structured comparison between K(t(L)) and B(L) at each L value, producing both data and formatted output.
**When to use:** HEAT-02 parameter identification and validation.
**Example:**
```python
@dataclass
class BarrierComparison:
    """Single comparison point between heat kernel trace and barrier."""
    L: float
    t: float                    # Heat kernel time parameter
    heat_kernel_value: float    # K(t)
    barrier_value: float        # B(L)
    discrete_sum: float         # Maass cusp form sum
    eisenstein_contrib: float   # Continuous spectrum integral
    constant_term: float        # 1/(4*pi*t) or area/t term
    digits_of_agreement: float  # -log10(|K-B|/|B|)
    n_maass_terms: int
    dual_validated: bool        # mpmath/flint agreement
```

### Anti-Patterns to Avoid
- **Ignoring Eisenstein continuous spectrum:** Every heat kernel trace on SL(2,Z)\H MUST include the continuous spectrum integral. A sum over Maass forms alone is incomplete and will not match the barrier.
- **Assuming B(L) IS the heat kernel trace:** B(L) is "morally" a heat kernel trace (Session 47). The exact relationship includes corrections. Do not claim identity before establishing it numerically.
- **Unvalidated truncation:** The 0.036 margin means truncation errors matter. Always compute tail bounds using Weyl's law asymptotics.
- **Mixing precision regimes:** D-09 requires both mpmath and python-flint for every computation. Never compute one quantity at 50 digits and its comparison target at 15 digits.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Maass eigenvalues | Hejhal's algorithm | LMFDB data in data/maass_forms.json (500 values, r up to ~91) | Hejhal's algorithm is hundreds of lines of careful numerics; LMFDB has tabulated values to 15+ digits |
| Arbitrary-precision zeta | Custom zeta implementation | mpmath.zeta (already used throughout) | mpmath.zeta handles complex arguments, high precision, and the critical strip reliably |
| Bessel K function | Custom Bessel implementation | mpmath.besselk (arbitrary precision) or scipy.special.kv (machine precision) | Both are mature, validated implementations |
| Digamma function | Series expansion | mpmath.digamma | Needed for phi'/phi in Eisenstein integral; mpmath handles complex arguments |
| Ball arithmetic | Custom interval arithmetic | python-flint arb/acb types | FLINT/Arb is the gold standard; python-flint wraps it cleanly |
| Parameter optimization | Grid search | scipy.optimize.minimize_scalar | D-05 numerical fitting; minimize_scalar handles 1D optimization robustly |
| Barrier B(L) computation | Rewrite from scratch | Promote session41g_uncapped_barrier.py into module system | Existing code is validated at 800+ points; extend, don't rewrite |

**Key insight:** The barrier computation (session41g) and Maass spectral data (LMFDB) are already validated. The new code is the bridge between them: the heat kernel trace computation, the Eisenstein integral, and the parameter mapping. Keep the new code focused on these three novel components.

## Mathematical Formulas (Implementation Reference)

### Heat Kernel Trace on SL(2,Z)\H

The trace of the heat kernel on the modular surface decomposes into three contributions:

```
K(t) = K_constant(t) + K_discrete(t) + K_continuous(t)
```

**Constant term (from the constant eigenfunction with lambda=0):**
```
K_constant(t) = vol(SL(2,Z)\H) / (4*pi*t) = (pi/3) / (4*pi*t) = 1/(12*t)
```
Note: vol(SL(2,Z)\H) = pi/3 in standard normalization.

**Discrete Maass cusp form sum:**
```
K_discrete(t) = sum_{j=1}^{N} exp(-(1/4 + r_j^2) * t)
```
where r_j are spectral parameters from LMFDB (r_1 ~ 9.534, ..., r_500 ~ 90.955).

**Continuous Eisenstein spectrum:**
```
K_continuous(t) = (1/(4*pi)) * integral_0^infinity exp(-(1/4 + r^2) * t) * psi_scatt(r) dr
```
where `psi_scatt(r) = -(phi'/phi)(1/2 + ir)` is the logarithmic derivative of the scattering determinant.

### Scattering Matrix for SL(2,Z)

The scattering determinant for SL(2,Z) (one cusp) is:
```
phi(s) = sqrt(pi) * Gamma(s - 1/2) / Gamma(s) * zeta(2s - 1) / zeta(2s)
```

Its logarithmic derivative:
```
-(phi'/phi)(s) = -digamma(s - 1/2) + digamma(s) - 2*zeta'(2s-1)/zeta(2s-1) + 2*zeta'(2s)/zeta(2s)
```

At s = 1/2 + ir:
```
-(phi'/phi)(1/2 + ir) = -digamma(ir) + digamma(1/2 + ir) - 2*(zeta'/zeta)(2ir) + 2*(zeta'/zeta)(1 + 2ir)
```

**Circularity note:** The terms `zeta'/zeta` involve the zeros of zeta directly. This is the continuous spectrum trap (Pitfall 3). Phase 5 computes the magnitude of this contribution; Phase 7 determines if it can be bounded non-circularly.

### Parameter Mapping Candidates

Four candidates for t = t(L), from Session 47 analysis:

| Candidate | Formula | Motivation |
|-----------|---------|------------|
| Direct | t = L | Simplest; barrier parameter L maps directly to heat time t |
| Selberg-normalized | t = L / (2*pi) | Selberg convention normalization |
| Quadratic | t = L^2 / (4*pi^2) | Lorentzian-to-Gaussian mapping |
| Implicit | t such that K(t) best fits B(L) | Numerical optimization per L point |

Phase 5 must determine which (if any) works. The analytic derivation track (D-05) starts from the Lorentzian structure: the barrier test function is `w_hat(n) = n / (L^2 + 16*pi^2*n^2)`, and the heat kernel test function is `h(r) = exp(-r^2*t)`. Matching these in the Selberg trace formula spectral side suggests a specific relationship.

### Convergence Estimates

Using Weyl's law for SL(2,Z)\H: N(lambda) ~ (pi/3)/(4*pi) * lambda = lambda/12 for the counting function of eigenvalues below lambda.

For the heat kernel sum truncated at N terms:
```
tail = sum_{j > N} exp(-lambda_j * t) <= integral_{lambda_N}^{inf} exp(-lambda * t) * (1/12) d(lambda)
     = exp(-lambda_N * t) / (12 * t)
```

With 500 eigenvalues (lambda_500 ~ 0.25 + 90.96^2 ~ 8274), the tail for t >= 0.01:
```
tail <= exp(-8274 * 0.01) / (12 * 0.01) ~ exp(-82.7) / 0.12 ~ 10^{-36}
```

This means 500 eigenvalues provide ~36 digits of convergence for t >= 0.01, far exceeding the 50-digit default precision. For smaller t (t ~ 0.001), more eigenvalues would be needed, but the operating range L ~ 1 to 50 likely maps to t values well above 0.01.

### Eisenstein Integral Computation

The continuous spectrum integral requires evaluating:
```
I(t) = (1/(4*pi)) * integral_0^{R} exp(-(1/4 + r^2) * t) * psi_scatt(r) dr
```

where R is a cutoff chosen so that the tail contribution is negligible. The integrand decays as exp(-r^2 * t), so for t >= 0.01, R = 100 gives tail ~ exp(-100^2 * 0.01) = exp(-100) ~ 10^{-44}.

The scattering phase psi_scatt(r) involves digamma and zeta'/zeta at complex arguments. Each evaluation requires:
- `mpmath.digamma(ir)` and `mpmath.digamma(1/2 + ir)` -- well-behaved
- `mpmath.zeta(2*ir, derivative=1) / mpmath.zeta(2*ir)` -- has poles at zeros of zeta(2s); for real r these are on the critical line of zeta(2s), which has zeros at s = rho/2 where rho are zeta zeros
- `mpmath.zeta(1 + 2*ir, derivative=1) / mpmath.zeta(1 + 2*ir)` -- well-behaved for r > 0 (no zeros of zeta on Re(s) > 1)

**Numerical integration method:** Use `mpmath.quad` with adaptive subdivision. The integrand oscillates due to the digamma/zeta terms but decays exponentially. Gauss-Legendre quadrature on [0, R] with R ~ 100-200 should suffice. Monitor quadrature error estimate.

## Common Pitfalls

### Pitfall 1: Eisenstein Integral Numerical Instability
**What goes wrong:** The scattering phase psi_scatt(r) involves zeta'/zeta(2ir), which has poles near values of r where 2ir is close to a zero of zeta(2s). For r near Im(rho)/2 where rho is a zeta zero, the integrand can spike.
**Why it happens:** The zeta function has zeros on (or near) the critical line; zeta'/zeta has poles there. The Eisenstein integrand inherits these singularities.
**How to avoid:** Use adaptive quadrature (mpmath.quad handles this); increase precision near problematic r values; compute the integral piecewise, subdividing around known pole locations (from the zeta zero database).
**Warning signs:** Quadrature returning NaN or Inf; python-flint ball radius exploding; digits of agreement dropping precipitously at specific L values.

### Pitfall 2: Parameter Mapping Degeneracy
**What goes wrong:** Multiple t(L) candidates give "reasonable" agreement (4-5 digits) but none gives proof-grade agreement. The mapping may not be a simple closed-form expression.
**Why it happens:** The barrier test function (Lorentzian) and the heat kernel (Gaussian) are different functions. The "moral" equivalence from Session 47 may be approximate, not exact.
**How to avoid:** Start with the implicit mapping (optimize t at each L to maximize agreement). If this gives 10+ digits, look for the pattern. If it only gives 4-5 digits, the heat kernel interpretation may be approximate -- this is data for the go/no-go decision.
**Warning signs:** Optimal t(L) having no clean functional form; digits of agreement varying wildly across L values; different L ranges requiring different formulas.

### Pitfall 3: python-flint Precision Context Confusion
**What goes wrong:** python-flint uses bit precision (prec), not decimal digits (dps). Confusing the two leads to insufficient precision or wasted computation.
**Why it happens:** mpmath uses dps (decimal), python-flint uses prec (bits). 50 dps ~ 166 bits. Setting prec=50 in flint gives only ~15 decimal digits.
**How to avoid:** Always convert: `prec = int(dps * 3.32193) + 20` (adding guard bits). The dual_precision wrapper must handle this conversion internally.
**Warning signs:** python-flint results having much lower precision than mpmath; ball radii being unexpectedly large; agreement digits being low despite high dps setting.

### Pitfall 4: Barrier Code Integration
**What goes wrong:** The barrier computation (session41g) uses numpy at machine precision. Comparing against mpmath 50-digit heat kernel values loses 35 digits of information.
**Why it happens:** session41g was designed for exploration, not proof-grade comparison. It uses float64 throughout.
**How to avoid:** Create an mpmath-precision version of the barrier computation for the comparison. The barrier formula (W02 - Mp) can be evaluated at arbitrary precision using mpmath. Keep the numpy version for bulk sweeps; use the mpmath version for the proof-grade comparison at each L.
**Warning signs:** Agreement capped at ~15 digits regardless of heat kernel precision; comparison table showing identical agreement_digits at all L values.

### Pitfall 5: Forgetting the Constant Term
**What goes wrong:** The heat kernel trace includes a constant term from the trivial eigenfunction (constant function with lambda=0). This is the 1/(12*t) term for SL(2,Z)\H. Omitting it gives systematically wrong values.
**Why it happens:** The constant function is not a "Maass cusp form" (it has eigenvalue 0, not 1/4+r^2). It is easy to forget when writing the spectral sum.
**How to avoid:** The heat kernel trace formula explicitly separates: constant term + cusp form sum + Eisenstein integral. Always verify that the constant term is present by checking K(t) as t approaches 0 (should diverge as 1/(12*t)).
**Warning signs:** Heat kernel trace being too small by a 1/(12*t) offset; comparison table showing systematic bias.

## Code Examples

### Loading Maass Spectral Parameters
```python
# Source: data/maass_forms.json (500 eigenvalues from LMFDB)
import json
from pathlib import Path

def load_maass_spectral_params(data_file=None):
    """Load Maass cusp form spectral parameters from LMFDB cache.
    
    Returns list of r_j values where lambda_j = 1/4 + r_j^2.
    """
    if data_file is None:
        data_file = Path(__file__).parent.parent.parent.parent / "data" / "maass_forms.json"
    
    with open(data_file) as f:
        data = json.load(f)
    
    return [entry["r"] for entry in data["spectral_parameters"]]
```

### Scattering Phase Evaluation
```python
# Source: Marklof "Selberg's Trace Formula: An Introduction"
# phi(s) = sqrt(pi) * Gamma(s-1/2) / Gamma(s) * zeta(2s-1) / zeta(2s)

def scattering_phase(r, *, dps=50):
    """Compute -(phi'/phi)(1/2 + ir) for the SL(2,Z) scattering matrix.
    
    This is the weight function in the continuous spectrum integral.
    
    Args:
        r: Real spectral parameter (r >= 0).
        dps: Decimal precision.
    
    Returns:
        Real value of -(phi'/phi)(1/2 + ir).
    """
    with mpmath.workdps(dps + 10):
        s = mpmath.mpc('0.5', str(r))  # s = 1/2 + ir
        
        # Digamma terms
        psi_term = -mpmath.digamma(s - mpmath.mpf('0.5')) + mpmath.digamma(s)
        
        # Zeta logarithmic derivative terms
        # -(zeta'/zeta)(w) = -zeta(w, derivative=1) / zeta(w)
        # But mpmath uses: zetaderiv = mpmath.zeta(s, derivative=1) = zeta'(s)
        w1 = 2 * s - 1  # = 2ir
        w2 = 2 * s       # = 1 + 2ir
        
        zeta_term1 = -2 * mpmath.zeta(w1, derivative=1) / mpmath.zeta(w1)
        zeta_term2 = 2 * mpmath.zeta(w2, derivative=1) / mpmath.zeta(w2)
        
        result = psi_term + zeta_term1 + zeta_term2
        return mpmath.re(result)  # Should be real for real r
```

### python-flint Ball Arithmetic Pattern
```python
# Source: python-flint 0.8.0 docs, FLINT acb_modular docs
from flint import arb, acb, ctx as flint_ctx

def heat_kernel_term_flint(r_j, t, *, prec=200):
    """Compute single Maass term exp(-(1/4 + r_j^2) * t) with ball arithmetic.
    
    Returns arb ball containing the true value.
    """
    # Set working precision
    old_prec = flint_ctx.prec
    flint_ctx.prec = prec
    try:
        r = arb(str(r_j))
        t_arb = arb(str(t))
        quarter = arb(1) / arb(4)
        lambda_j = quarter + r * r
        result = (-lambda_j * t_arb).exp()
        return result
    finally:
        flint_ctx.prec = old_prec
```

### Eisenstein Continuous Spectrum Integral
```python
def eisenstein_continuous_contribution(t, *, dps=50, R_cutoff=200):
    """Compute the continuous spectrum contribution to the heat kernel trace.
    
    K_continuous(t) = (1/(4*pi)) * integral_0^R exp(-(1/4+r^2)*t) * psi_scatt(r) dr
    
    Args:
        t: Heat kernel time parameter.
        dps: Decimal precision.
        R_cutoff: Upper integration limit (tail is exp(-R^2*t)).
    
    Returns:
        Tuple of (value, tail_bound, diagnostics_dict).
    """
    with mpmath.workdps(dps + 10):
        t_mp = mpmath.mpf(str(t))
        
        def integrand(r):
            exp_factor = mpmath.exp(-(mpmath.mpf('0.25') + r**2) * t_mp)
            phase = scattering_phase(float(r), dps=dps)
            return exp_factor * mpmath.mpf(str(phase))
        
        # Adaptive quadrature
        integral_val = mpmath.quad(integrand, [0, R_cutoff], error=True)
        
        # Normalize
        result = integral_val[0] / (4 * mpmath.pi)
        quad_error = integral_val[1] / (4 * mpmath.pi) if len(integral_val) > 1 else None
        
        # Tail bound: |tail| <= integral_R^inf exp(-(1/4+r^2)*t) * C dr
        # where C is max|psi_scatt| over [R, inf) (bounded by O(log(R)))
        tail_bound = mpmath.exp(-mpmath.mpf(str(R_cutoff))**2 * t_mp) / (2 * t_mp)
        
        return result, tail_bound, {
            'R_cutoff': R_cutoff,
            'quad_error': quad_error,
            'tail_bound': float(tail_bound),
        }
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Direct analytic proof of B(L) > 0 | Heat kernel structural positivity | Session 47, 2026 | Avoids the circularity trap that killed 40+ approaches |
| Separate mpmath / interval packages | python-flint unified ball arithmetic | python-flint 0.8.0, 2025 | Single library for both exploration and certification |
| Hardcoded 50 Maass eigenvalues | LMFDB cache of 500 eigenvalues | Phase 3 LMFDB integration | Sufficient eigenvalues for 36+ digits convergence |
| numpy float64 barrier computation | mpmath arbitrary-precision barrier | Phase 5 (to be built) | Enables proof-grade comparison |

**Deprecated/outdated:**
- Arb as a separate library: merged into FLINT 3.x; use python-flint which wraps FLINT 3.3.1
- mpmath.iv for interval arithmetic: functional but less developed than Arb; use python-flint arb type instead

## Open Questions

1. **Does the heat kernel trace actually match the barrier?**
   - What we know: Session 47 established a "moral" connection; the Lorentzian test function resembles a heat kernel at imaginary time; numerical evidence is partial.
   - What's unclear: The exact level of agreement (4 digits? 10 digits? exact identity?). This IS the central question Phase 5 answers.
   - Recommendation: Implement and compute. The answer determines the entire v2.0 trajectory.

2. **What is the precise parameter mapping t(L)?**
   - What we know: Four candidate formulas exist. None confirmed.
   - What's unclear: Whether a simple closed-form mapping exists, or the connection is mediated by the Selberg trace formula in a non-elementary way.
   - Recommendation: Start with numerical optimization (implicit t for each L), then look for patterns. D-05 mandates parallel analytic + numerical tracks.

3. **How large is the Eisenstein contribution?**
   - What we know: The scattering phase involves zeta'/zeta, which has poles at zeta zeros. The contribution has no definite sign.
   - What's unclear: Whether its magnitude is within the 0.036 error budget at all L values.
   - Recommendation: Compute numerically at every L value. D-02 says "no artificial kill threshold" -- report the magnitude as data.

4. **python-flint API for zeta'/zeta computation**
   - What we know: python-flint acb has `.zeta(s)` but the status of `.zeta(s, derivative=1)` needs verification.
   - What's unclear: Whether the logarithmic derivative of zeta can be computed directly via python-flint, or requires manual implementation using ball arithmetic.
   - Recommendation: Check python-flint docs/code at implementation time. Fallback: compute zeta'/zeta as `(zeta(s+h) - zeta(s-h))/(2*h)` with ball arithmetic tracking the approximation error.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | All | Yes | 3.13.7 | -- |
| mpmath | HEAT-01, HEAT-02, HEAT-03 | Yes | 1.3.0 | -- |
| python-flint | HEAT-04 | No (not installed) | 0.8.0 on PyPI | Must install; no fallback for ball arithmetic |
| numpy | Barrier computation | Yes | 2.4.3 | -- |
| scipy | Parameter fitting | Yes | 1.17.1 | -- |
| LMFDB API | Additional Maass data | Yes (internet) | REST API | data/maass_forms.json has 500 eigenvalues already |
| pytest | Testing | Yes | via dev deps | -- |

**Missing dependencies with no fallback:**
- python-flint: MUST be installed before Phase 5 execution. Add to pyproject.toml and `pip install python-flint>=0.7.0`.

**Missing dependencies with fallback:**
- None; all other dependencies are present.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2+ |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `pytest tests/test_analysis/test_heat_kernel.py -x` |
| Full suite command | `pytest tests/ -x --timeout=120` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| HEAT-01 | Heat kernel trace = constant + discrete + continuous | unit | `pytest tests/test_analysis/test_heat_kernel.py::TestHeatKernelTrace -x` | No -- Wave 0 |
| HEAT-01 | Constant term = 1/(12*t) | unit | `pytest tests/test_analysis/test_heat_kernel.py::TestConstantTerm -x` | No -- Wave 0 |
| HEAT-01 | Maass spectral sum convergence | unit | `pytest tests/test_analysis/test_heat_kernel.py::TestMaassSum -x` | No -- Wave 0 |
| HEAT-01 | Eisenstein integral computation | unit | `pytest tests/test_analysis/test_heat_kernel.py::TestEisensteinIntegral -x` | No -- Wave 0 |
| HEAT-02 | Parameter mapping t(L) gives 6+ digit agreement | integration | `pytest tests/test_analysis/test_heat_kernel.py::TestParameterMapping -x` | No -- Wave 0 |
| HEAT-03 | Configurable truncation with convergence diagnostics | unit | `pytest tests/test_analysis/test_heat_kernel.py::TestConvergenceDiagnostics -x` | No -- Wave 0 |
| HEAT-04 | Dual precision mpmath + python-flint agreement | unit | `pytest tests/test_engine/test_dual_precision.py -x` | No -- Wave 0 |
| HEAT-04 | Ball arithmetic gives certified enclosure | unit | `pytest tests/test_engine/test_dual_precision.py::TestBallArithmetic -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_analysis/test_heat_kernel.py tests/test_engine/test_dual_precision.py -x`
- **Per wave merge:** `pytest tests/ -x --timeout=120`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_analysis/test_heat_kernel.py` -- covers HEAT-01, HEAT-02, HEAT-03
- [ ] `tests/test_engine/test_dual_precision.py` -- covers HEAT-04
- [ ] Framework install: `pip install python-flint>=0.7.0` -- required dependency
- [ ] Barrier computation promotion: session41g must be accessible as a module for comparison tests

## Sources

### Primary (HIGH confidence)
- [python-flint 0.8.0 PyPI](https://pypi.org/project/python-flint/) -- v0.8.0 wheels confirmed for Python 3.13 Windows x86-64
- [python-flint acb docs](https://python-flint.readthedocs.io/en/latest/acb.html) -- modular_j, modular_eta, modular_delta, zeta methods on acb type
- [FLINT acb_modular docs](https://flintlib.org/doc/acb_modular.html) -- modular_j, eta, delta, eisenstein with ball arithmetic; verified stable API
- [data/maass_forms.json](data/maass_forms.json) -- 500 Maass spectral parameters for SL(2,Z) from LMFDB, r_1 = 9.534 to r_500 = 90.955
- [Marklof: Selberg's Trace Formula, An Introduction](https://people.maths.bris.ac.uk/~majm/bib/selberg.pdf) -- scattering matrix formula, test function conditions, four geometric terms
- Sessions 35-47 project history -- barrier computation, circularity audit, heat kernel hypothesis

### Secondary (MEDIUM confidence)
- [Goldfeld: Trace Formulae for SL(2,R)](https://www.math.columbia.edu/~goldfeld/TraceFormulae-4-12-2020.pdf) -- explicit four-term Selberg trace formula; detailed geometric contributions
- [Assing: The Selberg Trace Formula (Bonn lectures)](https://www.math.uni-bonn.de/people/assing/lectures/trace_formula.pdf) -- computational framework for SL(2,Z) trace formula
- [LMFDB Maass forms database](https://www.lmfdb.org/ModularForm/GL2/Q/Maass/) -- spectral parameters, coefficient data
- Training data on spectral theory of modular surfaces -- formulas for phi(s), heat kernel decomposition, Weyl's law; cross-verified with Marklof reference

### Tertiary (LOW confidence -- verify before use)
- python-flint zeta derivative capability -- need to verify whether `acb.zeta(s, derivative=1)` is supported or requires manual computation
- Eisenstein integral magnitude estimates -- scattering phase behavior near zeta zeros requires numerical investigation, not just training data

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- mpmath already validated in production; python-flint 0.8.0 confirmed on PyPI with Python 3.13 wheels
- Architecture: MEDIUM-HIGH -- single new module + dual precision adapter; follows established patterns; novel component is the Eisenstein integral
- Pitfalls: HIGH -- grounded in 47 sessions of project history; circularity traps documented from first-hand experience; 0.036 margin verified at 800+ points
- Mathematical formulas: MEDIUM -- spectral decomposition is standard textbook material; the novel connection between B(L) and K(t) is unverified (that IS Phase 5's job)

**Research date:** 2026-04-04
**Valid until:** 2026-05-04 (stable domain; mathematical formulas don't change; python-flint API stable)
