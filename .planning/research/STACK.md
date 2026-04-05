# Stack Research: Heat Kernel / Modular Barrier Proof Tools

**Domain:** Computational number theory -- heat kernel on modular surface, Selberg trace formula, Rankin-Selberg L-functions, CM point evaluation
**Researched:** 2026-04-04
**Confidence:** MEDIUM-HIGH (core approach uses mpmath which is already validated; new library python-flint verified with Windows wheels; heat kernel mathematics is closed-form and well-understood)

---

## Context: What Already Exists

The existing stack (mpmath, numpy, scipy, sympy, gmpy2) already covers the foundation. This research focuses ONLY on what is needed for the v2.0 modular barrier milestone. Validated capabilities NOT to re-add or duplicate:

- mpmath arbitrary-precision zeta, zeros, L-functions, special functions
- numpy/scipy eigenvalue computation, linear algebra, signal processing
- sympy symbolic manipulation, q-series algebra
- Existing `modular_forms.py`: Eisenstein series E_k, Ramanujan Delta, Hecke eigenvalues (level 1)
- Existing `spectral.py`: Berry-Keating Hamiltonian, eigenvalue computation
- Existing `trace_formula.py`: Weil explicit formula, Chebyshev psi
- Existing `lmfdb_client.py`: REST API with SQLite cache for modular form data
- Existing `padic.py`: p-adic arithmetic, Kubota-Leopoldt zeta

---

## Recommended Stack Additions

### 1. python-flint -- Certified Modular Form Arithmetic

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **python-flint** | >=0.7.0, recommend 0.8.0 | Rigorous ball arithmetic for modular j-invariant, Dedekind eta, Eisenstein series, theta functions | Wraps FLINT/Arb (the gold standard for certified number theory computation). Unlike mpmath's floating-point, python-flint returns ball arithmetic results with **rigorous error bounds** -- critical for a proof pipeline where you need to certify that corrections are bounded. Provides `acb.modular_j()`, `acb.modular_eta()`, `acb.modular_delta()`, `acb.modular_theta()`, and `acb.eisenstein()` with arbitrary-precision certified output. Windows x86-64 wheels available on PyPI. |

**Why not just mpmath?** mpmath computes the same functions but gives floating-point approximations without error certificates. For the barrier proof, we need to bound corrections rigorously. python-flint's ball arithmetic gives intervals `[a - eps, a + eps]` where the true value is guaranteed to lie. Use mpmath for exploration, python-flint for certification.

**Why not SageMath?** SageMath provides the most complete modular forms library in existence (Hecke algebras, modular symbols, newforms for arbitrary level/weight). However, SageMath is a 2+ GB installation with its own Python runtime that does not integrate cleanly into an existing pip/venv-based project. The specific functions needed (j-invariant at CM points, Eisenstein q-expansions, theta functions) are all available in python-flint at a fraction of the footprint. If the project later needs modular symbols or newforms at higher level, SageMath can be added as a subprocess oracle.

**Confidence:** HIGH -- python-flint 0.8.0 has Windows wheels, wraps FLINT 3.3.1, acb_modular API is stable and well-documented.

---

### 2. No New Libraries Required for Heat Kernel Computation

The heat kernel on the hyperbolic plane H^2 has an explicit closed-form:

```
K_H(d, t) = (sqrt(2)) / (4*pi*t)^(3/2) * exp(-t/4) * integral_{d}^{inf} (r * exp(-r^2/(4t))) / sqrt(cosh(r) - cosh(d)) dr
```

where `d` is the geodesic distance and `t` is time. For the modular surface SL(2,Z)\H, the heat kernel trace is:

```
K_{Gamma\H}(t) = sum_{gamma in Gamma} K_H(d(z, gamma*z), t)
```

This decomposes into identity, hyperbolic, elliptic, and parabolic orbital integrals -- each with known closed forms involving:

- `mpmath.exp`, `mpmath.sqrt`, `mpmath.log` (basic arithmetic)
- `mpmath.quad` (numerical integration for the identity term)
- `mpmath.besselk` (modified Bessel function K_v for hyperbolic orbital integrals)
- `mpmath.legenp`, `mpmath.legenq` (Legendre functions for spectral expansion)
- `scipy.special.kv` (fast machine-precision Bessel K for bulk exploration)
- `mpmath.gammainc` (incomplete gamma for parabolic terms)

**All of these already exist in the installed stack.** No new library is needed.

The Selberg trace formula computation similarly decomposes into:
- Spectral side: sum over eigenvalues lambda_j = 1/4 + r_j^2 of the Laplacian (Maass forms data from LMFDB or hardcoded, as in existing `_maass_hecke.py`)
- Geometric side: sum over conjugacy classes (identity + hyperbolic + elliptic + parabolic terms)

Each term uses elementary and special functions already in mpmath/scipy.

**Confidence:** HIGH -- the mathematics reduces to standard special function evaluation. No exotic dependencies needed.

---

### 3. Existing mpmath Functions for CM Point Evaluation

CM (complex multiplication) points on the modular surface are tau values where j(tau) is an algebraic integer. The nine Heegner discriminants D = -3, -4, -7, -8, -11, -19, -43, -67, -163 give tau values:

```
tau_D = (-D)^{1/2} * i / 2   (or shifts by 1/2 for odd D)
```

Evaluating the barrier at these points requires:
1. Computing j(tau_D) -- use `python-flint` `acb.modular_j()` for certified value
2. Recognizing the result as an algebraic number -- use `mpmath.identify()` or `mpmath.findpoly()` (PSLQ-based algebraic number recognition, already installed)
3. Evaluating Eisenstein series and eta function at CM points -- `acb.eisenstein()`, `acb.modular_eta()`

The algebraic number recognition pipeline:
- `mpmath.pslq(x, [1, x, x^2, ..., x^n])` finds the minimal polynomial
- `mpmath.findpoly(x, n)` convenience wrapper
- `mpmath.identify(x)` attempts symbolic closed form

**No new libraries needed.** mpmath's identification module handles algebraic number recognition up to degree ~20 with 50+ digits of precision.

**Confidence:** HIGH -- mpmath.identify and findpoly are mature, well-tested tools. CM point j-values are known algebraic integers; recognition at 100+ digits is straightforward.

---

### 4. Rankin-Selberg L-function: Build on Existing Infrastructure

The Rankin-Selberg L-function L(s, f x f-bar) for a cusp form f with Fourier coefficients a_n is:

```
L(s, f x f-bar) = sum_{n=1}^{N} |a_n|^2 / n^s
```

This is a Dirichlet series that can be evaluated with:
- Existing `modular_forms.py` for q-expansion coefficients a_n
- `mpmath.nsum` or direct partial sums for the L-series
- `mpmath.zeta`-style acceleration if needed (Euler product over primes)

The key identity to verify: `L(1, f x f-bar) = (4*pi)^k / ((k-1)! * vol(Gamma\H)) * <f, f>_Pet`

where `<f, f>_Pet` is the Petersson norm (always positive). This reduces to:
1. Compute a_n for n = 1..N via existing `compute_q_expansion()`
2. Sum `|a_n|^2 / n^s` at s = 1
3. Compare with Petersson norm formula

**No new library needed.** The Rankin-Selberg computation is a Dirichlet series over known coefficients.

**Confidence:** HIGH -- the computation is elementary given the q-expansion infrastructure already built.

---

## Supporting Libraries (Unchanged from Existing Stack)

These libraries from the existing pyproject.toml are directly useful for the new milestone features:

| Library | Version (installed) | Role in v2.0 |
|---------|---------------------|--------------|
| mpmath | >=1.3.0 | Heat kernel special functions (besselk, legenp/q, gammainc), CM point j-values, algebraic recognition (pslq, findpoly, identify), Rankin-Selberg series |
| gmpy2 | >=2.3.0 | Accelerates mpmath backend for high-precision heat kernel evaluation |
| numpy | >=2.4.3 | Matrix operations for Selberg trace formula computation, eigenvalue storage |
| scipy | >=1.17.1 | `scipy.special.kv` for fast Bessel K, `scipy.linalg.eigh` for Laplacian eigenvalues, `scipy.integrate.quad` for orbital integrals at machine precision |
| sympy | >=1.14.0 | Symbolic manipulation of trace formula terms, q-series algebra, prime enumeration |
| requests | >=2.32.0 | LMFDB API calls for Maass form spectral parameters |

---

## Installation

```bash
# Only new dependency
pip install python-flint>=0.7.0

# Or add to pyproject.toml:
# "python-flint>=0.7.0",
```

That is the complete installation delta. Everything else is already present in the existing pyproject.toml.

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| python-flint for certified modular forms | SageMath | If you need modular symbols, newforms at arbitrary level/weight, or Hecke algebra computations beyond what FLINT provides. SageMath is the nuclear option -- most complete but heaviest. Consider running as a subprocess (`sage -python script.py`) rather than integrating into the venv. |
| python-flint for certified arithmetic | mpmath alone | If you only need exploration (not proof certification). mpmath is faster for ad-hoc computation but lacks rigorous error bounds. Use mpmath for discovery, python-flint for verification. |
| mpmath.findpoly for algebraic recognition | LLL via fpLLL / fplll Python bindings | If PSLQ fails for high-degree algebraic numbers (degree > 15). fpLLL implements the LLL algorithm for lattice reduction which can find integer relations in contexts where PSLQ struggles. However, for CM j-values (known degree <= 3 for Heegner discriminants), mpmath.findpoly is more than sufficient. |
| Hand-coded heat kernel formulas | GeometricKernels package | If you need heat kernels on general Riemannian manifolds or need ML/GP integration. GeometricKernels supports hyperbolic spaces but is oriented toward machine learning (Gaussian processes), not number-theoretic proof. The heat kernel on SL(2,Z)\H has explicit formulas that are more efficient to implement directly. |
| Hardcoded Maass spectral data + LMFDB | psage (parallel sage) / hecke.py | If you need to compute Maass forms from scratch (Hejhal's algorithm). No standalone Python package exists for this. The existing hardcoded data (50 spectral parameters from _maass_hecke.py, LMFDB data) is sufficient for the barrier proof where you need finite sums over eigenvalues. |
| scipy.special for machine-precision Bessel | Cython/numba-accelerated custom | If bulk heat kernel evaluation becomes a bottleneck. scipy.special.kv is already compiled C (AMOS library) and is fast. Only consider custom acceleration if profiling reveals this as a bottleneck. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| SageMath as a full dependency | 2+ GB install, separate Python runtime, breaks venv isolation, overkill for the modular forms functions needed | python-flint for certified arithmetic; mpmath + existing modular_forms.py for q-expansions |
| pari/gp (via cypari2) | Another large C library dependency; python-flint already wraps the relevant FLINT functionality | python-flint |
| GeometricKernels | ML-oriented; computes heat kernels via truncated eigenfunction expansion, not the exact orbital decomposition needed for proof | Direct implementation using mpmath special functions |
| Custom Maass form solver | Hejhal's algorithm is complex (hundreds of lines of careful numerics); the project needs eigenvalues as data, not an eigenvalue solver | Use LMFDB data + hardcoded spectral parameters |
| Symbolic heat kernel via sympy.integrate | Symbolic integration of the heat kernel formula is slow and fragile; the integrals are best evaluated numerically | mpmath.quad for arbitrary precision, scipy.integrate.quad for machine precision |

---

## Stack Patterns by Feature

**Heat kernel on SL(2,Z)\H:**
- Exploration: mpmath (besselk, quad, exp) for arbitrary precision evaluation
- Bulk computation: scipy.special.kv + numpy for parameter sweeps
- Certification: python-flint acb ball arithmetic for rigorous error bounds on trace sums

**Modular form parametrization / q-series fitting:**
- Coefficient computation: existing modular_forms.py (Eisenstein, Delta, Hecke eigenvalues)
- Higher level/weight: LMFDB client for pre-computed data
- Certified evaluation: python-flint acb.modular_j(), acb.eisenstein()

**GL(1)->GL(2) lift (Selberg trace formula):**
- Spectral side: hardcoded Maass form data (50 spectral parameters from _maass_hecke.py) + LMFDB
- Geometric side: mpmath special functions for orbital integrals
- Comparison: numpy for vectorized sums, scipy.optimize for fitting

**Rankin-Selberg L-function:**
- Fourier coefficients: existing compute_q_expansion()
- L-series evaluation: mpmath.nsum or direct partial sums
- Petersson norm: mpmath.quad for the integral, or use L(1, f x f-bar) identity

**CM point evaluation:**
- j-invariant: python-flint acb.modular_j(tau) with certified bounds
- Algebraic recognition: mpmath.findpoly() / mpmath.identify()
- Barrier evaluation at CM points: mpmath at 100+ digit precision

**Correction bound estimation:**
- Upper bounds: python-flint ball arithmetic (guaranteed intervals)
- Numerical exploration: mpmath.quad for integrals, numpy for matrix norms
- Certification: combine python-flint error bounds with analytical estimates

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| python-flint 0.8.0 | Python 3.11-3.14 | Windows x86-64 wheels on PyPI; wraps FLINT 3.3.1, MPFR 4.2.2 |
| python-flint 0.8.0 | mpmath >=1.3.0 | No conflicts; they serve different roles (exploration vs certification). Both can evaluate modular_j but python-flint gives error bounds. |
| python-flint 0.8.0 | numpy >=2.0, scipy >=1.12 | No conflicts; python-flint uses its own numeric types (acb, arb), not numpy arrays. Convert with `float()` or `complex()` when passing to numpy. |
| gmpy2 >=2.3.0 | python-flint 0.8.0 | Both depend on GMP/MPFR. python-flint bundles its own copies in wheels; no conflict when installed via pip wheels. |
| Python 3.12 | All above | Recommended runtime; all libraries have 3.12 wheels |

---

## Integration Points with Existing Code

### How python-flint Connects to Existing Modules

1. **modular_forms.py** -- existing q-expansion code computes coefficients at machine precision. For proof certification, wrap key evaluations with python-flint:
   ```python
   from flint import acb
   tau = acb(0, 163**0.5 / 2)  # CM point for D=-163
   j_val = tau.modular_j()      # certified j-invariant
   # j_val is an acb with rigorous error bound
   ```

2. **trace_formula.py** -- existing Weil explicit formula maps to the geometric side of the Selberg trace formula. The lift GL(1)->GL(2) adds the spectral side (Maass form sum). Both sides use mpmath special functions.

3. **lmfdb_client.py** -- query `mf_maass_newforms` collection for Maass form spectral parameters beyond the hardcoded 50 in `_maass_hecke.py`. The existing caching infrastructure handles this.

4. **spectral.py** -- eigenvalue comparison methods apply directly to comparing Laplacian eigenvalues on the modular surface against barrier spectral data.

### Key mpmath Functions for New Features

| Function | Use Case | Module |
|----------|----------|--------|
| `mpmath.besselk(v, z)` | Modified Bessel K in hyperbolic orbital integrals | Heat kernel |
| `mpmath.legenp(n, m, z)` | Associated Legendre functions in spectral expansion | Selberg trace |
| `mpmath.legenq(n, m, z)` | Legendre Q in Green's function on H | Heat kernel |
| `mpmath.quad(f, [a, b])` | Numerical integration of orbital integrals | Heat kernel, Rankin-Selberg |
| `mpmath.nsum(f, [1, inf])` | Accelerated infinite series (Richardson, Euler-Maclaurin) | Rankin-Selberg L-values |
| `mpmath.findpoly(x, n)` | Find minimal polynomial of CM j-value | CM point recognition |
| `mpmath.identify(x)` | Recognize algebraic number as closed form | CM point recognition |
| `mpmath.pslq(vec)` | Integer relation detection | Barrier algebraic structure |
| `mpmath.gammainc(a, z)` | Incomplete gamma for parabolic orbital terms | Selberg trace |
| `mpmath.zeta(s)` | Zeta values in Selberg trace identity term | Selberg trace |

---

## Sources

- [mpmath identification docs](https://mpmath.org/doc/current/identification.html) -- PSLQ, findpoly, identify capabilities verified (HIGH confidence)
- [mpmath Bessel functions docs](https://mpmath.org/doc/current/functions/bessel.html) -- besselk, besseli confirmed (HIGH confidence)
- [mpmath Legendre functions docs](https://mpmath.org/doc/current/functions/orthogonal.html) -- legenp, legenq confirmed (HIGH confidence)
- [FLINT acb_modular docs](https://flintlib.org/doc/acb_modular.html) -- modular_j, eta, delta, theta, eisenstein with ball arithmetic (HIGH confidence)
- [python-flint PyPI](https://pypi.org/project/python-flint/) -- v0.8.0, Python 3.11-3.14, Windows wheels confirmed (HIGH confidence)
- [python-flint install docs](https://python-flint.readthedocs.io/en/latest/install.html) -- pip install verified (HIGH confidence)
- [Strohmaier-Uski algorithm paper](https://arxiv.org/abs/1110.2150) -- reference for eigenvalue computation methodology; no public code available (MEDIUM confidence)
- [LMFDB Maass forms](https://www.lmfdb.org) -- spectral parameter data source, already integrated via lmfdb_client.py (HIGH confidence)
- [Grigor'yan heat kernel on hyperbolic space](https://www.math.uni-bielefeld.de/~grigor/nog.pdf) -- closed-form heat kernel formula reference (HIGH confidence on math, PDF not machine-readable)
- [Boulanger heat kernel and Selberg pre-trace formula](https://arxiv.org/abs/1902.06580v2) -- orbital decomposition methodology (MEDIUM confidence)
- [GeometricKernels package](https://arxiv.org/html/2407.08086v2) -- evaluated and rejected for this use case (HIGH confidence on assessment)
- [Borthwick spectral geometry lectures](https://math.dartmouth.edu/~specgeom/Borthwick_slides.pdf) -- Selberg trace formula computational framework reference (MEDIUM confidence)

---
*Stack research for: Riemann v2.0 Modular Barrier -- Heat Kernel Proof Tools*
*Researched: 2026-04-04*
*Delta from existing stack: python-flint only*
