# Architecture: v2.0 Heat Kernel / Modular Barrier Proof Modules

**Domain:** Heat kernel interpretation of Connes barrier on SL(2,Z)\H, correction bounds, proof assembly
**Researched:** 2026-04-04
**Confidence:** MEDIUM (mathematical structure is well-understood; implementation patterns inferred from codebase analysis and domain knowledge)

## Existing Architecture Summary

The v1.0 platform has a clean, function-based architecture:

```
src/riemann/
  engine/         # Core computation: zeta, zeros, L-functions, precision
  analysis/       # Domain modules: spectral, trace_formula, modular_forms, ncg, rmt, ...
  embedding/      # N-dimensional embedding, HDF5 storage
  viz/            # Plotly visualization, projection theater
  workbench/      # SQLite-backed conjecture/experiment/evidence tracking
  formalization/  # Lean 4 translator, WSL2 builder, triage
  types.py        # ComputationResult, ZetaZero, EvidenceLevel
  config.py       # DEFAULT_DPS=50, project paths
```

**Key patterns to preserve:**
- Function-based APIs (no classes), returning dataclasses
- `validated_computation` for precision-critical math (P-vs-2P)
- `stress_test` for pattern verification across precision levels
- mpmath for arbitrary precision, numpy/scipy for matrix operations
- Workbench for conjecture/experiment/evidence chain tracking
- Lean 4 formalization pipeline via WSL2 with domain-aware Mathlib import selection

## Recommended Architecture for v2.0

### New Module Placement

All new v2.0 code lives in `src/riemann/analysis/` as new files, following the existing flat-module convention. No new top-level packages.

```
src/riemann/analysis/
  (existing)
  trace_formula.py       # Weil explicit formula, Chebyshev psi
  modular_forms.py       # Eisenstein series, Delta, q-expansion, Hecke eigenvalues
  spectral.py            # Berry-Keating Hamiltonian, spectrum comparison
  ncg.py                 # Bost-Connes system
  bost_connes_operator.py # Arithmetic operators (Hecke adjacency, BC Hamiltonian)

  (NEW for v2.0)
  heat_kernel.py         # Heat kernel on SL(2,Z)\H
  selberg_trace.py       # Selberg trace formula (GL(2) trace formula)
  rankin_selberg.py      # Rankin-Selberg L-functions, Petersson norm
  cm_evaluation.py       # CM point evaluation at Heegner discriminants
  barrier_bridge.py      # Correction bounds: heat trace minus actual barrier
  proof_assembly.py      # Orchestrate proof chain, drive formalization
```

### Component Boundaries

| Component | Responsibility | Reads From | Writes To |
|-----------|---------------|------------|-----------|
| `heat_kernel.py` | Compute K_t(z,z) on SL(2,Z)\H via spectral expansion; each term manifestly positive | `modular_forms.py` (Eisenstein), `data/maass_forms.json` (spectral params) | Returns `HeatKernelResult` dataclass |
| `selberg_trace.py` | Selberg trace formula: spectral side <-> geometric side; GL(1)->GL(2) lift of Weil explicit formula | `trace_formula.py` (Weil formula for comparison), `engine/zeros.py` (zeta zeros) | Returns `SelbergTraceResult` dataclass |
| `rankin_selberg.py` | Compute L(s, f x f-bar) for cusp forms; verify Petersson norm positivity; check barrier = L-value identity | `modular_forms.py` (q-expansion, Hecke eigenvalues), `engine/lfunctions.py` (Dirichlet L) | Returns `RankinSelbergResult` dataclass |
| `cm_evaluation.py` | Evaluate barrier and heat kernel at CM points tau_D for Heegner discriminants; extract algebraic values | `heat_kernel.py`, `modular_forms.py`, `engine/zeta.py` | Returns `CMEvaluationResult` dataclass |
| `barrier_bridge.py` | Bound |K_t(trace) - B(L)|; the correction between heat kernel trace and actual Connes barrier | `heat_kernel.py`, `selberg_trace.py`, `engine/precision.py` (validated_computation) | Returns `CorrectionBoundResult` dataclass |
| `proof_assembly.py` | Chain: heat positivity + correction bound -> barrier positivity; register conjectures/experiments in workbench; drive Lean formalization | `barrier_bridge.py`, `workbench/` (conjecture, experiment, evidence), `formalization/` (translator, tracker) | Workbench DB entries, Lean files |

## Data Flow

### Primary Proof Pipeline

```
                    MATHEMATICAL DEPENDENCY CHAIN
                    =============================

1. SPECTRAL DATA (Foundation)
   +-----------------------+     +-------------------------+
   | engine/zeros.py       |     | data/maass_forms.json   |
   | (zeta zeros rho_n)    |     | (Maass eigenvalues r_j) |
   +-----------+-----------+     +------------+------------+
               |                              |
               v                              v
2. MODULAR SURFACE COMPUTATIONS
   +---------------------------+     +------------------------+
   | heat_kernel.py            |     | selberg_trace.py       |
   | K_t(z,z) = spectral sum  |     | GL(2) trace formula    |
   | = 1/(4pi*t)              |     | spectral <-> geometric |
   |   + Sigma_j h(r_j,t)     |     | lifts Weil explicit    |
   |   + Eisenstein integral   |     +----------+-------------+
   +-----------+---------------+                |
               |                                |
               v                                v
3. POSITIVITY VERIFICATION
   +---------------------------+     +------------------------+
   | rankin_selberg.py         |     | cm_evaluation.py       |
   | L(1, f x f-bar) > 0      |     | Algebraic values at    |
   | Petersson norm check      |     | tau_D, D = -3,-4,...   |
   +---------------------------+     +-----------+------------+
                                                 |
               +------ all feed into ------+     |
               v                           v     v
4. CORRECTION BOUNDS
   +--------------------------------------------------+
   | barrier_bridge.py                                 |
   | |heat_trace - barrier| < epsilon                  |
   | Uses validated_computation for rigorous bounds    |
   +---------------------------+-----------------------+
                               |
                               v
5. PROOF ASSEMBLY
   +--------------------------------------------------+
   | proof_assembly.py                                 |
   | Chain: (heat positivity) + (correction < margin)  |
   |   -> barrier > 0 for all L                        |
   | Registers in workbench, drives Lean formalization  |
   +--------------------------------------------------+
```

### Integration Points with Existing Code

**1. heat_kernel.py -> modular_forms.py**

The heat kernel on SL(2,Z)\H decomposes as:

```
K_t(z,z) = (constant term) + (Maass cusp form contribution) + (Eisenstein contribution)
```

The Eisenstein contribution needs the existing `eisenstein_series()` function from `modular_forms.py` for the holomorphic part. The Maass form contribution uses spectral parameters already cached in `data/maass_forms.json` (fetched from LMFDB by the existing `lmfdb_client.py`).

**Integration pattern:**
```python
# heat_kernel.py uses existing modular_forms infrastructure
from riemann.analysis.modular_forms import eisenstein_series, compute_q_expansion
from riemann.analysis.lmfdb_client import query_lmfdb  # for Maass form data
from riemann.engine.precision import validated_computation
```

**2. selberg_trace.py -> trace_formula.py**

The existing `trace_formula.py` implements the Weil explicit formula (GL(1) trace formula). The Selberg trace formula is the GL(2) analog. The new module extends — not replaces — the existing one.

Key relationship: The Weil explicit formula connects zeta zeros to primes. The Selberg trace formula connects Laplacian eigenvalues on SL(2,Z)\H to closed geodesics. The GL(1)->GL(2) lift shows how the barrier, originally a GL(1) object (sum over zeta zeros), becomes a GL(2) object (heat kernel trace on the modular surface).

**Integration pattern:**
```python
# selberg_trace.py references existing GL(1) trace formula
from riemann.analysis.trace_formula import weil_explicit_psi, chebyshev_psi_exact
from riemann.engine.zeros import compute_zero  # zeta zeros for comparison
```

The Selberg trace formula has three sides:
- **Spectral side:** Sum over Laplacian eigenvalues (Maass forms + Eisenstein series)
- **Identity contribution:** Volume term (analogous to "x" in Weil explicit formula)
- **Hyperbolic contribution:** Sum over closed geodesics (analogous to sum over primes)
- **Elliptic/parabolic:** Correction terms from elliptic fixed points and cusps

The lift GL(1)->GL(2) maps: zeta zeros -> Maass eigenvalues, primes -> geodesic lengths.

**3. rankin_selberg.py -> modular_forms.py + engine/lfunctions.py**

The Rankin-Selberg L-function L(s, f x f-bar) is computed from Hecke eigenvalues:

```
L(s, f x f-bar) = Product_p (1 - alpha_p^2 * p^{-s})^{-1} (1 - |alpha_p|^2 * p^{-s})^{-1} (1 - beta_p^2 * p^{-s})^{-1}
```

where alpha_p, beta_p come from Hecke eigenvalues a_p of the cusp form f.

**Integration pattern:**
```python
# rankin_selberg.py builds on existing Hecke eigenvalue computation
from riemann.analysis.modular_forms import hecke_eigenvalues, compute_q_expansion
from riemann.engine.lfunctions import dirichlet_l  # for comparison/cross-checks
from riemann.engine.precision import validated_computation
```

At s=1, this gives L(1, f x f-bar) = (4*pi)^k * ||f||^2 / (k-1)! where ||f|| is the Petersson norm. This is manifestly positive (norm squared), which is the key fact.

**4. cm_evaluation.py -> heat_kernel.py + modular_forms.py**

CM points are tau_D = (-b + sqrt(D)) / (2a) for negative discriminants D. The nine Heegner numbers D = -3, -4, -7, -8, -11, -19, -43, -67, -163 have class number 1, giving unique CM points.

At these points, modular forms take algebraic values (Kronecker's theorem). This gives exact, algebraic values for the heat kernel and barrier, bypassing numerical approximation entirely.

**Integration pattern:**
```python
# cm_evaluation.py combines heat kernel with modular form evaluation
from riemann.analysis.heat_kernel import heat_kernel_diagonal
from riemann.analysis.modular_forms import compute_q_expansion, eisenstein_series
from riemann.engine.zeta import zeta  # for barrier computation at CM points
```

**5. barrier_bridge.py -> heat_kernel.py + selberg_trace.py + engine/precision.py**

This is the critical module: it bounds the difference between the heat kernel trace (which is manifestly positive, each term being positive) and the actual Connes barrier B(L).

The correction has the form:
```
B(L) = Tr(K_t) - epsilon(t, L)
```

If we can show epsilon(t, L) < Tr(K_t) for all L, then B(L) > 0. The bridge module computes epsilon rigorously using `validated_computation`.

**Integration pattern:**
```python
# barrier_bridge.py is the convergence point
from riemann.analysis.heat_kernel import heat_kernel_trace
from riemann.analysis.selberg_trace import selberg_spectral_sum, selberg_geometric_sum
from riemann.engine.precision import validated_computation
from riemann.engine.validation import stress_test
```

**6. proof_assembly.py -> workbench + formalization**

This module orchestrates the proof chain and connects to the existing workbench and Lean 4 pipeline.

**Integration pattern:**
```python
# proof_assembly.py uses the full existing infrastructure
from riemann.analysis.barrier_bridge import compute_correction_bound
from riemann.workbench.conjecture import create_conjecture, update_conjecture
from riemann.workbench.experiment import save_experiment
from riemann.workbench.evidence import link_evidence
from riemann.formalization.translator import generate_lean_file
from riemann.formalization.tracker import update_formalization_state
from riemann.formalization.builder import run_lake_build
```

The proof chain produces three conjectures for the workbench:
1. "Heat kernel trace on SL(2,Z)\H is positive" (evidence_level=2, conditional on spectral expansion convergence)
2. "Correction |K_t - B(L)| bounded by C(t)" (evidence_level=1 or 2, depending on rigor of bound)
3. "B(L) > 0 for all L" (evidence_level=2, conditional on 1+2)

Each conjecture links to experiments (numerical verification at specific L values) via the evidence chain.

## Dataclass Definitions

New result types follow the existing pattern (see `TraceFormulaResult`, `SpectralResult`, `ModularFormResult`):

```python
@dataclass
class HeatKernelResult:
    """Result of heat kernel computation on SL(2,Z)\H."""
    z: complex                    # Point on upper half-plane
    t: float                      # Heat time parameter
    kernel_value: float           # K_t(z,z)
    constant_term: float          # 1/(4*pi*t) contribution
    maass_contribution: float     # Sum over Maass cusp forms
    eisenstein_contribution: float # Eisenstein integral
    n_maass_terms: int            # Number of Maass forms used
    metadata: dict = field(default_factory=dict)

@dataclass
class SelbergTraceResult:
    """Result of Selberg trace formula evaluation."""
    test_function_name: str       # Which h(r) was used
    spectral_sum: float           # Sum over eigenvalues
    geometric_sum: float          # Sum over geodesics
    identity_term: float          # Volume contribution
    elliptic_term: float          # Elliptic fixed point contribution
    parabolic_term: float         # Cusp contribution
    discrepancy: float            # |spectral - geometric|
    n_eigenvalues: int
    n_geodesics: int
    metadata: dict = field(default_factory=dict)

@dataclass
class RankinSelbergResult:
    """Result of Rankin-Selberg L-function computation."""
    weight: int
    level: int
    s: complex                    # Evaluation point
    l_value: complex              # L(s, f x f-bar)
    petersson_norm_sq: float | None  # ||f||^2 if s=1
    n_euler_factors: int          # Number of primes in product
    metadata: dict = field(default_factory=dict)

@dataclass
class CMEvaluationResult:
    """Result of evaluation at a CM point."""
    discriminant: int             # D (negative)
    tau: complex                  # CM point in upper half-plane
    heat_kernel_value: float      # K_t(tau, tau)
    barrier_value: float          # B(L) at corresponding L
    j_invariant: complex          # j(tau) — should be algebraic
    is_algebraic: bool            # Whether result recognized as algebraic
    algebraic_expression: str | None  # String representation if algebraic
    metadata: dict = field(default_factory=dict)

@dataclass
class CorrectionBoundResult:
    """Result of bounding heat trace minus actual barrier."""
    L: float                      # Barrier parameter
    t: float                      # Heat time parameter
    heat_trace: float             # Tr(K_t) = spectral sum
    barrier_value: float          # B(L) from direct computation
    correction: float             # heat_trace - barrier_value
    correction_bound: float       # Rigorous upper bound on |correction|
    margin: float                 # heat_trace - correction_bound
    proof_viable: bool            # margin > 0 means B(L) > 0 proved
    validated: bool               # Whether P-vs-2P validation passed
    metadata: dict = field(default_factory=dict)
```

## Patterns to Follow

### Pattern 1: Validated Spectral Sums

Heat kernel and Selberg trace formulas involve infinite sums that must be truncated. Use `validated_computation` to verify truncation is adequate.

**When:** Any computation involving spectral sums or geodesic sums.

**Example:**
```python
def heat_kernel_diagonal(z, t, *, dps=None, validate=True, n_maass=50):
    """Compute K_t(z,z) with P-vs-2P validation."""
    if dps is None:
        dps = DEFAULT_DPS

    def _compute():
        # Constant term
        constant = mpmath.mpf(1) / (4 * mpmath.pi * mpmath.mpf(t))
        # Maass contribution
        maass_sum = _maass_spectral_sum(z, t, n_maass)
        # Eisenstein contribution
        eis = _eisenstein_contribution(z, t)
        return constant + maass_sum + eis

    return validated_computation(
        _compute, dps=dps, validate=validate,
        algorithm="heat_kernel_spectral_expansion",
    )
```

### Pattern 2: Convergence Monitoring

Since truncated sums are the core operation, every spectral sum function should return both the sum value and convergence diagnostics.

**When:** All functions computing truncated infinite sums.

**Example:**
```python
def _maass_spectral_sum(z, t, n_terms):
    """Sum over Maass cusp form contributions with convergence tracking."""
    terms = []
    for j in range(n_terms):
        term_j = _single_maass_term(z, t, spectral_params[j])
        terms.append(term_j)

    partial_sums = list(itertools.accumulate(terms))
    # Check: last few partial sums should be converging
    tail_variation = abs(partial_sums[-1] - partial_sums[-2])
    return partial_sums[-1], {"tail_variation": tail_variation, "n_terms": n_terms}
```

### Pattern 3: Proof Chain via Workbench

Each link in the proof chain creates a conjecture, runs experiments at multiple parameter values, and links evidence.

**When:** proof_assembly.py orchestrating the full proof attempt.

**Example:**
```python
def attempt_heat_positivity_proof(L_values, t_values, dps=100):
    """Attempt to prove B(L) > 0 via heat kernel positivity."""
    # Step 1: Create conjecture
    conj_id = create_conjecture(
        statement="B(L) = Tr(K_t) - eps(t,L) > 0 for all L > 0",
        description="Heat kernel trace on SL(2,Z)\\H minus correction...",
        evidence_level=0,  # Start as observation
        tags=["heat_kernel", "barrier", "modular_surface"],
    )

    # Step 2: Run experiments at each (L, t) pair
    for L in L_values:
        for t in t_values:
            result = compute_correction_bound(L, t, dps=dps)
            exp_id = save_experiment(
                description=f"Correction bound at L={L}, t={t}",
                parameters={"L": L, "t": t, "dps": dps},
                result_summary=f"margin={result.margin}, viable={result.proof_viable}",
            )
            link_evidence(conj_id, exp_id,
                relationship="supports" if result.proof_viable else "neutral",
                strength=min(result.margin / 0.1, 1.0),
            )

    # Step 3: If all viable, upgrade evidence level
    # ...
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Circular Computation

**What:** Using B(L) > 0 (which IS the Riemann Hypothesis) anywhere in the proof chain.

**Why bad:** Sessions 35-42 showed every direct analytic approach is circular. The heat kernel approach works precisely because positivity comes from the structure of the spectral expansion (each term is positive), not from assuming RH.

**Instead:** The proof chain must be: (1) heat kernel trace is positive because each Maass form term is |phi_j(z)|^2 * e^{-lambda_j * t} >= 0, (2) correction is bounded, (3) therefore barrier > 0. At no point should the code assume RH or use it as an intermediate step.

**Detection:** Any import of or reference to barrier positivity or zero location assumptions in heat_kernel.py, selberg_trace.py, or barrier_bridge.py is a red flag.

### Anti-Pattern 2: Unvalidated Truncation

**What:** Truncating a spectral sum at N terms without verifying convergence or bounding the tail.

**Why bad:** The margin between heat trace and correction is only ~0.036 (from Session 43). If the truncation error exceeds this margin, the proof breaks.

**Instead:** Always use `validated_computation` and always compute tail bounds. The heat kernel sum converges exponentially in t (each term has factor e^{-lambda_j * t}), so the tail can be bounded using Weyl's law for eigenvalue asymptotics.

### Anti-Pattern 3: Mixing Precision Regimes

**What:** Computing the heat kernel at 50 digits but the correction bound at 15 digits, or vice versa.

**Why bad:** The proof requires the margin (heat_trace - correction_bound) to be certifiably positive. If the two quantities are computed at different precisions, the margin might be a precision artifact.

**Instead:** Always compute heat_trace and correction_bound at the same precision, both validated. The `CorrectionBoundResult.validated` flag must be True for the result to be used in proof assembly.

### Anti-Pattern 4: Premature Lean Formalization

**What:** Generating Lean 4 code before the numerical evidence is solid.

**Why bad:** The formalization pipeline works well for translating established conjectures. If the mathematics is still uncertain, sorry-heavy Lean files waste time.

**Instead:** proof_assembly.py should only trigger Lean generation after: (a) numerical experiments pass at multiple (L, t) values, (b) stress tests confirm stability, (c) evidence_level >= 2 (conditional).

## Build Order (Dependency-Respecting)

The modules have strict mathematical dependencies that dictate build order:

```
Phase 1: Foundation          Phase 2: Verification       Phase 3: Assembly
=================           ====================        ================
heat_kernel.py       -->    cm_evaluation.py      -->   barrier_bridge.py
selberg_trace.py     -->    rankin_selberg.py     -->   proof_assembly.py
```

**Detailed order:**

1. **heat_kernel.py** (first) -- No v2.0 dependencies. Depends only on existing modular_forms.py and maass_forms.json. This is the mathematical foundation: if the heat kernel computation does not work, nothing else matters.

2. **selberg_trace.py** (parallel with 1) -- No v2.0 dependencies. Depends only on existing trace_formula.py and engine/zeros.py. Provides the GL(2) trace formula that validates the heat kernel computation via spectral-geometric duality.

3. **rankin_selberg.py** (after 1) -- Depends on modular_forms.py (existing) for Hecke eigenvalues. Independent of heat_kernel.py but logically follows it because the Petersson norm positivity is part of the "why positivity is automatic" story.

4. **cm_evaluation.py** (after 1) -- Depends on heat_kernel.py for evaluating K_t at CM points. Also uses modular_forms.py for algebraic value recognition.

5. **barrier_bridge.py** (after 1, 2) -- Depends on heat_kernel.py for the trace and selberg_trace.py for the spectral/geometric decomposition. This is the critical module that bounds corrections.

6. **proof_assembly.py** (last) -- Depends on barrier_bridge.py and all workbench/formalization infrastructure. Only built after the computational modules are validated.

## Lean 4 Formalization Plan

The existing `formalization/translator.py` has domain-aware Mathlib import selection. For v2.0, add a new domain:

```python
# In translator.py _DOMAIN_IMPORTS:
"heat_kernel": [
    "Mathlib.NumberTheory.LSeries.RiemannZeta",
    "Mathlib.NumberTheory.ModularForms.JacobiTheta.Basic",
    "Mathlib.Analysis.SpecialFunctions.Gamma.Deligne",
    "Mathlib.Topology.MetricSpace.Basic",  # for hyperbolic metric
],
```

The Lean proofs would build on the existing `ConnesPSD.lean` (Gram matrix / Schur complement paths) and `SelbergClass.lean` (Selberg class axioms). New Lean files:

- `HeatKernelPositivity.lean` -- Each term in the spectral expansion is non-negative
- `CorrectionBound.lean` -- The correction epsilon(t,L) is bounded
- `ModularBarrier.lean` -- Combines the above: B(L) = Tr(K_t) - eps > 0

## Existing Data Assets

| Asset | Location | How v2.0 Uses It |
|-------|----------|------------------|
| Zeta zeros | `data/zeros.db` (SQLite) | Barrier computation, comparison with Selberg trace |
| Maass form spectral parameters | `data/maass_forms.json` | Heat kernel spectral expansion (r_j values) |
| LMFDB cache | `data/cache/` | Additional Maass forms, modular form data |
| Computed arrays | `data/computed/` (npz) | Experiment result storage |
| Workbench DB | `data/zeros.db` | Conjecture/experiment/evidence tracking |

**New data needed:**
- Geodesic length spectrum for SL(2,Z)\H (computable from the group structure; the primitive geodesic lengths are log(N(gamma)) where N(gamma) runs over norms of hyperbolic elements)
- Higher Maass form spectral parameters (may need to fetch more from LMFDB or compute via Hejhal's algorithm)

## Scalability Considerations

| Concern | At 10 Maass terms | At 100 Maass terms | At 1000 Maass terms |
|---------|--------------------|--------------------|---------------------|
| Heat kernel sum | Instant, ~10ms | Fast, ~100ms | Moderate, ~2s (Bessel function evaluations dominate) |
| Selberg geometric side | ~50 geodesics sufficient | ~200 geodesics | Geodesic enumeration becomes O(e^L) |
| Rankin-Selberg Euler product | ~20 primes | ~100 primes | ~500 primes, convergence excellent |
| CM evaluation | 9 Heegner points, instant | Same 9 points | Same 9 points |
| Correction bound | Single validated_computation | Same | Same, but precision demands increase |

The bottleneck is the heat kernel sum for large t (many Maass terms needed when t is small, since convergence factor is e^{-lambda_j * t}). For the proof, we choose t optimally: large enough for fast convergence but not so large that the heat kernel approximation to the barrier degrades.

## Sources

- [Selberg trace formula - Wikipedia](https://en.wikipedia.org/wiki/Selberg_trace_formula) -- Overview of spectral/geometric duality
- [How Selberg's trace formula and the Riemann-Weil explicit formula are related](https://empslocal.ex.ac.uk/people/staff/mrwatkin/zeta/STF-WEF.htm) -- GL(1)->GL(2) connection
- [Effective computation of Maass cusp forms](https://www.math.ias.edu/~akshay/research/bsv.pdf) -- Hejhal's algorithm for Maass form computation
- [Numerical computations with the trace formula](https://www2.math.uu.se/~ast10761/papers/stfz14march06.pdf) -- Numerical Selberg trace formula methods
- [Rankin-Selberg method user's guide](https://mathweb.ucsd.edu/~apollack/rankin-selberg.pdf) -- Rankin-Selberg L-function computation
- [Lecture on CM point values](https://www.math.ucla.edu/~hida/Kyoto2.pdf) -- Values of modular forms at CM points
- [Selberg's trace formula introduction (Marklof)](https://people.maths.bris.ac.uk/~majm/bib/selberg.pdf) -- Pedagogical introduction
- [Heat kernel on Cayley graph of PSL2Z (2025)](https://arxiv.org/html/2506.02340) -- Recent heat kernel work on the modular group
- Session 43 spectral dominance result -- barrier = spectral sum - correction, spectral always wins
- Session 45 Wick rotation -- Lorentzian test function as heat kernel at imaginary time
