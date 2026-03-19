# Phase 3: Deep Domain Modules and Cross-Disciplinary Synthesis - Research

**Researched:** 2026-03-19
**Domain:** Spectral theory, modular forms, p-adic arithmetic, topological data analysis, dynamical systems, noncommutative geometry, AI-guided conjecture generation
**Confidence:** MEDIUM (individual module domains are well-understood; cross-disciplinary synthesis patterns are novel)

## Summary

Phase 3 builds 8 independent domain modules and an AI-guided conjecture generator, all following the established function-based API pattern from Phases 1-2. The core technical challenge is NOT building complex infrastructure -- it is implementing the correct mathematics in each domain while maintaining the 70/30 exploration-to-infrastructure ratio. Every module follows the same pattern: function-based API in `analysis/`, returns data (not plots), integrates with the existing workbench and anomaly system.

The phase spans 11 requirements across 8 mathematical domains (spectral operators, trace formulas, modular forms, LMFDB, p-adic, analogy engine, TDA, dynamical systems, noncommutative geometry, AI conjecture generation). The biggest risk is scope explosion -- each domain is deep enough to consume the entire phase. The key mitigation is: each module should be minimal-viable (implements the core computation, returns useful data) before any module gets polished. Libraries exist for the heavy computational lifting (scipy.linalg for eigenvalues, ripser for persistent homology, pyadic for p-adic numbers, nolds for Lyapunov exponents). Custom implementations are needed only where no library covers the specific mathematical object (Berry-Keating Hamiltonian, Bost-Connes system, analogy engine).

**Primary recommendation:** Build modules as independent, testable units following the Phase 2 pattern. Use existing libraries (ripser, pyadic, nolds, requests for LMFDB). Custom-build only the domain-specific mathematics (operator construction, trace formula sums, analogy mappings, Bost-Connes). Prioritize getting all 8 domains minimally functional before deepening any single one.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- All modules follow the established Phase 2 pattern: function-based API, returns data (not plots), pluggable into the analysis pipeline
- Each domain module is independent -- can be developed and tested in isolation
- Modules integrate with existing infrastructure: workbench for tracking, anomaly system for flagging, embedding framework for higher-dimensional views
- New visualization functions go in `viz/` following the comparison.py pattern from Phase 2
- Spectral operators: Berry-Keating Hamiltonian H = xp (and regularized variants) as finite-dimensional matrices; numpy.linalg for moderate sizes, scipy.sparse.linalg for large; chi-squared fit metric
- Trace formulas: Riemann-von Mangoldt and Weil explicit formulas; interactive visualization of zeros-to-primes duality
- Modular forms via direct Fourier expansion (q-series) -- no SageMath dependency; Hecke eigenvalues via matrix representation; LMFDB via REST API; cache with SQLite
- Custom p-adic number class with configurable precision; Kubota-Leopoldt p-adic zeta via Bernoulli interpolation; fractal tree layout visualization
- Analogy engine: formal AnalogyMapping structure {source_domain, target_domain, correspondences, unknowns, evidence}
- TDA via giotto-tda or ripser; dynamical systems tools (Lyapunov, fixed points, orbits); Bost-Connes system for noncommutative geometry
- AI conjecture pipeline: observation -> hypothesis -> experiment design -> execution -> evidence evaluation -> conjecture formalization
- Claude as active scientist: autonomous exploration loops
- Carrying forward: JupyterLab, Claude-driven, speed over polish, SQLite+numpy+HDF5, strict evidence hierarchy, 50-digit default, function-based API, "surprise me"

### Claude's Discretion
- Operator discretization strategies and matrix sizes
- Fourier expansion truncation orders for modular forms
- p-adic precision defaults
- TDA filtration parameters and persistence diagram interpretation
- Dynamical systems iteration depths and convergence criteria
- Noncommutative geometry representation choices
- LMFDB query strategies and caching policies
- Analogy mapping initialization and unknown-inference algorithms
- AI conjecture generation prompting strategies and confidence scoring
- All performance optimization decisions

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| SPEC-01 | Construct, discretize, and analyze candidate self-adjoint operators (Berry-Keating and variants); compare eigenvalue spectra against zeta zeros | Spectral operator module: scipy.linalg.eigh for moderate matrices, scipy.sparse.linalg.eigsh for large; Berry-Keating H=xp discretization via finite-difference or spectral methods |
| SPEC-02 | Explore trace formula connections (Selberg, Weil explicit formula); compute partial sums; visualize zeros-to-primes duality | Trace formula module: mpmath for high-precision sums; truncated explicit formula with configurable terms; Chebyshev psi function computation |
| MOD-01 | Compute modular forms, Hecke eigenvalues, Fourier coefficients; visualize in upper half-plane | Modular forms module: q-series via mpmath; Hecke operator matrices via linear algebra; domain coloring adaptation for upper half-plane |
| MOD-02 | Query LMFDB for L-function data, modular form data, number field data; integrate with platform tools | LMFDB client: requests library with JSON format; SQLite caching; REST API at lmfdb.org/api/ confirmed accessible |
| ADEL-01 | Perform p-adic arithmetic and compute p-adic zeta functions | p-adic module: custom PadicNumber class (simpler than pyadic for our needs) or pyadic library; Kubota-Leopoldt via Bernoulli number interpolation using mpmath |
| ADEL-02 | Visualize p-adic structures (fractal geometry); connect p-adic and archimedean pictures | p-adic visualization: fractal tree layout via recursive plotting in matplotlib; dual-view linking same L-function data |
| XDISC-01 | Define and test analogy mappings between domains; computationally explore unknown correspondences | Analogy engine: dataclass-based AnalogyMapping; computational testing of correspondences via existing analysis modules |
| XDISC-02 | Apply TDA (persistent homology) to zero distributions and mathematical objects | TDA module: ripser (actively maintained, v0.6.14) for persistent homology computation; persim for persistence diagram distances |
| XDISC-03 | Analyze zeta/zero dynamics through dynamical systems tools (Lyapunov, phase portraits, attractors) | Dynamics module: nolds for Lyapunov exponents; scipy.integrate for ODE iteration; custom zeta-map definitions |
| XDISC-04 | Compute in noncommutative geometric frameworks (Connes' approach) | NCG module: Bost-Connes system as custom implementation (no library exists); numpy matrices for finite-dimensional approximations of C*-algebras |
| RSRCH-03 | AI-guided analysis: examine results, identify patterns, generate conjectures, suggest experiments | AI conjecture module: structured functions (suggest_experiments, analyze_results, generate_conjecture) that read workbench state and produce structured output for Claude to interpret |
</phase_requirements>

## Standard Stack

### Core (Already Installed)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | >=2.4.3 | Matrix operations for operators, embeddings | Already in project; foundation for all numerical work |
| scipy | >=1.17.1 | Eigenvalue computation (linalg.eigh, sparse.linalg.eigsh), ODE integration | Already in project; standard for spectral problems |
| mpmath | >=1.3.0 | Arbitrary-precision trace formula sums, Bernoulli numbers, modular forms | Already in project; required for precision-sensitive computations |
| matplotlib | >=3.10.8 | p-adic fractal trees, upper half-plane coloring, phase portraits | Already in project; best for static mathematical plots |
| plotly | >=6.6.0 | Interactive persistence diagrams, trace formula exploration | Already in project; interactive 3D/2D |
| scikit-learn | >=1.8.0 | Feature extraction support for TDA pipeline | Already in project |

### New Dependencies
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| ripser | >=0.6.14 | Persistent homology (Vietoris-Rips) | XDISC-02: TDA on zero point clouds. Actively maintained (Dec 2025 release). Lean, fast C++ core. |
| persim | >=0.3.7 | Persistence diagram distances (bottleneck, Wasserstein) | XDISC-02: Comparing persistence diagrams across embeddings |
| requests | >=2.31 | LMFDB REST API client | MOD-02: HTTP queries to lmfdb.org/api/. Likely already available as transitive dependency. |
| nolds | >=0.6.3 | Lyapunov exponents, correlation dimension | XDISC-03: Dynamical systems analysis. numpy-only dependency, lightweight. |

### Considered but NOT Adding
| Library | Why Not |
|---------|---------|
| giotto-tda | AGPLv3 license; heavier than ripser; last release May 2024 (less active). ripser is more focused and actively maintained. |
| pyadic | Version 0.3.0 (March 2026) exists and works, but for our use case (Q_p arithmetic to configurable precision, p-adic zeta via Bernoulli), a custom ~100-line class is simpler, has zero extra dependencies, and integrates better with our mpmath-based precision infrastructure. Recommend custom implementation per CONTEXT.md decision. |
| python-flint | Windows binary availability flagged as concern in STATE.md. Not needed for Phase 3 computations. |
| SageMath | Explicitly rejected in CONTEXT.md for modular forms. q-series approach is sufficient. |

**Installation:**
```bash
uv add ripser persim nolds requests
```

## Architecture Patterns

### Recommended Project Structure
```
src/riemann/
  analysis/
    spectral.py        # SPEC-01: Berry-Keating operators, eigenvalue computation
    trace_formula.py   # SPEC-02: Weil explicit formula, Riemann-von Mangoldt
    modular_forms.py   # MOD-01: q-series, Hecke operators, Fourier coefficients
    lmfdb_client.py    # MOD-02: LMFDB REST API wrapper with SQLite cache
    padic.py           # ADEL-01: PadicNumber class, p-adic zeta functions
    tda.py             # XDISC-02: Persistent homology via ripser, diagram analysis
    dynamics.py        # XDISC-03: Zeta dynamics, Lyapunov, orbits, fixed points
    ncg.py             # XDISC-04: Bost-Connes system, spectral triples
    analogy.py         # XDISC-01: AnalogyMapping, correspondence testing
    conjecture_gen.py  # RSRCH-03: AI-guided experiment suggestion, pattern analysis
  viz/
    trace_formula_viz.py   # Zeros-to-primes duality visualization
    modular_forms_viz.py   # Upper half-plane domain coloring
    padic_viz.py           # Fractal tree layout for p-adic structures
    tda_viz.py             # Persistence diagram plots
    dynamics_viz.py        # Phase portraits, orbit plots
tests/
  test_analysis/
    test_spectral.py
    test_trace_formula.py
    test_modular_forms.py
    test_lmfdb_client.py
    test_padic.py
    test_tda.py
    test_dynamics.py
    test_ncg.py
    test_analogy.py
    test_conjecture_gen.py
```

### Pattern 1: Domain Module (Phase 2 Pattern Continued)
**What:** Each analysis module is a standalone `.py` file with function-based API. Functions accept data, return results. No classes except dataclasses for structured output.
**When to use:** Every new module in this phase.
**Example:**
```python
# Source: Established pattern from analysis/spacing.py, analysis/rmt.py
from dataclasses import dataclass
import numpy as np

@dataclass
class SpectralResult:
    """Result from spectral operator analysis."""
    eigenvalues: np.ndarray
    operator_name: str
    matrix_size: int
    chi_squared_fit: float  # vs zeta zero spacings
    metadata: dict

def construct_berry_keating(n: int, regularization: str = "box") -> np.ndarray:
    """Construct discretized Berry-Keating Hamiltonian H = xp.

    Args:
        n: Matrix dimension (number of grid points).
        regularization: "box" (hard wall), "smooth" (soft potential).

    Returns:
        Hermitian matrix (n x n) representing the discretized operator.
    """
    # Implementation details...
    pass

def compare_spectrum_to_zeros(
    eigenvalues: np.ndarray,
    zero_spacings: np.ndarray,
) -> float:
    """Chi-squared distance between eigenvalue spacings and zero spacings."""
    pass
```

### Pattern 2: LMFDB Client with SQLite Cache
**What:** HTTP client wrapping the LMFDB REST API with local SQLite caching to avoid repeated API calls.
**When to use:** MOD-02 requirement.
**Example:**
```python
# LMFDB API format: https://www.lmfdb.org/api/{collection}/?{query}&_format=json
import json
import requests
import sqlite3
from pathlib import Path

LMFDB_BASE = "https://www.lmfdb.org/api"

def query_lmfdb(
    collection: str,
    params: dict,
    fields: list[str] | None = None,
    cache_db: str | Path | None = None,
) -> list[dict]:
    """Query LMFDB with caching.

    Args:
        collection: e.g., "lfunc_lfunctions", "mf_newforms"
        params: Query parameters (key=value pairs)
        fields: Optional list of fields to retrieve
        cache_db: SQLite cache path

    Returns:
        List of result dicts.
    """
    # Check cache first, then HTTP, then cache response
    pass
```

### Pattern 3: AnalogyMapping as Experiment
**What:** Analogy mappings stored as workbench experiments, enabling versioning and evidence tracking.
**When to use:** XDISC-01 analogy engine.
**Example:**
```python
from dataclasses import dataclass, field

@dataclass
class AnalogyMapping:
    """Formal correspondence between two mathematical domains."""
    source_domain: str              # e.g., "spectral_theory"
    target_domain: str              # e.g., "zeta_zeros"
    correspondences: dict[str, str] # Known: {"eigenvalues": "zeros", "trace_formula": "explicit_formula"}
    unknowns: list[str]             # Unknown: ["Hamiltonian", "boundary_conditions"]
    evidence: list[str]             # experiment_ids supporting this mapping
    confidence: float = 0.0         # 0.0-1.0

def test_correspondence(
    mapping: AnalogyMapping,
    source_data: np.ndarray,
    target_data: np.ndarray,
    metric: str = "chi_squared",
) -> dict:
    """Computationally test whether a proposed correspondence holds."""
    pass
```

### Pattern 4: AI Conjecture Pipeline
**What:** Structured functions that read workbench state and produce actionable suggestions. These are NOT autonomous agents -- they are functions Claude calls in notebooks.
**When to use:** RSRCH-03 AI-guided analysis.
**Example:**
```python
def suggest_experiments(
    db_path: str | Path | None = None,
    max_suggestions: int = 5,
) -> list[dict]:
    """Examine workbench state and suggest next experiments.

    Reads: conjectures (especially speculative ones), anomalies,
    recent experiments, analogy mappings with unknowns.

    Returns list of dicts: {
        "type": "spectral" | "trace" | "modular" | "analogy" | "tda",
        "description": str,
        "rationale": str,
        "parameters": dict,
        "priority": float,  # 0.0-1.0
    }
    """
    pass

def analyze_results(
    experiment_id: str,
    db_path: str | Path | None = None,
) -> dict:
    """Interpret an experiment's results and suggest follow-ups.

    Returns: {
        "summary": str,
        "patterns_detected": list[str],
        "anomalies": list[str],
        "suggested_conjectures": list[dict],
        "next_experiments": list[dict],
    }
    """
    pass

def generate_conjecture(
    observations: list[str],
    evidence_ids: list[str] | None = None,
    db_path: str | Path | None = None,
) -> str:
    """Synthesize observations into a formal conjecture and save to workbench.

    Returns: conjecture_id
    """
    pass
```

### Anti-Patterns to Avoid
- **Framework-first for each domain:** Do NOT build elaborate class hierarchies per domain. Flat functions, dataclass results. The analogy engine is a dict and a test function, not an AbstractAnalogyFramework.
- **Perfect mathematical implementation:** Each module needs to produce useful, correct output for the specific use cases defined in requirements. It does NOT need to handle every edge case of the mathematical theory. A Berry-Keating discretization at size 100-500 that shows the eigenvalue-zero correspondence is far more valuable than a perfectly rigorous discretization that takes a month to implement.
- **Sequential deepening:** Do NOT finish spectral operators perfectly before starting modular forms. Get all 8 domains minimally functional, then deepen the most promising ones.
- **AI conjecture as magic:** The `suggest_experiments` and `generate_conjecture` functions produce structured data that Claude interprets. They are NOT expected to autonomously discover proofs. They aggregate workbench state into a format Claude can reason about.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Persistent homology | Custom simplicial complex code | `ripser` (0.6.14) | C++ core is orders of magnitude faster; correct edge cases in boundary operators |
| Persistence diagram distances | Bottleneck/Wasserstein distance | `persim` | Optimal transport matching is subtle to implement correctly |
| Lyapunov exponents | Custom QR decomposition iteration | `nolds.lyap_r` / `nolds.lyap_e` | Rosenstein and Eckmann algorithms handle numerical stability issues |
| HTTP client with retries | Raw urllib | `requests` | Connection pooling, retries, encoding handled correctly |
| Eigenvalue computation | Custom power iteration | `scipy.linalg.eigh` / `scipy.sparse.linalg.eigsh` | LAPACK/ARPACK implementations are battle-tested |
| Bernoulli numbers | Manual recurrence | `mpmath.bernoulli(n)` | mpmath computes arbitrary-precision Bernoulli numbers correctly and efficiently |
| q-series evaluation | Manual power series | `mpmath.nsum` or direct array ops | mpmath handles convergence and precision; numpy handles vectorized evaluation |

**Key insight:** This phase has 8 mathematical domains. The only way to cover them all is to use existing libraries for computational primitives and focus custom code on the domain-specific mathematical objects (operator construction, analogy mappings, Bost-Connes algebra) that no library provides.

## Common Pitfalls

### Pitfall 1: Scope Explosion Across 8 Domains
**What goes wrong:** Each domain is fascinating and deep. The spectral operator module alone could consume weeks of refinement. Multiply by 8 domains and the phase never completes.
**Why it happens:** Cross-disciplinary exploration is genuinely exciting, and each connection feels like "the one." This is the #1 risk flagged in PITFALLS.md and CONTEXT.md.
**How to avoid:** Define "minimally functional" for each module upfront. A spectral module is minimally functional when it constructs one operator variant, computes eigenvalues, and returns a chi-squared fit. A TDA module is minimally functional when it computes persistence diagrams for a point cloud. Build ALL minimally functional modules before deepening ANY single one.
**Warning signs:** More than 3 days spent on a single module before any other module exists; adding configuration options for scenarios not yet encountered.

### Pitfall 2: Berry-Keating Discretization Instability
**What goes wrong:** The Berry-Keating Hamiltonian H = xp is not compact, so naive discretization produces eigenvalues that depend strongly on matrix size and boundary conditions. Different discretization choices yield wildly different spectra that may or may not approximate zeta zeros.
**Why it happens:** The operator H = xp on L^2(R) has continuous spectrum. The connection to zeta zeros requires specific regularization (confinement potential, boundary conditions on [0, infinity)). The mathematical subtlety is in the regularization, not the matrix diagonalization.
**How to avoid:** Implement multiple regularization strategies (hard-wall box, smooth potential, Bender-Brody-Mueller approach) and compare. Report fit quality honestly -- if eigenvalues don't match zeros, that is a valid result. Use moderate matrix sizes (N=100-500) initially; convergence studies later.
**Warning signs:** Eigenvalues that change by large amounts when N changes by 10%; chi-squared fit that is much worse than random; spending more time on discretization than on analyzing results.

### Pitfall 3: LMFDB API Rate Limits and Data Volume
**What goes wrong:** LMFDB returns at most 100 results per query with a ~10,000 overall limit. Naive queries fetch too much or too little. API availability is not guaranteed (it is a community-run academic service).
**Why it happens:** LMFDB is not a commercial API with SLAs. It is a research database with modest infrastructure.
**How to avoid:** Cache aggressively in SQLite. Design queries to be specific (use `_fields` to fetch only needed columns). Handle HTTP errors gracefully (retry with exponential backoff, fall back to cached data). Test with small queries first. Store raw JSON responses for debugging.
**Warning signs:** Queries timing out; hitting the 100-result limit without pagination; cache growing to GBs from storing full records.

### Pitfall 4: p-adic Precision Semantics
**What goes wrong:** p-adic numbers have fundamentally different precision semantics from archimedean numbers. Precision is lost through addition (not multiplication), and "precision to N digits" means O(p^N), not N decimal digits. Mixing p-adic and archimedean precision concepts leads to confusion.
**Why it happens:** The project's precision infrastructure (precision_scope, validated_computation) is designed for archimedean (mpmath) arithmetic. p-adic precision is a different concept entirely.
**How to avoid:** The PadicNumber class must track its own precision independently of mpmath's dps. Document clearly: "p-adic precision N means known modulo p^N." Never convert p-adic to float for comparison -- compare in the p-adic metric.
**Warning signs:** PrecisionError from validated_computation when working with p-adics (wrong tool); p-adic values that change when increasing "precision" (may be correct behavior due to carries).

### Pitfall 5: Persistent Homology on Wrong Metric Space
**What goes wrong:** Persistence diagrams are sensitive to the choice of distance metric and point cloud embedding. Computing persistent homology of zeta zeros using the wrong embedding (e.g., just imaginary parts as 1D points) produces trivial results.
**Why it happens:** TDA requires thoughtful embedding of mathematical objects as point clouds in a metric space. The embedding choices (which features, what metric) determine what structure TDA can detect.
**How to avoid:** Use multiple embeddings from the Phase 2 embedding framework (spectral_basic, spectral_full, information_space, kitchen_sink). Compare persistence diagrams across embeddings. Non-trivial persistent features that appear in multiple embeddings are more likely genuine.
**Warning signs:** All persistence diagrams showing only short-lived features near the diagonal; identical diagrams from very different embeddings; features that disappear when the point cloud is permuted.

### Pitfall 6: Noncommutative Geometry Without Clear Computation Target
**What goes wrong:** The Bost-Connes system and Connes' spectral triple approach to RH are mathematically sophisticated but computationally opaque. Implementing "noncommutative geometry" without a concrete computation to perform results in a framework with no useful output.
**Why it happens:** The NCG approach to RH is a theoretical framework, not an algorithm. The connection between Bost-Connes partition function and the Riemann zeta function is exact but the computational implications for a proof are unclear.
**How to avoid:** Focus on concrete computations: (1) KMS states of the Bost-Connes system at various temperatures; (2) partition function matching zeta function values; (3) phase transition behavior. These are computable and produce data that can be compared with other domain outputs.
**Warning signs:** Code that defines algebraic structures but never evaluates them numerically; module with no test cases because "the math is too abstract to test."

## Code Examples

### Berry-Keating Hamiltonian Construction
```python
# Source: Berry-Keating (1999), discretization via finite differences
import numpy as np
from scipy.linalg import eigh

def construct_berry_keating_box(n: int, L: float = 10.0) -> np.ndarray:
    """Berry-Keating H = xp with hard-wall boundary conditions on [epsilon, L].

    Discretizes on a uniform grid with n points. The operator xp is
    symmetrized as (xp + px)/2 = xp + i*hbar/2.

    Args:
        n: Number of grid points.
        L: Right boundary (left boundary is L/n to avoid x=0 singularity).

    Returns:
        Real symmetric matrix (n x n).
    """
    # Grid points avoiding x=0
    epsilon = L / n
    x = np.linspace(epsilon, L, n)
    dx = x[1] - x[0]

    # p = -i d/dx discretized as central difference
    # xp symmetrized: H = (xp + px)/2 = -i(x d/dx + 1/2)
    # Finite difference: d/dx -> (f[i+1] - f[i-1]) / (2*dx)
    H = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            H[i, i - 1] = -x[i] / (2.0 * dx)
        if i < n - 1:
            H[i, i + 1] = x[i] / (2.0 * dx)

    # Symmetrize: H_sym = (H + H^T) / 2 gives real eigenvalues
    H_sym = (H + H.T) / 2.0
    return H_sym

def spectral_comparison(eigenvalues: np.ndarray, zero_heights: np.ndarray) -> dict:
    """Compare operator eigenvalues against zeta zero heights.

    Returns chi-squared fit and Kolmogorov-Smirnov test of spacing distributions.
    """
    from scipy.stats import ks_2samp

    # Normalize both to mean spacing 1
    eig_spacings = np.diff(np.sort(eigenvalues))
    if len(eig_spacings) > 0 and np.mean(eig_spacings) > 0:
        eig_spacings = eig_spacings / np.mean(eig_spacings)

    zero_spacings = np.diff(np.sort(zero_heights))
    if len(zero_spacings) > 0 and np.mean(zero_spacings) > 0:
        zero_spacings = zero_spacings / np.mean(zero_spacings)

    # Chi-squared on histograms
    bins = np.linspace(0, 4, 41)
    hist_eig, _ = np.histogram(eig_spacings, bins=bins, density=True)
    hist_zero, _ = np.histogram(zero_spacings, bins=bins, density=True)

    mask = hist_zero > 0
    chi2 = float(np.sum((hist_eig[mask] - hist_zero[mask])**2 / hist_zero[mask]))

    # KS test
    ks_stat, ks_pvalue = ks_2samp(eig_spacings, zero_spacings)

    return {
        "chi_squared": chi2,
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "n_eigenvalues": len(eigenvalues),
        "n_zeros": len(zero_heights),
    }
```

### Weil Explicit Formula Partial Sums
```python
# Source: Explicit formulae for L-functions (Riemann, Weil)
import mpmath
import numpy as np

def weil_explicit_psi(x: float, zeros: list[complex], n_terms: int | None = None) -> float:
    """Compute Chebyshev psi(x) via Weil's explicit formula using zero sum.

    psi(x) = x - sum_rho (x^rho / rho) - log(2*pi) - (1/2)*log(1 - x^{-2})

    Truncated to the first n_terms zeros (conjugate pairs).

    Args:
        x: Evaluation point (x > 1).
        zeros: List of zero heights (imaginary parts, positive only).
        n_terms: Number of zero pairs to include (default: all).

    Returns:
        Approximation to psi(x).
    """
    if n_terms is None:
        n_terms = len(zeros)

    # Main term
    result = x

    # Zero contribution: sum over conjugate pairs
    for t in zeros[:n_terms]:
        rho = complex(0.5, t)
        rho_conj = complex(0.5, -t)
        term = x**rho / rho + x**rho_conj / rho_conj
        result -= term.real

    # Trivial zero contribution and constant
    result -= np.log(2 * np.pi)
    if x > 1:
        result -= 0.5 * np.log(1 - x**(-2))

    return result
```

### PadicNumber Arithmetic
```python
# Source: p-adic number theory, Koblitz "p-adic Numbers"
from dataclasses import dataclass

@dataclass
class PadicNumber:
    """Element of Q_p to finite precision.

    Represents a = sum_{i=v}^{v+prec-1} a_i * p^i where v is the valuation.
    """
    p: int                    # prime
    digits: list[int]         # coefficients a_i, least significant first
    valuation: int            # p-adic valuation (order of p)
    precision: int            # number of known digits

    def __add__(self, other: "PadicNumber") -> "PadicNumber":
        """p-adic addition with carry propagation."""
        assert self.p == other.p
        # Align valuations and add digit-by-digit with carry
        # ... (implementation)
        pass

    def __mul__(self, other: "PadicNumber") -> "PadicNumber":
        """p-adic multiplication (convolution of digit sequences)."""
        assert self.p == other.p
        # Multiply as polynomials in p, reduce digits mod p
        # ... (implementation)
        pass

    def norm(self) -> float:
        """p-adic absolute value: |a|_p = p^{-v(a)}."""
        if not self.digits or all(d == 0 for d in self.digits):
            return 0.0
        return float(self.p ** (-self.valuation))

def kubota_leopoldt_zeta(s: int, p: int, precision: int = 20) -> PadicNumber:
    """Compute the Kubota-Leopoldt p-adic zeta function at negative odd integer s.

    Uses interpolation of Bernoulli numbers: zeta_p(1-n) = -(1 - p^{n-1}) * B_n / n
    for n >= 2 even.
    """
    # mpmath.bernoulli for arbitrary-precision Bernoulli numbers
    # Interpolate to get p-adic values
    pass
```

### Persistent Homology via ripser
```python
# Source: ripser documentation, scikit-tda
import numpy as np
from ripser import ripser

def compute_persistence(
    points: np.ndarray,
    max_dim: int = 2,
    max_edge: float | None = None,
) -> dict:
    """Compute persistent homology of a point cloud.

    Args:
        points: (N, D) array of points in R^D.
        max_dim: Maximum homology dimension (0=components, 1=loops, 2=voids).
        max_edge: Maximum edge length for Rips complex.

    Returns:
        Dict with keys "dgms" (persistence diagrams per dimension),
        "num_features" per dimension, "total_persistence" per dimension.
    """
    thresh = max_edge if max_edge is not None else np.inf
    result = ripser(points, maxdim=max_dim, thresh=thresh)

    diagrams = result["dgms"]
    summary = {
        "dgms": diagrams,
        "num_features": {},
        "total_persistence": {},
    }
    for dim, dgm in enumerate(diagrams):
        finite = dgm[dgm[:, 1] < np.inf]
        lifetimes = finite[:, 1] - finite[:, 0]
        summary["num_features"][dim] = len(finite)
        summary["total_persistence"][dim] = float(np.sum(lifetimes))

    return summary
```

### Bost-Connes System (Finite Approximation)
```python
# Source: Connes-Marcolli, "Noncommutative Geometry, Quantum Fields and Motives"
import numpy as np

def bost_connes_partition(beta: float, n_max: int = 100) -> float:
    """Compute Bost-Connes partition function Z(beta) = sum_{n=1}^{n_max} n^{-beta}.

    This equals the Riemann zeta function zeta(beta) for beta > 1.
    The KMS state at inverse temperature beta has partition function zeta(beta).

    Args:
        beta: Inverse temperature (must be > 1 for convergence).
        n_max: Truncation of the sum.

    Returns:
        Approximate partition function value.
    """
    return float(sum(n**(-beta) for n in range(1, n_max + 1)))

def bost_connes_kms_values(beta: float, n_max: int = 50) -> np.ndarray:
    """Compute KMS state expectation values for Bost-Connes generators.

    For beta > 1 (low temperature), the unique KMS_beta state gives:
    phi_beta(e_n) = n^{-beta} / zeta(beta)

    For 0 < beta <= 1 (high temperature), there is a unique KMS state
    but the symmetry group acts trivially.

    Phase transition at beta = 1 is the critical phenomenon.
    """
    Z = bost_connes_partition(beta, n_max)
    values = np.array([n**(-beta) / Z for n in range(1, n_max + 1)])
    return values
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| SageMath for modular forms | Direct q-series + LMFDB REST API | Project decision | Avoids 8GB dependency; q-series sufficient for exploration |
| giotto-tda for TDA | ripser + persim | ripser 0.6.14 (Dec 2025) | More actively maintained, MIT license, leaner |
| Custom p-adic from scratch | pyadic library OR minimal custom class | pyadic 0.3.0 (Mar 2026) | Custom class preferred for tighter mpmath integration |
| Manual Lyapunov computation | nolds library | nolds 0.6.3 (Nov 2025) | Validated algorithms (Rosenstein, Eckmann) |

**Deprecated/outdated:**
- giotto-tda's pyflagser dependency has had build issues on Windows; ripser avoids this entirely
- LMFDB's old MongoDB-based API format has been superseded by the current REST API with _format=json parameter

## Open Questions

1. **Berry-Keating optimal regularization**
   - What we know: Multiple regularization strategies exist (box, smooth potential, Bender-Brody-Mueller). Each gives different eigenvalue spectra.
   - What's unclear: Which regularization best approximates zeta zeros for moderate matrix sizes (N=100-500). This is an active research question.
   - Recommendation: Implement 2-3 regularizations, compare empirically. The comparison IS the science.

2. **LMFDB API stability and rate limits**
   - What we know: API is accessible, returns JSON, has 100-result pagination limit with ~10K overall cap.
   - What's unclear: Whether the API has rate limits, whether it handles sustained querying. Flagged as concern in STATE.md.
   - Recommendation: Aggressive SQLite caching from the start. First run fetches and caches; subsequent runs use cache. Include a `refresh_cache` parameter for explicit re-fetching.

3. **Analogy engine inference algorithm**
   - What we know: The AnalogyMapping structure is defined. Correspondences can be tested computationally.
   - What's unclear: How to computationally "fill in" unknowns in an analogy mapping. This is genuinely novel.
   - Recommendation: Start with brute-force search over candidate correspondences, scored by cross-domain metric correlation. Claude's judgment is the inference engine here -- the code provides the data for Claude to reason about.

4. **Noncommutative geometry: what is computable?**
   - What we know: Bost-Connes partition function = zeta function. KMS states exist. Phase transition at beta=1.
   - What's unclear: What computation beyond reproducing known zeta values advances the proof search.
   - Recommendation: Implement the finite-dimensional Bost-Connes system, compute KMS values, and look for surprises in the phase transition neighborhood. If no surprises emerge, this module stays minimal.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=9.0.2 |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `uv run pytest tests/test_analysis/test_spectral.py -x` |
| Full suite command | `uv run pytest tests/ -x --timeout=120` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SPEC-01 | Berry-Keating eigenvalues match known spectra; chi-squared fit | unit | `uv run pytest tests/test_analysis/test_spectral.py -x` | Wave 0 |
| SPEC-02 | Weil explicit formula partial sums converge to psi(x) | unit | `uv run pytest tests/test_analysis/test_trace_formula.py -x` | Wave 0 |
| MOD-01 | q-series coefficients match known modular form values | unit | `uv run pytest tests/test_analysis/test_modular_forms.py -x` | Wave 0 |
| MOD-02 | LMFDB queries return valid JSON; cache hits avoid HTTP | unit | `uv run pytest tests/test_analysis/test_lmfdb_client.py -x` | Wave 0 |
| ADEL-01 | p-adic addition/multiplication correct; Kubota-Leopoldt matches known values | unit | `uv run pytest tests/test_analysis/test_padic.py -x` | Wave 0 |
| ADEL-02 | Fractal tree produces valid matplotlib figure; dual-view data matches | unit | `uv run pytest tests/test_analysis/test_padic.py -x` | Wave 0 |
| XDISC-01 | AnalogyMapping serialization round-trip; correspondence test returns dict | unit | `uv run pytest tests/test_analysis/test_analogy.py -x` | Wave 0 |
| XDISC-02 | ripser produces correct persistence diagrams for known point clouds | unit | `uv run pytest tests/test_analysis/test_tda.py -x` | Wave 0 |
| XDISC-03 | Lyapunov exponent positive for known chaotic map; zero for regular | unit | `uv run pytest tests/test_analysis/test_dynamics.py -x` | Wave 0 |
| XDISC-04 | Bost-Connes partition function matches zeta(beta) to 10 digits | unit | `uv run pytest tests/test_analysis/test_ncg.py -x` | Wave 0 |
| RSRCH-03 | suggest_experiments returns well-formed dicts; generate_conjecture creates DB record | unit | `uv run pytest tests/test_analysis/test_conjecture_gen.py -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/test_analysis/ -x --timeout=60`
- **Per wave merge:** `uv run pytest tests/ -x --timeout=120`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_analysis/test_spectral.py` -- covers SPEC-01
- [ ] `tests/test_analysis/test_trace_formula.py` -- covers SPEC-02
- [ ] `tests/test_analysis/test_modular_forms.py` -- covers MOD-01
- [ ] `tests/test_analysis/test_lmfdb_client.py` -- covers MOD-02 (mock HTTP)
- [ ] `tests/test_analysis/test_padic.py` -- covers ADEL-01, ADEL-02
- [ ] `tests/test_analysis/test_tda.py` -- covers XDISC-02
- [ ] `tests/test_analysis/test_dynamics.py` -- covers XDISC-03
- [ ] `tests/test_analysis/test_ncg.py` -- covers XDISC-04
- [ ] `tests/test_analysis/test_analogy.py` -- covers XDISC-01
- [ ] `tests/test_analysis/test_conjecture_gen.py` -- covers RSRCH-03
- [ ] Framework install: `uv add ripser persim nolds requests` -- new dependencies

## Sources

### Primary (HIGH confidence)
- Existing codebase: `src/riemann/analysis/`, `src/riemann/workbench/`, `src/riemann/embedding/` -- patterns, APIs, integration points
- [ripser PyPI](https://pypi.org/project/ripser/) -- v0.6.14, Dec 2025, actively maintained, MIT license
- [persim PyPI](https://pypi.org/project/persim/) -- persistence diagram utilities
- [LMFDB API](https://www.lmfdb.org/api/) -- REST API confirmed accessible, JSON format, pagination limits verified
- [pyadic PyPI](https://pypi.org/project/pyadic/) -- v0.3.0, Mar 2026, GPL-3.0
- [nolds PyPI](https://pypi.org/project/nolds/) -- v0.6.3, Nov 2025, Lyapunov and nonlinear dynamics

### Secondary (MEDIUM confidence)
- [Berry-Keating (1999)](https://empslocal.ex.ac.uk/people/staff/mrwatkin/zeta/berry-keating1.pdf) -- Riemann zeros and eigenvalue asymptotics
- [Bender-Brody-Mueller (2017)](https://arxiv.org/abs/1608.03679) -- Hamiltonian for zeros of zeta
- [Kubota-Leopoldt construction](https://en.wikipedia.org/wiki/P-adic_L-function) -- p-adic interpolation of Bernoulli numbers
- [Weil explicit formula](https://en.wikipedia.org/wiki/Explicit_formulae_for_L-functions) -- zeros-to-primes duality
- [Bost-Connes system](https://en.wikipedia.org/wiki/Bost%E2%80%93Connes_system) -- quantum statistical mechanics and zeta
- [Selberg trace formula](https://en.wikipedia.org/wiki/Selberg_trace_formula) -- eigenvalues and geodesics
- [giotto-tda vs ripser comparison](https://pypi.org/project/giotto-tda/) -- giotto-tda 0.6.2 (May 2024, AGPLv3); ripser preferred

### Tertiary (LOW confidence)
- Connes-Marcolli NCG approach: theoretical framework is well-documented but computational implications for our platform are uncertain
- Analogy engine inference algorithms: no established methodology exists; this is genuinely novel territory

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries verified on PyPI with recent releases, existing codebase patterns well-understood
- Architecture: HIGH -- direct extension of established Phase 2 patterns (function-based modules, dataclass results, workbench integration)
- Individual domain mathematics: MEDIUM -- mathematical foundations are textbook but numerical implementation choices (discretization, truncation, precision) require empirical tuning
- Cross-disciplinary synthesis (analogy engine, AI conjecture): LOW -- genuinely novel territory; no established methodology
- Pitfalls: HIGH -- scope explosion is well-documented; domain-specific pitfalls based on numerical analysis literature

**Research date:** 2026-03-19
**Valid until:** 2026-04-19 (libraries are stable; mathematical content does not expire)
