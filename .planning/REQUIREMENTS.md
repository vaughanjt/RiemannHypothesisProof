# Requirements: Riemann

**Defined:** 2026-03-18
**Core Value:** Discover a novel proof pathway for the Riemann Hypothesis by exploring unconventional cross-disciplinary approaches, with computational tools that can operate in higher-dimensional spaces and project insights down to human-interpretable forms.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Computation

- [ ] **COMP-01**: User can evaluate the Riemann zeta function at any complex point to arbitrary precision (hundreds/thousands of digits)
- [ ] **COMP-02**: User can compute and catalog non-trivial zeros of the zeta function, with verification against known tables (Odlyzko)
- [ ] **COMP-03**: User can evaluate related functions (Dirichlet L-functions, Hardy's Z-function, xi function, Selberg zeta) to arbitrary precision
- [ ] **COMP-04**: User can stress-test any discovered pattern against expanded data (more zeros, higher precision, varied parameters) to distinguish genuine structure from numerical artifacts

### Visualization

- [ ] **VIZ-01**: User can visualize |zeta(1/2+it)| along the critical line with interactive zoom and pan
- [ ] **VIZ-02**: User can view domain coloring of the zeta function in the complex plane (phase→hue, magnitude→brightness) with zoomable critical strip regions
- [ ] **VIZ-03**: User can interactively rotate through higher-dimensional mathematical spaces, watching how structures project into different 2D/3D views (projection theater)

### Research Infrastructure

- [ ] **RSRCH-01**: User can track conjectures (formal statement, evidence for/against, status, confidence) in a structured research workbench
- [ ] **RSRCH-02**: User can save, annotate, and revisit experiments with full parameter reproducibility (seeds, serialization, checksums)
- [ ] **RSRCH-03**: User can invoke AI-guided analysis that examines computational results, identifies patterns, generates formal conjectures, and suggests next experiments

### Zero Analysis

- [ ] **ZERO-01**: User can compute zero distribution statistics (nearest-neighbor spacing, pair correlation, n-level density) and compare against GUE predictions
- [ ] **ZERO-02**: User can detect anomalies in zero structure — deviations from expected behavior (GUE statistics, Riemann-von Mangoldt formula) are automatically flagged

### Higher-Dimensional Framework

- [ ] **HDIM-01**: User can represent mathematical objects (zeros, function values, operator spectra) in N-dimensional spaces with configurable coordinate mappings
- [ ] **HDIM-02**: User can apply multiple projection methods (PCA, UMAP, t-SNE, stereographic, custom mathematical projections) and compare results side-by-side

### Spectral Theory

- [ ] **SPEC-01**: User can construct, discretize, and analyze candidate self-adjoint operators (Berry-Keating Hamiltonian and variants) and compare their eigenvalue spectra against zeta zeros
- [ ] **SPEC-02**: User can explore trace formula connections — Selberg trace formula, Weil explicit formula — interactively computing partial sums and visualizing zeros↔primes duality

### Random Matrix Theory

- [ ] **RMT-01**: User can generate GUE/GOE/GSE random matrix ensembles, compute eigenvalue statistics, and overlay with zeta zero statistics in interactive linked views
- [ ] **RMT-02**: User can vary matrix size and ensemble type and observe how the fit to zero statistics changes

### Information Theory

- [ ] **INFO-01**: User can apply information-theoretic measures (entropy, mutual information, Kolmogorov complexity estimates, compression-based distances) to zero sequences and related data
- [ ] **INFO-02**: User can compare information-theoretic signatures across different mathematical objects (zeros, eigenvalues, primes) to surface hidden structural similarities

### Modular Forms & Automorphic Representations

- [ ] **MOD-01**: User can compute modular forms, Hecke eigenvalues, and Fourier coefficients, and visualize them in the upper half-plane
- [ ] **MOD-02**: User can query LMFDB for known L-function data, modular form data, and number field data and integrate it with the platform's analysis tools

### Adelic & p-adic Computation

- [ ] **ADEL-01**: User can perform p-adic arithmetic and compute p-adic zeta functions for various primes
- [ ] **ADEL-02**: User can visualize p-adic structures (fractal geometry) and connect p-adic and archimedean pictures of zeta

### Cross-Disciplinary Synthesis

- [ ] **XDISC-01**: User can define and test analogy mappings between domains (e.g., {eigenvalues:zeros, Hamiltonian:???, trace formula:explicit formula}) and computationally explore unknown correspondences
- [ ] **XDISC-02**: User can apply topological data analysis (persistent homology) to zero distributions and other mathematical objects to detect hidden topological structure
- [ ] **XDISC-03**: User can analyze the zeta function and zero dynamics through dynamical systems tools (Lyapunov exponents, phase portraits, strange attractors)
- [ ] **XDISC-04**: User can compute in noncommutative geometric frameworks relevant to Connes' approach to RH

### Formalization

- [ ] **FORM-01**: User can translate conjectures from the research workbench into Lean 4 theorem statements
- [ ] **FORM-02**: User can track formalization progress (statement formalized → proof attempted → proof complete) with Mathlib integration for existing formalized mathematics

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Advanced Visualization

- **AVIZ-01**: Real-time WebGL rendering for projection theater (v1 uses Plotly 3D)
- **AVIZ-02**: VR/AR integration for immersive higher-dimensional exploration

### Advanced Computation

- **ACOMP-01**: GPU-accelerated zero computation at scale (billions of zeros)
- **ACOMP-02**: Distributed computation across multiple machines

### Collaboration

- **COLLAB-01**: Export research sessions for sharing with collaborators
- **COLLAB-02**: Standardized conjecture format for publication

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| General-purpose CAS | SageMath, Mathematica, SymPy exist — reimplementing symbolic algebra is a multi-decade project |
| General-purpose proof assistant | Lean 4 exists and is actively maintained — we build a pipeline TO it, not a replacement |
| Reimplemented zeta function | mpmath and Arb have decades of work; only implement custom evaluation for novel function variants |
| Classical proof strategy tooling | Deliberately avoided — well-studied approaches that haven't worked |
| Web deployment / multi-user | Single-user local research tool; web adds complexity with zero benefit |
| Publication/typesetting | LaTeX and Overleaf exist; irrelevant to proof discovery |
| Teaching/tutorial system | Claude is the teacher on demand; no curriculum infrastructure |
| GPU-accelerated mass zero verification | Use published databases (LMFDB, Odlyzko) for large datasets; compute targeted zeros only |
| Mobile app | Desktop research tool |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| COMP-01 | — | Pending |
| COMP-02 | — | Pending |
| COMP-03 | — | Pending |
| COMP-04 | — | Pending |
| VIZ-01 | — | Pending |
| VIZ-02 | — | Pending |
| VIZ-03 | — | Pending |
| RSRCH-01 | — | Pending |
| RSRCH-02 | — | Pending |
| RSRCH-03 | — | Pending |
| ZERO-01 | — | Pending |
| ZERO-02 | — | Pending |
| HDIM-01 | — | Pending |
| HDIM-02 | — | Pending |
| SPEC-01 | — | Pending |
| SPEC-02 | — | Pending |
| RMT-01 | — | Pending |
| RMT-02 | — | Pending |
| INFO-01 | — | Pending |
| INFO-02 | — | Pending |
| MOD-01 | — | Pending |
| MOD-02 | — | Pending |
| ADEL-01 | — | Pending |
| ADEL-02 | — | Pending |
| XDISC-01 | — | Pending |
| XDISC-02 | — | Pending |
| XDISC-03 | — | Pending |
| XDISC-04 | — | Pending |
| FORM-01 | — | Pending |
| FORM-02 | — | Pending |

**Coverage:**
- v1 requirements: 30 total
- Mapped to phases: 0
- Unmapped: 30 ⚠️

---
*Requirements defined: 2026-03-18*
*Last updated: 2026-03-18 after initial definition*
