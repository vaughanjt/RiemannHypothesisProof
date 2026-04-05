# Requirements: Riemann

**Defined:** 2026-03-18
**Core Value:** Discover a novel proof pathway for the Riemann Hypothesis by exploring unconventional cross-disciplinary approaches, with computational tools that can operate in higher-dimensional spaces and project insights down to human-interpretable forms.

## v2.0 Requirements — The Modular Barrier

Requirements for proving RH via heat kernel positivity on the modular surface. Each maps to roadmap phases.

### Heat Kernel Foundation

- [x] **HEAT-01**: User can compute the heat kernel trace Tr(e^{-tΔ}) on SL(2,Z)\H including both discrete Maass eigenvalue sum and continuous Eisenstein spectrum
- [ ] **HEAT-02**: User can identify the precise parameter mapping t = t(L) between barrier parameter L and heat kernel time t, validated by numerical agreement to 6+ digits
- [x] **HEAT-03**: User can compute the discrete spectral sum over Maass form eigenvalues using LMFDB data, with configurable truncation and convergence diagnostics
- [x] **HEAT-04**: User can run dual-precision computation: mpmath for exploratory evaluation and python-flint certified ball arithmetic for rigorous bounds

### Selberg Trace Formula

- [ ] **SELB-01**: User can compute the Selberg trace formula on SL(2,Z)\H with all four orbital integral types (identity, hyperbolic, elliptic, parabolic)
- [ ] **SELB-02**: User can verify that the Lorentzian test function satisfies Selberg trace formula admissibility conditions (Paley-Wiener, decay bounds)
- [ ] **SELB-03**: User can compute the primitive geodesic length spectrum for SL(2,Z)\H and evaluate hyperbolic orbital integrals

### Verification & Bounds

- [ ] **BOUND-01**: User can compute the correction |B(L) - K(t)| with explicit constants (no big-O), verified against known barrier values
- [ ] **BOUND-02**: User can generate a formal circularity dependency graph proving no step in the proof chain assumes RH
- [ ] **BOUND-03**: User can compute Rankin-Selberg L-values L(1, f×f̄) and test whether B(L) matches a Petersson norm (parallel proof path)
- [ ] **BOUND-04**: User can evaluate the barrier at CM points (Heegner discriminants) and recognize algebraic values via PSLQ

### Proof Assembly

- [ ] **PROOF-01**: User can generate a rigorous informal proof document with all steps, explicit constants, bounds, and circularity certification
- [ ] **PROOF-02**: User can translate the proof to Lean 4 theorem statements with Mathlib integration, gated behind informal proof completion

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Computation

- [x] **COMP-01**: User can evaluate the Riemann zeta function at any complex point to arbitrary precision (hundreds/thousands of digits)
- [x] **COMP-02**: User can compute and catalog non-trivial zeros of the zeta function, with verification against known tables (Odlyzko)
- [x] **COMP-03**: User can evaluate related functions (Dirichlet L-functions, Hardy's Z-function, xi function, Selberg zeta) to arbitrary precision
- [x] **COMP-04**: User can stress-test any discovered pattern against expanded data (more zeros, higher precision, varied parameters) to distinguish genuine structure from numerical artifacts

### Visualization

- [x] **VIZ-01**: User can visualize |zeta(1/2+it)| along the critical line with interactive zoom and pan
- [x] **VIZ-02**: User can view domain coloring of the zeta function in the complex plane (phase->hue, magnitude->brightness) with zoomable critical strip regions
- [x] **VIZ-03**: User can interactively rotate through higher-dimensional mathematical spaces, watching how structures project into different 2D/3D views (projection theater)

### Research Infrastructure

- [x] **RSRCH-01**: User can track conjectures (formal statement, evidence for/against, status, confidence) in a structured research workbench
- [x] **RSRCH-02**: User can save, annotate, and revisit experiments with full parameter reproducibility (seeds, serialization, checksums)
- [x] **RSRCH-03**: User can invoke AI-guided analysis that examines computational results, identifies patterns, generates formal conjectures, and suggests next experiments

### Zero Analysis

- [x] **ZERO-01**: User can compute zero distribution statistics (nearest-neighbor spacing, pair correlation, n-level density) and compare against GUE predictions
- [ ] **ZERO-02**: User can detect anomalies in zero structure -- deviations from expected behavior (GUE statistics, Riemann-von Mangoldt formula) are automatically flagged

### Higher-Dimensional Framework

- [x] **HDIM-01**: User can represent mathematical objects (zeros, function values, operator spectra) in N-dimensional spaces with configurable coordinate mappings
- [x] **HDIM-02**: User can apply multiple projection methods (PCA, UMAP, t-SNE, stereographic, custom mathematical projections) and compare results side-by-side

### Spectral Theory

- [x] **SPEC-01**: User can construct, discretize, and analyze candidate self-adjoint operators (Berry-Keating Hamiltonian and variants) and compare their eigenvalue spectra against zeta zeros
- [x] **SPEC-02**: User can explore trace formula connections -- Selberg trace formula, Weil explicit formula -- interactively computing partial sums and visualizing zeros-to-primes duality

### Random Matrix Theory

- [x] **RMT-01**: User can generate GUE/GOE/GSE random matrix ensembles, compute eigenvalue statistics, and overlay with zeta zero statistics in interactive linked views
- [x] **RMT-02**: User can vary matrix size and ensemble type and observe how the fit to zero statistics changes

### Information Theory

- [ ] **INFO-01**: User can apply information-theoretic measures (entropy, mutual information, Kolmogorov complexity estimates, compression-based distances) to zero sequences and related data
- [x] **INFO-02**: User can compare information-theoretic signatures across different mathematical objects (zeros, eigenvalues, primes) to surface hidden structural similarities

### Modular Forms & Automorphic Representations

- [x] **MOD-01**: User can compute modular forms, Hecke eigenvalues, and Fourier coefficients, and visualize them in the upper half-plane
- [x] **MOD-02**: User can query LMFDB for known L-function data, modular form data, and number field data and integrate it with the platform's analysis tools

### Adelic & p-adic Computation

- [x] **ADEL-01**: User can perform p-adic arithmetic and compute p-adic zeta functions for various primes
- [x] **ADEL-02**: User can visualize p-adic structures (fractal geometry) and connect p-adic and archimedean pictures of zeta

### Cross-Disciplinary Synthesis

- [x] **XDISC-01**: User can define and test analogy mappings between domains (e.g., {eigenvalues:zeros, Hamiltonian:???, trace formula:explicit formula}) and computationally explore unknown correspondences
- [x] **XDISC-02**: User can apply topological data analysis (persistent homology) to zero distributions and other mathematical objects to detect hidden topological structure
- [x] **XDISC-03**: User can analyze the zeta function and zero dynamics through dynamical systems tools (Lyapunov exponents, phase portraits, strange attractors)
- [x] **XDISC-04**: User can compute in noncommutative geometric frameworks relevant to Connes' approach to RH

### Formalization

- [x] **FORM-01**: User can translate conjectures from the research workbench into Lean 4 theorem statements
- [x] **FORM-02**: User can track formalization progress (statement formalized -> proof attempted -> proof complete) with Mathlib integration for existing formalized mathematics

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
| General-purpose CAS | SageMath, Mathematica, SymPy exist -- reimplementing symbolic algebra is a multi-decade project |
| General-purpose proof assistant | Lean 4 exists and is actively maintained -- we build a pipeline TO it, not a replacement |
| Reimplemented zeta function | mpmath and Arb have decades of work; only implement custom evaluation for novel function variants |
| Classical proof strategy tooling | Deliberately avoided -- well-studied approaches that haven't worked |
| Web deployment / multi-user | Single-user local research tool; web adds complexity with zero benefit |
| Publication/typesetting | LaTeX and Overleaf exist; irrelevant to proof discovery |
| Teaching/tutorial system | Claude is the teacher on demand; no curriculum infrastructure |
| GPU-accelerated mass zero verification | Use published databases (LMFDB, Odlyzko) for large datasets; compute targeted zeros only |
| Mobile app | Desktop research tool |
| Direct analytic proof of B(L) > 0 | Sessions 35-42 proved every direct approach circular |
| Reimplementing Hejhal's algorithm | Use LMFDB tabulated Maass eigenvalues instead |
| SageMath dependency | 2+ GB footprint, venv incompatible; python-flint covers needed functions |
| Full Lean 4 proof without informal proof | Formalizing false identity wastes months; gate behind PROOF-01 |
| Compact quotient Gamma(N)\H approach | Interesting but orthogonal; save for v3.0 if needed |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| COMP-01 | Phase 1 | Complete |
| COMP-02 | Phase 1 | Complete |
| COMP-03 | Phase 1 | Complete |
| COMP-04 | Phase 1 | Complete |
| VIZ-01 | Phase 1 | Complete |
| VIZ-02 | Phase 1 | Complete |
| VIZ-03 | Phase 2 | Complete |
| RSRCH-01 | Phase 1 | Complete |
| RSRCH-02 | Phase 1 | Complete |
| RSRCH-03 | Phase 3 | Complete |
| ZERO-01 | Phase 2 | Complete |
| ZERO-02 | Phase 2 | Pending |
| HDIM-01 | Phase 2 | Complete |
| HDIM-02 | Phase 2 | Complete |
| SPEC-01 | Phase 3 | Complete |
| SPEC-02 | Phase 3 | Complete |
| RMT-01 | Phase 2 | Complete |
| RMT-02 | Phase 2 | Complete |
| INFO-01 | Phase 2 | Pending |
| INFO-02 | Phase 2 | Complete |
| MOD-01 | Phase 3 | Complete |
| MOD-02 | Phase 3 | Complete |
| ADEL-01 | Phase 3 | Complete |
| ADEL-02 | Phase 3 | Complete |
| XDISC-01 | Phase 3 | Complete |
| XDISC-02 | Phase 3 | Complete |
| XDISC-03 | Phase 3 | Complete |
| XDISC-04 | Phase 3 | Complete |
| FORM-01 | Phase 4 | Complete |
| FORM-02 | Phase 4 | Complete |
| HEAT-01 | Phase 5 | Complete |
| HEAT-02 | Phase 5 | Pending |
| HEAT-03 | Phase 5 | Complete |
| HEAT-04 | Phase 5 | Complete |
| SELB-01 | Phase 6 | Pending |
| SELB-02 | Phase 6 | Pending |
| SELB-03 | Phase 6 | Pending |
| BOUND-01 | Phase 7 | Pending |
| BOUND-02 | Phase 7 | Pending |
| BOUND-03 | Phase 7 | Pending |
| BOUND-04 | Phase 7 | Pending |
| PROOF-01 | Phase 8 | Pending |
| PROOF-02 | Phase 8 | Pending |

**Coverage:**
- v1 requirements: 30 total (28 complete, 2 pending)
- v2.0 requirements: 13 total (13 mapped)
- Mapped to phases: 30 (v1) + 13 (v2.0) = 43 total
- Unmapped: 0

---
*Requirements defined: 2026-03-18*
*Last updated: 2026-04-04 after v2.0 roadmap creation*
