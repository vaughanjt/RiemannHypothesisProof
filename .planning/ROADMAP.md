# Roadmap: Riemann

## Overview

This roadmap delivers a hybrid computational research platform for exploring the Riemann Hypothesis through unconventional cross-disciplinary lenses. The build order follows strict dependency flow: first establish reliable computation and visualization so the user can see and trust results, then layer on higher-dimensional analysis and cross-disciplinary modules that are the platform's unique differentiator, then add deep domain-specific modules and synthesis tools, and finally build the Lean 4 formalization pipeline only after genuine conjectures worth formalizing have emerged. Every phase delivers a coherent, usable research capability -- the platform is useful for mathematical exploration from Phase 1 onward.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Computational Foundation and Research Workbench** - Arbitrary-precision zeta computation, zero-finding, interactive visualization of the critical line and complex plane, and structured research tracking
- [ ] **Phase 2: Higher-Dimensional Analysis** - N-dimensional embedding of mathematical objects, projection theater, zero distribution statistics, random matrix comparison, and information-theoretic analysis
- [x] **Phase 3: Deep Domain Modules and Cross-Disciplinary Synthesis** - Spectral operators, trace formulas, modular forms, p-adic computation, analogy engine, topological data analysis, dynamical systems tools, and AI-guided conjecture generation (completed 2026-03-19)
- [x] **Phase 4: Lean 4 Formalization Pipeline** - Translate conjectures to Lean 4 theorem statements, track formalization progress, integrate with Mathlib (completed 2026-03-19)

## Phase Details

### Phase 1: Computational Foundation and Research Workbench
**Goal**: User can compute zeta function values and zeros to arbitrary precision, visualize results interactively, and track research progress in a structured workbench -- establishing a trusted computational foundation for all downstream exploration
**Depends on**: Nothing (first phase)
**Requirements**: COMP-01, COMP-02, COMP-03, COMP-04, VIZ-01, VIZ-02, RSRCH-01, RSRCH-02
**Success Criteria** (what must be TRUE):
  1. User can evaluate the Riemann zeta function at any complex point with configurable precision (up to thousands of digits) and see results validated against known values
  2. User can compute non-trivial zeros, store them in a persistent database, and verify them against published tables (Odlyzko)
  3. User can visualize |zeta(1/2+it)| along the critical line with interactive zoom/pan, and view domain coloring of the complex plane with zoomable critical strip regions
  4. User can create, annotate, and revisit experiments with full parameter reproducibility, and track conjectures with formal status and evidence levels
  5. User can stress-test any observed pattern against expanded data (more zeros, higher precision, varied parameters) to distinguish genuine structure from numerical artifacts
**Plans**: 5 plans

Plans:
- [x] 01-01-PLAN.md -- Project initialization, type system, precision management, test scaffolds
- [ ] 01-02-PLAN.md -- Zeta evaluation engine and zero computation/cataloging with Odlyzko validation
- [ ] 01-03-PLAN.md -- Related functions (Hardy Z, Dirichlet L, xi, Selberg stub) and stress-test framework
- [ ] 01-04-PLAN.md -- Critical line visualization and domain coloring
- [ ] 01-05-PLAN.md -- Research workbench: conjecture tracking and experiment reproducibility

### Phase 2: Higher-Dimensional Analysis
**Goal**: User can embed mathematical objects in N-dimensional spaces, explore them through interactive projections, and compare zero distributions against random matrix theory and information-theoretic baselines -- unlocking the platform's core differentiator of seeing structure invisible in lower dimensions
**Depends on**: Phase 1
**Requirements**: ZERO-01, ZERO-02, HDIM-01, HDIM-02, VIZ-03, RMT-01, RMT-02, INFO-01, INFO-02
**Success Criteria** (what must be TRUE):
  1. User can represent zeros (and other mathematical objects) as points in configurable N-dimensional feature spaces and apply multiple projection methods (PCA, UMAP, t-SNE, stereographic, custom) with side-by-side comparison
  2. User can interactively rotate through higher-dimensional mathematical spaces in a projection theater, watching how structures change across different 2D/3D views
  3. User can compute zero spacing statistics (nearest-neighbor, pair correlation, n-level density) and overlay them with GUE random matrix predictions in linked interactive views
  4. User can apply information-theoretic measures (entropy, mutual information, compression-based distances) to zero sequences and compare signatures across different mathematical objects to surface hidden structural similarities
  5. User sees automatic anomaly flags when zero distributions deviate from expected behavior (GUE statistics, Riemann-von Mangoldt formula)
**Plans**: 5 plans

Plans:
- [ ] 02-01-PLAN.md -- Zero statistics engine (spacing, pair correlation, n-level density) and embedding configuration registry
- [ ] 02-02-PLAN.md -- Random matrix theory laboratory (GUE/GOE/GSE generation, eigenvalue statistics, Wigner surmise)
- [ ] 02-03-PLAN.md -- Embedding pipeline (feature extractors, HDF5 storage) and projection methods (PCA, t-SNE, UMAP, stereographic)
- [ ] 02-04-PLAN.md -- Information-theoretic analysis (entropy, MI, LZ complexity) and SPC anomaly detection with workbench auto-logging
- [ ] 02-05-PLAN.md -- Projection theater (3D visualization, animation, dimension slicing) and comparison views (RMT overlay, info heatmap)

### Phase 3: Deep Domain Modules and Cross-Disciplinary Synthesis
**Goal**: User can explore the deepest cross-disciplinary connections to the Riemann Hypothesis -- spectral operators, trace formulas, modular forms, p-adic structures, topological invariants, dynamical systems, and noncommutative geometry -- with an analogy engine and AI-guided analysis that synthesize insights across all domains
**Depends on**: Phase 2
**Requirements**: SPEC-01, SPEC-02, MOD-01, MOD-02, ADEL-01, ADEL-02, XDISC-01, XDISC-02, XDISC-03, XDISC-04, RSRCH-03
**Success Criteria** (what must be TRUE):
  1. User can construct candidate self-adjoint operators (Berry-Keating Hamiltonian and variants), compute their eigenvalue spectra, and compare against zeta zeros with quantitative fit metrics
  2. User can explore trace formula connections (Selberg, Weil explicit formula) interactively, computing partial sums and visualizing the zeros-to-primes duality
  3. User can compute modular forms and Hecke eigenvalues, query LMFDB for known data, and perform p-adic arithmetic with visualized fractal structure connecting p-adic and archimedean pictures
  4. User can define analogy mappings between domains, apply topological data analysis to detect hidden structure in zero distributions, analyze zeta dynamics through dynamical systems tools, and compute in noncommutative geometric frameworks
  5. User can invoke AI-guided analysis that examines computational results across all domains, identifies cross-domain patterns, generates formal conjectures, and suggests next experiments
**Plans**: 5 plans

Plans:
- [ ] 03-01-PLAN.md -- Spectral operators (Berry-Keating Hamiltonian, spectrum comparison) and trace formulas (Weil explicit formula, Chebyshev psi)
- [ ] 03-02-PLAN.md -- Modular forms (q-series, Hecke eigenvalues) and LMFDB REST API client with SQLite caching
- [ ] 03-03-PLAN.md -- p-adic arithmetic with Kubota-Leopoldt zeta, TDA via ripser, and dynamical systems tools
- [ ] 03-04-PLAN.md -- Noncommutative geometry (Bost-Connes system) and analogy engine with correspondence testing
- [ ] 03-05-PLAN.md -- AI-guided conjecture generation and Phase 3 module integration wiring

### Phase 4: Lean 4 Formalization Pipeline
**Goal**: User can translate mature conjectures from the research workbench into machine-verified Lean 4 proofs, with Mathlib integration and progress tracking -- closing the loop from computational exploration to rigorous mathematics
**Depends on**: Phase 3
**Requirements**: FORM-01, FORM-02
**Success Criteria** (what must be TRUE):
  1. User can select a conjecture from the research workbench and translate it into a Lean 4 theorem statement (with sorry placeholders for unproven parts)
  2. User can track formalization progress per conjecture (statement formalized, proof attempted, proof complete) with Mathlib integration for leveraging existing formalized mathematics
  3. User can build and check Lean 4 proofs from within the platform, with structured error reporting and sorry-count tracking
**Plans**: 3 plans

Plans:
- [x] 04-01-PLAN.md -- WSL2 Lean 4 environment setup, build runner, and output parser
- [x] 04-02-PLAN.md -- Formalization tracker (state machine, sorry tracking, auto-promotion) and conjecture-to-Lean translator
- [x] 04-03-PLAN.md -- Triage module, package exports, and full formalization assault with human verification

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Computational Foundation and Research Workbench | 1/5 | In progress | - |
| 2. Higher-Dimensional Analysis | 4/5 | In Progress|  |
| 3. Deep Domain Modules and Cross-Disciplinary Synthesis | 0/5 | Complete    | 2026-03-19 |
| 4. Lean 4 Formalization Pipeline | 3/3 | Complete   | 2026-03-19 |
