# Roadmap: Riemann

## Milestones

- :white_check_mark: **v1.0 Research Platform** - Phases 1-4 (shipped 2026-03-19)
- :construction: **v2.0 The Modular Barrier** - Phases 5-8 (in progress)

## Phases

<details>
<summary>v1.0 Research Platform (Phases 1-4) - SHIPPED 2026-03-19</summary>

- [x] **Phase 1: Computational Foundation and Research Workbench** - Arbitrary-precision zeta computation, zero-finding, interactive visualization of the critical line and complex plane, and structured research tracking
- [x] **Phase 2: Higher-Dimensional Analysis** - N-dimensional embedding of mathematical objects, projection theater, zero distribution statistics, random matrix comparison, and information-theoretic analysis
- [x] **Phase 3: Deep Domain Modules and Cross-Disciplinary Synthesis** - Spectral operators, trace formulas, modular forms, p-adic computation, analogy engine, topological data analysis, dynamical systems tools, and AI-guided conjecture generation (completed 2026-03-19)
- [x] **Phase 4: Lean 4 Formalization Pipeline** - Translate conjectures to Lean 4 theorem statements, track formalization progress, integrate with Mathlib (completed 2026-03-19)

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

</details>

### v2.0 The Modular Barrier (In Progress)

**Milestone Goal:** Express the Connes barrier as a heat kernel trace on the modular surface SL(2,Z)\H where positivity is structural (each spectral term non-negative), then bound the corrections to close the proof of B(L) > 0 for all L.

- [ ] **Phase 5: Heat Kernel Feasibility Gate** - Compute heat kernel trace on SL(2,Z)\H, establish parameter mapping t(L), and validate that K(t) matches B(L) to 6+ digits -- kill the approach early if it fails
- [ ] **Phase 6: Selberg Trace Formula and Geometric Decomposition** - Implement the full Selberg trace formula with all four orbital contributions, verify spectral-geometric duality, and decompose the correction into identifiable geometric terms
- [ ] **Phase 7: Correction Bounds and Parallel Proof Paths** - Bound |K(t) - B(L)| with explicit constants, certify non-circularity, run Rankin-Selberg as parallel proof path, and evaluate CM points if modular structure is found
- [ ] **Phase 8: Proof Assembly and Lean 4 Formalization** - Assemble the rigorous informal proof with all explicit constants, then translate to Lean 4 theorem statements gated behind informal proof completion

## Phase Details

### Phase 5: Heat Kernel Feasibility Gate
**Goal**: User can compute the heat kernel trace on SL(2,Z)\H and confirm (or refute) that it matches the Connes barrier B(L) to 6+ significant figures -- providing a go/no-go decision for the entire v2.0 approach before investing in downstream phases
**Depends on**: Phase 4 (v1.0 platform complete)
**Requirements**: HEAT-01, HEAT-02, HEAT-03, HEAT-04
**Success Criteria** (what must be TRUE):
  1. User can compute the heat kernel trace Tr(e^{-t*Delta}) on SL(2,Z)\H with both the discrete Maass eigenvalue sum and the continuous Eisenstein spectrum contribution, and see convergence diagnostics for the truncated spectral sum
  2. User can run the barrier-to-heat-kernel comparison at 50+ values of L and see agreement to 6+ significant figures, with the parameter mapping t = t(L) identified and validated
  3. User can see the magnitude of the Eisenstein continuous spectrum contribution at each L value, compared against the 0.036 error budget threshold, with a clear verdict on whether the "manifestly positive heat kernel" argument survives
  4. User can toggle between mpmath exploratory precision and python-flint ball arithmetic for any computation, with dual-precision validation confirming that exploratory results are trustworthy
**Plans**: TBD

### Phase 6: Selberg Trace Formula and Geometric Decomposition
**Goal**: User can compute both sides of the Selberg trace formula for the Lorentzian test function, verify spectral-geometric duality to 10+ digits, and see the correction epsilon(L) decomposed into its four geometric contributions (identity, hyperbolic, elliptic, parabolic) with magnitudes established
**Depends on**: Phase 5 (parameter mapping t(L) needed for test function)
**Requirements**: SELB-01, SELB-02, SELB-03
**Success Criteria** (what must be TRUE):
  1. User can compute the Selberg trace formula on SL(2,Z)\H with all four orbital integral types and verify that spectral and geometric sides agree to 10+ significant digits at 20+ parameter values
  2. User can see confirmation (or a smoothed-variant workaround) that the Lorentzian test function h(r) = 1/(L^2+r^2) satisfies Selberg trace formula admissibility conditions, with decay bounds explicitly checked
  3. User can compute the primitive geodesic length spectrum, evaluate hyperbolic orbital integrals, and see the magnitude of each geometric contribution (identity, hyperbolic, elliptic, parabolic) to the correction epsilon(L)
**Plans**: TBD

### Phase 7: Correction Bounds and Parallel Proof Paths
**Goal**: User can either (a) see a rigorous bound |epsilon(L)| < K(t(L)) proving B(L) > 0 with explicit constants and certified non-circularity, or (b) see the Rankin-Selberg identification B(L) = L(1, f x f-bar) with positivity from Petersson norm, or (c) see a documented obstruction explaining exactly which step requires RH
**Depends on**: Phase 5, Phase 6 (geometric decomposition needed to identify which terms to bound)
**Requirements**: BOUND-01, BOUND-02, BOUND-03, BOUND-04
**Success Criteria** (what must be TRUE):
  1. User can see the correction |B(L) - K(t)| computed with explicit constants (no big-O) at every L value, with each error term allocated from the 0.036 budget and verified using python-flint ball arithmetic
  2. User can view a formal circularity dependency graph that traces every step from "heat kernel trace > 0" to "B(L) > 0" and certifies that no step assumes RH, zero-free regions, or GRH-conditional results
  3. User can compute Rankin-Selberg L-values L(1, f x f-bar) for low-lying Maass forms and see whether B(L) matches a Petersson norm, providing an independent parallel proof path
  4. User can evaluate the barrier at the nine Heegner CM points and see algebraic recognition results via PSLQ, conditional on Phase 7's Rankin-Selberg analysis finding modular structure
**Plans**: TBD

### Phase 8: Proof Assembly and Lean 4 Formalization
**Goal**: User has a complete rigorous proof document (informal) with all steps, explicit constants, and circularity certification -- and if the proof succeeds, Lean 4 theorem statements with Mathlib integration generated from the informal proof
**Depends on**: Phase 7 (proof path must be established)
**Requirements**: PROOF-01, PROOF-02
**Success Criteria** (what must be TRUE):
  1. User can generate a self-contained informal proof document that traces the complete chain from heat kernel positivity through correction bounds to B(L) > 0, with every constant explicit and every dependency certified non-circular
  2. User can translate the proof to Lean 4 theorem statements using the existing formalization pipeline with a new "heat_kernel" domain, with Mathlib integration for available lemmas and sorry placeholders for gaps -- gated behind successful completion of the informal proof
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Computational Foundation | v1.0 | 1/5 | In progress | - |
| 2. Higher-Dimensional Analysis | v1.0 | 4/5 | In Progress | - |
| 3. Deep Domain Modules | v1.0 | 0/5 | Complete | 2026-03-19 |
| 4. Lean 4 Formalization | v1.0 | 3/3 | Complete | 2026-03-19 |
| 5. Heat Kernel Feasibility Gate | v2.0 | 0/0 | Not started | - |
| 6. Selberg Trace Formula | v2.0 | 0/0 | Not started | - |
| 7. Correction Bounds | v2.0 | 0/0 | Not started | - |
| 8. Proof Assembly | v2.0 | 0/0 | Not started | - |
