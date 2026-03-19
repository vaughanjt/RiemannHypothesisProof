# Phase 3: Deep Domain Modules and Cross-Disciplinary Synthesis - Context

**Gathered:** 2026-03-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver spectral operator analysis (Berry-Keating and variants), trace formula workbench (Selberg, Weil explicit formula), modular forms toolkit with LMFDB integration, p-adic arithmetic and visualization, cross-disciplinary analogy engine, topological data analysis (persistent homology), dynamical systems tools, noncommutative geometry toolkit, and AI-guided conjecture generation. All follow the pluggable analysis module pattern from Phase 2. This phase unlocks the full cross-disciplinary arsenal.

</domain>

<decisions>
## Implementation Decisions

### Module Architecture
- All modules follow the established Phase 2 pattern: function-based API, returns data (not plots), pluggable into the analysis pipeline
- Each domain module is independent — can be developed and tested in isolation
- Modules integrate with existing infrastructure: workbench for tracking, anomaly system for flagging, embedding framework for higher-dimensional views
- New visualization functions go in `viz/` following the comparison.py pattern from Phase 2

### Spectral Operators (SPEC-01, SPEC-02)
- Construct Berry-Keating Hamiltonian H = xp (and regularized variants) as finite-dimensional matrices
- Eigenvalue computation via numpy.linalg for moderate sizes, scipy.sparse.linalg for large matrices
- Quantitative fit metrics: chi-squared distance between eigenvalue spacings and zero spacings
- Trace formula module: compute partial sums of Riemann-von Mangoldt and Weil explicit formulas
- Interactive visualization of zeros↔primes duality through explicit formula truncation

### Modular Forms & LMFDB (MOD-01, MOD-02)
- Compute modular forms using direct Fourier expansion (q-series) — no SageMath dependency
- Hecke eigenvalues via matrix representation on spaces of modular forms
- LMFDB integration via their REST API (https://www.lmfdb.org/api/) for L-function data, modular form data, number field data
- Upper half-plane visualization using domain coloring adapted from Phase 1
- Cache LMFDB responses locally (SQLite) to avoid repeated API calls

### Adelic & p-adic (ADEL-01, ADEL-02)
- Custom p-adic number class: represents elements of Q_p to configurable precision
- p-adic zeta functions via interpolation of Bernoulli numbers (Kubota-Leopoldt)
- Visualization of p-adic structures using fractal tree layout (standard p-adic visualization)
- Connect p-adic and archimedean pictures: same L-function data shown in both completions

### Cross-Disciplinary Synthesis (XDISC-01 through XDISC-04)
- Analogy engine: formal `AnalogyMapping` structure {source_domain, target_domain, correspondences: dict, unknowns: list, evidence: list}
- Compute partial analogies: given known correspondences, test computationally whether unknown entries can be inferred
- TDA via giotto-tda or ripser (persistent homology): compute persistence diagrams of zero point clouds in various embeddings
- Dynamical systems: iterate zeta-related maps, compute Lyapunov exponents, detect fixed points and periodic orbits
- Noncommutative geometry: implement Connes' spectral triple for the integers (Bost-Connes system), compute KMS states

### AI-Guided Conjecture Generation (RSRCH-03)
- Claude operates as an **active scientist**: proactively forms hypotheses, designs experiments, pursues leads
- Structured conjecture pipeline: observation → hypothesis → experiment design → execution → evidence evaluation → conjecture formalization
- `suggest_experiments(context)` function that examines workbench state (conjectures, anomalies, experiments) and proposes next steps
- `analyze_results(experiment_id)` function that interprets experiment outcomes and updates conjecture confidence
- `generate_conjecture(observations)` function that synthesizes patterns into formal conjecture statements
- All suggestions tracked in workbench with evidence chains
- Integration with anomaly system: newly flagged anomalies trigger suggestion generation

### Carrying Forward from Phases 1-2
- JupyterLab, Claude-driven exploration
- Speed over visual polish, Claude picks per-module tooling
- SQLite + numpy + HDF5, strict evidence hierarchy
- 50-digit default, always-validate precision
- Function-based API, pluggable module pattern
- "Surprise me" — widest net, no bias toward expected structure

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

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Context
- `.planning/PROJECT.md` — Core value, constraints, key decisions
- `.planning/REQUIREMENTS.md` — SPEC-01, SPEC-02, MOD-01, MOD-02, ADEL-01, ADEL-02, XDISC-01..04, RSRCH-03

### Prior Phases
- `.planning/phases/01-computational-foundation-and-research-workbench/01-CONTEXT.md` — Phase 1 decisions
- `.planning/phases/02-higher-dimensional-analysis/02-CONTEXT.md` — Phase 2 decisions

### Research
- `.planning/research/STACK.md` — Technology choices
- `.planning/research/FEATURES.md` — Feature details for D2, D4, D5, D6, D7, D10, D12
- `.planning/research/ARCHITECTURE.md` — Pluggable module architecture
- `.planning/research/PITFALLS.md` — Scope explosion in cross-disciplinary approaches, infrastructure addiction

### Existing Code (Phase 1-2 outputs)
- `src/riemann/engine/` — Computation engine (zeta, zeros, lfunctions, precision, validation)
- `src/riemann/analysis/` — Statistics (spacing, rmt, information, anomaly)
- `src/riemann/embedding/` — Coordinate extraction, HDF5 storage, registry
- `src/riemann/viz/` — Visualization (critical_line, domain_coloring, projection, theater, comparison)
- `src/riemann/workbench/` — Conjecture CRUD, experiment save/load, evidence chains
- `src/riemann/types.py` — Shared types (ZetaZero, EvidenceLevel, etc.)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `engine/precision.py`: validated_computation, precision_scope — reuse for all new computations
- `analysis/spacing.py`: normalized_spacings — input for spectral operator comparison
- `analysis/rmt.py`: generate_gue, eigenvalue_spacings — comparison baseline for operators
- `analysis/anomaly.py`: detect_anomalies, log_anomalies_to_workbench — auto-flag unusual results
- `analysis/information.py`: cross_object_comparison — extend to new mathematical objects
- `embedding/coordinates.py`: FEATURE_EXTRACTORS registry — add new extractors from domain modules
- `embedding/storage.py`: HDF5 save/load — store new embeddings
- `viz/comparison.py`: create_spacing_comparison — template for new comparison views
- `viz/domain_coloring.py`: dual-mode pattern — adapt for upper half-plane visualization
- `workbench/conjecture.py`: create_conjecture — AI conjecture generation target
- `workbench/experiment.py`: save/load experiments — store analogy mappings and TDA results

### Established Patterns
- Function-based API, no classes (except dataclasses for data)
- TDD with pytest red-green cycle
- Dual-mode computation (fast numpy + precise mpmath)
- Pluggable module pattern: independent modules, shared integration points

### Integration Points
- New domain modules → analysis/ directory (spectral, modular, padic, tda, dynamics, ncg)
- New visualizations → viz/ directory
- Analogy engine → workbench (store mappings as experiments)
- AI conjecture generation → workbench (read anomalies + experiments, write conjectures)
- TDA results → embedding framework (persistence diagrams as new embedding coordinates)

</code_context>

<specifics>
## Specific Ideas

- Claude as active scientist: the AI conjecture module should support autonomous exploration loops where Claude examines the current state of knowledge, identifies the most promising lead, designs and runs an experiment, and updates the research state — without waiting to be asked
- The analogy engine is potentially the most powerful tool: if we can formalize the structure-preserving maps between spectral theory, RMT, and zeta zeros, the gaps in those maps point directly at what's missing from a proof
- Pitfall warning from research: this phase has 11 requirements across 8 different mathematical domains — maintain the 70/30 exploration-to-infrastructure ratio, don't build perfect frameworks for each domain before doing actual mathematics

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-deep-domain-modules-and-cross-disciplinary-synthesis*
*Context gathered: 2026-03-19*
