# Project Research Summary

**Project:** Riemann -- Hybrid Computational Math Research Platform & Formal Proof Workbench
**Domain:** Exploratory mathematical research platform targeting the Riemann Hypothesis
**Researched:** 2026-03-18
**Confidence:** MEDIUM (all research from training data; web verification unavailable; core library recommendations are stable and HIGH confidence; specific versions and Lean 4/Mathlib ecosystem details are MEDIUM)

## Executive Summary

Riemann is a single-user computational research workbench designed to explore the Riemann Hypothesis through unconventional cross-disciplinary lenses: random matrix theory, spectral operator theory, higher-dimensional geometry, information-theoretic analysis, and formal verification. No existing tool unifies arbitrary-precision zeta computation, interactive visualization, cross-disciplinary analysis pipelines, structured research tracking, AI-guided conjecture generation, and Lean 4 formalization in one place. The recommended approach is a layered Python platform built around mpmath (arbitrary-precision arithmetic) and Lean 4 (formal verification), delivered through JupyterLab notebooks as the primary interface. The entire stack is built from mature, stable libraries -- there are no experimental technology bets required.

The architecture must be built bottom-up following strict dependency order: computation engine first, then visualization, then cross-disciplinary analysis, then higher-dimensional exploration, then Lean 4 formalization. The project's unique differentiator -- the higher-dimensional projection framework and cross-disciplinary analogy engine -- cannot be attempted until the foundational computation and analysis infrastructure is solid. The Lean 4 formalization layer is the final phase and should not begin until the exploration pipeline has produced genuine conjectures worth formalizing. Premature formalization is a critical failure mode documented extensively by the Lean community.

The three most dangerous risks are (1) silent precision collapse in zeta function evaluation if float64/scipy is used naively near the critical strip -- this produces phantom patterns with no warning and can contaminate weeks of exploration; (2) the infrastructure trap of building a perfect framework while never doing actual mathematical exploration, which is particularly dangerous for this project because software engineering feels productive while mathematical exploration feels uncertain; and (3) confusing numerical computational evidence with mathematical proof, which the history of analytic number theory shows can be misleading even across astronomically large test ranges. All three must be designed against from day one.

## Key Findings

### Recommended Stack

The Python scientific stack around mpmath is the only viable choice for this project. No alternative provides the combination of arbitrary-precision special functions (essential for zeta evaluation near the critical line), machine-precision bulk computation (essential for random matrix ensembles and spectral analysis), symbolic manipulation, and interactive visualization that this project requires. The stack is mature, stable, and well-documented. SageMath should be explicitly avoided as a primary dependency -- it conflicts with modern Python packaging (uv, virtual environments), has inferior notebook integration, and its constituent libraries (mpmath, SymPy, NumPy/SciPy) are better used directly without the 8GB dependency weight.

For formal verification, Lean 4 with Mathlib4 is the unambiguous choice given the active mathematical formalization community, dependent type theory expressiveness, and the best-maintained mathematical library (100,000+ theorems). The Lean 4 project lives in a separate `lean/` subdirectory and communicates with Python exclusively through file-based exchange (Python writes `.lean` files, invokes lake CLI, parses structured output). There is no stable Python-Lean FFI and none should be attempted.

**Core technologies:**
- **mpmath + gmpy2**: Arbitrary-precision arithmetic and zeta evaluation -- the only production-quality option in Python; gmpy2 provides 2-10x acceleration as mpmath's C backend; use for all critical strip evaluation
- **NumPy + SciPy**: Machine-precision bulk computation for random matrix ensembles, spectral analysis, and FFTs -- non-negotiable scientific Python foundation; use where float64 (~15 digits) suffices
- **SymPy**: Symbolic computation for formula derivation and algebraic manipulation before numerical evaluation
- **Matplotlib + Plotly + ipywidgets**: Static 2D publication plots (Matplotlib), interactive 3D visualization (Plotly), and parameter controls (ipywidgets); Plotly is essential for the higher-dimensional projection theater
- **JupyterLab**: Primary user interface for all exploration; required, not optional; classic Jupyter Notebook is in maintenance mode and must not be used
- **Lean 4 + Mathlib4**: Formal verification layer; installed via elan/lake; Mathlib ~3GB compiled; use `lake exe cache get` to avoid 30-60 minute from-source builds
- **uv**: Modern Python package and project management replacing pip+venv; 10-100x faster dependency resolution with proper lockfile support
- **numba**: JIT compilation for machine-precision numerical loops; does NOT work with mpmath; use only for float64 bottlenecks in random matrix sampling or Monte Carlo
- **SQLite + h5py**: Zero database (SQLite, zero-config, handles millions of rows) and large array storage (h5py/HDF5)
- **pandas**: Tabular organization of computed results as the intermediate format between computation and visualization
- **scikit-learn + umap-learn**: PCA, t-SNE, UMAP for dimensionality reduction in the N-dimensional projection pipeline
- **panel (HoloViz)**: Dashboard and widget framework for Jupyter; preferred over Streamlit (forces separate server) and Dash (Plotly-only)
- **Ruff + pytest**: Python linting/formatting and testing; pytest validates computational modules against known zero values

### Expected Features

The MVP must answer: "Can this platform show me something about the Riemann zeta function that I cannot trivially see in Mathematica or SageMath?" The critical path running T1 (zeta evaluation) --> T2 (zero computation) --> T8 (zero statistics) --> D3 (RMT comparison) is the longest dependency chain gating meaningful research. Nothing downstream functions without this foundation.

**Must have (table stakes -- Phase 1):**
- Arbitrary-precision zeta function evaluation (T1) -- via mpmath wrapping with unified backend API; never float64 for critical strip evaluation
- Non-trivial zero computation (T2) -- mpmath.zetazero() initially, with SQLite caching and validation against Odlyzko tables
- Critical line visualization + domain coloring (T3, T4) -- interactive zoom/pan with parameter sliders; perceptually uniform colormaps only
- Session-based research workbench -- structured experiment logging, annotation, evidence-level tagging from day one; not bolted on later
- Zero distribution statistics (T8) -- nearest-neighbor spacing, GUE comparison; the first cross-disciplinary connection and genuinely differentiating result

**Should have (competitive differentiators -- Phases 2-4):**
- Higher-dimensional computation framework (D1) -- even a prototype delivers unique value: zeros as R^k feature points, PCA/t-SNE/UMAP projection, structure detection
- Random matrix theory laboratory (D3) -- generate GUE/GOE ensembles, compare eigenvalue statistics to zero statistics interactively side-by-side
- Spectral operator analysis (D2) -- construct and diagonalize candidate operators (Berry-Keating Hamiltonian); compare eigenvalue spectra to zeros
- Related function evaluation (T7) -- Dirichlet L-functions, xi function, Hardy's Z-function; essential for cross-disciplinary connections
- Numerical verification framework (T9) -- stress-test every "interesting" pattern against expanded data before excitement; distinguish structure from artifact
- Information-theoretic analysis of zero distributions (D5) -- entropy, mutual information, compression complexity; novel unconventional angle
- Anomaly detection in zero structure (D11) -- automated flagging of deviations from GUE statistics or Riemann-von Mangoldt formula
- Trace formula workbench (D10) -- explicit formulas connecting zeros to prime-counting functions
- Experiment reproducibility (T6) -- full parameter serialization, result caching with checksums, seed management

**Defer (Phases 5+):**
- Lean 4 formalization pipeline (D8) -- no discoveries to formalize yet; scaffold the pipeline but do not use it until genuine conjectures with maturity threshold are met
- Modular forms toolkit (D4) -- rich but self-contained; add as a module after cross-disciplinary analysis is established
- Adelic/p-adic computation (D12) -- specialized; defer until exploration suggests need
- Dimensional projection theater real-time rendering (D9) -- static projections from D1 suffice initially; real-time WebGL is Phase 5+ polish
- Cross-disciplinary analogy engine (D6) -- requires multiple analysis domains to exist first; highest-value feature but last to be buildable
- AI conjecture generation pipeline (D7) -- needs accumulated structured data and mature workbench infrastructure first

**Explicitly out of scope (anti-features):**
- General-purpose CAS (SageMath/Mathematica already exist; integrate as backends, do not replicate)
- Reimplementing zeta from scratch (use mpmath; correct arbitrary-precision zeta is a multi-decade problem)
- GPU-accelerated large-scale zero verification (use LMFDB/Odlyzko tables; HPC infrastructure is out of scope)
- Web deployment or multi-user features (single-user local tool by design)
- Teaching/tutorial infrastructure (Claude is the teacher; no curriculum needed)

### Architecture Approach

The platform follows a six-layer architecture with strict downward dependency flow. The Formalization Pipeline sits beside the main stack as a separate Lean 4 process communicating via file exchange rather than on top of it. Boundary rules between layers are non-negotiable: the computation engine never visualizes; the visualization layer never computes mathematical results; analysis modules receive data and return results without persisting or rendering. Violating these boundaries creates the "God Notebook" anti-pattern where computation logic entangles with UI state, destroying reproducibility and cacheability.

The two-precision-regime principle runs through all computation: use NumPy/SciPy at machine precision for bulk exploration (random matrix ensembles, spectral statistics, FFTs) and mpmath at arbitrary precision for targeted verification (zeta values, zero location, formal results). Mixing these regimes without clear boundaries is the most common technical mistake in this domain. The architecture enforces this boundary through typed mathematical objects (ZetaZero, AnalysisResult dataclasses distinguishing mpf/mpc from float/complex) and a precision context manager that makes scope explicit and prevents global state leakage.

**Major components:**
1. **Computation Engine (Layer 1)** -- all numerical evaluation: zeta evaluator, zero finder, L-function evaluator, spectral operator engine, random matrix generator, modular form evaluator; exposes a `precision()` context manager; functions tagged as arbitrary-precision or machine-precision; conversion between regimes is always explicit with logged warnings
2. **Data / Object Store (Layer 0)** -- SQLite zero database with SQL query support, content-addressed computation cache keyed by (function, parameters, precision), YAML session state files, NumPy .npy/.npz files for large array storage; dumb persistence layer that never computes or interprets
3. **Analysis Modules (Layer 2)** -- pluggable cross-disciplinary analysis via a standard `AnalysisModule` protocol and plugin registry; modules register at startup, new modules added without modifying existing code; planned: ZeroSpacingAnalyzer, GUEComparator, SpectralOperatorModule, EntropyAnalyzer, ModularFormBridge, HighDimGeometryModule
4. **Visualization Layer (Layer 3)** -- composable N-dimensional projection pipeline (custom steps + scikit-learn), Plotly 3D interactive renderer, Matplotlib 2D publication renderer, Panel dashboards, domain coloring complex plane viewer; artifacts warnings and quantitative metrics built into every projection output
5. **Research Workbench (Layer 4)** -- Jupyter magic commands, session management, experiment runner with full parameter capture, conjecture tracker with enforced epistemological hierarchy (Level 0-3), insight journal with dead-ends registry, Claude interface layer for structured formalism prompts
6. **Formalization Pipeline (Layer 5, separate Lean 4 process)** -- Lean 4 project under `lean/` with lakefile.lean and pinned toolchain, Python-side statement translator (Python conjecture --> Lean 4 syntax), proof skeleton generator with sorry placeholders, lake CLI runner with structured output parsing, proof state tracker (sorry count vs. proven theorems)

Key architectural patterns: mathematical objects as typed first-class citizens, precision context manager, plugin registry for extensible analysis, composable projection pipeline, experiment-as-reproducible-unit dataclass, file-exchange Lean protocol.

### Critical Pitfalls

1. **Silent precision collapse in zeta evaluation** -- Using float64/scipy near the critical strip silently returns garbage for large Im(s); for Im(s) ~ 10^6 you need 50+ digit precision; build precision canary tests that run before every session; validate all computation against known Odlyzko zero tables before any exploration begins. This is a Phase 1 blocker -- do not explore until this is verified.

2. **Building infrastructure instead of doing mathematics** -- Software engineering feels productive; mathematical exploration feels uncertain. Enforce a 70/30 exploration-to-infrastructure ratio; define mathematical milestones (not just engineering milestones) in every phase; compute zeta zeros in raw Jupyter on Day 1 before any platform abstraction exists; time-box any infrastructure feature to two days maximum.

3. **Confusing numerical evidence with mathematical proof** -- The RH has been verified for 10^13 zeros; that is not a proof. Build a strict epistemological hierarchy into the workbench schema from day one: Level 0 (computational observation) through Level 3 (Lean 4 verified); every conjecture tagged with its level; actively seek counterexamples before investing in understanding why a pattern holds.

4. **Premature Lean 4 formalization** -- Gate formalization behind a maturity threshold: result must be Level 2 (conditional, heuristic justification) and stable for 2+ weeks before formalization begins; start Lean 4 learning with known established results (functional equation, Euler product) not novel conjectures; formalization effort is typically 10-100x the informal statement.

5. **Visualization artifacts mistaken for mathematical structure** -- Human visual pattern recognition finds structure in noise. Always use perceptually uniform colormaps (viridis/inferno/cividis, never jet/rainbow); show every high-dimensional projection with at least two independent projection methods; display quantitative metrics alongside every visualization; build automated artifact detection (render at two resolutions, flag discrepancies) into the visualization module from Phase 2.

## Implications for Roadmap

Based on combined research, the architecture's build-order dependency graph maps directly to phase structure. The architecture document establishes a clear 6-phase build order grounded in dependency analysis; the features research confirms the critical path; the pitfalls research identifies which risks are phase-specific and must be addressed before proceeding.

### Phase 1: Computational Foundation

**Rationale:** Nothing else can function without numerical results. The critical path runs T1 --> T2 --> T8; blocking any step blocks all downstream work. Precision correctness must be established in this phase -- retrofitting precision validation after patterns have been "discovered" is a project-ending mistake that undermines trust in all prior results. The computation engine must also expose correct boundaries (precision context manager, typed objects) so that later layers cannot accidentally violate precision contracts.

**Delivers:** Working arbitrary-precision zeta evaluation with mpmath/gmpy2 backend; non-trivial zero computation with mpmath.zetazero(); SQLite zero database with caching; precision validation framework with canary tests that run at startup; functional equation symmetry checks as assertions; Euler product independent verification path; test suite validating all computation against known zero locations (Odlyzko tables); mathematical exploration in raw Jupyter notebooks from Day 1.

**Addresses features:** T1 (zeta evaluation), T2 (zero computation), T6 (reproducibility foundations), T9 (numerical verification core)

**Avoids:** Pitfall 1 (precision collapse -- validation framework built in, not bolted on), Pitfall 13 (reinventing existing wheels -- audit mpmath/scipy/FLINT before writing any custom algorithm), Pitfall 3 (infrastructure trap -- mathematical exploration in raw notebooks begins Day 1, not after "the engine is ready")

**Research flag:** Standard patterns (mpmath, SQLite, gmpy2, pytest). Skip research-phase for this phase.

---

### Phase 2: Visualization and Research Workbench

**Rationale:** The user is an explorer who builds intuition visually. Without seeing results, exploration cannot be directed and validated. Early visualization also provides sanity-checking of computation results -- visual anomalies often reveal precision issues or implementation bugs before they contaminate analysis. The research workbench must launch here, not later, because the epistemological hierarchy and dead-ends registry must be present from the first real exploration session. Retrofitting evidence-level tagging after conjectures have already been formed is psychologically difficult and architecturally messy.

**Delivers:** Critical line magnitude and phase plots with interactive zoom/pan (T3); complex plane domain coloring with perceptually uniform colormaps and zoom (T4); Panel dashboards with parameter sliders and linked views; session management with evidence-level tagging enforced by workbench schema; experiment runner with full parameter capture and result linking; conjecture tracker (Level 0-3 taxonomy); insight journal with dead-ends registry; Jupyter magic commands for common operations; rich display formatters for mathematical objects.

**Uses:** Matplotlib + Plotly + ipywidgets, Panel, JupyterLab, ipywidgets

**Implements:** Visualization Layer (Layer 3) + Research Workbench (Layer 4)

**Avoids:** Pitfall 5 (visualization artifacts -- perceptual colormaps and multiple projection methods built in from start; quantitative metrics alongside every plot), Pitfall 2 (evidence-level confusion -- workbench schema enforces hierarchy, cannot be skipped), Pitfall 10 (research amnesia -- structured session logging and dead-ends registry required by workflow, not optional)

**Research flag:** Panel + current JupyterLab compatibility should be verified before starting. May warrant a quick research-phase on the HoloViz ecosystem current state.

---

### Phase 3: Cross-Disciplinary Analysis Modules

**Rationale:** With computation and visualization working, cross-disciplinary analysis modules can be built against real data and immediately visualized. These modules are the platform's differentiating value over Mathematica and SageMath. The analogy engine (D6) -- the highest-value feature of the entire platform -- requires multiple analysis domains to exist first, so this phase systematically builds those domains. The pluggable module architecture means each module can be added independently without disrupting existing ones.

**Delivers:** Zero spacing statistics and GUE comparison (T8, D3 -- Montgomery-Odlyzko law); higher-dimensional framework prototype with zeros as R^k feature points and PCA/t-SNE/UMAP projection (D1 prototype); information-theoretic analysis with entropy and mutual information of zero sequences (D5); anomaly detection flagging deviations from GUE predictions (D11); 3D interactive Plotly visualizations with rotation/zoom; related function evaluation including Dirichlet L-functions and Hardy's Z-function (T7); pluggable analysis module registry fully operational.

**Addresses features:** D1 (prototype), D3, D5, D7 (early), D11, T7, T8

**Avoids:** Pitfall 7 (scope explosion -- investigation stack with maximum depth of 3; time-boxed tangents with explicit success criteria defined before each exploration begins; quarterly pruning of backlog), Pitfall 9 (performance death spiral -- profile every computation from day one; use precision hierarchy; aggressive caching with LRU eviction; multiprocessing for parallel zero evaluation because GIL blocks threading)

**Research flag:** UMAP integration (separate `umap-learn` package), Panel advanced dashboard features, and higher-dimensional feature space design may warrant research-phase. The HighDimGeometryModule is inherently exploratory -- the right feature space for embedding zeros is unknown until exploration reveals it; architecture must remain flexible.

---

### Phase 4: Spectral Operators and Full Higher-Dimensional Expansion

**Rationale:** Spectral operator analysis (D2) and the complete higher-dimensional framework (D1 full, D9 projection theater) are the most computationally demanding and architecturally complex features. They require the Phase 3 cross-disciplinary infrastructure to be solid before adding another layer of complexity. The Berry-Keating Hamiltonian and related operators require careful numerical linear algebra that is sensitive to discretization choices -- getting this wrong produces plausible but incorrect spectra. Modular forms (D4) and trace formula workbench (D10) also land here as they depend on related function evaluation from Phase 3.

**Delivers:** Candidate operator construction and diagonalization (Berry-Keating Hamiltonian and variants) with discretization error reporting; N-dimensional projection pipeline with composable steps (PCA, UMAP, stereographic, custom geometric projections) and artifact metadata; full interactive 3D projection theater with rotation/zoom/selection; modular form evaluator with Hecke eigenvalues and Fourier coefficients; modular form bridge to L-function zero structure; trace formula workbench connecting zero contributions to prime-counting functions with truncation effect visualization.

**Avoids:** Pitfall 9 (performance at high dimensions with high precision -- use NumPy/SciPy at machine precision for structure detection in high-dim spaces, mpmath only for targeted verification of interesting findings; this combination makes high-dimensional exploration tractable)

**Research flag:** Spectral operator discretization and convergence properties are mathematically subtle. Modular form computation library options (python-flint maturity on Windows, LMFDB API for precomputed data, potential selective SageMath component reuse) need research-phase before starting. Verify LMFDB API availability for precomputed zero and modular form data.

---

### Phase 5: Lean 4 Formalization Pipeline

**Rationale:** Formalization only makes sense after the exploration pipeline has generated genuine conjectures -- statements that have survived counterexample searches, are at evidence Level 2 or above, and have been stable for multiple weeks. Beginning formalization before this wastes enormous effort (10-100x overhead) on mathematics that may shift. The Lean 4 learning curve is also steep; this phase must include explicit dedicated learning time on known established results before any project-specific formalization begins.

**Delivers:** Lean 4 project under `lean/` with Mathlib4 dependency and pinned toolchain; Python-side statement translator (Python conjecture format --> Lean 4 syntax); proof skeleton generator with sorry placeholders and Mathlib references; lake CLI runner with structured error parsing; proof state tracker (sorry-to-proof ratio, per-lemma status); initial formalizations of established results (functional equation, Euler product, basic zeta properties in the critical strip) to build Lean 4 skill on known-correct mathematics; first attempted formalization of a platform-generated conjecture that has met maturity threshold.

**Avoids:** Pitfall 4 (premature formalization -- maturity threshold gate enforced by workbench; formalization readiness checklist required), Pitfall 6 (Lean 4 learning curve -- explicit 2-4 week learning period on tutorials and known results before any project code; Claude handles Lean 4 writing; pin Lean 4 and Mathlib versions and do not upgrade mid-phase), Pitfall 11 (Python-Lean mismatch -- interval arithmetic in Python for formalization-bound computations; ARB/FLINT for rigorous enclosures rather than point estimates)

**Research flag:** This phase needs a research-phase before planning. Key questions: What does Mathlib4 currently have formalized for complex analysis, Dirichlet series, and zeta function properties? What is the current state of lake file-exchange integration? Does elan work correctly on Windows Server 2025 (the development platform)? What is the Python-to-Lean statement translation best practice for analytic number theory results?

---

### Phase 6: Advanced Integration and Proof Discovery

**Rationale:** The cross-disciplinary analogy engine (D6) and AI conjecture generation pipeline (D7) are the most synthesis-intensive features -- they require all preceding analysis domains to be operational and populated with results. Adelic/p-adic computation (D12) is the most specialized feature with the most uncertain implementation path. This phase is deliberately left loosely defined because its design must be shaped by what Phases 1-5 discover. Do not detail-plan Phase 6 until Phase 4 is complete.

**Delivers:** Cross-disciplinary analogy engine mapping structures between spectral, RMT, modular form, and information-theoretic domains; AI conjecture generation pipeline with structured evidence chains stored in workbench; adelic/p-adic computation module; automated background anomaly surfacing; publication-quality dashboard output for research reporting.

**Research flag:** This phase requires a full research-phase when approached. Its scope and design will be entirely determined by what prior phases discover. Do not attempt to research it in advance.

---

### Phase Ordering Rationale

- **Dependency-driven order:** The computation engine (Layer 1) must precede visualization (Layer 3) which must precede analysis modules (Layer 2 consuming computed data and producing visualizable results); the formalization pipeline is an independent branch that runs beside the main stack and comes last. This ordering is derived from the architecture's component dependency graph, not from arbitrary sequencing.
- **User-facing feedback as early as possible:** Phase 2 delivers visualization immediately after computation, so the user can see results and direct research before any cross-disciplinary infrastructure exists. This prevents the platform from becoming a black box that produces numbers without insight.
- **Pitfall mitigation requires early foundation:** Precision validation (Phase 1), evidence-level hierarchy (Phase 2), and projection artifact warnings (Phase 2) must be designed into the system early. Retrofitting these after "patterns have been discovered" is psychologically resisted and architecturally disruptive.
- **Lean 4 deliberately last:** The Lean 4 learning curve is steep, formalization effort is 10-100x the informal statement, and premature formalization is the second most dangerous pitfall identified. Lean 4 work belongs only after the platform has generated genuine conjectures that have survived scrutiny.
- **Higher-dimensional work after basics:** The platform's core differentiator (D1, D6, D9) cannot be validated or sensibly designed without first understanding what structures emerge from basic zero analysis (Phases 1-3). Attempting N-dimensional projection before basic 2D/3D works produces confusing results that cannot be distinguished from projection artifacts.

### Research Flags

Phases needing deeper research during planning:
- **Phase 2:** Panel + current JupyterLab compatibility; verify current HoloViz widget ecosystem state before committing to Panel as the dashboard framework
- **Phase 4:** Modular form computation library options (python-flint Windows binary availability, LMFDB API, selective SageMath component reuse without full SageMath dependency); spectral operator discretization best practices and convergence validation
- **Phase 5:** Current Mathlib4 analytic number theory coverage (what is already formalized); current lake project setup and file-exchange patterns; elan installation and Lean 4 toolchain on Windows Server 2025; Python-to-Lean translation best practices for analytic number theory
- **Phase 6:** Design is inherently shaped by prior discoveries; requires full research-phase when approached; do not pre-research

Phases with standard patterns (skip research-phase):
- **Phase 1:** mpmath, SQLite, gmpy2, pytest, uv patterns are extremely well-documented with stable APIs; no research needed before starting
- **Phase 3:** NumPy/SciPy analysis patterns, scikit-learn dimensionality reduction, pandas organization -- all highly standard; UMAP package integration is minor; higher-dimensional feature space design is exploratory but the projection infrastructure is straightforward

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM-HIGH | Core library choices (mpmath, NumPy/SciPy, Lean 4, JupyterLab, uv) are HIGH confidence; specific version numbers need validation with `uv pip index versions`; Panel/Jupyter compatibility is MEDIUM; python-flint Windows binary availability is LOW |
| Features | MEDIUM | Library capabilities and feature priorities are HIGH confidence based on domain knowledge of RH research; actual user workflow and which cross-disciplinary connections prove fruitful are inherently LOW confidence given the exploratory nature; higher-dimensional approach is genuinely novel with LOW confidence on specific implementation details |
| Architecture | MEDIUM-HIGH | Layered architecture with plugin modules is well-established for research platforms; component boundaries and data flow are clear; SQLite for zeros and file-exchange Lean protocol are well-reasoned and conservative; Panel dashboard framework compatibility is MEDIUM |
| Pitfalls | HIGH | Core pitfalls (precision collapse, infrastructure trap, evidence confusion, Lean learning curve) are universally documented in the relevant communities with consistent warnings; specific performance characteristics are MEDIUM depending on hardware and exact algorithms chosen |

**Overall confidence:** MEDIUM

### Gaps to Address

- **Lean 4 on Windows Server 2025:** elan installation and Lean 4 toolchain behavior on Windows needs explicit validation before Phase 5 planning. The Lean 4 community is primarily Linux/macOS and Windows support may have rough edges. Validate early so Phase 5 can be planned around actual constraints.
- **Mathlib4 analytic number theory coverage:** The extent to which Mathlib has formalized complex analysis, Dirichlet series, and zeta function properties determines how much from-scratch formalization Phase 5 requires versus building on existing Mathlib infrastructure. Must be audited before Phase 5 planning begins.
- **python-flint Windows binaries:** python-flint (FLINT/Arb bindings) is needed for performance-critical zeta evaluation paths and for rigorous interval arithmetic required by the Lean 4 formalization pipeline. Binary availability on Windows should be checked before Phase 3 (where performance optimization may first be needed) and confirmed before Phase 5.
- **Panel + JupyterLab current compatibility:** Panel is the recommended dashboard framework but Jupyter widget compatibility changes with JupyterLab major versions. Verify with live documentation before starting Phase 2 to avoid mid-phase framework pivots.
- **LMFDB API current state and data availability:** LMFDB is the primary source for large-scale precomputed zero data, modular form data, and L-function databases. The API, data availability, and query interface should be confirmed before Phase 3-4 to avoid reimplementing what is already queryable.
- **Higher-dimensional feature space design:** The HighDimGeometryModule is genuinely novel. The research provides a starting point (zeros as R^k points with spacing/derivative/local-density features, projected via PCA/UMAP), but the right feature space is unknown until exploration reveals it. The architecture must remain flexible to redesign this module based on what Phase 3 discovers.
- **Modular forms computation without SageMath:** Modular form evaluation outside SageMath requires either python-flint (FLINT has modular form support), LMFDB API queries for precomputed data, or very careful custom implementation. The right approach depends on python-flint maturity and LMFDB coverage, which must be checked before Phase 4.

## Sources

### Primary (HIGH confidence)
- mpmath official documentation and source (training data through v1.3; stable library, infrequent breaking changes) -- zeta evaluation, zetazero(), arbitrary-precision arithmetic, gmpy2 backend
- NumPy/SciPy official documentation (training data through NumPy 2.0, SciPy 1.12-1.13) -- machine-precision computation, spectral analysis, linear algebra, statistics
- Lean 4 / Mathlib community documentation (training data through early 2025) -- theorem prover setup, Mathlib structure, lake build system
- Odlyzko zero tables (publicly available, stable) -- known zero data for validation and testing
- LMFDB database (lmfdb.org, stable public database) -- precomputed zeros, modular forms, L-functions

### Secondary (MEDIUM confidence)
- Lean 4 formalization community experience reports -- learning curve estimates, Mathlib naming conventions, sorry-to-proof ratios
- HoloViz/Panel documentation (training data) -- dashboard framework capabilities and Jupyter integration
- Scientific visualization best practices literature (Borland and Taylor on rainbow colormaps; Wattenberg on dimensionality reduction distortions) -- visualization artifact prevention methodology
- Riemann Hypothesis computational literature (Montgomery-Odlyzko, Keating-Snaith, Berry-Keating) -- domain knowledge for feature prioritization and approach validation

### Tertiary (LOW confidence, needs live validation)
- python-flint binding maturity and Windows binary availability -- check PyPI before Phase 3
- Panel + JupyterLab current compatibility matrix -- check HoloViz documentation before Phase 2
- elan and Lean 4 toolchain on Windows Server 2025 -- test environment before Phase 5 planning
- Specific latest version numbers for all packages -- training data cutoff May 2025; validate with `uv pip index versions <pkg>` before installation

---
*Research completed: 2026-03-18*
*Ready for roadmap: yes*
