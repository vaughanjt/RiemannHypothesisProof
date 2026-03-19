# Phase 2: Higher-Dimensional Analysis - Context

**Gathered:** 2026-03-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver N-dimensional embedding of mathematical objects (zeros, function values, operator spectra), multiple projection methods with side-by-side comparison, an interactive projection theater, zero distribution statistics (spacing, pair correlation, n-level density) with GUE comparison, random matrix ensemble generation with linked interactive views, information-theoretic analysis of zero sequences, and automatic anomaly detection. This is the platform's core differentiator — seeing structure in dimensions beyond human spatial intuition.

</domain>

<decisions>
## Implementation Decisions

### Embedding Strategy
- Multiple embedding schemes, not just one — maximize diversity of views to avoid bias toward expected outcomes
- Embedding coordinates for zeros: imaginary part, spacing to neighbors (left/right), derivative |zeta'(rho)|, local zero density deviation, pair correlation at multiple scales, Hardy Z sign changes
- Additional embeddings: zeros as points in spectral space (eigenvalue-gap analogy), zeros in information-theoretic space (local entropy, compression distance)
- User directive: "Surprise me" — cast the widest net, don't bias toward any particular expected structure
- Each embedding is a named, reproducible configuration stored in the workbench

### Projection Methods
- Implement all: PCA, t-SNE, UMAP, stereographic, custom mathematical projections (e.g., Hopf fibration for S^3 data)
- Side-by-side comparison: same data, multiple projections, linked highlighting
- Claude selects which projections to try based on mathematical intuition about the data
- Progressive: start with PCA (linear, fast, trustworthy), then nonlinear methods to reveal manifold structure

### Projection Theater
- Interactive 3D visualization using Plotly with rotation, zoom, parameter controls
- "Projection path" animations — smoothly interpolate between projection methods to see how structure transforms
- Dimension slicing: fix some dimensions, project remaining — systematic exploration of high-dim structure
- Speed over polish: computation speed matters more than frame rate or rendering quality

### Zero Distribution Statistics
- Nearest-neighbor spacing (normalized by mean spacing)
- Pair correlation function r_2(x) compared against GUE sine kernel
- n-level density for n=2,3,4
- Number variance and Sigma_2 statistic
- All statistics computed with configurable zero ranges and overlap with Odlyzko-verified zeros

### Random Matrix Theory Laboratory
- Generate GUE ensembles at configurable matrix sizes (N=10 to N=1000+)
- Compute eigenvalue statistics matching every zero statistic above
- Linked views: zero statistics and RMT statistics side-by-side, interactive N slider
- Residual analysis: where do zeros deviate from GUE? At what scale? This is where proof structure might hide.

### Information-Theoretic Analysis
- Shannon entropy of zero spacing sequences (binned and kernel-density estimated)
- Mutual information between consecutive spacings at multiple lags
- Lempel-Ziv complexity / compression-based distance metrics on zero sequences
- Compare information signatures: zeros vs GUE eigenvalues vs Poisson random points vs primes
- Cross-object comparison is key — differences in information content point to unique structure

### Anomaly Detection
- Statistical process control (SPC) applied to zero data streams
- Flag any window where local statistics deviate >3σ from GUE prediction
- Flag unusual spacing patterns, local density anomalies, unexpected correlations
- Anomaly severity levels: info / warning / critical (integrate with evidence hierarchy)
- Every anomaly auto-logged as an "observation" in the research workbench (evidence level 0)
- Claude decides sensitivity thresholds and tuning

### Carrying Forward from Phase 1
- JupyterLab, Claude-driven exploration
- Speed over visual polish, Claude picks per-plot tooling
- SQLite + numpy + HDF5 for data storage
- Strict evidence hierarchy (observation / heuristic / conditional / formal)
- 50-digit default, always-validate precision
- Phase 1 computation engine, visualization layer, and workbench are the integration substrate

### Claude's Discretion
- Exact embedding coordinate engineering and feature scaling
- Projection hyperparameters (t-SNE perplexity, UMAP n_neighbors, etc.)
- Statistical test selection and significance thresholds
- Matrix ensemble sampling strategy
- Information-theoretic estimator choices (binning vs KDE vs k-NN)
- Anomaly detection window sizes and threshold tuning
- Notebook organization for Phase 2 explorations
- HDF5 schema for high-dimensional data
- Performance optimization (vectorization, caching, lazy computation)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Context
- `.planning/PROJECT.md` — Core value, constraints, key decisions
- `.planning/REQUIREMENTS.md` — ZERO-01, ZERO-02, HDIM-01, HDIM-02, VIZ-03, RMT-01, RMT-02, INFO-01, INFO-02

### Prior Phase
- `.planning/phases/01-computational-foundation-and-research-workbench/01-CONTEXT.md` — Phase 1 decisions (interface, storage, precision)

### Research
- `.planning/research/STACK.md` — Technology choices
- `.planning/research/FEATURES.md` — Feature details for D1, D3, D5, D9, D11, T8
- `.planning/research/ARCHITECTURE.md` — Component boundaries, projection pipeline architecture
- `.planning/research/PITFALLS.md` — Visualization artifacts from dimensionality reduction, confusing evidence with proof

### Existing Code (Phase 1 outputs)
- `src/riemann/engine/zeta.py` — zeta_eval, zeta_on_critical_line (computation substrate)
- `src/riemann/engine/zeros.py` — compute_zero, compute_zeros_range, ZeroCatalog (zero data source)
- `src/riemann/engine/precision.py` — validated_computation, precision_scope (precision infrastructure)
- `src/riemann/types.py` — ZetaZero, ComputationResult, EvidenceLevel (shared types)
- `src/riemann/viz/critical_line.py` — plotting patterns to follow
- `src/riemann/viz/domain_coloring.py` — dual-mode computation pattern (fast numpy + precise mpmath)
- `src/riemann/workbench/conjecture.py` — conjecture CRUD for anomaly logging
- `src/riemann/workbench/experiment.py` — experiment save/load for embedding configs

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ZetaZero` dataclass: has `n`, `value` (mpf), `imaginary_part` (mpf) — natural starting point for embedding coordinates
- `ZeroCatalog`: SQLite-backed zero storage with bulk retrieval — data source for all statistics
- `validated_computation`: P-vs-2P pattern — reuse for any new computation near critical strip
- `precision_scope`: context manager for temporary precision changes
- `workbench/conjecture.py`: CRUD with evidence levels — anomaly auto-logging target
- `workbench/experiment.py`: experiment save/load with checksums — store embedding configurations
- `viz/styles.py`: analytical clarity color palette — extend for multi-dimensional visualizations

### Established Patterns
- TDD with pytest: red-green cycle, xfail scaffolds replaced by real tests
- Dual-mode computation: fast numpy for overview, mpmath for precision-critical paths
- SQLite for structured metadata, numpy/HDF5 for numerical arrays
- Function-based API (not class-based) — Phase 1 workbench uses functions, not classes

### Integration Points
- Zero data flows from `engine/zeros.py` → embedding module → projection module → visualization
- Statistics computed from zero data, compared against RMT module output
- Anomalies detected → auto-logged to workbench as observations
- Embedding configs saved as experiments in workbench
- HDF5 for large embedding arrays (new dependency for Phase 2)

</code_context>

<specifics>
## Specific Ideas

- User believes the proof is structural, not in extreme precision — higher-dimensional projections are the primary discovery mechanism
- "Surprise me" — don't optimize for any particular expected structure; maximize diversity of views
- The projection theater should let Claude systematically sweep through projection spaces looking for unexpected patterns
- Residual analysis (where zeros deviate from GUE) is a particularly promising angle — the deviations are where proof structure might hide

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-higher-dimensional-analysis*
*Context gathered: 2026-03-19*
