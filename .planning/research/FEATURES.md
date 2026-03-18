# Feature Landscape

**Domain:** Mathematical research platform for Riemann Hypothesis proof exploration
**Researched:** 2026-03-18
**Confidence:** MEDIUM (training data only -- web verification was unavailable)

---

## Table Stakes

Features users expect. Missing = platform is useless for serious RH research.

| # | Feature | Why Expected | Complexity | Notes |
|---|---------|--------------|------------|-------|
| T1 | **Arbitrary-precision zeta function evaluation** | Cannot study RH without computing zeta(s) to hundreds/thousands of digits. Floating-point is insufficient -- zeros cluster and cancel near the critical line. | Medium | Wrap mpmath (pure Python, arbitrary precision) for exploration. Use Arb/flint via python-flint for performance-critical paths. Do NOT implement zeta from scratch. |
| T2 | **Non-trivial zero computation** | The hypothesis IS about zeros. Must locate, verify, and catalog them. Odlyzko's method (Riemann-Siegel formula + Newton refinement) is standard. | Medium | mpmath.zetazero(n) gives the nth zero. For large-scale computation, need direct Riemann-Siegel formula implementation or binding to Andrew Odlyzko's C code / David Platt's verified computations. |
| T3 | **Critical line visualization** | The user is an explorer, not a trained mathematician. Visualizing |zeta(1/2 + it)| along the critical line, phase portraits, and zero locations is how they will build intuition. | Low-Medium | 2D plots (matplotlib/plotly) of |zeta(s)| on critical line, Argand diagrams, domain coloring of zeta in the critical strip. |
| T4 | **Complex plane domain coloring** | Standard technique for visualizing complex functions. Maps phase to hue, magnitude to brightness. Reveals structure invisible in magnitude-only plots. | Low | Well-understood technique. Libraries exist (cplot, custom matplotlib). Must support zoom into critical strip regions. |
| T5 | **Session-based research workbench** | Without persistent state (conjectures, observations, experiment results), exploration is aimless repetition. Must record what was tried, what was found, what was surprising. | Medium | Jupyter-like notebook paradigm or custom session management. Store structured data (experiment parameters, results, annotations) not just text notes. |
| T6 | **Experiment reproducibility** | Mathematical exploration demands exact reproducibility. Every computation must be re-runnable with identical parameters and verifiable results. | Low-Medium | Seed management, parameter serialization, result caching with checksums. Each "experiment" is a reproducible unit. |
| T7 | **Related function evaluation** | RH connects to Dirichlet L-functions, Dedekind zeta functions, Selberg zeta function, xi function, Z function (Hardy's function). Cannot explore cross-disciplinary connections without computing these. | Medium | mpmath provides many. Dirichlet L-functions via characters. Selberg zeta requires spectral data (harder). Hardy's Z-function is essential for zero studies. |
| T8 | **Zero distribution statistics** | GUE statistics, nearest-neighbor spacing, pair correlation, n-level density. These connect zeros to random matrix theory -- one of the deepest RH connections. | Medium | Compute spacing distributions from zero data. Compare against GUE predictions (Montgomery-Odlyzko). Statistical testing framework needed. |
| T9 | **Numerical verification framework** | Every "interesting finding" must be numerically stress-tested before getting excited. False patterns emerge constantly in finite computations. Need systematic tools to test whether a pattern holds to higher precision / more zeros / different parameters. | Medium | Hypothesis testing: given a conjectured pattern, automatically test it against expanded data. Distinguish genuine structure from numerical artifacts. |

---

## Differentiators

Features that set the platform apart. These are the unconventional, cross-disciplinary tools that justify building a custom platform rather than using SageMath or Mathematica.

| # | Feature | Value Proposition | Complexity | Notes |
|---|---------|-------------------|------------|-------|
| D1 | **Higher-dimensional computation framework** | The project's core thesis: proof structures may live in dimensions beyond human intuition. Must compute in N-dimensional spaces and project results to 2D/3D. No existing tool does this with RH focus. | High | Represent objects in arbitrary-dimensional spaces (adelic spaces, high-dim hyperbolic manifolds). Implement projection operators (PCA, UMAP, t-SNE, custom mathematical projections). This is the most novel and hardest component. |
| D2 | **Spectral operator analysis** | The Hilbert-Polya conjecture (RH equivalent: zeros = eigenvalues of a self-adjoint operator) is a leading proof strategy. Need tools to construct, analyze, and visualize candidate operators. | High | Discretize candidate operators, compute eigenvalues, compare against zeta zeros. Berry-Keating Hamiltonian (xp + corrections). Requires careful numerics -- operator spectra are sensitive to discretization. |
| D3 | **Random matrix theory (RMT) laboratory** | Montgomery-Odlyzko law: zeros behave like GUE eigenvalues. But WHY? Platform should let user explore this connection interactively -- sample random matrices, compare statistics, visualize both side-by-side. | Medium | Generate GUE/GOE/GSE ensembles, compute eigenvalue statistics, overlay with zero statistics. Interactive: change matrix size, ensemble type, see how fit changes. NumPy/SciPy sufficient for matrix operations. |
| D4 | **Modular forms and automorphic representations toolkit** | RH is deeply connected to modular forms via the Langlands program. The modularity theorem (Wiles) and connections between L-functions and automorphic forms are potential proof pathways. | High | Compute modular forms, Hecke eigenvalues, Fourier coefficients. LMFDB integration for known data. Visualize in the upper half-plane. Connecting modular form data to zeta zero structure. |
| D5 | **Information-theoretic analysis of zero distributions** | Unconventional: treat zero distributions as signals. Compute entropy, mutual information, compression complexity. Look for hidden structure that traditional analytic number theory misses. | Medium | Apply entropy measures, Kolmogorov complexity estimates, compression-based distance metrics to zero sequences. Novel and exploratory -- no established methodology, which is the point. |
| D6 | **Cross-disciplinary analogy engine** | Map structures from one domain (e.g., quantum chaos) to RH structures. If zeros behave like quantum energy levels, what does the "Hamiltonian" look like? Systematic framework for importing and testing analogies. | High | Define "analogy mappings" between domains. E.g., {eigenvalues : zeros, Hamiltonian : ???, trace formula : explicit formula}. Let user fill in unknowns and test computationally. Highly novel. |
| D7 | **AI-guided conjecture generation** | Use Claude to analyze computational results, spot patterns humans miss, generate formal conjectures, and suggest what to explore next. The user explores; Claude does formalism AND pattern recognition. | Medium | Structured prompting: feed Claude experimental data, ask for pattern identification, conjecture formulation, and suggested next experiments. Store conjectures with evidence chains. |
| D8 | **Lean 4 formalization pipeline** | When exploration finds something promising, formalize it in Lean 4 for machine-verified proof. This is the bridge from "interesting computation" to "rigorous mathematics." | High | Translate conjectures to Lean 4 statements. Use Mathlib for existing formalized mathematics. Track formalization progress (statement formalized, proof attempted, proof complete). Lean 4 interop from Python is non-trivial. |
| D9 | **Dimensional projection theater** | Interactive visualization that lets user "rotate" through higher-dimensional spaces, watching how structures project into different 2D/3D views. Like a planetarium for mathematical objects. | High | Real-time rendering of high-dimensional mathematical objects with interactive projection controls. WebGL or Plotly 3D with custom projection math. Must handle mathematical objects, not just point clouds. |
| D10 | **Trace formula workbench** | The explicit formulas of prime number theory (Riemann-von Mangoldt, Weil explicit formula, Selberg trace formula) connect zeros to primes. Interactive tools to explore these dualities. | Medium-High | Compute partial sums of explicit formulas, visualize how zeros contribute to prime-counting functions and vice versa. Truncation effects are subtle and must be handled carefully. |
| D11 | **Anomaly detection in zero structure** | Automated detection of deviations from expected behavior (GUE statistics, Riemann-von Mangoldt formula). Anomalies could point toward proof structure or reveal computational errors. | Medium | Statistical process control applied to zero data. Flag unexpected spacing, unusual local statistics, deviations from smooth N(T) approximation. |
| D12 | **Adelic/p-adic computation** | The adelic viewpoint unifies archimedean and p-adic completions. Tate's thesis reproves the functional equation adelically. Platform should compute in p-adic fields and visualize p-adic structure. | High | p-adic number arithmetic, p-adic zeta functions, visualization of p-adic spaces (fractal structure). Connect p-adic and archimedean pictures. SageMath has p-adic types; may need to adapt. |

---

## Anti-Features

Features to explicitly NOT build. Building these would be a waste of effort or actively harmful to the project.

| # | Anti-Feature | Why Avoid | What to Do Instead |
|---|--------------|-----------|-------------------|
| A1 | **General-purpose computer algebra system** | SageMath, Mathematica, and SymPy already exist. Reimplementing symbolic algebra is a multi-decade project that would consume all resources. | Integrate with mpmath, SymPy, and SageMath as computational backends. Focus platform effort on the research workflow and cross-disciplinary tools. |
| A2 | **General-purpose proof assistant** | Lean 4 exists and is actively maintained by a large community. Building a custom proof system would be absurd. | Build a pipeline TO Lean 4: translate conjectures, manage formalization tasks, track proof state. Do not replicate Lean's type theory. |
| A3 | **Reimplemented zeta function from scratch** | Correct arbitrary-precision zeta evaluation is extremely subtle (Riemann-Siegel requires careful error bounds, Euler-Maclaurin has convergence issues). mpmath and Arb have decades of work in this. | Use mpmath.zeta() and arb's acb_dirichlet_zeta(). Only implement custom evaluation for novel function variants not in existing libraries. |
| A4 | **Classical proof strategy tooling** | PROJECT.md explicitly scopes this out. De la Vallee-Poussin methods, moment methods for percentage of zeros on critical line -- these are well-studied and haven't worked. | Focus all tooling on unconventional approaches: higher-dimensional, cross-disciplinary, information-theoretic. If a user wants classical tools, they can use SageMath. |
| A5 | **Web deployment / multi-user collaboration** | This is a local research tool for one user + Claude. Web infrastructure adds complexity with zero benefit for the research mission. | Single-user desktop application. Python scripts, Jupyter notebooks, local file storage. |
| A6 | **Publication/typesetting system** | LaTeX exists. Overleaf exists. Writing a paper formatter is irrelevant to proving RH. | Export results in formats compatible with LaTeX. Store conjectures in structured data that could feed into a paper later. |
| A7 | **Teaching or tutorial system** | The user directs exploration while Claude explains. Building pedagogical infrastructure is scope creep. | Claude itself is the "teacher" -- it explains concepts on demand in the conversation. No curriculum, no lesson plans, no graded exercises. |
| A8 | **GPU-accelerated zero computation at scale** | Large-scale zero verification (billions of zeros) has been done by Platt, Gourdon, etc. Reproducing this infrastructure requires specialized HPC knowledge and months of engineering. The goal is insight, not exhaustive verification. | Use published zero databases (LMFDB, Odlyzko's tables) for large datasets. Compute fresh zeros only in targeted regions where the platform's analysis suggests something interesting. |
| A9 | **Real-time collaboration features** | Multiplayer editing, shared sessions, conflict resolution -- massive engineering for a single-researcher tool. | The collaboration partner is Claude, which operates in the same session. File-based persistence is sufficient. |

---

## Feature Dependencies

```
FOUNDATION LAYER (must exist first):
  T1 (zeta evaluation) ──────────────────────────────────────┐
  T7 (related functions) ─────────────────────────────────────┤
                                                               │
VISUALIZATION LAYER (needs computation):                       │
  T3 (critical line viz) ← T1                                 │
  T4 (domain coloring) ← T1                                   │
  D9 (projection theater) ← D1 (higher-dim framework)         │
                                                               │
ANALYSIS LAYER (needs computation + data):                     │
  T2 (zero computation) ← T1                                  │
  T8 (zero statistics) ← T2                                   │
  D3 (RMT laboratory) ← T8                                    │
  D11 (anomaly detection) ← T8                                │
  D5 (information theory) ← T2, T8                            │
  D2 (spectral operators) ← T1                                │
  D10 (trace formulas) ← T1, T2                               │
  D4 (modular forms) ← T7                                     │
  D12 (adelic computation) ← T7                               │
                                                               │
CROSS-DISCIPLINARY LAYER (needs analysis):                     │
  D1 (higher-dim framework) ← T1 (independent core)           │
  D6 (analogy engine) ← D2, D3, D4, D5 (needs multiple       │
                         domains to draw analogies between)    │
                                                               │
RESEARCH INFRASTRUCTURE (parallel development):                │
  T5 (research workbench) ← (independent, but more useful     │
                              with computation features)       │
  T6 (reproducibility) ← T5                                   │
  T9 (verification framework) ← T1, T2                        │
  D7 (AI conjecture generation) ← T5, T9                      │
                                                               │
FORMALIZATION LAYER (last -- needs discoveries to formalize):  │
  D8 (Lean 4 pipeline) ← D7, T9 (needs conjectures worth     │
                          formalizing)                          │
```

### Critical Path

The longest dependency chain that gates meaningful research:

```
T1 (zeta eval) --> T2 (zeros) --> T8 (statistics) --> D3 (RMT) --> D6 (analogies)
     |                                                    ^
     └--> D2 (spectral) ─────────────────────────────────┘
```

This means: Phase 1 MUST deliver zeta evaluation and zero computation. Without these, nothing downstream works.

### Parallel Tracks

These can be developed independently and composed later:

- **Track A (computation):** T1 --> T2 --> T8 --> D3, D5, D11
- **Track B (visualization):** T3, T4 --> D9 (when D1 is ready)
- **Track C (higher-dim):** D1 (can start with toy examples before T1 is production-ready)
- **Track D (research infra):** T5, T6, T9, D7
- **Track E (formalization):** D8 (can scaffold pipeline before discoveries exist)

---

## MVP Recommendation

The MVP must answer one question: **"Can this platform show me something about the Riemann zeta function that I cannot trivially see in Mathematica or SageMath?"**

### MVP Feature Set (Phase 1 target)

**Prioritize (build first):**

1. **T1 -- Arbitrary-precision zeta evaluation** via mpmath wrapping. This is the foundation for everything. Without it, the platform computes nothing.

2. **T2 -- Non-trivial zero computation.** Immediate payoff: the user can compute and examine zeros. Use mpmath.zetazero() initially; optimize later if needed.

3. **T3 + T4 -- Critical line visualization + domain coloring.** The user is an explorer who builds intuition visually. These are how they "see" the mathematics. Interactive plots with zoom, pan, parameter controls.

4. **T5 -- Session-based research workbench (minimal).** At minimum: save/load experiment state, annotate results, track what was explored. Can start as structured JSON files + Jupyter notebooks.

5. **T8 -- Zero distribution statistics (basic).** Nearest-neighbor spacing, comparison against GUE. This is the first cross-disciplinary connection (RMT) and gives the user something genuinely interesting to explore.

### MVP Differentiator

6. **D1 -- Higher-dimensional framework (prototype).** Even a basic version -- represent zeros in higher-dimensional spaces, apply projections, see what structures emerge -- would demonstrate the platform's unique value. Start with the zeros as points in R^n (using spacing, derivative values, nearby zero structure as coordinates) and project via PCA/t-SNE.

### Defer (build later)

- **D2 (spectral operators):** Requires careful numerical linear algebra; high complexity for Phase 1.
- **D4 (modular forms):** Rich but self-contained; can be added as a module later.
- **D6 (analogy engine):** Needs multiple analysis domains to exist first.
- **D8 (Lean 4 pipeline):** No discoveries to formalize yet.
- **D9 (projection theater):** Real-time rendering is polish; static projections from D1 suffice initially.
- **D12 (adelic computation):** Specialized; defer until exploration suggests it is needed.

---

## Feature Prioritization Matrix

Prioritized by: (Impact on RH research) x (Enables other features) / (Complexity)

| Priority | Feature | Impact | Enables | Complexity | Score | Phase |
|----------|---------|--------|---------|------------|-------|-------|
| 1 | T1: Zeta evaluation | Critical | Everything | Medium | Highest | 1 |
| 2 | T2: Zero computation | Critical | T8, D3, D5, D10, D11 | Medium | Highest | 1 |
| 3 | T3: Critical line viz | High | User intuition | Low-Med | Very High | 1 |
| 4 | T4: Domain coloring | High | User intuition | Low | Very High | 1 |
| 5 | T5: Research workbench | High | D7, T6, T9 | Medium | High | 1 |
| 6 | T8: Zero statistics | High | D3, D5, D11 | Medium | High | 1 |
| 7 | D1: Higher-dim framework | High | D9, D6 | High | High (strategic) | 1-2 |
| 8 | T7: Related functions | Medium-High | D4, D10, D12 | Medium | Medium-High | 2 |
| 9 | D3: RMT laboratory | High | D6 | Medium | Medium-High | 2 |
| 10 | D2: Spectral operators | High | D6 | High | Medium | 2 |
| 11 | T9: Verification framework | Medium-High | D7, D8 | Medium | Medium | 2 |
| 12 | D5: Information theory | Medium | D6 | Medium | Medium | 2 |
| 13 | T6: Reproducibility | Medium | All experiments | Low-Med | Medium | 2 |
| 14 | D10: Trace formulas | Medium-High | D6 | Med-High | Medium | 3 |
| 15 | D11: Anomaly detection | Medium | Research velocity | Medium | Medium | 2-3 |
| 16 | D7: AI conjecture gen | High | D8 | Medium | Medium (needs data) | 3 |
| 17 | D4: Modular forms | Medium-High | D6 | High | Medium-Low | 3 |
| 18 | D9: Projection theater | Medium | User experience | High | Medium-Low | 3 |
| 19 | D6: Analogy engine | Very High | Proof discovery | High | Low (needs prereqs) | 4 |
| 20 | D8: Lean 4 pipeline | Critical (later) | Proof rigor | High | Low (needs results) | 4 |
| 21 | D12: Adelic computation | Medium | D6 | High | Low | 4+ |

---

## Key Feature Design Decisions

### Zeta Evaluation: Wrapping, Not Reimplementing

mpmath's `zeta()` supports arbitrary precision and is pure Python (portable, debuggable). For performance-critical paths (computing thousands of zeros), python-flint provides bindings to Arb/FLINT, which is orders of magnitude faster. The platform should abstract over both backends with a unified API:

```python
# User-facing API
zeta(s, precision=50)          # 50 decimal digits, uses mpmath
zeta(s, precision=50, fast=True)  # uses Arb if available, falls back to mpmath
```

### Visualization: Interactive, Not Static

The user explores visually. Static matplotlib plots are table stakes but insufficient. Need:
- **Zoom:** Into critical strip regions, around specific zeros
- **Parameter sliders:** Vary precision, number of terms, function parameters
- **Linked views:** Selecting a zero in one view highlights it in others
- **Animation:** Watch how structures change as parameters vary

Plotly for interactive 2D/3D. Matplotlib for publication-quality statics. Custom WebGL only if Plotly proves insufficient for projection theater (Phase 3+).

### Higher-Dimensional Framework: Start Concrete, Abstract Later

Don't build a general N-dimensional computing framework first. Start with specific higher-dimensional representations of zeta zeros:

1. **Zeros as points in R^k:** Each zero gets k coordinates (imaginary part, spacing to neighbors, derivative values, local statistics). Project to 2D via PCA, t-SNE, UMAP.
2. **Zeta as function on higher-dim spaces:** Evaluate on adelic completions, Hilbert spaces.
3. **Abstract framework:** Generalize once we know what representations are useful.

### Research Workbench: Structured Data, Not Just Notebooks

Jupyter notebooks are good for exploration but poor for structured research tracking. The workbench should store:

- **Conjectures:** Formal statement, evidence for/against, status, confidence
- **Experiments:** Parameters, results, interpretation, links to related experiments
- **Observations:** Freeform notes tagged to specific computations
- **Evidence chains:** Which experiments support/contradict which conjectures

This is essentially a research knowledge graph with computation nodes.

---

## Competitive Landscape (What Already Exists)

Existing tools the platform must exceed in the RH research niche:

| Tool | Strengths | Gaps (our opportunity) |
|------|-----------|----------------------|
| **SageMath** | Comprehensive CAS, number theory, modular forms, L-functions | No cross-disciplinary workflow, no higher-dim projection, no research tracking, clunky visualization |
| **Mathematica** | Beautiful visualization, symbolic + numeric, Manipulate for interactivity | Proprietary, no formalization pipeline, no AI guidance, not designed for research tracking |
| **LMFDB** | Massive database of L-functions, modular forms, number fields, elliptic curves | Read-only database, no computation, no exploration workflow |
| **mpmath** | Best arbitrary-precision Python library, excellent zeta support | Library not platform, no visualization, no workflow |
| **PARI/GP** | Fast number theory computation, designed for research | Archaic UI, limited visualization, no cross-disciplinary tools |
| **Lean 4 + Mathlib** | Machine-verified proofs, growing number theory library | No computational exploration, formalization only, steep learning curve |

**The gap we fill:** No existing tool combines arbitrary-precision zeta computation, interactive visualization, cross-disciplinary analysis (spectral, RMT, information theory, higher-dim geometry), structured research tracking, AI-guided conjecture generation, and formal verification pipeline -- all focused on RH.

---

## Sources and Confidence Notes

All findings in this document are based on training data (cutoff May 2025). Web verification tools were unavailable during research.

- **mpmath capabilities:** HIGH confidence -- well-established library, stable API, personally familiar from training data
- **Lean 4 / Mathlib status:** MEDIUM confidence -- rapidly evolving ecosystem, specific capabilities may have changed
- **LMFDB data availability:** HIGH confidence -- stable public database with well-documented API
- **Feature priorities:** MEDIUM confidence -- based on domain knowledge of RH research approaches, but actual user workflow may differ
- **Higher-dimensional approach novelty:** HIGH confidence that this is genuinely uncommon in existing tools; LOW confidence on specific implementation approach (this is inherently exploratory)
- **Competitive landscape:** MEDIUM confidence -- tools are well-known but exact current feature sets may have evolved
