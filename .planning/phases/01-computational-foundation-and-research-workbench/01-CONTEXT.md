# Phase 1: Computational Foundation and Research Workbench - Context

**Gathered:** 2026-03-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver arbitrary-precision zeta function evaluation, non-trivial zero computation/cataloging, related function evaluation (L-functions, Hardy's Z, xi, Selberg), a numerical verification framework, interactive critical line visualization, complex plane domain coloring, a structured research workbench, and experiment reproducibility. This is the foundational layer — everything in Phases 2-4 depends on these capabilities being correct and trustworthy.

</domain>

<decisions>
## Implementation Decisions

### Interface Paradigm
- JupyterLab is the primary interface — all exploration happens in notebooks
- Claude is the primary driver: Claude builds notebooks, runs computations, analyzes results, presents findings
- The user directs exploration ("what if we look at X?") and Claude does the mathematical heavy lifting
- Claude has full discretion on notebook organization (by topic, by session, or whatever makes mathematical sense)

### Visualization Style
- Optimize for speed over visual polish — Claude is the primary consumer of visualizations
- Start coarse, zoom to refine on demand (progressive resolution for domain coloring)
- Claude picks visualization tools per use case (matplotlib for static analysis, Plotly for interactive when needed)
- Claude picks color schemes optimized for analytical clarity, not aesthetics

### Research Workbench Data Model
- SQLite for structured research tracking (conjectures, experiments, evidence chains, metadata)
- numpy files for numerical data persistence (computed zeros, function values)
- HDF5 (h5py) for large array storage when needed (Phase 2+ primarily)
- Strict evidence-level hierarchy from day 1: every finding tagged as observation / heuristic / conditional / formal proof
- The user emphasized: "the only way to understand this problem is to hold in context a whole mess of variables — go strict and heavy"

### Precision Management
- Default precision: 50 decimal digits (user intuition: "this problem will be cracked in under 50 digits")
- Always-validate mode: every computation runs at P and 2P digits, results compared to catch silent precision collapse
- mpmath for all critical strip evaluation — never use float64/scipy near the critical line
- Claude manages precision escalation as needed without user intervention
- gmpy2 as mpmath's C backend for 2-10x acceleration

### Claude's Discretion
- Notebook organization strategy
- Color scheme selection per visualization type
- Specific matplotlib vs Plotly decisions per plot
- Domain coloring resolution levels at each zoom stage
- Evidence hierarchy subcategories beyond the four main levels
- Workbench schema design (SQLite table structure)
- Precision escalation thresholds
- Error handling and edge case management
- File organization for computed data

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Context
- `.planning/PROJECT.md` — Core value, constraints, key decisions
- `.planning/REQUIREMENTS.md` — COMP-01 through COMP-04, VIZ-01, VIZ-02, RSRCH-01, RSRCH-02

### Research
- `.planning/research/STACK.md` — Technology choices: mpmath, gmpy2, numpy, scipy, sympy, matplotlib, plotly, jupyterlab
- `.planning/research/FEATURES.md` — Feature details for T1-T9 (table stakes)
- `.planning/research/ARCHITECTURE.md` — Component boundaries, data flow, build order
- `.planning/research/PITFALLS.md` — Critical: silent precision collapse, infrastructure addiction, confusing evidence with proof
- `.planning/research/SUMMARY.md` — Executive synthesis of all research

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- None — greenfield project, no existing code

### Established Patterns
- None — patterns will be established in this phase

### Integration Points
- This phase establishes the foundation. All subsequent phases integrate with:
  - Computation engine (zeta evaluation, zero-finding APIs)
  - Visualization layer (plot generation functions)
  - Research workbench (conjecture/experiment tracking database)
  - Data store (SQLite + numpy file conventions)

</code_context>

<specifics>
## Specific Ideas

- User believes the proof is structural, not hiding in extreme precision — hence 50-digit default
- Claude-as-mathematician model: Claude doesn't just compute, it actively analyzes, forms hypotheses, and drives exploration
- The strict evidence hierarchy is non-negotiable — the user sees it as essential to managing the complexity of the problem

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-computational-foundation-and-research-workbench*
*Context gathered: 2026-03-18*
