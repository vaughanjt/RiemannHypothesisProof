# Phase 5: Heat Kernel Feasibility Gate - Context

**Gathered:** 2026-04-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Compute the heat kernel trace Tr(e^{-tΔ}) on SL(2,Z)\H with both discrete Maass eigenvalue sum and continuous Eisenstein spectrum. Identify the parameter mapping t = t(L) connecting barrier parameter L to heat kernel time t. Validate agreement between K(t(L)) and B(L) at 100+ L values. Establish dual-precision computation (mpmath + python-flint). Determine whether the heat kernel interpretation of the Connes barrier is viable for a proof.

</domain>

<decisions>
## Implementation Decisions

### Kill Gate Criteria
- **D-01:** Agreement threshold is proof-grade — whatever is necessary for a rigorous proof, not an arbitrary digit count. The feasibility check must demonstrate that the heat kernel trace and barrier agree well enough to build a proof on.
- **D-02:** No artificial kill threshold on the Eisenstein continuous spectrum contribution. The magnitude is data — Phase 7 (correction bounds) determines if it's fatal. Don't pre-judge.
- **D-03:** Test at 100+ L values spanning the full range (L ~ 1 to 50+), with density concentrated where the barrier margin is smallest.
- **D-04:** Diagnostic output is BOTH summary table (L, K(t), B(L), digits of agreement, verdict) AND convergence plots (spectral sum convergence, agreement heatmap).

### Parameter Discovery
- **D-05:** Pursue analytic derivation AND numerical fitting in parallel. Derive t(L) from the Lorentzian test function structure; simultaneously fit numerically by optimizing t for each L. Cross-validate: analytic formula must match numerical fit.
- **D-06:** Claude judges whether t(L) mapping complexity is acceptable for downstream proof assembly. Simple expressions preferred, but special functions acceptable if well-defined and computable.

### Maass Eigenvalue Source
- **D-07:** Claude's discretion on data source (LMFDB, published tables, or combination). Existing `lmfdb_client.py` should be extended if LMFDB is chosen.
- **D-08:** Claude's discretion on eigenvalue count, driven by convergence diagnostics. Start with enough for convergence at the operating range, expand if needed.

### Dual Precision Pattern
- **D-09:** Always-dual computation: every computation runs in both mpmath and python-flint, flagging disagreement. This catches subtle precision bugs early.
- **D-10:** Claude's discretion on the integration pattern (wrapper functions vs backend flag vs other), picking what integrates best with the existing `validated_computation` and function-based API conventions.

### Carrying Forward from Phases 1-4
- Function-based API, returns data (not plots), pluggable into analysis pipeline
- mpmath + `validated_computation` (P-vs-2P precision validation) pattern
- SQLite workbench for tracking experiments and conjectures
- LMFDB client with SQLite caching (`analysis/lmfdb_client.py`)
- Modular forms module with Eisenstein E_k, q-series, Hecke eigenvalues (`analysis/modular_forms.py`)
- Trace formula module with Weil explicit formula (`analysis/trace_formula.py`)
- Stress-test framework for pattern validation (`engine/validation.py`)
- JupyterLab, Claude-driven exploration
- 50-digit default precision, always-validate

### Claude's Discretion
- Maass eigenvalue data source and count
- Dual-precision integration pattern
- t(L) mapping complexity acceptance
- Heat kernel spectral sum truncation strategy
- Continuous spectrum numerical integration method
- Module organization within `src/riemann/analysis/`
- All performance optimization decisions

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Context
- `.planning/PROJECT.md` — Core value, constraints, current milestone goals
- `.planning/REQUIREMENTS.md` — HEAT-01 through HEAT-04
- `.planning/ROADMAP.md` — Phase 5 goal and success criteria

### Research
- `.planning/research/SUMMARY.md` — Executive summary with roadmap implications
- `.planning/research/STACK.md` — python-flint 0.8.0 recommendation, no other new deps
- `.planning/research/FEATURES.md` — Continuous spectrum circularity flag, feature dependencies
- `.planning/research/PITFALLS.md` — Circularity traps, 7-question master checklist, 0.036 margin
- `.planning/research/ARCHITECTURE.md` — Module placement in `src/riemann/analysis/`, integration points

### Prior Phase Context
- `.planning/phases/01-computational-foundation-and-research-workbench/01-CONTEXT.md` — Precision, workbench, evidence hierarchy
- `.planning/phases/03-deep-domain-modules-and-cross-disciplinary-synthesis/03-CONTEXT.md` — Module architecture pattern, modular forms decisions, LMFDB integration
- `.planning/phases/04-lean-4-formalization-pipeline/04-CONTEXT.md` — Lean 4 pipeline decisions, WSL2 setup

### Existing Code
- `src/riemann/analysis/modular_forms.py` — Eisenstein series, q-series, Hecke eigenvalues
- `src/riemann/analysis/trace_formula.py` — Weil explicit formula, Chebyshev psi
- `src/riemann/analysis/spectral.py` — Berry-Keating Hamiltonian, eigenvalue computation
- `src/riemann/analysis/lmfdb_client.py` — REST API client with SQLite caching
- `src/riemann/engine/lfunctions.py` — L-function evaluation (Hardy Z, Dirichlet L, xi)
- `src/riemann/engine/precision.py` — validated_computation (P-vs-2P)
- `src/riemann/engine/validation.py` — stress_test framework
- `src/riemann/types.py` — ComputationResult, ZetaZero, EvidenceLevel

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `analysis/modular_forms.py`: Eisenstein series E_k computation — extend for continuous spectrum Eisenstein integral
- `analysis/trace_formula.py`: Weil explicit formula — connect to Selberg trace (GL(1)->GL(2) bridge in Phase 6)
- `analysis/lmfdb_client.py`: REST API + SQLite cache — extend for `mf_maass_newforms` collection queries
- `engine/precision.py`: `validated_computation` — wrap new heat kernel functions in same P-vs-2P pattern
- `engine/validation.py`: `stress_test` — use for verifying parameter mapping t(L) across precision levels

### Established Patterns
- All analysis functions return dataclasses (e.g., `ModularFormResult`, `TraceFormulaResult`)
- Function-based API, no classes for computation (consistent across all 4 phases)
- mpmath for arbitrary precision; string-serialize inputs before closure capture
- SQLite with `get_connection` context manager for all persistent storage
- `data/` directory for static reference data (maass_forms.json, zeros.db)

### Integration Points
- New heat kernel module connects to `modular_forms.py` (Eisenstein), `lmfdb_client.py` (Maass data)
- Results feed into workbench experiments for tracking
- Convergence diagnostics connect to stress-test framework
- python-flint ball arithmetic is a new backend alongside mpmath — needs adapter layer

</code_context>

<specifics>
## Specific Ideas

- Session 47 key insight: barrier's Lorentzian test function w_hat(n) = n/(L^2 + 16*pi^2*n^2) is morally the heat kernel on the modular surface. Heat kernel trace = sum exp(-lambda_k * L) > 0 always.
- Ramanujan-level convergence (~14 digits/term) vs Euler product (~1 digit/many terms) is the critical advantage.
- The barrier B(L) = W02(L) - Mp(L) has been verified positive at 800+ points up to lambda^2 = 50,000.
- Margin-drain gap is 0.036 (margin=0.264, drain=0.228) — bounds must be extremely tight.
- Must be non-circular: cannot assume RH to prove RH.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 05-heat-kernel-feasibility-gate*
*Context gathered: 2026-04-04*
