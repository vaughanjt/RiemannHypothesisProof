# Phase 4: Lean 4 Formalization Pipeline - Context

**Gathered:** 2026-03-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver a complete pipeline from research workbench conjectures to machine-verified Lean 4 proofs: conjecture selection, Lean 4 theorem generation with Mathlib integration, build/check via WSL2, structured error reporting, sorry-count tracking, automatic evidence-level promotion, and formalization progress tracking. Then execute a full assault — attempt to formalize every viable conjecture in the workbench.

</domain>

<decisions>
## Implementation Decisions

### Lean 4 Runtime Environment
- Lean 4 runs inside WSL2 (already installed and working on the machine)
- Install elan + lake inside WSL2 as a Wave 0 validation task
- Python calls Lean via subprocess: writes .lean files to a shared path, calls `wsl lake build`, parses output
- Lean project lives on Windows filesystem accessible from both Python and WSL
- Wave 0 must validate the full chain: elan install -> lake init -> hello-world theorem -> `wsl lake build` from Python succeeds

### Translation Approach
- Claude generates Lean 4 code directly from conjecture statements (no template scaffolding)
- Full Mathlib integration: Lean project depends on Mathlib, Claude uses existing formalized number theory (primes, zeta, complex analysis, measure theory) as building blocks
- Evidence mapping: each .lean file includes docstrings with linked experiment IDs, computational evidence summaries, and confidence scores from the workbench
- Storage: .lean files in a Lean project directory + workbench DB stores file path, formalization status, sorry count, last build result

### Proof Workflow & Tracking
- 4-state pipeline per conjecture: `not_formalized` -> `statement_formalized` (theorem + sorry) -> `proof_attempted` (partial tactics, reduced sorry count) -> `proof_complete` (zero sorries, clean build)
- Automatic sorry-count parsing after every `lake build`: extract sorry count, error count, warning count, build success from Lean compiler output
- Track sorry reduction over time per conjecture
- Structured error reporting: parse Lean output into {file, line, error_type, message, suggestion} for Claude to iterate on proof fixes
- Store build history in workbench (every build attempt with timestamp, sorry count, errors)
- Auto-promote: zero sorries + clean build = evidence_level automatically set to 3 (FORMAL_PROOF), closing the loop from Phase 1's type system

### Scope & Ambition
- Full assault: build the pipeline AND aggressively attempt to formalize every viable conjecture
- Claude triages the workbench and picks optimal attack order per conjecture (considers confidence, Mathlib proximity, novelty)
- Time-box per conjecture: if sorry count isn't decreasing after N attempts, move to next and revisit later
- Success = pipeline works end-to-end AND at least some conjectures advance through the formalization states

### Carrying Forward from Phases 1-3
- JupyterLab, Claude-driven exploration
- Function-based API, SQLite workbench pattern
- EvidenceLevel.FORMAL_PROOF = 3 ("X is verified in Lean 4") already defined in types.py
- Conjecture CRUD with versioning already exists in workbench/conjecture.py
- Strict evidence hierarchy enforcement

### Claude's Discretion
- elan/Lean 4 version selection (latest stable)
- Lean project structure and lakefile configuration
- Mathlib module selection per conjecture
- Proof tactic strategies and lemma decomposition
- Time-box threshold (how many attempts before moving on)
- Conjecture triage ordering algorithm
- Build output parsing heuristics
- Whether to decompose a conjecture into sub-lemmas before formalizing

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Context
- `.planning/PROJECT.md` -- Core value, constraints, key decisions
- `.planning/REQUIREMENTS.md` -- FORM-01, FORM-02

### Prior Phases
- `.planning/phases/01-computational-foundation-and-research-workbench/01-CONTEXT.md` -- Phase 1 decisions (interface, workbench, precision, evidence hierarchy)
- `.planning/phases/02-higher-dimensional-analysis/02-CONTEXT.md` -- Phase 2 decisions (embedding, statistics)
- `.planning/phases/03-deep-domain-modules-and-cross-disciplinary-synthesis/03-CONTEXT.md` -- Phase 3 decisions (domain modules, AI conjecture generation)

### Existing Code (Phases 1-3 outputs)
- `src/riemann/types.py` -- EvidenceLevel enum (FORMAL_PROOF = 3), ZetaZero, ComputationResult
- `src/riemann/workbench/conjecture.py` -- Conjecture CRUD with versioning and evidence levels
- `src/riemann/workbench/db.py` -- SQLite connection management, schema initialization
- `src/riemann/workbench/experiment.py` -- Experiment save/load with checksums
- `src/riemann/workbench/evidence.py` -- Evidence chain tracking
- `src/riemann/analysis/conjecture_gen.py` -- AI-guided conjecture generation (suggest_experiments, analyze_results, generate_conjecture)

### Research
- `.planning/research/STACK.md` -- Technology choices
- `.planning/research/PITFALLS.md` -- "Lean 4 / elan on Windows Server 2025 needs validation" blocker

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `workbench/conjecture.py`: create_conjecture, update_conjecture, list_conjectures, get_conjecture_history -- direct integration for formalization tracking
- `workbench/db.py`: get_connection context manager, init_db -- extend schema for formalization tables
- `types.py`: EvidenceLevel enum with FORMAL_PROOF = 3 -- auto-promotion target
- `analysis/conjecture_gen.py`: suggest_experiments, generate_conjecture -- source of conjectures to formalize

### Established Patterns
- Function-based API with dataclass results
- SQLite for structured data with context manager connections
- TDD with pytest red-green cycle
- Subprocess execution for external tools (pattern needed for `wsl lake build`)

### Integration Points
- Formalization module reads from workbench DB (conjectures table)
- New `formalizations` table tracks: conjecture_id, lean_file_path, formalization_state, sorry_count, last_build_result, build_history
- Build runner calls `wsl lake build` via subprocess, parses output
- Auto-promotion updates conjecture evidence_level via existing update_conjecture function
- Lean project directory: `lean_proofs/` at project root (contains lakefile.lean, Mathlib dependency, generated .lean files)

</code_context>

<specifics>
## Specific Ideas

- The full assault is the point -- this isn't infrastructure for its own sake, it's the final push to turn computational discoveries into rigorous mathematics
- Claude triages and attacks conjectures autonomously, time-boxing to avoid rabbit holes
- The auto-promotion from clean build to FORMAL_PROOF evidence level closes the loop defined in Phase 1's type system -- the whole platform was building toward this moment
- Build history tracking enables Claude to learn which proof strategies work and which don't across multiple attempts

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 04-lean-4-formalization-pipeline*
*Context gathered: 2026-03-19*
