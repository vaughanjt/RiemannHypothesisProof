---
phase: 04-lean-4-formalization-pipeline
plan: 03
subsystem: formalization
tags: [lean4, mathlib, triage, assault, scoring, tdd, state-machine]

# Dependency graph
requires:
  - phase: 04-lean-4-formalization-pipeline
    provides: "tracker.py (FormalizationState, record_build, auto_promote_if_clean), translator.py (generate_lean_file, TranslationResult), builder.py (run_lake_build, LakeBuildResult)"
provides:
  - "Conjecture triage with 4-factor scoring (confidence, Mathlib proximity, continuation, novelty)"
  - "Formalization assault runner with translation, state advancement, building, and time-boxing"
  - "Complete formalization package exports (24 public names from 5 submodules)"
  - "TriageEntry, AssaultResult, AssaultOutcome dataclasses"
  - "10-domain Mathlib proximity map for scoring"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns: ["4-factor weighted triage scoring formula", "assault runner with time-boxing on stalling sorry counts", "state machine bridge: explicit proof_attempted advancement before build loop"]

key-files:
  created:
    - src/riemann/formalization/triage.py
    - tests/test_formalization/test_triage.py
  modified:
    - src/riemann/formalization/__init__.py

key-decisions:
  - "Triage scoring formula: 0.4*confidence + 0.3*mathlib_proximity + 0.2*continuation_bonus + 0.1*novelty with 10-domain proximity map"
  - "State machine bridge: assault runner explicitly advances state from statement_formalized to proof_attempted before build loop, enabling auto_promote_if_clean to reach proof_complete"
  - "Time-boxing: assault stops building a conjecture when sorry count is not decreasing across attempts"

patterns-established:
  - "Weighted multi-factor scoring for prioritization with domain-specific proximity values"
  - "Assault runner pattern: triage -> translate -> advance state -> build loop -> record -> time-box"
  - "Complete package __init__.py with __all__ list for public API surface"

requirements-completed: [FORM-01, FORM-02]

# Metrics
duration: 5min
completed: 2026-03-19
---

# Phase 4 Plan 3: Triage and Formalization Assault Summary

**Conjecture triage with 4-factor scoring and full formalization assault runner that translates, builds, time-boxes, and auto-promotes conjectures through the complete Lean 4 pipeline**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-19T20:20:00Z
- **Completed:** 2026-03-19T20:25:00Z
- **Tasks:** 2 (Task 1 TDD with RED+GREEN, Task 2 human-verify checkpoint)
- **Files modified:** 3

## Accomplishments
- Triage module scores conjectures by confidence (40%), Mathlib proximity (30%), continuation bonus (20%), and novelty (10%) across 10 mathematical domains
- Assault runner translates conjectures to Lean 4, advances state machine to proof_attempted before build loop, records builds, and time-boxes on stalling sorry counts
- Complete package __init__.py exports 24 public names from all 5 submodules (builder, parser, tracker, translator, triage)
- Full state machine traversal verified: not_formalized -> statement_formalized -> proof_attempted -> proof_complete on zero-sorry clean builds
- End-to-end pipeline verified via human checkpoint: Lean 4 builds from Python, 55/55 formalization tests pass, 20/20 workbench tests pass

## Task Commits

Each task was committed atomically (TDD: RED then GREEN):

1. **Task 1 (RED): Failing tests for triage and assault** - `261da2d` (test)
2. **Task 1 (GREEN): Triage module, package exports, assault runner** - `2fc0d0e` (feat)
3. **Task 2: Human verification checkpoint** - approved (no commit)

## Files Created/Modified
- `src/riemann/formalization/triage.py` - Conjecture triage scoring (triage_conjectures) and formalization assault runner (run_formalization_assault) with TriageEntry, AssaultOutcome, AssaultResult dataclasses
- `src/riemann/formalization/__init__.py` - Complete package public API with 24 exports from all 5 submodules and __all__ list
- `tests/test_formalization/test_triage.py` - 12 tests covering triage scoring, exclusion, domain ranking, continuation bonus, assault runner, state advancement, and auto-promotion

## Decisions Made
- Triage scoring formula: 0.4*confidence + 0.3*mathlib_proximity + 0.2*continuation_bonus + 0.1*novelty -- weights emphasize confidence and Mathlib proximity as primary drivers
- 10-domain Mathlib proximity map (spectral 0.9, modular 0.85, trace 0.8, padic 0.75, ncg 0.5, dynamics 0.4, tda 0.3, cross_domain 0.3, analogy 0.2, default 0.5)
- State machine bridge: assault runner must explicitly advance from statement_formalized to proof_attempted before the build loop because _VALID_TRANSITIONS only allows proof_attempted -> proof_complete
- Time-boxing: if sorry count does not decrease across consecutive build attempts, the assault moves on to the next conjecture

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Phase 4 is complete: all 3 plans executed, all 75 tests green (55 formalization + 20 workbench)
- The full Lean 4 formalization pipeline is operational: triage -> translate -> build -> track -> auto-promote
- Platform is feature-complete for v1 milestone across all 4 phases (18 plans total)

## Self-Check: PASSED

- All 3 files verified on disk (triage.py, __init__.py, test_triage.py)
- All 2 task commits verified in git history (261da2d, 2fc0d0e)

---
*Phase: 04-lean-4-formalization-pipeline*
*Completed: 2026-03-19*
