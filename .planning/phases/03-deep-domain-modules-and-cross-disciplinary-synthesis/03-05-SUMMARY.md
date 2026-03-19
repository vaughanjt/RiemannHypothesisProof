---
phase: 03-deep-domain-modules-and-cross-disciplinary-synthesis
plan: 05
subsystem: analysis
tags: [conjecture-generation, experiment-suggestion, workbench-synthesis, package-exports]

# Dependency graph
requires:
  - phase: 03-01
    provides: spectral and trace formula modules
  - phase: 03-02
    provides: modular forms and LMFDB client modules
  - phase: 03-03
    provides: p-adic, TDA, dynamics modules
  - phase: 03-04
    provides: NCG, analogy modules
provides:
  - AI-guided conjecture generation (suggest_experiments, analyze_results, generate_conjecture)
  - Complete analysis package exports (all 10 Phase 3 domain modules, 61 public names)
affects: [phase-04, research-workflows, jupyter-notebooks]

# Tech tracking
tech-stack:
  added: []
  patterns: [workbench-synthesis-layer, keyword-based-anomaly-detection, bootstrap-vs-context-aware-suggestion]

key-files:
  created:
    - src/riemann/analysis/conjecture_gen.py
    - tests/test_analysis/test_conjecture_gen.py
  modified:
    - src/riemann/analysis/__init__.py

key-decisions:
  - "Keyword-based anomaly detection in result summaries (deviat/unexpect/anomal/surpris) for lightweight NLP without dependencies"
  - "Bootstrap suggestions cover 5 priority domains (spectral 0.9, analogy 0.85, tda 0.8, trace 0.75, ncg 0.7)"
  - "Context-aware suggestions use domain coverage ratio to prioritize under-explored areas"

patterns-established:
  - "Synthesis layer pattern: read workbench state -> analyze -> suggest next steps"
  - "All 10 Phase 3 analysis modules importable via from riemann.analysis import ..."

requirements-completed: [RSRCH-03]

# Metrics
duration: 5min
completed: 2026-03-19
---

# Phase 3 Plan 5: AI-Guided Conjecture Generation and Package Wiring Summary

**Conjecture synthesis layer with experiment suggestion (bootstrap + context-aware), result analysis with pattern/anomaly detection, evidence-linked conjecture generation, and complete Phase 3 package exports (61 names across 10 modules)**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-19T12:39:53Z
- **Completed:** 2026-03-19T12:45:05Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Implemented conjecture generation module with three core functions: suggest_experiments (prioritized next-step suggestions), analyze_results (structured experiment interpretation), and generate_conjecture (observation synthesis with workbench persistence)
- Wired all 10 Phase 3 analysis modules into the analysis package __init__.py with 61 alphabetically sorted public names
- Full test suite passes: 367 tests, zero regressions across all phases

## Task Commits

Each task was committed atomically:

1. **Task 1: Create AI-guided conjecture generation module** - `90a73e3` (test) + `c04d9d4` (feat)
2. **Task 2: Wire all Phase 3 modules into analysis package exports** - `5eb3cbf` (feat)

_Note: Task 1 was TDD with separate RED (test) and GREEN (feat) commits._

## Files Created/Modified
- `src/riemann/analysis/conjecture_gen.py` - ExperimentSuggestion dataclass, suggest_experiments, analyze_results, generate_conjecture
- `tests/test_analysis/test_conjecture_gen.py` - 12 tests covering all module functionality
- `src/riemann/analysis/__init__.py` - Added padic, tda, dynamics, conjecture_gen exports; updated __all__ to 61 names

## Decisions Made
- Keyword-based anomaly detection in result summaries (deviat/unexpect/anomal/surpris) provides lightweight NLP without adding ML dependencies
- Bootstrap suggestions cover 5 priority domains with descending priority (spectral 0.9, analogy 0.85, tda 0.8, trace 0.75, ncg 0.7) based on research impact
- Context-aware suggestions use domain coverage ratio to prioritize under-explored areas and speculative conjectures needing evidence

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Phase 3 is now complete: all 5 plans executed, all 10 analysis modules implemented and exported
- Full analysis package (riemann.analysis) provides unified access to spectral, trace formula, modular forms, LMFDB, p-adic, TDA, dynamics, NCG, analogy, and conjecture generation
- Research workbench integration is complete: experiments, conjectures, and evidence chains are all connected
- Ready for Phase 4: Lean 4 formalization and proof pathway construction

## Self-Check: PASSED

All files verified present. All commits verified in git log.

---
*Phase: 03-deep-domain-modules-and-cross-disciplinary-synthesis*
*Completed: 2026-03-19*
