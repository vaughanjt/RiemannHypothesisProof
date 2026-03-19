---
phase: 02-higher-dimensional-analysis
plan: 02
subsystem: analysis
tags: [numpy, random-matrix-theory, gue, goe, gse, eigenvalues, wigner-surmise, statistics]

# Dependency graph
requires:
  - phase: 01-computational-foundation-and-research-workbench
    provides: numpy, scipy, project structure, ZetaZero types
provides:
  - GUE/GOE/GSE random matrix ensemble generation
  - Eigenvalue spacing statistics with semicircle-law unfolding
  - Wigner surmise probability densities for beta=1,2,4
  - fit_effective_n residual analysis tool for zero-vs-RMT comparison
affects: [02-03, 02-04, 02-05, anomaly-detection, information-theory]

# Tech tracking
tech-stack:
  added: []
  patterns: [semicircle-law unfolding, index-based spectral bulk selection, seeded RNG for reproducibility]

key-files:
  created:
    - src/riemann/analysis/rmt.py
    - tests/test_analysis/test_rmt.py
    - src/riemann/analysis/__init__.py
    - tests/test_analysis/__init__.py
  modified: []

key-decisions:
  - "Index-based bulk trimming (central 80%) for semicircle unfolding instead of value-based cutoff -- more stable across matrix sizes"
  - "Convergence test measures chi-squared distance from Wigner surmise rather than raw variance -- better captures universal limit approach"
  - "GSE uses 2N x 2N block quaternion structure with degenerate pair collapsing (every-other eigenvalue)"

patterns-established:
  - "Ensemble generation pattern: generate_{ensemble}(n, num_matrices, seed) -> list[np.ndarray]"
  - "Seeded RNG via np.random.default_rng(seed) for all stochastic computation"
  - "eigenvalue_spacings returns np.ndarray of normalized spacings matching spacing.py convention"

requirements-completed: [RMT-01, RMT-02]

# Metrics
duration: 4min
completed: 2026-03-19
---

# Phase 02 Plan 02: Random Matrix Theory Summary

**GUE/GOE/GSE ensemble generation with semicircle-law unfolded eigenvalue spacings, Wigner surmise for beta=1,2,4, and fit_effective_n residual analysis for zero-RMT comparison**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-19T02:24:26Z
- **Completed:** 2026-03-19T02:28:32Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 4

## Accomplishments
- GUE/GOE/GSE ensemble generation at configurable matrix size N with seeded RNG
- Semicircle-law eigenvalue unfolding producing normalized spacings with mean=1.000
- Wigner surmise exact formulas for all three Dyson indices (GOE beta=1, GUE beta=2, GSE beta=4)
- fit_effective_n tool that scans GUE(N) for best chi-squared match against target spacings
- Statistical verification: GUE(N=100) spacing histogram matches Wigner surmise within chi-squared tolerance
- Convergence verification: GUE(N=200) fits Wigner surmise better than GUE(N=10)

## Task Commits

Each task was committed atomically (TDD RED then GREEN):

1. **Task 1 RED: Failing tests for RMT module** - `fde144b` (test)
2. **Task 1 GREEN: RMT implementation passing all tests** - `aac2055` (feat)

## Files Created/Modified
- `src/riemann/analysis/rmt.py` - GUE/GOE/GSE generation, eigenvalue spacings, Wigner surmise, fit_effective_n
- `src/riemann/analysis/__init__.py` - Analysis module exports (rmt functions)
- `tests/test_analysis/__init__.py` - Test package init
- `tests/test_analysis/test_rmt.py` - 14 tests covering all RMT functions

## Decisions Made
- **Index-based bulk trimming** for semicircle unfolding: trim 10% of eigenvalues from each spectral edge (by sorted index) rather than using a fixed value cutoff. This gives more consistent behavior across matrix sizes since the fraction of bulk eigenvalues is stable regardless of N.
- **Convergence test via chi-squared distance from Wigner surmise** rather than raw spacing variance. The Wigner surmise variance is a fixed constant; what changes with N is how closely the empirical distribution matches theory. Chi-squared distance captures this directly.
- **GSE quaternion structure** uses 2N x 2N block representation [[A, B], [-B*, A*]] with degenerate pair collapsing (every-other eigenvalue from eigvalsh).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed convergence test metric**
- **Found during:** Task 1 GREEN phase
- **Issue:** Original test compared raw spacing variance between GUE(N=10) and GUE(N=50). Due to semicircle unfolding edge effects, smaller N artificially had lower variance because more aggressive trimming kept only the most central eigenvalues.
- **Fix:** Changed test to measure chi-squared distance from theoretical Wigner surmise distribution, which is the correct convergence metric. Also increased N contrast (N=10 vs N=200) for clearer separation.
- **Files modified:** tests/test_analysis/test_rmt.py
- **Verification:** Test passes; GUE(N=200) chi-squared is demonstrably lower than GUE(N=10)
- **Committed in:** aac2055 (part of GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix in test)
**Impact on plan:** Test metric improved to correctly capture convergence behavior. No scope creep.

## Issues Encountered
None beyond the test metric adjustment documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- RMT module complete and ready for comparison against zero distribution statistics (Plan 01 spacing.py)
- eigenvalue_spacings output format matches spacing.normalized_spacings convention for direct overlay
- fit_effective_n enables the residual analysis workflow the user requested ("at what effective N do zeros best match GUE?")
- Phase 1 tests verified: all 85 pass unchanged

## Self-Check: PASSED

- All 4 created files verified on disk
- Both commits (fde144b, aac2055) verified in git log
- 14/14 tests pass
- All 6 exported functions present in rmt.py
- Phase 1 tests: 85/85 pass

---
*Phase: 02-higher-dimensional-analysis*
*Completed: 2026-03-19*
