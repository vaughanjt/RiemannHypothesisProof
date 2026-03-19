---
phase: 03-deep-domain-modules-and-cross-disciplinary-synthesis
plan: 04
subsystem: analysis
tags: [ncg, bost-connes, kms-states, analogy-engine, statistical-testing, scipy, mpmath]

# Dependency graph
requires:
  - phase: 01-computational-foundation-and-research-workbench
    provides: workbench experiment system (save_experiment, load_experiment), precision engine, types
provides:
  - Bost-Connes quantum statistical mechanical system (partition function, KMS states, phase transition)
  - Analogy engine for formal correspondence mappings between mathematical domains
  - Statistical correspondence testing via KS, chi-squared, and correlation metrics
  - Workbench persistence for analogy mappings
affects: [04-lean4-formalization, cross-disciplinary-synthesis, proof-pathway-discovery]

# Tech tracking
tech-stack:
  added: [scipy.stats.ks_2samp, scipy.stats.pearsonr]
  patterns: [Euler-Maclaurin tail correction for partition sums, dataclass serialization round-trip via workbench]

key-files:
  created:
    - src/riemann/analysis/ncg.py
    - src/riemann/analysis/analogy.py
    - tests/test_analysis/test_ncg.py
    - tests/test_analysis/test_analogy.py
  modified:
    - src/riemann/analysis/__init__.py

key-decisions:
  - "Euler-Maclaurin tail correction for partition function: improves convergence to zeta(beta) from ~1e-3 to <1e-4 at n_max=1000"
  - "KMS normalization uses raw partial sum (not tail-corrected) so probabilities sum to exactly 1.0"
  - "Analogy confidence update uses p-value thresholds: >0.05 increase, <0.01 decrease, between = no change"

patterns-established:
  - "Tail-corrected partial sums: separate raw partial sum from corrected version for different use cases"
  - "Dataclass to_dict/from_dict pattern for workbench serialization round-trip"

requirements-completed: [XDISC-04, XDISC-01]

# Metrics
duration: 8min
completed: 2026-03-19
---

# Phase 3 Plan 4: NCG Module and Analogy Engine Summary

**Bost-Connes partition function matching zeta(beta) to 4+ decimal places with KMS states, plus analogy engine with KS/chi-squared/correlation correspondence testing and workbench persistence**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-19T12:26:57Z
- **Completed:** 2026-03-19T12:35:00Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Bost-Connes partition function with Euler-Maclaurin tail correction converging to zeta(beta) within 1e-4 for n_max >= 1000
- KMS equilibrium state probabilities summing to exactly 1.0 with phase transition scanning
- AnalogyMapping dataclass with full serialization round-trip through workbench experiment system
- Statistical correspondence testing distinguishing similar from different distributions via three metrics
- 36 new tests all passing, zero regressions in existing 94 analysis tests

## Task Commits

Each task was committed atomically:

1. **Task 1: Create NCG module with Bost-Connes system** - `3f4f4b5` (feat) -- committed by prior agent execution
2. **Task 2: Create analogy engine with correspondence testing** - `fe963e4` (feat) -- committed by prior agent execution

_Note: Both modules were created by concurrent agent executions of other plans in the same wave. This plan validated their correctness, verified all 36 tests pass, and updated the analysis __init__.py exports._

## Files Created/Modified
- `src/riemann/analysis/ncg.py` - Bost-Connes system: BostConnesResult dataclass, partition function with tail correction, KMS states, phase transition scan
- `src/riemann/analysis/analogy.py` - Analogy engine: AnalogyMapping dataclass with serialization, correspondence testing (KS/chi2/correlation), workbench persistence, confidence updating
- `tests/test_analysis/test_ncg.py` - 18 tests: partition convergence, KMS normalization, phase transition entropy, error handling
- `tests/test_analysis/test_analogy.py` - 18 tests: dataclass round-trip, statistical testing, workbench persistence, confidence bounds
- `src/riemann/analysis/__init__.py` - Updated with NCG and analogy exports

## Decisions Made
- Euler-Maclaurin tail correction applied to partition function for better zeta(beta) convergence: integral approximation from N+0.5 to infinity yields sub-1e-4 accuracy at n_max=1000
- KMS values normalized by raw partial sum (not tail-corrected) so they form an exact probability distribution summing to 1.0
- Analogy confidence update rule: p > 0.05 means correspondence supported (+0.1), p < 0.01 means contradicted (-0.1), middle range is inconclusive (no change)
- Correlation metric uses sorted, length-matched arrays for meaningful comparison of distribution shapes

## Deviations from Plan

None - plan executed exactly as written. The files were created by concurrent agent executions but matched the plan specification exactly.

## Issues Encountered
- Both ncg.py and analogy.py were already committed by prior agent runs (plans 03-01 and 03-03 respectively). The implementations matched the plan spec, all tests pass, and no additional commits were needed for Task 1 or Task 2.
- Pre-existing test failures in test_dynamics.py, test_lmfdb_client.py (modules from other plans not yet complete) -- these are out of scope and do not affect this plan.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- NCG module provides computational access to Connes' approach to RH via the Bost-Connes system
- Analogy engine ready for cross-disciplinary synthesis work in Phase 4
- Statistical correspondence testing can be used to validate proposed analogies between domains
- All analysis modules now have comprehensive test coverage

## Self-Check: PASSED

- All 5 key files exist on disk
- Commit 3f4f4b5 (Task 1 NCG module) exists in git history
- Commit fe963e4 (Task 2 analogy engine) exists in git history
- SUMMARY.md created at expected path
- 36 tests passing, 0 regressions

---
*Phase: 03-deep-domain-modules-and-cross-disciplinary-synthesis*
*Completed: 2026-03-19*
