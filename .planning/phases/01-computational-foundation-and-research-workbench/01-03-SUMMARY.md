---
phase: 01-computational-foundation-and-research-workbench
plan: 03
subsystem: computation
tags: [mpmath, siegelz, dirichlet, xi-function, selberg-zeta, stress-test, validation, lfunctions]

# Dependency graph
requires:
  - phase: 01-01
    provides: "Precision management layer (validated_computation, precision_scope), type system (ComputationResult, PrecisionError), config (DEFAULT_DPS)"
provides:
  - "Hardy Z-function evaluation via mpmath.siegelz with validated_computation"
  - "Dirichlet L-function evaluation via mpmath.dirichlet with validated_computation"
  - "Xi function hand-built from zeta + gamma with symmetry xi(s) = xi(1-s)"
  - "Selberg zeta stub with informative NotImplementedError (Phase 2/3 placeholder)"
  - "Stress-test framework: stress_test() runs computations at escalating precisions to detect artifacts"
  - "StressTestResult dataclass with consistency flag, per-level results, max_deviation, metadata"
affects: [01-04, 01-05, phase-02, phase-03]

# Tech tracking
tech-stack:
  added: []
  patterns: [string-serialized inputs for precision-safe closures, cross-level consistency checking, predicate-based validation]

key-files:
  created:
    - src/riemann/engine/lfunctions.py
    - src/riemann/engine/validation.py
  modified:
    - tests/test_engine/test_lfunctions.py
    - tests/test_engine/test_validation.py

key-decisions:
  - "All function inputs string-serialized before closure capture to avoid precision truncation when validated_computation runs at 2P"
  - "Xi function tolerance set to dps-10 (not dps-5) because gamma/zeta/power chain amplifies precision loss"
  - "Stress-test fake pattern uses call-counter approach since dps-threshold approach cannot satisfy P-vs-2P validation"
  - "Selberg zeta implemented as stub with NotImplementedError per RESEARCH.md recommendation"

patterns-established:
  - "String-serialized closure inputs: convert mpmath numbers to strings before capturing in lambdas passed to validated_computation"
  - "stress_test(func, dps_levels=[...]): standard pattern for verifying any observed pattern persists at higher precision"
  - "StressTestResult.consistent: boolean flag indicating genuine pattern (True) vs artifact (False)"

requirements-completed: [COMP-03, COMP-04]

# Metrics
duration: 5min
completed: 2026-03-18
---

# Phase 1 Plan 3: Related Functions and Stress-Test Framework Summary

**Hardy Z, Dirichlet L, xi function with symmetry verification, Selberg zeta stub, and stress_test framework for distinguishing genuine patterns from numerical artifacts**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-18T22:38:51Z
- **Completed:** 2026-03-18T22:44:37Z
- **Tasks:** 2 (both TDD)
- **Files modified:** 4 (2 created, 2 modified)

## Accomplishments
- Four related function evaluators: hardy_z (mpmath.siegelz), dirichlet_l (mpmath.dirichlet), xi_function (hand-built from zeta+gamma), selberg_zeta_stub (NotImplementedError placeholder)
- Stress-test framework (stress_test + StressTestResult) that re-runs computations at escalating precisions to catch numerical artifacts -- the primary defense against Pitfall 2
- All inputs string-serialized before closure capture to prevent precision truncation in validated_computation's P-vs-2P pattern
- 20 new tests covering: real-valued Hardy Z, trivial character = zeta, xi symmetry, cross-level consistency, predicate checking, always-validate integration

## Task Commits

Each task was committed atomically (TDD RED then GREEN):

1. **Task 1: Implement Hardy Z, Dirichlet L, xi function, Selberg zeta stub**
   - `ef9dc84` (test: TDD RED - failing tests)
   - `978dbc3` (feat: TDD GREEN - implementation)
2. **Task 2: Implement stress-test framework for pattern verification**
   - `4aa20c2` (test: TDD RED - failing tests)
   - `8e652a7` (feat: TDD GREEN - implementation)

## Files Created/Modified
- `src/riemann/engine/lfunctions.py` - Hardy Z, Dirichlet L, xi function, Selberg zeta stub with validated_computation integration
- `src/riemann/engine/validation.py` - StressTestResult dataclass and stress_test() function for pattern verification
- `tests/test_engine/test_lfunctions.py` - 11 tests replacing 5 xfail scaffolds: real-valued Z, trivial character, xi symmetry, near-zero at zeta zeros, Selberg stub
- `tests/test_engine/test_validation.py` - 9 tests replacing 2 xfail scaffolds: default/custom levels, genuine/fake patterns, predicates, timing, P-vs-2P

## Decisions Made
- All function inputs converted to strings before closure capture to ensure precision is reconstructed at the working precision inside validated_computation (avoids truncation when P=50 captures input created at dps=55 but 2P=100 needs full precision)
- Xi function tolerance set to dps-10 (not default dps-5) because the 5-operation chain (multiply, power, gamma, zeta, multiply) amplifies precision loss
- Fake pattern test uses call-counter approach: a dps-threshold approach is fundamentally incompatible with validated_computation's P-vs-2P check (any threshold has a level where P is below and 2P is above)
- Xi symmetry tests use string-based mpmath construction (e.g., `mpmath.mpc("0.3", "5")`) and compute `1-s` via mpmath arithmetic, because Python float `0.7` is not exactly `7/10`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed xi function closure precision truncation**
- **Found during:** Task 1 (xi_function implementation)
- **Issue:** Input `s_mp` was constructed at caller's precision (55 dps) but the lambda was evaluated at P=55 and 2P=105; the xi formula amplified the truncation error, causing the symmetry test to fail (only 17 digits agreement instead of 40)
- **Fix:** String-serialize all inputs before closure capture, reconstruct mpmath numbers at the working precision inside the lambda
- **Files modified:** src/riemann/engine/lfunctions.py (all three functions: hardy_z, dirichlet_l, xi_function)
- **Verification:** Xi symmetry test passes with 40+ digit agreement
- **Committed in:** 978dbc3

**2. [Rule 1 - Bug] Fixed xi symmetry test float64 input precision**
- **Found during:** Task 1 (test debugging)
- **Issue:** Test constructed `mpmath.mpc(0.7, -5)` from Python float `0.7`, which is not exactly `7/10`. This meant `s=0.3+5i` and `1-s=0.7-5i` were NOT exactly related by the symmetry transformation.
- **Fix:** Use `mpmath.mpc("0.3", "5")` with string construction and compute `1-s` via mpmath arithmetic for exact pairs
- **Files modified:** tests/test_engine/test_lfunctions.py
- **Verification:** Symmetry test achieves 40+ digit agreement
- **Committed in:** 978dbc3

**3. [Rule 1 - Bug] Fixed fake pattern test incompatibility with P-vs-2P validation**
- **Found during:** Task 2 (stress_test tests)
- **Issue:** Original approach used dps-threshold to make function return different values at different levels, but this is fundamentally incompatible with validated_computation: at ANY threshold, some level has P below and 2P above, triggering PrecisionError
- **Fix:** Used call-counter approach where function changes after the first two levels (4 calls = 2 levels x 2 calls each)
- **Files modified:** tests/test_engine/test_validation.py
- **Verification:** Fake pattern correctly detected as inconsistent (consistent=False)
- **Committed in:** 8e652a7

---

**Total deviations:** 3 auto-fixed (3 bugs)
**Impact on plan:** All auto-fixes address precision-related edge cases inherent to arbitrary-precision computation. The closure serialization pattern is a net improvement applicable to all future function wrappers. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviations above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Computation engine complete: zeta evaluation (01-02), zero-finding (01-02), related functions (01-03), and stress-testing (01-03) all operational
- String-serialized closure pattern established for any future function wrappers that pass lambdas to validated_computation
- stress_test() ready for use in notebooks and downstream exploration to verify any observed patterns
- Full test suite: 85 passed, 0 failures

## Self-Check: PASSED

- All 5 files verified present on disk (2 created, 2 modified, 1 summary)
- All 4 commits (ef9dc84, 978dbc3, 4aa20c2, 8e652a7) verified in git log
- Full test suite: 85 passed, 0 failures

---
*Phase: 01-computational-foundation-and-research-workbench*
*Completed: 2026-03-18*
