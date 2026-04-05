---
phase: 05-heat-kernel-feasibility-gate
plan: 01
subsystem: engine
tags: [python-flint, dual-precision, ball-arithmetic, mpmath, arb]

# Dependency graph
requires:
  - phase: 01-computational-foundation-and-research-workbench
    provides: validated_computation pattern, mpmath precision management, ComputationResult type
provides:
  - python-flint dependency installed and importable
  - dual_compute function for running mpmath + python-flint in parallel
  - DualResult, BarrierComparison, ConvergenceDiagnostic dataclasses
  - Test scaffolds for all HEAT-01 through HEAT-04 requirements
affects: [05-02-PLAN, 05-03-PLAN, heat-kernel-trace, barrier-comparison]

# Tech tracking
tech-stack:
  added: [python-flint 0.8.0]
  patterns: [dual-precision comparison via mpmath string conversion, arb midpoint extraction via regex]

key-files:
  created:
    - src/riemann/engine/dual_precision.py
    - tests/test_heat_kernel.py
  modified:
    - pyproject.toml
    - src/riemann/types.py
    - src/riemann/engine/__init__.py

key-decisions:
  - "Full-precision comparison via mpmath, not float64 (float caps at ~15 digits, dual_compute needs dps-level)"
  - "Flint arb midpoint extracted via str(n_digits) + regex parse to avoid ball-format string issues"
  - "Catastrophic threshold at dps-20, flag threshold at dps-10 (user-configurable)"
  - "Set flint_ctx.prec before calling func_flint, restore after"

patterns-established:
  - "dual_compute pattern: func_mpmath (no args, runs in workdps) + func_flint (takes prec arg)"
  - "arb-to-mpmath conversion via decimal string intermediate"

requirements-completed: [HEAT-04]

# Metrics
duration: 7min
completed: 2026-04-04
---

# Phase 5 Plan 01: Dual-Precision Foundation Summary

**python-flint 0.8.0 installed with dual_compute backend that runs mpmath + flint arb in parallel, compares at full precision, and flags disagreement**

## Performance

- **Duration:** 7 min
- **Started:** 2026-04-05T02:07:41Z
- **Completed:** 2026-04-05T02:15:20Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- python-flint 0.8.0 installed and verified (arb ball arithmetic on Windows)
- dual_compute function compares mpmath and python-flint results at full arbitrary precision (not float64-limited)
- DualResult, BarrierComparison, and ConvergenceDiagnostic types ready for Plans 02-03
- Test scaffolds: 2 real tests passing, 6 stubs for downstream plans

## Task Commits

Each task was committed atomically:

1. **Task 1: Install python-flint, create DualResult type, scaffold tests** - `398f7f7` (feat)
2. **Task 2: Create dual-precision computation backend** - `8fcbb94` (feat, TDD)

## Files Created/Modified
- `pyproject.toml` - Added python-flint>=0.7.0 dependency
- `src/riemann/types.py` - Added DualResult, BarrierComparison, ConvergenceDiagnostic dataclasses
- `src/riemann/engine/dual_precision.py` - dual_compute, DualPrecisionError, dps_to_prec, _arb_to_mpmath helper
- `src/riemann/engine/__init__.py` - Exports dual_compute, DualPrecisionError, dps_to_prec
- `tests/test_heat_kernel.py` - 2 real tests + 6 stubs for Plans 02-03

## Decisions Made
- **Full-precision comparison via mpmath:** Converting flint arb midpoint to mpmath number via decimal string intermediate, because float64 truncation caps comparison at ~15 digits (insufficient for dps=50 computations).
- **Regex extraction of arb midpoint:** Flint arb.str(n_digits) returns `[midpoint +/- error]` format; regex parse extracts the midpoint string for mpmath conversion.
- **Catastrophic vs flag thresholds:** Catastrophic disagreement (raises) at dps-20, flag threshold (sets flagged=True) at dps-10. Both are configurable.
- **flint_ctx.prec management:** Set global flint context precision before func_flint call, restore after (similar to mpmath.workdps pattern).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed float64 truncation in agreement computation**
- **Found during:** Task 2 (dual_compute implementation)
- **Issue:** Plan specified using `float()` conversion for comparison, but float64 is limited to ~15 digits. With dps=50, the catastrophic threshold (dps-20=30) was impossible to satisfy via floats.
- **Fix:** Convert flint arb midpoint to mpmath number via decimal string (arb.str(n_digits) + regex parse), compare entirely in mpmath at full precision.
- **Files modified:** src/riemann/engine/dual_precision.py
- **Verification:** test_dual_compute_basic passes with agreement_digits > 10 (actually ~50)
- **Committed in:** 8fcbb94

**2. [Rule 1 - Bug] Fixed disagreement test threshold interaction**
- **Found during:** Task 2 (test_dual_compute_flags_disagreement)
- **Issue:** Test used dps=50 with threshold=1, but catastrophic threshold (dps-20=30) triggers before flag threshold for completely wrong values. Test expected flagged=True, got DualPrecisionError.
- **Fix:** Changed test to use dps=15 (catastrophic threshold becomes -5, always satisfied), letting flag threshold work as intended.
- **Files modified:** tests/test_heat_kernel.py
- **Verification:** test_dual_compute_flags_disagreement passes, flagged=True, agreement_digits < 1
- **Committed in:** 8fcbb94

---

**Total deviations:** 2 auto-fixed (2 bug fixes)
**Impact on plan:** Both fixes necessary for correctness. The float64 limitation is fundamental -- plan's algorithm was correct in spirit but needed mpmath for the comparison step. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviations above.

## User Setup Required
None - python-flint installed automatically via pip.

## Next Phase Readiness
- dual_compute ready for use in Plan 02 (heat kernel trace module)
- DualResult, BarrierComparison, ConvergenceDiagnostic types ready
- Test stubs in place for Plans 02 and 03
- All 6 stub tests skip cleanly, ready to be filled in

## Self-Check: PASSED

- All 5 created/modified files exist on disk
- Commit 398f7f7 (Task 1) found in git log
- Commit 8fcbb94 (Task 2) found in git log
- `python -m pytest tests/test_heat_kernel.py -x -q` returns 2 passed, 6 skipped

---
*Phase: 05-heat-kernel-feasibility-gate*
*Completed: 2026-04-04*
