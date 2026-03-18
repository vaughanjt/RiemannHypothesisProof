---
phase: 01-computational-foundation-and-research-workbench
plan: 02
subsystem: computation
tags: [mpmath, zeta, zeros, odlyzko, sqlite, precision, validation]

# Dependency graph
requires:
  - phase: 01-01
    provides: "Type system (ZetaZero, ComputationResult), precision management (validated_computation), config, Odlyzko data"
provides:
  - "zeta_eval(s) -- evaluate Riemann zeta at any complex point with always-validate P-vs-2P pattern"
  - "zeta_on_critical_line(t) -- evaluate zeta(0.5+it) using mpmath.mpf('0.5') precision"
  - "compute_zero(n) -- compute nth non-trivial zero via mpmath.zetazero with validation"
  - "compute_zeros_range(start, end) -- compute range of zeros"
  - "validate_against_odlyzko -- compare computed zeros to 1026-digit Odlyzko table"
  - "ZeroCatalog -- SQLite store/get/get_range/count with precision tracking"
  - "zero_count(T) -- N(T) via mpmath.nzeros for Riemann-von Mangoldt counting"
affects: [01-03, 01-04, 01-05]

# Tech tracking
tech-stack:
  added: []
  patterns: [zeta_eval always-validate wrapper, compute_zero returning validated ZetaZero, ZeroCatalog SQLite with precision-based replacement, Odlyzko validation pipeline]

key-files:
  created:
    - src/riemann/engine/zeta.py
    - src/riemann/engine/zeros.py
  modified:
    - tests/test_engine/test_zeta.py
    - tests/test_engine/test_zeros.py

key-decisions:
  - "zeta_eval delegates entirely to validated_computation(mpmath.zeta) -- no custom algorithm selection, mpmath handles Borwein/RS/Euler-Maclaurin internally"
  - "compute_zero checks on_critical_line by comparing Re(zero) to 0.5 with threshold 10^(-(dps-5))"
  - "ZeroCatalog stores string representations via mpmath.nstr at precision_digits+5 for SQLite portability"
  - "zero_count wraps mpmath.nzeros directly rather than implementing Riemann-von Mangoldt formula"
  - "validate_against_odlyzko defaults to min(dps-5, 45) tolerance digits"

patterns-established:
  - "zeta_eval(s, dps=N): standard entry point for all zeta evaluation -- never call mpmath.zeta directly"
  - "compute_zero(n, dps=N): standard entry point for zero computation -- never call mpmath.zetazero directly"
  - "ZeroCatalog with never-delete policy: only add or replace with higher precision"
  - "validate_against_odlyzko as verification pipeline: compare list of ZetaZero against reference table"

requirements-completed: [COMP-01, COMP-02]

# Metrics
duration: 5min
completed: 2026-03-18
---

# Phase 1 Plan 2: Zeta Evaluation Engine and Zero Catalog Summary

**zeta_eval with P-vs-2P always-validate at any complex point, compute_zero returning validated ZetaZero objects matching Odlyzko to 45+ digits, and SQLite ZeroCatalog with precision-tracked storage**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-18T22:38:45Z
- **Completed:** 2026-03-18T22:43:46Z
- **Tasks:** 2 (4 TDD commits: 2 RED + 2 GREEN)
- **Files modified:** 4

## Accomplishments
- zeta_eval evaluates Riemann zeta at any complex point with always-validate P-vs-2P, confirmed against zeta(2)=pi^2/6 to 45+ digits, functional equation verified, near-zero at first non-trivial zero
- compute_zero finds nth zero via mpmath.zetazero with validation, first 10 zeros match Odlyzko table to 45+ digits, all on critical line
- ZeroCatalog provides SQLite-based zero storage with precision-tracked replacement, store/get/get_range/count operations
- 17 total tests pass covering known values, functional equation, Odlyzko validation, catalog CRUD, and N(T) counting

## Task Commits

Each task was committed atomically (TDD: RED then GREEN):

1. **Task 1: Implement zeta evaluation engine with always-validate wrapper**
   - `34e4912` (test: TDD RED -- 7 failing tests)
   - `5d11d89` (feat: TDD GREEN -- zeta_eval and zeta_on_critical_line implemented)
2. **Task 2: Implement zero computation, Odlyzko validation, and SQLite catalog**
   - `0f764f5` (test: TDD RED -- 10 failing tests)
   - `6879596` (feat: TDD GREEN -- compute_zero, validate_against_odlyzko, ZeroCatalog, zero_count)

## Files Created/Modified
- `src/riemann/engine/zeta.py` - zeta_eval and zeta_on_critical_line with always-validate pattern
- `src/riemann/engine/zeros.py` - compute_zero, compute_zeros_range, validate_against_odlyzko, load_odlyzko_zeros, zero_count, ZeroCatalog
- `tests/test_engine/test_zeta.py` - 7 tests replacing xfail scaffolds (known values, functional equation, metadata, high precision)
- `tests/test_engine/test_zeros.py` - 10 tests replacing xfail scaffolds (Odlyzko validation, critical line, catalog CRUD, N(T))

## Decisions Made
- zeta_eval uses a single lambda wrapping mpmath.zeta rather than implementing algorithm selection -- mpmath internally dispatches to the optimal algorithm based on the argument region
- ZeroCatalog stores zero values as string representations (mpmath.nstr) rather than binary for SQLite portability and human readability
- validate_against_odlyzko caps tolerance at 45 digits by default since Odlyzko table precision varies and 45 is safe for 50-dps computation
- Added zero_count(T) function wrapping mpmath.nzeros to match existing test scaffold expectation for N(T) counting

## Deviations from Plan

None - plan executed exactly as written.

Note: The conftest.py odlyzko_zeros fixture required parsing at 1050 dps to read the 1026-digit Odlyzko values correctly (was truncating to float64 precision). This fix was already present in the working tree from a prior commit (054d34f) and did not require a separate fix in this plan.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- zeta_eval and compute_zero are the foundation for all downstream computation (visualization, stress-testing, L-functions)
- ZeroCatalog ready for use by workbench experiment tracking
- 17 tests provide regression safety for any refactoring of the computation engine
- The always-validate pattern ensures precision collapse is detected automatically in all downstream usage

## Self-Check: PASSED

- All 4 created/modified files verified present on disk
- All 4 commits (34e4912, 5d11d89, 0f764f5, 6879596) verified in git log
- Full test suite for this plan: 17 passed, 0 failures

---
*Phase: 01-computational-foundation-and-research-workbench*
*Completed: 2026-03-18*
