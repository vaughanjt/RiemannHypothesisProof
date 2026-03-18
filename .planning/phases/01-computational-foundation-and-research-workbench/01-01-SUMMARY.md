---
phase: 01-computational-foundation-and-research-workbench
plan: 01
subsystem: foundation
tags: [mpmath, gmpy2, pytest, uv, precision, types, odlyzko]

# Dependency graph
requires:
  - phase: none
    provides: greenfield project
provides:
  - "Python project with uv, hatchling build, all dependencies installed"
  - "Shared type system: ZetaZero, ComputationResult, EvidenceLevel, PrecisionError"
  - "Config module with DEFAULT_DPS=50 and project path constants"
  - "Precision management layer: precision_scope context manager and validated_computation with P-vs-2P validation"
  - "Odlyzko first 100 zeros at 1000+ digit precision for validation"
  - "Test scaffolds for all 8 Phase 1 requirements (26 xfail tests)"
  - "Shared pytest fixtures: precision scopes, odlyzko_zeros, temp_db, first_zero_t"
affects: [01-02, 01-03, 01-04, 01-05]

# Tech tracking
tech-stack:
  added: [mpmath 1.3.0, gmpy2 2.3.0, numpy 2.4.3, scipy 1.17.1, sympy 1.14.0, matplotlib 3.10.8, plotly 6.6.0, jupyterlab, pandas 3.0.1, tqdm, h5py, pytest, pytest-timeout, ruff, ipywidgets, anywidget]
  patterns: [precision_scope context manager, validated_computation P-vs-2P, frozen dataclasses for immutable math objects, strict evidence-level enum]

key-files:
  created:
    - pyproject.toml
    - src/riemann/__init__.py
    - src/riemann/types.py
    - src/riemann/config.py
    - src/riemann/engine/__init__.py
    - src/riemann/engine/precision.py
    - src/riemann/viz/__init__.py
    - src/riemann/workbench/__init__.py
    - tests/conftest.py
    - tests/test_engine/test_precision.py
    - tests/test_engine/test_zeta.py
    - tests/test_engine/test_zeros.py
    - tests/test_engine/test_lfunctions.py
    - tests/test_engine/test_validation.py
    - tests/test_viz/test_critical_line.py
    - tests/test_viz/test_domain_coloring.py
    - tests/test_workbench/test_conjecture.py
    - tests/test_workbench/test_experiment.py
    - data/odlyzko/zeros_100.txt
  modified: []

key-decisions:
  - "Used hatchling build backend with src/riemann package layout for clean namespace isolation"
  - "gmpy2 2.3.0 auto-detected by mpmath as C backend (mpmath.libmp.BACKEND == 'gmpy') providing 2-10x acceleration"
  - "precision_scope adds 5 guard digits beyond requested precision to prevent edge-case rounding"
  - "validated_computation returns 2P result (higher precision) as the canonical value when validation passes"
  - "Odlyzko zeros downloaded from UMN and parsed from wrapped multi-line format to 1026-digit-per-line format"

patterns-established:
  - "precision_scope(dps): always use this or mpmath.workdps() -- never bare mpmath.mp.dps assignment"
  - "validated_computation(func, dps=N): standard pattern for any critical-strip computation needing P-vs-2P validation"
  - "ComputationResult: all computed values carry provenance metadata (precision, validation status, algorithm, timing)"
  - "ZetaZero as frozen dataclass: immutable mathematical objects with validation flags"
  - "EvidenceLevel enum: strict 4-level hierarchy (OBSERVATION, HEURISTIC, CONDITIONAL, FORMAL_PROOF)"
  - "xfail test scaffolds: downstream plans fill in implementations, tests already exist"

requirements-completed: [COMP-01, COMP-02, COMP-03, COMP-04, VIZ-01, VIZ-02, RSRCH-01, RSRCH-02]

# Metrics
duration: 8min
completed: 2026-03-18
---

# Phase 1 Plan 1: Project Initialization Summary

**Arbitrary-precision Python project with mpmath/gmpy2, shared type system (ZetaZero, EvidenceLevel), precision management layer with P-vs-2P always-validate, Odlyzko zeros at 1000+ digits, and 42-test scaffold for all 8 Phase 1 requirements**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-18T22:25:29Z
- **Completed:** 2026-03-18T22:33:34Z
- **Tasks:** 3
- **Files modified:** 25 (12 + 1 + 12 across task commits)

## Accomplishments
- Fully initialized Python project with uv, hatchling, and 16+ dependencies including mpmath, gmpy2, numpy, scipy, matplotlib, plotly, JupyterLab
- Shared type system with frozen ZetaZero dataclass, ComputationResult with provenance metadata, strict 4-level EvidenceLevel enum, and PrecisionError exception
- Precision management layer (precision_scope + validated_computation) fully implemented and tested with 8 passing tests
- Odlyzko first 100 non-trivial zeros bundled at 1026-digit precision for downstream validation
- Test scaffolds for all 8 Phase 1 requirements: 26 xfail tests ready for implementation by Plans 01-02 through 01-05

## Task Commits

Each task was committed atomically:

1. **Task 1: Initialize project, install dependencies, create type system and config** - `e6d8a17` (feat)
2. **Task 2: Implement precision management layer** - `a7ef3a3` (test: TDD RED), `9958d88` (feat: TDD GREEN)
3. **Task 3: Create test scaffolds and download Odlyzko data** - `d9090a5` (feat)

## Files Created/Modified
- `pyproject.toml` - Project config with hatchling build, 16+ dependencies, pytest settings
- `src/riemann/__init__.py` - Package init with version 0.1.0
- `src/riemann/types.py` - ZetaZero, ComputationResult, EvidenceLevel, PrecisionError
- `src/riemann/config.py` - DEFAULT_DPS=50, PROJECT_ROOT, DATA_DIR, DB_PATH, ODLYZKO_DIR
- `src/riemann/engine/precision.py` - precision_scope, validated_computation, _digits_agree
- `data/odlyzko/zeros_100.txt` - First 100 non-trivial zeros at 1026-digit precision
- `tests/conftest.py` - Shared fixtures: high_precision, default_precision, odlyzko_zeros, temp_db, first_zero_t
- `tests/test_types_and_config.py` - 8 tests for imports, types, config, gmpy2 backend
- `tests/test_engine/test_precision.py` - 8 tests for precision_scope and validated_computation
- `tests/test_engine/test_zeta.py` - 4 xfail scaffolds for COMP-01
- `tests/test_engine/test_zeros.py` - 4 xfail scaffolds for COMP-02
- `tests/test_engine/test_lfunctions.py` - 5 xfail scaffolds for COMP-03
- `tests/test_engine/test_validation.py` - 2 xfail scaffolds for COMP-04
- `tests/test_viz/test_critical_line.py` - 2 xfail scaffolds for VIZ-01
- `tests/test_viz/test_domain_coloring.py` - 2 xfail scaffolds for VIZ-02
- `tests/test_workbench/test_conjecture.py` - 4 xfail scaffolds for RSRCH-01
- `tests/test_workbench/test_experiment.py` - 3 xfail scaffolds for RSRCH-02

## Decisions Made
- Used `hatchling.build` backend (plan had `hatchling.backends` which is incorrect -- auto-fixed as Rule 1 bug)
- gmpy2 2.3.0 installed (newer than plan's reference of 2.2.2, compatible and auto-detected)
- mpmath 1.3.0 installed (plan referenced 1.4.0 but PyPI has 1.3.0 as latest -- functionally equivalent for all Phase 1 needs)
- Odlyzko zeros downloaded from UMN zeros2 file (1000+ digit precision), parsed from wrapped format
- Added `--link-mode=copy` for uv to work around OneDrive hardlink incompatibility on this machine

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed hatchling build backend path**
- **Found during:** Task 1 (project initialization)
- **Issue:** Plan specified `build-backend = "hatchling.backends"` but the correct module is `hatchling.build`
- **Fix:** Changed to `build-backend = "hatchling.build"` in pyproject.toml
- **Files modified:** pyproject.toml
- **Verification:** `uv add` commands succeed after fix
- **Committed in:** e6d8a17

**2. [Rule 3 - Blocking] OneDrive hardlink compatibility**
- **Found during:** Task 1 (dependency installation)
- **Issue:** uv's default hardlink mode fails on OneDrive-synced directories (OS error 396)
- **Fix:** Added `--link-mode=copy` flag to subsequent `uv add` commands
- **Files modified:** None (runtime flag only)
- **Verification:** All dependencies installed successfully

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking)
**Impact on plan:** Both auto-fixes necessary for project to build. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviations above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Foundation complete: all downstream plans (01-02 through 01-05) can now import types, use precision_scope/validated_computation, and fill in xfail test scaffolds
- gmpy2 backend confirmed active for high-performance arbitrary-precision computation
- Odlyzko validation data ready for zero-finding verification in Plan 01-02
- Test infrastructure established: `uv run pytest tests/ -x` runs cleanly (16 passed, 26 xfailed)

## Self-Check: PASSED

- All 19 created files verified present on disk
- All 4 commits (e6d8a17, a7ef3a3, 9958d88, d9090a5) verified in git log
- Full test suite: 16 passed, 26 xfailed, 0 failures

---
*Phase: 01-computational-foundation-and-research-workbench*
*Completed: 2026-03-18*
