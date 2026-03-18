---
phase: 01-computational-foundation-and-research-workbench
plan: 05
subsystem: workbench
tags: [sqlite, conjecture, experiment, evidence, checksum, sha256, reproducibility]

# Dependency graph
requires:
  - phase: 01-01
    provides: "Python project with types.py (EvidenceLevel enum), config.py (DB_PATH, COMPUTED_DIR), temp_db fixture, workbench __init__.py"
provides:
  - "SQLite schema with 5 tables: conjectures, experiments, evidence_links, observations, conjecture_history"
  - "Conjecture CRUD with strict evidence-level enforcement (0-3 only)"
  - "Version history: updates archive old version before modifying"
  - "Evidence chain management: link experiments to conjectures with validated relationship types"
  - "Experiment save/load with full parameter serialization and SHA-256 checksums"
  - "Numpy result data storage as compressed .npz files"
  - "Tamper detection via checksum verification"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns: [get_connection context manager with auto-close, init_db idempotent schema creation, SHA-256 checksum for experiment reproducibility, version history via conjecture_history table]

key-files:
  created:
    - src/riemann/workbench/db.py
    - src/riemann/workbench/conjecture.py
    - src/riemann/workbench/experiment.py
    - src/riemann/workbench/evidence.py
  modified:
    - tests/test_workbench/test_conjecture.py
    - tests/test_workbench/test_experiment.py
    - tests/conftest.py

key-decisions:
  - "Function-based API (create_conjecture, save_experiment) instead of class-based (ConjectureTracker, ExperimentManager) -- simpler, matches plan specification"
  - "get_connection as contextmanager that auto-closes on exit -- critical for Windows where unclosed SQLite connections hold file locks"
  - "Separate conjecture_history table instead of BEFORE UPDATE trigger -- more explicit control over version archiving in application code"
  - "Checksum covers parameters + result_summary + result_data + precision_digits -- precision-stable per RESEARCH.md Open Question 4"

patterns-established:
  - "get_connection(db_path) as context manager: yields conn, auto-commits on success, rollback on exception, always closes"
  - "init_db(db_path): idempotent schema creation, safe to call multiple times"
  - "Evidence level enforcement: validate in application layer before INSERT (0-3 only, ValueError otherwise)"
  - "Version history: archive current to conjecture_history before UPDATE, increment version, set parent_version_id"
  - "Experiment checksums: SHA-256 of params_json + result_summary + precision_digits + result_data.tobytes()"

requirements-completed: [RSRCH-01, RSRCH-02]

# Metrics
duration: 6min
completed: 2026-03-18
---

# Phase 1 Plan 5: Research Workbench Summary

**SQLite research workbench with strict 4-level evidence hierarchy, conjecture version history, experiment parameter serialization with SHA-256 tamper detection, and numpy result storage**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-18T22:38:52Z
- **Completed:** 2026-03-18T22:45:01Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- SQLite schema with 5 tables (conjectures, experiments, evidence_links, observations, conjecture_history) and strict CHECK constraints for evidence levels (0-3) and relationship types
- Conjecture CRUD with strict evidence-level enforcement: values outside 0-3 raise ValueError. Updates never overwrite -- old versions archived to conjecture_history before modification
- Experiment save/load with full parameter JSON serialization, SHA-256 checksums, and numpy .npz storage. verify_checksum detects any post-save modification of parameters or results
- Evidence chain management linking experiments to conjectures with validated relationship types (supports/contradicts/neutral/extends)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create SQLite schema, conjecture CRUD, and evidence chain management**
   - `fc6744f` (test: TDD RED - 12 failing tests)
   - `054d34f` (feat: TDD GREEN - implementation passing all 12 tests)
2. **Task 2: Implement experiment save/load with parameter serialization and checksums**
   - `e1e9ee2` (test: TDD RED - 8 failing tests)
   - `44af00f` (feat: TDD GREEN - implementation passing all 8 tests)

## Files Created/Modified
- `src/riemann/workbench/db.py` - SQLite schema (5 tables), init_db, get_connection context manager
- `src/riemann/workbench/conjecture.py` - Conjecture CRUD with evidence-level enforcement and version history
- `src/riemann/workbench/evidence.py` - Evidence chain: link_evidence, get_evidence_for_conjecture
- `src/riemann/workbench/experiment.py` - Experiment save/load, parameter serialization, SHA-256 checksums, numpy storage
- `tests/test_workbench/test_conjecture.py` - 12 tests replacing xfail scaffolds (schema, CRUD, evidence levels, versioning, links)
- `tests/test_workbench/test_experiment.py` - 8 tests replacing xfail scaffolds (save/load, checksums, tamper detection, reproducibility)
- `tests/conftest.py` - Updated temp_db fixture for Windows file locking robustness

## Decisions Made
- Used function-based API (create_conjecture, save_experiment) instead of the class-based pattern in the original xfail scaffolds -- plan specification was authoritative
- Changed get_connection from returning a raw Connection to a contextmanager that auto-closes -- prevents Windows file locking issues with SQLite
- Used a separate conjecture_history table instead of a BEFORE UPDATE trigger -- more explicit version archiving logic in application code
- Checksum covers params_json + result_summary + precision_digits + result_data.tobytes() for precision-stable tamper detection

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed sqlite3.Connection context manager not closing on Windows**
- **Found during:** Task 1 (initial test run)
- **Issue:** sqlite3.Connection used as context manager (`with conn:`) commits but does NOT close the connection. On Windows, unclosed SQLite connections hold file locks, causing PermissionError when temp_db fixture tries to clean up
- **Fix:** Changed get_connection() to a @contextmanager that explicitly closes the connection in a finally block. Also hardened temp_db fixture to handle PermissionError gracefully
- **Files modified:** src/riemann/workbench/db.py, tests/conftest.py
- **Verification:** All 20 workbench tests pass without file locking errors
- **Committed in:** 054d34f (Task 1 commit)

**2. [Rule 1 - Bug] Replaced class-based test API with function-based API**
- **Found during:** Task 1 (test design)
- **Issue:** Original xfail scaffolds used ConjectureTracker/ExperimentManager class-based API, but plan specified function-based API (create_conjecture, save_experiment etc.)
- **Fix:** Rewrote all test scaffolds to use the function-based API matching the plan specification
- **Files modified:** tests/test_workbench/test_conjecture.py, tests/test_workbench/test_experiment.py
- **Verification:** All 20 tests pass with the function-based implementation
- **Committed in:** fc6744f, e1e9ee2 (TDD RED commits)

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both fixes necessary for correctness on Windows and API consistency. No scope creep.

## Issues Encountered
- The plan's smoke test commands using `:memory:` as db_path fail because each `get_connection(':memory:')` call creates a separate in-memory database (SQLite behavior). init_db creates schema in one connection, but the subsequent CRUD call opens a new empty database. This is expected SQLite behavior -- real usage always uses file-based databases.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Research workbench infrastructure complete: conjectures, experiments, evidence chains, and observations all trackable in SQLite
- All 85 tests pass across the full project (no regressions)
- Phase 1 workbench module is ready for use in notebooks via `from riemann.workbench.conjecture import create_conjecture` etc.
- Future phases can build on this for research tracking without modification

## Self-Check: PASSED

- All 7 created/modified files verified present on disk
- All 4 commits (fc6744f, 054d34f, e1e9ee2, 44af00f) verified in git log
- Full test suite: 85 passed, 0 failures

---
*Phase: 01-computational-foundation-and-research-workbench*
*Completed: 2026-03-18*
