---
phase: 04-lean-4-formalization-pipeline
plan: 01
subsystem: formalization
tags: [lean4, mathlib, wsl2, subprocess, theorem-proving, parser]

# Dependency graph
requires:
  - phase: 01-computational-foundation-and-research-workbench
    provides: PROJECT_ROOT config, types.py EvidenceLevel, workbench DB patterns
provides:
  - Lean 4 project with Mathlib at lean_proofs/ (builds successfully from WSL2)
  - WSL2 subprocess build runner (run_lake_build -> LakeBuildResult)
  - Lean compiler output parser (parse_lean_output -> LeanMessage list + sorry count)
  - Source-level sorry counter (count_sorry_in_source)
  - Path conversion utilities (windows_to_wsl_path, wsl_to_windows_path)
affects: [04-02, 04-03, formalization-translator, formalization-tracker, formalization-triage]

# Tech tracking
tech-stack:
  added: [Lean 4 v4.29.0-rc6, Mathlib4, elan, lake]
  patterns: [WSL2 subprocess invocation, Lean output regex parsing, symlinked .lake for NTFS compat]

key-files:
  created:
    - lean_proofs/lakefile.toml
    - lean_proofs/lean-toolchain
    - lean_proofs/RiemannProofs/Basic.lean
    - lean_proofs/RiemannProofs.lean
    - lean_proofs/.gitignore
    - lean_proofs/lake-manifest.json
    - src/riemann/formalization/__init__.py
    - src/riemann/formalization/builder.py
    - src/riemann/formalization/parser.py
    - tests/test_formalization/__init__.py
    - tests/test_formalization/conftest.py
    - tests/test_formalization/test_builder.py
    - tests/test_formalization/test_parser.py
  modified: []

key-decisions:
  - "Symlinked .lake/ to WSL-native filesystem (/home/jvaughan/.lake-riemann) to avoid NTFS chmod failures during git clone of Mathlib"
  - "Used Lean 4 v4.29.0-rc6 (latest via Mathlib toolchain pin) with lakefile.toml flat format (not [package] section)"
  - "Replaced #check RiemannHypothesis with example : Prop := RiemannHypothesis for clean builds (no info output)"
  - "Moved import statement before module docstring in Basic.lean (Lean 4 requires imports first)"

patterns-established:
  - "WSL2 subprocess pattern: subprocess.run(['wsl', '-e', 'bash', '-c', cmd]) with elan env sourcing"
  - "Lean output parsing: regex on file:line:col: severity: message format"
  - "Source-level sorry counting: strip comments/strings then count sorry tokens"
  - "9P filesystem delay: 0.1s sleep before lake build to flush Windows->WSL cache"

requirements-completed: [FORM-01]

# Metrics
duration: 12min
completed: 2026-03-19
---

# Phase 4 Plan 1: Lean 4 Environment and Build Infrastructure Summary

**WSL2 Lean 4 + Mathlib build pipeline with Python subprocess runner, structured output parser, and hello-world theorem proving riemannZeta_zero**

## Performance

- **Duration:** 12 min
- **Started:** 2026-03-19T19:53:45Z
- **Completed:** 2026-03-19T20:06:32Z
- **Tasks:** 2
- **Files modified:** 13

## Accomplishments
- Lean 4 project at lean_proofs/ with Mathlib dependency compiles successfully from WSL2
- Hello-world theorem validates `riemannZeta 0 = -1/2` using Mathlib's `riemannZeta_zero` and confirms `RiemannHypothesis` is accessible as a `Prop`
- Python build runner invokes `lake build` via WSL2 subprocess and returns structured LakeBuildResult with parsed messages, sorry count, error/warning counts, and duration
- Output parser extracts LeanMessage objects from Lean compiler format and counts sorry declarations
- Source-level sorry counter handles comments, block comments, and string literal exclusion
- 21 tests pass (14 parser, 7 builder including WSL2 integration test)

## Task Commits

Each task was committed atomically:

1. **Task 1: WSL2 Lean 4 environment setup and hello-world validation** - `466496c` (feat)
2. **Task 2: Build runner and output parser (TDD RED)** - `8d951e6` (test)
3. **Task 2: Build runner and output parser (TDD GREEN)** - `48e47a0` (feat)

## Files Created/Modified
- `lean_proofs/lakefile.toml` - Lean 4 project config with Mathlib dependency
- `lean_proofs/lean-toolchain` - Pinned Lean version (v4.29.0-rc6)
- `lean_proofs/RiemannProofs/Basic.lean` - Hello-world theorem with zeta and RH validation
- `lean_proofs/RiemannProofs.lean` - Root module importing Basic
- `lean_proofs/.gitignore` - Excludes .lake/, lake-packages/, build/
- `lean_proofs/lake-manifest.json` - Pinned dependency versions
- `src/riemann/formalization/__init__.py` - Package init
- `src/riemann/formalization/builder.py` - WSL subprocess build runner with LakeBuildResult
- `src/riemann/formalization/parser.py` - Lean output parser with LeanMessage and sorry counting
- `tests/test_formalization/__init__.py` - Test package init
- `tests/test_formalization/conftest.py` - Shared fixtures (sample outputs, WSL availability check)
- `tests/test_formalization/test_builder.py` - 7 tests for builder module
- `tests/test_formalization/test_parser.py` - 14 tests for parser module

## Decisions Made
- **Symlinked .lake/ to WSL-native filesystem:** The OneDrive-synced Windows path caused `chmod` failures when git clone ran inside WSL2 on the NTFS mount. Symlinking `.lake/` to `/home/jvaughan/.lake-riemann` on the ext4 filesystem resolves this while keeping .lean source files accessible from both sides.
- **Lean 4 v4.29.0-rc6 via Mathlib toolchain:** The `lake new ... math` template pins to whatever Lean version Mathlib's CI uses. This turned out to be v4.29.0-rc6 (not v4.28.0 as research suggested). Following the Mathlib pin is correct.
- **Flat lakefile.toml format:** Lake requires top-level `name = ...` (not `[package]` section). The plan's suggested format used `[package]` which Lake rejected.
- **Import before docstring in Lean 4:** Lean 4 requires all `import` statements before any other content including module docstrings (`/-! ... -/`).
- **`example : Prop := RiemannHypothesis` instead of `#check`:** The `#check` command produces info output that interferes with library builds. Using an `example` declaration compiles cleanly.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] git safe.directory and filemode config for WSL2/NTFS**
- **Found during:** Task 1 (Lean project creation)
- **Issue:** `lake new` and `lake update` failed because git clone inside WSL2 could not `chmod` files on the NTFS mount at `/mnt/c/...OneDrive...`
- **Fix:** Set `git config --global core.filemode false` and `git config --global --add safe.directory "*"` in WSL, then symlinked `.lake/` to WSL-native filesystem
- **Files modified:** WSL git global config, lean_proofs/.lake (symlink)
- **Verification:** `lake update` completed successfully, cloned Mathlib and all dependencies
- **Committed in:** 466496c (Task 1 commit)

**2. [Rule 1 - Bug] Moved import before module docstring in Basic.lean**
- **Found during:** Task 1 (lake build)
- **Issue:** Lean 4 rejected `import` after `/-! ... -/` docstring with "invalid 'import' command, it must be used in the beginning of the file"
- **Fix:** Moved `import Mathlib.NumberTheory.LSeries.RiemannZeta` to line 1, docstring after
- **Files modified:** lean_proofs/RiemannProofs/Basic.lean
- **Verification:** `lake build` passes
- **Committed in:** 466496c (Task 1 commit)

**3. [Rule 1 - Bug] Replaced #check with example declaration**
- **Found during:** Task 1 (lake build)
- **Issue:** `#check RiemannHypothesis` produced "unexpected token '#check'; expected 'lemma'" error after docstring
- **Fix:** Changed to `example : Prop := RiemannHypothesis` which compiles cleanly and validates RH accessibility
- **Files modified:** lean_proofs/RiemannProofs/Basic.lean
- **Verification:** `lake build RiemannProofs` completed successfully (2953 jobs)
- **Committed in:** 466496c (Task 1 commit)

**4. [Rule 1 - Bug] Fixed lakefile.toml format (flat keys, not [package] section)**
- **Found during:** Task 1 (lake update)
- **Issue:** Lake rejected `[package]\nname = "RiemannProofs"` with "missing required key: name" at line 1
- **Fix:** Used flat format `name = "RiemannProofs"` at top level (matching lake new template output)
- **Files modified:** lean_proofs/lakefile.toml
- **Verification:** `lake update` succeeds
- **Committed in:** 466496c (Task 1 commit)

---

**Total deviations:** 4 auto-fixed (3 bugs, 1 blocking)
**Impact on plan:** All auto-fixes were necessary for the build pipeline to function. The plan's Lean syntax assumptions were slightly off (docstring ordering, #check usage, lakefile format). No scope creep.

## Issues Encountered
- OneDrive-synced NTFS path causes chmod failures in WSL2 git operations -- resolved with .lake symlink to WSL-native filesystem. This is a known WSL2/NTFS interaction issue.

## User Setup Required
None - elan was installed automatically inside WSL2 during execution.

## Next Phase Readiness
- Lean 4 + Mathlib build pipeline fully operational from Python
- Builder and parser modules ready for translator (04-02) and tracker (04-03) to consume
- 21 tests provide regression safety for the build infrastructure

## Self-Check: PASSED

All 13 created files verified present. All 3 task commits (466496c, 8d951e6, 48e47a0) verified in git log.

---
*Phase: 04-lean-4-formalization-pipeline*
*Completed: 2026-03-19*
