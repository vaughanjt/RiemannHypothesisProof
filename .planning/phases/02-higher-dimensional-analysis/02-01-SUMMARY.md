---
phase: 02-higher-dimensional-analysis
plan: 01
subsystem: analysis
tags: [numpy, scipy, scikit-learn, umap-learn, statistics, spacing, pair-correlation, gue, embedding, dataclass]

# Dependency graph
requires:
  - phase: 01-computational-foundation-and-research-workbench
    provides: "ZetaZero dataclass, workbench experiment save/load, config paths"
provides:
  - "normalized_spacings function for zero spacing analysis"
  - "pair_correlation and gue_pair_correlation for GUE comparison"
  - "n_level_density and number_variance statistics"
  - "EmbeddingConfig frozen dataclass for N-dimensional embedding specs"
  - "FEATURE_EXTRACTORS registry with 9 feature extractor stubs"
  - "4 preset configs: spectral_basic, spectral_full, information_space, kitchen_sink"
  - "scikit-learn and umap-learn available as Phase 2 dependencies"
affects: [02-02-rmt-comparison, 02-03-feature-extractors, 02-04-anomaly-detection, 02-05-projection-theater]

# Tech tracking
tech-stack:
  added: [scikit-learn, umap-learn, numba, llvmlite, pynndescent, joblib, threadpoolctl]
  patterns: [function-based-statistics-api, frozen-dataclass-config, feature-extractor-registry, stub-with-not-implemented]

key-files:
  created:
    - src/riemann/analysis/spacing.py
    - src/riemann/embedding/registry.py
    - src/riemann/embedding/__init__.py
    - tests/test_analysis/test_spacing.py
    - tests/test_embedding/test_registry.py
    - tests/test_embedding/__init__.py
  modified:
    - src/riemann/analysis/__init__.py
    - pyproject.toml
    - uv.lock

key-decisions:
  - "Function-based statistics API: all spacing functions are standalone, no classes, return numpy arrays, never plot"
  - "Local mean spacing via asymptotic formula 2*pi/log(t/(2*pi)) for normalization at each midpoint"
  - "Pair correlation computed from cumulative pairwise gaps with adaptive normalization to approach 1.0 at large x"
  - "Feature extractor stubs raise NotImplementedError with descriptive message naming the feature -- Plan 02-03 fills them in"
  - "EmbeddingConfig is a frozen dataclass for safe sharing and workbench persistence"
  - "4 preset configs covering 3D fast (spectral_basic) to 9D exhaustive (kitchen_sink) embedding strategies"

patterns-established:
  - "Statistics functions: accept list[ZetaZero] or np.ndarray, return np.ndarray, raise ValueError on empty input"
  - "Feature registry pattern: dict[str, Callable] mapping names to extractors, stubs replaced incrementally"
  - "Config-as-experiment: frozen dataclass serialized to dict, stored via save_experiment, round-trips through load_experiment"

requirements-completed: [ZERO-01, HDIM-01]

# Metrics
duration: 7min
completed: 2026-03-19
---

# Phase 2 Plan 1: Zero Statistics Engine and Embedding Registry Summary

**Zero distribution statistics (5 functions: normalized spacings, pair correlation, GUE sine kernel, n-level density, number variance) plus EmbeddingConfig registry with 9 feature stubs and 4 presets, backed by scikit-learn/umap-learn**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-19T02:24:10Z
- **Completed:** 2026-03-19T02:31:00Z
- **Tasks:** 2
- **Files modified:** 10

## Accomplishments
- Five zero distribution statistics functions fully implemented and tested: normalized_spacings, pair_correlation, gue_pair_correlation, n_level_density, number_variance
- EmbeddingConfig frozen dataclass with 9 feature extractor stubs and 4 preset configurations for downstream embedding work
- scikit-learn and umap-learn installed as Phase 2 dependencies (with numba, llvmlite, pynndescent transitive deps)
- All 133 tests pass (48 new + 85 Phase 1 preserved)

## Task Commits

Each task was committed atomically:

1. **Task 1a: RED tests for spacing module** - `b67785a` (test)
2. **Task 1b: GREEN implementation of spacing module + Phase 2 deps** - `fa1ad0a` (feat)
3. **Task 2: Embedding config registry with stubs and presets** - `dbc380a` (feat)

## Files Created/Modified
- `src/riemann/analysis/spacing.py` - Zero distribution statistics engine (5 functions)
- `src/riemann/analysis/__init__.py` - Package exports for spacing and rmt modules
- `src/riemann/embedding/registry.py` - EmbeddingConfig, FEATURE_EXTRACTORS, PRESET_CONFIGS, save/load helpers
- `src/riemann/embedding/__init__.py` - Package exports for embedding module
- `tests/test_analysis/test_spacing.py` - 20 tests for all spacing functions
- `tests/test_embedding/test_registry.py` - 14 tests for embedding config and registry
- `tests/test_analysis/__init__.py` - Test package init
- `tests/test_embedding/__init__.py` - Test package init
- `pyproject.toml` - Added scikit-learn and umap-learn dependencies
- `uv.lock` - Updated lockfile with new dependencies

## Decisions Made
- Function-based statistics API (no classes) consistent with Phase 1 pattern: all functions accept ZetaZero lists or numpy arrays, return numpy arrays, never plot
- Local mean spacing normalization uses asymptotic formula 2*pi/log(t/(2*pi)) at midpoints -- standard in the literature, works well for t > ~50
- Pair correlation uses cumulative pairwise gaps (not just nearest-neighbor) with adaptive far-field normalization
- Feature extractor stubs explicitly raise NotImplementedError with the feature name in the message -- Plan 02-03 provides real implementations
- EmbeddingConfig frozen dataclass enables safe sharing and immutable workbench persistence via save_experiment round-trip
- Four preset configs designed for progressive complexity: spectral_basic (3D fast) through kitchen_sink (9D "surprise me" mode)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Pre-existing rmt.py required __init__.py adjustment**
- **Found during:** Task 1 (package structure creation)
- **Issue:** `src/riemann/analysis/__init__.py` already existed with imports from `rmt.py` (from a prior run). Overwriting would break existing test suite.
- **Fix:** Merged spacing imports alongside existing rmt imports in __init__.py
- **Files modified:** src/riemann/analysis/__init__.py
- **Verification:** All 133 tests pass including pre-existing rmt tests
- **Committed in:** fa1ad0a (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Minor -- __init__.py adapted to coexist with pre-existing rmt module. No scope creep.

## Issues Encountered
- OneDrive hardlink issue caused first `uv add` to fail with "Access is denied" on dist-info directory. Second attempt succeeded (known Windows/OneDrive interaction, mitigated by `--link-mode=copy` flag).

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Zero statistics engine ready for RMT comparison (Plan 02-02): pair_correlation and gue_pair_correlation enable direct GUE vs zeta zero comparison
- Embedding registry ready for feature extraction (Plan 02-03): FEATURE_EXTRACTORS stubs are the implementation targets
- n_level_density and number_variance ready for anomaly detection (Plan 02-04)
- All 4 preset configs available for projection theater (Plan 02-05)

## Self-Check: PASSED

- All 4 key source files exist (spacing.py, registry.py, __init__.py x2)
- All 4 test files exist (test_spacing.py, test_registry.py, __init__.py x2)
- Commits verified: b67785a (RED tests), fa1ad0a (GREEN spacing), dbc380a (embedding registry)
- 133 tests pass (48 new + 85 Phase 1)

---
*Phase: 02-higher-dimensional-analysis*
*Completed: 2026-03-19*
