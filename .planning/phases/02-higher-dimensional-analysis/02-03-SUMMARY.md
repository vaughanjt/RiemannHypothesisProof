---
phase: 02-higher-dimensional-analysis
plan: 03
subsystem: embedding
tags: [hdf5, sklearn, umap, pca, tsne, hopf-fibration, feature-extraction, stereographic, projection]

# Dependency graph
requires:
  - phase: 02-higher-dimensional-analysis/02-01
    provides: "EmbeddingConfig, FEATURE_EXTRACTORS stubs, PRESET_CONFIGS, spacing statistics"
provides:
  - "9 feature extractors (imag_part, spacing_left/right, zeta_derivative_magnitude, local_density_deviation, pair_correlation_local, hardy_z_sign_changes, local_entropy, compression_distance)"
  - "compute_embedding pipeline with configurable feature selection and scaling"
  - "HDF5 storage for embeddings and projections (save/load/list)"
  - "5 projection methods: PCA, t-SNE, UMAP, stereographic, Hopf fibration"
  - "ProjectionResult dataclass for uniform projection output"
affects: [02-05-projection-theater, phase-3-synthesis]

# Tech tracking
tech-stack:
  added: [h5py, sklearn.decomposition.PCA, sklearn.manifold.TSNE, umap-learn, sklearn.preprocessing.StandardScaler, sklearn.preprocessing.RobustScaler]
  patterns: [registration-pattern-for-extractors, cache-to-disk-for-expensive-features, context-manager-hdf5, hopf-fibration-S3-to-S2]

key-files:
  created:
    - src/riemann/embedding/coordinates.py
    - src/riemann/embedding/storage.py
    - src/riemann/viz/projection.py
    - tests/test_embedding/test_coordinates.py
    - tests/test_embedding/test_storage.py
    - tests/test_viz/test_projection.py
  modified:
    - src/riemann/embedding/__init__.py
    - tests/test_embedding/test_registry.py

key-decisions:
  - "Registration pattern: coordinates.py imports FEATURE_EXTRACTORS dict from registry.py and replaces stubs at import time -- avoids circular imports"
  - "Zeta derivative caching: expensive |zeta'(rho)| computation cached to DATA_DIR/cache with SHA-256 hash key from zero indices + dps"
  - "Hopf fibration: custom mathematical projection S^3 -> S^2 with fiber phase metadata stored for downstream coloring"
  - "HDF5 single-writer pattern: always use context managers, close before reading"
  - "Auto-reduce t-SNE perplexity and UMAP n_neighbors for small datasets to prevent sklearn errors"

patterns-established:
  - "Registration pattern: module-level function replaces stubs in shared dict at import time"
  - "Cache-to-disk: expensive computations cached as .npy files with hash-based keys"
  - "ProjectionResult dataclass: uniform output for all projection methods with metadata dict"
  - "HDF5 context managers: always close file handle before returning data"

requirements-completed: [HDIM-01, HDIM-02]

# Metrics
duration: 12min
completed: 2026-03-19
---

# Phase 2 Plan 3: Embedding Pipeline and Projection Methods Summary

**9 feature extractors with configurable scaling, HDF5 storage with gzip compression, and 5 projection methods (PCA, t-SNE, UMAP, stereographic, Hopf fibration S^3->S^2) forming a complete extract->store->project pipeline**

## Performance

- **Duration:** 12 min
- **Started:** 2026-03-19T02:36:03Z
- **Completed:** 2026-03-19T02:48:00Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- Implemented 9 feature extractors replacing all stubs in FEATURE_EXTRACTORS registry, turning ZetaZero lists into N-dimensional float64 arrays
- Built compute_embedding pipeline with configurable feature selection (EmbeddingConfig) and 3 scaling modes (standard, robust, none)
- Implemented HDF5 storage with gzip compression, embedding and projection round-trip, and directory-based listing
- Built 5 projection methods (PCA, t-SNE, UMAP, stereographic, Hopf fibration) all returning uniform ProjectionResult dataclass
- Hopf fibration implements custom mathematical projection S^3 -> S^2 with fiber phase metadata for downstream coloring in projection theater
- Full end-to-end pipeline verified: extract features -> compute embedding -> store HDF5 -> project -> get coordinates

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement feature extractors and compute_embedding pipeline** - `673fac5` (feat) - included in prior commit with 02-04 work
2. **Task 2: Implement HDF5 storage and projection pipeline** - pending commit (feat)

_Note: Task 1 code was committed alongside 02-04 files in commit 673fac5. Task 2 files await commit._

## Files Created/Modified
- `src/riemann/embedding/coordinates.py` - 9 feature extractors + compute_embedding pipeline with scaling
- `src/riemann/embedding/storage.py` - HDF5 save/load/list for embeddings and projections
- `src/riemann/viz/projection.py` - ProjectionResult dataclass + 5 projection methods (PCA, t-SNE, UMAP, stereographic, Hopf)
- `src/riemann/embedding/__init__.py` - Updated exports for coordinates, storage, and compute_embedding
- `tests/test_embedding/test_coordinates.py` - 15 tests for feature extractors and compute_embedding
- `tests/test_embedding/test_storage.py` - 6 tests for HDF5 save/load/list round-trips
- `tests/test_viz/test_projection.py` - 15 tests for all projection methods including Hopf fiber structure
- `tests/test_embedding/test_registry.py` - Updated stale test (stubs replaced with real implementations)

## Decisions Made
- **Registration pattern:** coordinates.py imports FEATURE_EXTRACTORS dict from registry.py and calls `.update()` at module load time to replace stubs with real functions. Avoids circular imports by having __init__.py import registry first, then coordinates.
- **Zeta derivative caching:** |zeta'(rho)| is the most expensive feature, so results are cached to DATA_DIR/cache as .npy files keyed by SHA-256 hash of zero indices + dps.
- **Hopf fibration implementation:** Custom mathematical projection per user decision. Maps 4D data (S^3) to 3D (S^2) using the standard Hopf map: (z1, z2) -> (2*Re(z1*conj(z2)), 2*Im(z1*conj(z2)), |z1|^2-|z2|^2). Stores fiber phase (relative angle between z1 and z2) in metadata for coloring.
- **Auto-parameter adjustment:** t-SNE and UMAP auto-reduce their key parameters (perplexity and n_neighbors) for small datasets to prevent sklearn/umap-learn errors.
- **HDF5 single-writer pattern:** All HDF5 file operations use context managers (with h5py.File(...)) to ensure handles are closed before returning. Data is copied into memory before the file is closed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated stale test_registry.py test**
- **Found during:** Overall verification
- **Issue:** `test_each_stub_raises_not_implemented` expected stubs to raise NotImplementedError, but stubs are now replaced with real implementations
- **Fix:** Changed test to `test_extractors_are_callable` which verifies all extractors are callable
- **Files modified:** tests/test_embedding/test_registry.py
- **Verification:** All 50 embedding/projection tests pass
- **Committed in:** 673fac5 (prior commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix)
**Impact on plan:** Trivially necessary -- the old test tested the pre-implementation state.

## Issues Encountered
- Git add/commit permissions intermittently denied during execution, requiring workarounds for atomic commits. Task 2 files created and verified but commit pending.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Embedding pipeline is complete: extract -> store -> project works end-to-end
- All 5 projection methods ready for the projection theater (Plan 02-05)
- HDF5 storage ready for large-scale zero embeddings
- ProjectionResult dataclass provides uniform interface for theater visualization
- Hopf fibration fiber_phase metadata ready for coloring in Plotly 3D scatter

## Self-Check: PASSED

All 7 key files verified to exist on disk:
- src/riemann/embedding/coordinates.py - FOUND
- src/riemann/embedding/storage.py - FOUND
- src/riemann/viz/projection.py - FOUND
- tests/test_embedding/test_coordinates.py - FOUND
- tests/test_embedding/test_storage.py - FOUND
- tests/test_viz/test_projection.py - FOUND
- .planning/phases/02-higher-dimensional-analysis/02-03-SUMMARY.md - FOUND

Commit 673fac5 verified (Task 1 code).
50 tests passing across all embedding/projection test files.
85 Phase 1 regression tests still passing.

---
*Phase: 02-higher-dimensional-analysis*
*Completed: 2026-03-19*
