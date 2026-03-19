---
phase: 02-higher-dimensional-analysis
plan: 05
subsystem: visualization
tags: [plotly, 3d-scatter, animation, heatmap, rmt, number-variance, pair-correlation]

# Dependency graph
requires:
  - phase: 02-higher-dimensional-analysis
    provides: "ProjectionResult dataclass and PCA/t-SNE/UMAP/Hopf projection functions (Plan 03)"
  - phase: 02-higher-dimensional-analysis
    provides: "GUE/GOE/GSE ensemble generation, eigenvalue_spacings, wigner_surmise (Plan 02)"
  - phase: 02-higher-dimensional-analysis
    provides: "Spacing statistics: pair_correlation, number_variance, gue_pair_correlation (Plan 01)"
  - phase: 02-higher-dimensional-analysis
    provides: "Information-theoretic analysis: cross_object_comparison (Plan 04)"
provides:
  - "Projection theater: 3D Plotly scatter, projection path animation, dimension slicing, side-by-side comparison"
  - "RMT comparison views: spacing histogram overlay, pair correlation with residual, N-slider"
  - "Info-theory heatmap: normalized cross-object comparison visualization"
  - "Number variance comparison: empirical + GUE theory + Poisson reference"
affects: [phase-03, dashboards, notebooks]

# Tech tracking
tech-stack:
  added: [plotly.graph_objects, plotly.subplots]
  patterns: [go.Figure return convention, Plotly slider for parameter exploration, make_subplots for multi-panel views]

key-files:
  created:
    - src/riemann/viz/theater.py
    - src/riemann/viz/comparison.py
    - tests/test_viz/test_theater.py
    - tests/test_viz/test_comparison.py
  modified:
    - src/riemann/viz/__init__.py

key-decisions:
  - "All viz functions return go.Figure -- user calls fig.show() in Jupyter, no side effects"
  - "RMT slider uses Plotly native sliders (works in static HTML export, not ipywidgets)"
  - "Side-by-side uses 2D projections (first 2 components) for consistent panel layout"
  - "Number variance includes Poisson reference line (Sigma_2 = L) for universal comparison"

patterns-established:
  - "Plotly-first visualization: all interactive views return go.Figure for JupyterLab display"
  - "Precomputed data pattern: slider precomputes all N values upfront, animation precomputes all frames"
  - "Layered annotation: method info, fixed dims, chi-squared fit annotated on figures"

requirements-completed: [VIZ-03, RMT-01, RMT-02, INFO-02]

# Metrics
duration: 7min
completed: 2026-03-19
---

# Phase 2 Plan 05: Visualization Theater & Comparison Views Summary

**Projection theater with 3D Plotly scatter, animation interpolation, dimension slicing plus RMT comparison views (spacing overlay, pair correlation, N-slider) and info-theoretic heatmap**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-19T02:51:31Z
- **Completed:** 2026-03-19T02:58:41Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Projection theater renders 3D/2D interactive Plotly figures from any ProjectionResult, with color_by support, PCA variance labels, and method metadata annotations
- Projection path animation smoothly interpolates between projection methods with play/pause controls and normalized coordinate ranges
- Dimension slicing selects points within tolerance of fixed dimension values and projects remaining dimensions
- RMT comparison views: spacing histogram overlay with Wigner surmise, pair correlation with residual subplot, interactive N-slider precomputing GUE(N) for multiple matrix sizes
- Information-theoretic heatmap normalizes cross-object comparison metrics for visual pattern detection
- Number variance comparison plots empirical Sigma_2(L) against GUE theoretical formula and Poisson reference

## Task Commits

Each task was committed atomically:

1. **Task 1: Build projection theater with 3D visualization, animation, and dimension slicing** - `9e3c016` (feat)
2. **Task 2: Build RMT comparison views, info-theoretic heatmap, and smoke tests** - `8e6a81e` (feat)

## Files Created/Modified
- `src/riemann/viz/theater.py` - Projection theater: create_theater_figure, create_projection_path_animation, create_dimension_slice_view, create_side_by_side
- `src/riemann/viz/comparison.py` - Comparison views: create_spacing_comparison, create_pair_correlation_comparison, create_rmt_slider_figure, create_info_comparison_heatmap, create_number_variance_comparison
- `tests/test_viz/test_theater.py` - 7 smoke tests for theater module
- `tests/test_viz/test_comparison.py` - 5 smoke tests for comparison module (one per function)
- `src/riemann/viz/__init__.py` - Updated with theater and comparison module documentation

## Decisions Made
- All viz functions return go.Figure -- user calls fig.show() in Jupyter, no side effects in the functions themselves
- RMT slider uses Plotly native sliders rather than ipywidgets so figures work in static HTML export
- Side-by-side comparison uses 2D projections (first 2 components) for consistent panel layout across methods
- Number variance includes Poisson reference line (Sigma_2 = L) providing universal comparison baseline
- Dimension slice tolerance set to 0.5 * std per fixed dimension, selecting roughly 68% of points near target

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Phase 2 is now complete: all 5 plans (spacing statistics, RMT ensembles, embedding/projection, information theory, visualization theater) are delivered
- Full import chain verified: projection -> theater -> comparison all interoperate
- 205 tests passing across entire codebase
- Ready for Phase 3: all analysis and visualization infrastructure in place for advanced investigation

## Self-Check: PASSED

All 5 created/modified files verified present on disk. Both task commits (9e3c016, 8e6a81e) verified in git log.

---
*Phase: 02-higher-dimensional-analysis*
*Completed: 2026-03-19*
