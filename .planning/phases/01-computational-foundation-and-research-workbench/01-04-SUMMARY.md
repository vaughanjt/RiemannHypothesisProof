---
phase: 01-computational-foundation-and-research-workbench
plan: 04
subsystem: visualization
tags: [mpmath, siegelz, matplotlib, plotly, domain-coloring, hsv-to-rgb, numpy, critical-line]

# Dependency graph
requires:
  - phase: 01-computational-foundation-and-research-workbench
    provides: "Python project with mpmath, styles constants, viz package __init__.py"
provides:
  - "Critical line data generation using Hardy Z-function (mpmath.siegelz)"
  - "Static matplotlib and interactive Plotly critical line plots"
  - "Vectorized numpy domain coloring for fast complex plane visualization"
  - "High-precision mpmath domain coloring for critical strip accuracy"
  - "Analytical clarity color palette and matplotlib style defaults"
affects: [01-05, notebooks, downstream-exploration]

# Tech tracking
tech-stack:
  added: []
  patterns: [compute-then-render separation, progressive resolution, dual-mode domain coloring (numpy fast / mpmath precise)]

key-files:
  created:
    - src/riemann/viz/styles.py
    - src/riemann/viz/critical_line.py
    - src/riemann/viz/domain_coloring.py
  modified:
    - tests/test_viz/test_critical_line.py
    - tests/test_viz/test_domain_coloring.py

key-decisions:
  - "dps=15 default for visualization data (not 50) since float64 display precision is sufficient for plots"
  - "Hardy Z-function (siegelz) used directly rather than engine wrappers for visualization speed"
  - "Dual domain coloring modes: numpy vectorized for overview speed, mpmath per-point for critical strip accuracy"
  - "Brightness formula V = 1 - 1/(1 + 0.3*log1p(mag)) makes zeros dark and poles bright"

patterns-established:
  - "Compute-then-render: critical_line_data() returns arrays, plot_*() renders them -- never compute inside plot callbacks"
  - "Progressive resolution: vary resolution parameter (50 for fast, 200+ for detail) rather than computing everything at max"
  - "Dual precision: numpy complex128 for overview, mpmath for critical strip -- choose per use case"
  - "NaN/Inf handling: detect non-finite values and render as white (poles) to avoid corrupting RGB output"

requirements-completed: [VIZ-01, VIZ-02]

# Metrics
duration: 3min
completed: 2026-03-18
---

# Phase 1 Plan 4: Visualization Summary

**Critical line plot via Hardy Z-function with matplotlib/Plotly, domain coloring with dual numpy-fast/mpmath-precise modes, and analytical clarity color palette**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-18T22:38:52Z
- **Completed:** 2026-03-18T22:42:13Z
- **Tasks:** 2
- **Files modified:** 5 (3 created, 2 replaced)

## Accomplishments
- Critical line data generation using mpmath.siegelz with configurable precision and resolution; zero crossings visible as sign changes near known zero locations (t=14.134)
- Static matplotlib and interactive Plotly critical line plots with analytical clarity colors, zero reference line, and hover data
- Vectorized numpy domain coloring producing valid RGB arrays with phase-to-hue and log-magnitude-to-brightness mapping
- High-precision mpmath domain coloring mode for critical strip evaluation where float64 would produce garbage
- All 12 visualization tests pass within 2 seconds

## Task Commits

Each task was committed atomically:

1. **Task 1: Critical line data generation and plotting**
   - `e46689a` (test: TDD RED - failing tests for critical line)
   - `3e44efc` (feat: TDD GREEN - implement critical line module)
2. **Task 2: Domain coloring with progressive resolution**
   - `feaa7e7` (test: TDD RED - failing tests for domain coloring)
   - `eea5df0` (feat: TDD GREEN - implement domain coloring module)

## Files Created/Modified
- `src/riemann/viz/styles.py` - ANALYTICAL_PALETTE color constants and MATPLOTLIB_DEFAULTS style dict
- `src/riemann/viz/critical_line.py` - critical_line_data(), plot_critical_line_static(), plot_critical_line_interactive()
- `src/riemann/viz/domain_coloring.py` - domain_coloring(), domain_coloring_mpmath(), plot_domain_coloring()
- `tests/test_viz/test_critical_line.py` - 6 tests replacing 2 xfail scaffolds (data generation, real values, zero crossing, static plot, interactive plot, precision speed)
- `tests/test_viz/test_domain_coloring.py` - 6 tests replacing 2 xfail scaffolds (RGB output, zeros dark, no NaN, mpmath mode, progressive resolution, plot rendering)

## Decisions Made
- Used dps=15 as default for visualization functions (not DEFAULT_DPS=50) since display-resolution float64 is sufficient for plots and 15 digits runs significantly faster
- Imported mpmath.siegelz directly rather than depending on engine wrappers from Plan 01-03, per plan's note that visualization code can call mpmath directly
- Brightness formula uses 0.3 scaling factor on log1p(magnitude) for good visual contrast between zeros and poles

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Visualization infrastructure complete: all critical line and domain coloring functions ready for notebook exploration
- Both fast-overview (numpy) and precise (mpmath) domain coloring available for different use cases
- Tests cover correctness (shapes, ranges, zero behavior) and performance (resolution scaling, precision tradeoff)
- Ready for integration with notebooks and workbench in Plan 01-05

## Self-Check: PASSED

- All 5 files verified present on disk (3 created, 2 modified)
- All 4 commits (e46689a, 3e44efc, feaa7e7, eea5df0) verified in git log
- Full visualization test suite: 12 passed, 0 failures

---
*Phase: 01-computational-foundation-and-research-workbench*
*Completed: 2026-03-18*
