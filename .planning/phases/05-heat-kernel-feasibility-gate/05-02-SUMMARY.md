---
phase: 05-heat-kernel-feasibility-gate
plan: 02
subsystem: analysis
tags: [heat-kernel, maass-forms, eisenstein, scattering-phase, spectral-sum, mpmath, dual-precision]

# Dependency graph
requires:
  - phase: 05-heat-kernel-feasibility-gate
    provides: dual_compute backend, DualResult/ConvergenceDiagnostic types, test scaffolds
provides:
  - heat_kernel_trace function computing all three spectral contributions
  - maass_spectral_sum with Weyl's law tail bound and auto-truncation
  - eisenstein_continuous_integral via mpmath.quad adaptive quadrature
  - scattering_phase -(phi'/phi)(1/2+ir) using digamma and zeta'/zeta
  - load_maass_spectral_params with module-level cache
affects: [05-03-PLAN, barrier-comparison, parameter-mapping]

# Tech tracking
tech-stack:
  added: []
  patterns: [inlined scattering phase for quadrature efficiency, mpmath.quad adaptive integration for continuous spectrum]

key-files:
  created:
    - src/riemann/analysis/heat_kernel.py
  modified:
    - src/riemann/analysis/__init__.py
    - tests/test_heat_kernel.py

key-decisions:
  - "Eisenstein integral is mpmath-only because python-flint arb lacks digamma and zeta special functions"
  - "Scattering phase inlined in quadrature integrand to avoid per-point function call overhead"
  - "Auto-truncation threshold: lambda_j < dps*ln(10)/t with minimum 10 terms"
  - "Integration domain [0.01, 200] avoids r=0 pole while capturing dominant contributions"

patterns-established:
  - "Inlined special function evaluation inside mpmath.quad integrand for quadrature efficiency"
  - "dual_compute wraps constant and discrete terms; continuous is mpmath-only with clear documentation"

requirements-completed: [HEAT-01, HEAT-03]

# Metrics
duration: 8min
completed: 2026-04-04
---

# Phase 5 Plan 02: Heat Kernel Trace Summary

**Heat kernel trace Tr(e^{-tDelta}) on SL(2,Z)\\H with discrete Maass spectral sum (Weyl tail bound), continuous Eisenstein integral (mpmath.quad), and dual-precision cross-validation on constant/discrete terms**

## Performance

- **Duration:** 8 min
- **Started:** 2026-04-05T02:19:50Z
- **Completed:** 2026-04-05T02:28:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Full heat kernel trace K(t) = 1/(12t) + Maass sum + Eisenstein integral, all three contributions computed and returned in structured dict
- Maass spectral sum with auto-truncation based on precision threshold and Weyl's law tail bound in ConvergenceDiagnostic
- Scattering phase -(phi'/phi)(1/2+ir) evaluated using mpmath.digamma and mpmath.zeta(derivative=True)
- Eisenstein continuous integral via mpmath.quad adaptive quadrature with inlined scattering phase
- Dual-precision cross-validation (D-09) on constant and discrete terms via flint arb; Eisenstein integral documented as mpmath-only

## Task Commits

Each task was committed atomically:

1. **Task 1: Maass spectral sum with convergence diagnostics and scattering phase** - `771226e` (feat, TDD)
2. **Task 2: Eisenstein continuous integral and full heat kernel trace** - `bdbec8d` (feat, TDD)

## Files Created/Modified
- `src/riemann/analysis/heat_kernel.py` - 337 lines: load_maass_spectral_params, maass_spectral_sum, scattering_phase, eisenstein_continuous_integral, heat_kernel_trace
- `src/riemann/analysis/__init__.py` - Added heat_kernel exports (5 functions)
- `tests/test_heat_kernel.py` - 6 real tests replacing 3 Plan-02 stubs: convergence diagnostics, scattering phase real output, param loading, Eisenstein finite, trace all terms, constant dominates small t

## Decisions Made
- **Eisenstein integral mpmath-only:** python-flint arb does not expose digamma or zeta special functions needed for the scattering phase, so the continuous spectrum integral runs only in mpmath. Dual-precision validation covers the constant term and discrete sum (which together dominate the trace). Documented as intentional limitation.
- **Inlined scattering phase in quadrature:** Rather than calling the standalone `scattering_phase()` function at each quadrature point (function call overhead per evaluation), the formula is inlined directly in the mpmath.quad integrand for efficiency.
- **Integration domain [0.01, 200]:** The lower bound epsilon=0.01 avoids the digamma(ir) pole at r=0; the upper bound R=200 is sufficient because exp(-(1/4+R^2)*t) decays super-exponentially. Tail estimate is computed but not added to the result (conservative approach).

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Package resolution in worktree initially pointed to a different worktree (`agent-add9b3a2`) because `pip install -e .` was installed for that directory. Fixed by re-running `pip install -e .` in the current worktree. This is a development environment issue, not a code issue.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- heat_kernel_trace ready for Plan 03 (parameter mapping t(L) and barrier comparison)
- All 5 exported functions available via `from riemann.analysis import heat_kernel_trace`
- 3 Plan 03 test stubs remain: parameter_mapping_cross_validation, barrier_comparison_100_values, dual_precision_all_computations
- Eisenstein integral limitation (mpmath-only) documented; Plan 03 should note this when running dual-precision validation across all computations

## Self-Check: PASSED

- All 3 created/modified files exist on disk
- Commit 771226e (Task 1) found in git log
- Commit bdbec8d (Task 2) found in git log
- `python -m pytest tests/test_heat_kernel.py -k "not parameter_mapping and not barrier_comparison and not dual_precision_all" -x -q` returns 8 passed, 3 deselected

---
*Phase: 05-heat-kernel-feasibility-gate*
*Completed: 2026-04-04*
