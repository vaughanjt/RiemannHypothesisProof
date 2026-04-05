---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: The Modular Barrier
status: modular_angle_decisively_dead_awaiting_pivot
stopped_at: "Session 50 complete: modular angle killed with full confidence. R^2 = 1.0 to machine precision with L-modulated cosine basis at 60 zeros + 30 log primes. B_full is ENTIRELY Weil explicit formula content. Conjecture 3 in paper should be retracted."
last_updated: "2026-04-06T00:30:00.000Z"
last_activity: 2026-04-05
progress:
  total_phases: 4
  completed_phases: 1
  total_plans: 3
  completed_plans: 3
  percent: 50
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-04)

**Core value:** Discover a novel proof pathway for the Riemann Hypothesis by exploring unconventional cross-disciplinary approaches, with computational tools that can operate in higher-dimensional spaces and project insights down to human-interpretable forms.
**Current focus:** Phase 05 — heat-kernel-feasibility-gate

## Current Position

Milestone v2.0: Heat kernel KILLED (Phase 5), conjugate-Poisson reframing
probed in Session 49 and mostly dead. No active phase -- awaiting pivot
decision.
Last activity: 2026-04-05 (Session 49 complete)

Progress: [##########..........] 50% (v1.0 done, v2.0 stalled post-kill)

## Performance Metrics

**Velocity:**

- Total plans completed: 4 (v1.0: 01-01, 04-01, 04-02, 04-03)
- Average duration: —
- Total execution time: —

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| v1.0 Phase 1 | 1/5 | — | — |
| v1.0 Phase 4 | 3/3 | — | — |

*Updated after each plan completion*
| Phase 05 P02 | 8min | 2 tasks | 3 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [v1.0 Roadmap]: 4-phase coarse structure; computation+viz+workbench in Phase 1; HD in Phase 2; Lean 4 last
- [v1.0 Complete]: All platform phases complete -- zeta computation, HD analysis, domain modules, Lean 4 pipeline
- [v2.0 Direction]: Heat kernel interpretation of Connes barrier -- Lorentzian test function ~ heat kernel at imaginary time (Session 47)
- [v2.0 Constraint]: Must be non-circular (cannot assume RH to prove RH)
- [v2.0 Roadmap]: 4 phases (5-8), feasibility gate first, Selberg trace second, bounds third, proof last
- [Phase 05]: Eisenstein integral mpmath-only: flint arb lacks digamma/zeta special functions
- [Phase 05]: Scattering phase inlined in mpmath.quad integrand for quadrature efficiency
- [Phase 05]: Auto-truncation: lambda_j < dps*ln(10)/t with min 10 terms for Maass spectral sum

### Pending Todos

None yet.

### Blockers/Concerns

- Every direct analytic approach to B(L)>0 proved circular in Sessions 35-42
- Margin-drain gap is only 0.036 -- bounds must be tight with explicit constants
- Eisenstein continuous spectrum involves zeta'/zeta -- circularity risk in continuous spectrum contribution
- Phase 5 is a KILL GATE: if K(t) does not match B(L) to 6+ digits, entire v2.0 approach is dead

## Session Continuity

Last session: 2026-04-05 (Session 49)
Stopped at: Session 49 complete -- modular angle probably dead, awaiting pivot

Session 49 arc:
  49a/b -- Modular structure probe on B(y). Two findings:
    (1) B_partial = W02 - M_prime is SMOOTH and monotonic, converging to ~3.1.
        Heegner points sit on the smooth curve. No modular signal.
    (2) Session 48's "y=0.8 non-monotonic dip" lives entirely in the W_R +
        alpha_offdiag archimedean corrections, NOT in the prime/W02 interplay.
        This corrects the Session 48 interpretation.
  49c -- Weil residual probe. Fit B_full(L) to constant-amplitude cosines at
         known zeros + log primes. R^2 = 0.940 at K=40 zeros + 50 log primes.
         The 6% unexplained is consistent with amplitude mis-spec (true Weil
         amplitudes are L-dependent, not constants). ESPRIT on residual finds
         no frequencies clearly outside the zero set. No obvious modular signal.

Infrastructure produced:
  - build_all_fast (in session49c_weil_residual.py) -- ~100x faster drop-in
    replacement for connes_crossterm.build_all, validated to 1.8e-16 vs slow.
    Reusable for any future B_full computation at L <= 6.5.

Open thread (if modular angle revisited): derive the exact L-dependent
Weil amplitudes f(L, gamma_n) for the Lorentzian test function, refit B_full.
If residual drops to <1% -> modular angle fully killed clean. If it stays
at 6% -> genuine lead.

Commits: ade2b7c (49a/b), a228bde (49c fast evaluator + probe)

Memory entries (see ~/.claude .../memory/):
  - project_session49_modular_probe.md
  - project_session49_weil_residual.md
  - (correction appended to project_session48_summary.md)

Session 50 ran same day. Resolution below.

## Session 50 Resolution

**DECISIVE KILL. R^2 = 1.0000000000 to machine precision.**

Superset basis test using L-polynomial-modulated cosines at known zero
frequencies + log prime frequencies. Sweep over (K_zeros, d_zero):

  K_z  d_z   R^2
  20   0     0.928  (Session 49c baseline constant amplitudes)
  30   2     0.993
  50   2     0.99993
  60   2     0.9999997
  60   3     1.0000000000  (residual 5.17e-11, 13 orders below signal)

Crisp monotone convergence. ESPRIT on the 5e-11 residual still finds
only further zero-frequencies -- zero content all the way down to the
float64 noise floor.

**B_full(L) is ENTIRELY the Weil explicit formula content.** Smooth
archimedean trend + zero oscillations with L-dependent amplitudes +
prime oscillations with L-dependent amplitudes. Nothing else.

**What this kills (with full confidence):**
  - Conjecture 3 in structural_analysis_draft.tex (modular structure
    of B(y) via conjugate Poisson reframing)
  - Heegner-point algebraic recognition of B values (Session 49)
  - Any reading of the conjugate-Poisson reframing as unlocking a
    proof path

**What remains alive (for next session):**
  - Margin-drain formulation of RH from the paper (pre-v2.0 direction)
  - Info-theoretic / prime-equidistribution approaches (Session 43 memo)
  - Consolidation: update structural_analysis_draft.tex with Session 49-50
    findings (retract Conjecture 3, document fast evaluator, add clean
    kill result for the modular reading)

Commits: 4d3de87 (Session 50 decisive kill)
Memory:  project_session50_modular_kill.md

## Session 51 (complete)

Parallel probes after Session 50 killed the modular angle.

**Thread 51a (dense margin-drain scan):**
251 L values in [1.5, 6.5]. Minimum gap = +0.01679 at L = 4.68 (lam^2 ~ 108).
Gap never negative. Important correction to Session 46f: asymptotic gap
of 0.029 is NOT a tight lower bound over finite L -- there's an interior
minimum at L ~ 4.68 where |drain|/margin reaches 0.935 (94% of the way
to blowing the bound). Max |drain| = 0.245 at L = 6.46.

**Thread 51b (conditional Cramer):**
Fix primes p <= K at actual positions, randomize rest under count-matched
Cramer. At every K (from 2 to 500), the actual drain value 0.211 is within
0.01-0.5 sigma of the Cramer mean, but the residual std dev is 0.8-1.6 --
much larger than the gap of 0.017. Meaning: the drain is Cramer-TYPICAL,
not dominated by small primes.

**Combined implications:**
- Numerical verification holds: margin > |drain| in [1.5, 6.5], gap >= 0.017.
- "Condition on small primes + bound tail" strategy is DEAD -- the tail
  variance is ~80x the gap, so per-prime bounds are necessarily loose.
- Any proof path through margin-drain must use EQUIDISTRIBUTION / large-sieve
  / discrepancy bounds on the whole prime sum, not per-prime accounting.
- The asymptotic 0.029 is not a lower bound; focus on the L=4.68 "hot spot".

Commits: 74499fd (Session 51)
Memory: project_session51_margin_drain.md

## Session 52 (complete)

Hot-spot characterization: investigated why Session 51 found min gap at L=4.68.

Initial hypothesis (refuted): "drain jumps at prime entry, hot spots are at
log(p) + offset". Wrong. Drain is continuous through most prime entries;
the correlation in Session 51 data was coincidental.

Step-by-step findings:
  1. Fine scan around log(107): drain is continuous, peaks at lam^2=107-108
     at 0.2412, no discontinuity.
  2. Per-prime decomposition at L=4.68: dominant contributors are p in
     {17, 19, 23, 29, 31, 37, 41, 43, 47, ...}. Top 15 primes account for
     85.6% of M_prime. The "newly entered" p=107 contributes 0.00005 --
     NEGLIGIBLE. The hot spot is a COLLECTIVE effect of ~30 primes, not
     a last-prime-entered phenomenon.
  3. Drain profiles at p in {31, 73, 107, 151, 251, 503}: drain rises at
     entry for small primes but falls for larger ones. No universal pattern.
  4. Global search over 322 primes p <= 10000 at L = log(p)+0.008:
     minimum gap is STILL 0.01679 at p=107. Plateau of hot spots 0.017-0.020
     across the whole scanned range. No tighter point found up to L=9.17.

Important correction to Session 46f:
  Paper states asymptotic gap = 0.269 - 0.240 = 0.029. This is WRONG as a
  finite-L lower bound. Actual plateau of hot-spot gaps is 0.017-0.020 over
  lam^2 in [100, 10000]. Max |drain(L)| at large L hovers around 0.246-0.247,
  not 0.240. Asymptotic gap is closer to 0.022, and finite-L transient
  minima reach 0.017.

Infrastructure: session52 used session41g (fast M_prime) + session49c
build_all_fast (fast W02, wr_diag, alpha via vectorized numpy). Step 4
scan of 322 primes completed in 99 seconds vs estimated hours with the
original scalar code.

Commits: d09b7b8 (Session 52)
Memory: project_session52_hot_spot.md

## Arc 49-52 summary (all committed, all killed cleanly)

Four sessions produced four clean kills plus one infrastructure asset:
  - Session 49: naive modular reading of conjugate-Poisson reframing
                (B_partial smooth, Heegner unremarkable, y=0.8 dip
                 relocated to W_R)
  - Session 49c + 50: modular angle DECISIVELY killed, R^2 = 1.0 to
                machine precision with L-modulated basis. B_full is
                entirely Weil explicit formula.
  - Session 51: margin-drain strategy "condition on small primes +
                bound tail" killed via conditional Cramer (drain is
                Cramer-typical; tail variance 80x the gap).
  - Session 52: margin-drain strategy "discrete bound at prime entries"
                killed (p=107 hot spot is collective phase-alignment of
                ~30 primes; p=107 contributes 0.00005, negligible).

  Infrastructure win: build_all_fast (~100x speedup, validated to 1.8e-16),
  reusable for any future B_full or margin-drain computation.

  Paper corrections accumulated (retracted/revised in future session):
    - Conjecture 3 (modular structure of B(y)): retract
    - Asymptotic gap 0.269 - 0.240 = 0.029: revise (real plateau 0.017-0.020)
    - Max |drain| -> 0.240: revise (actual plateau ~0.247)

## Next Session options (await user selection)

Option A: Paper consolidation (overdue, non-discovery but decision-clearing)
  Update structural_analysis_draft.tex with Sessions 49-52 findings.
  Specifically: retract Conjecture 3, fix the asymptotic-gap numbers,
  document build_all_fast, add the hot-spot plateau observation.

Option B: Extend hot-spot scan beyond lam^2 = 10000 (cheap follow-up)
  Verify the 0.017-0.020 plateau extends to lam^2 >= 10^6 or so.
  Uses build_all_fast + session41g. ~5 min per large lam^2 point.

Option C: Genuinely fresh direction (not margin-drain, not modular)
  Session 43 memo suggested prime-equidistribution / info-theoretic
  approaches that were never tried. Blank slate, higher risk.

Recommendation (from end of Session 52 report): Option A. Four sessions
of accumulated corrections deserve to be written up before starting the
next attack. Paper consolidation also clarifies WHAT the next attack
should target (the corrected gap of ~0.017, not the 0.029 from 46f).
