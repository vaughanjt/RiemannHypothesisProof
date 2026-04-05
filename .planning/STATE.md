---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: The Modular Barrier
status: session54_complete_landscape_audit
stopped_at: "Session 54: Cramer concentration KILLED. std~L^1.39 vs constant margin~0.264. 93% of Cramer trials violate |drain|<margin at L=12. Margin-drain proof via generic concentration is dead. Deeper circularity: real primes satisfy the bound BECAUSE zeros enforce cancellations, not due to generic randomness. Margin-drain framework likely circular for proof purposes. Landscape audit pending."
last_updated: "2026-04-05T14:00:00.000Z"
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

## Next Session options (historical -- from end of Session 52)

Option A: Paper consolidation -- DONE in Session 53, committed 2b74314.
Option B: Extend hot-spot scan beyond lam^2 = 10000 -- still open.
Option C: Fresh equidistribution direction -- recommended, awaiting scope.

See "Next-attack decision point" above for current state.

## Session 53 (complete, committed 2b74314) -- paper consolidation

User selected Option A at session start. Edits applied to
docs/structural_analysis_draft.tex:

  1. Abstract: gap "approx 0.036" replaced with dense-scan plateau
     0.017-0.020 + transient min 0.01679 at lam^2 ~ 108.
  2. Observation "Margin exceeds drain" (obs:margin-drain): coarse table
     preserved but concluding text now flags it as coarse sampling and
     forwards to the new dense-scan observation. Asymptotic gap revised
     to 0.022, max|drain| revised to ~0.247.
  3. New Observation: Hot-spot plateau (obs:hot-spot-plateau).
     Documents 251-point dense scan, 322-prime global search, per-prime
     decomposition at L=4.68 showing p=107 contributes only 5e-5.
     build_all_fast (~100x speedup, 1.8e-16 agreement) documented in
     footnote.
  4. New Observation: Conditional Cramer (obs:conditional-cramer).
     Drain at hot spot is Cramer-typical, tail std 0.8-1.6 vs gap 0.017,
     per-prime bounds ~80x looser than needed.
  5. Heat kernel section: "margin-drain budget of 0.036" -> "0.017-0.020",
     Eisenstein ratio updated from "3-5x" to "5-10x".
  6. Structural-picture bullet 5: margin/drain asymptotes replaced with
     dense-scan plateau, forwards to obs:hot-spot-plateau.
  7. Conjecture [Margin-drain gap] (conj:gap): specific numerical targets
     0.269/0.240 removed, replaced with "asymptotic gap ~0.022 and
     finite-L plateau gap at least 0.017"; appended note on
     equidistribution/large-sieve requirement per conditional Cramer.
  8. Conjecture [Modular structure] (conj:modular): marked RETRACTED
     in title, introductory clause added pointing to retraction remark.
  9. New Observation: Retraction of Conjecture 3 (rem:conj-modular-
     retraction). Documents R^2 = 1.0 at K_z=60, K_p=30, d_zero=3,
     residual 5.17e-11, ESPRIT-on-residual finds only further zero
     frequencies. Conjecture withdrawn.
  10. Final closing paragraphs: "prove |d(L)| < 0.269" -> "prove
      |d(L)| < m(L)" with finite-L lower bound 0.017 annotated;
      reference to conj:modular replaced with pointer to retraction
      observation.

Resolved in Session 53 (all items closed by end of session):
  - Paper edits committed (2b74314).
  - Memory entry project_session53_paper_consolidation.md written.
  - Stale Session-45 HANDOFF.json / .continue-here.md verified absent
    from .planning/ (not present at resume; already cleaned).

## Next-attack decision point (2026-04-05, post Session 53)

Arc 49-53 closed. No active phase. Options on the table:

Option B -- Extend hot-spot scan beyond lam^2 = 10000 (cheap follow-up)
  Verify the 0.017-0.020 plateau extends to lam^2 >= 10^6. Uses
  build_all_fast + session41g. Low risk, low information yield.

Option C -- Fresh equidistribution / large-sieve direction (RECOMMENDED)
  Session 51 conditional-Cramer explicitly signposts this route:
  drain is Cramer-typical at every K with residual std 0.8-1.6 vs
  gap 0.017, meaning per-prime bounds are doomed and any proof must
  use large-sieve / discrepancy bounds on the full prime Fourier sum.
  Session 43 memo sketched four sub-routes (probabilistic, sieve,
  ergodic, info-theoretic); Session 51 narrows this to
  "variance/concentration of the Cramer model + deviation bound".

  Concrete first step under discussion: characterize the Cramer-model
  distribution of drain(L) -- if we can prove |drain_Cramer(L)| < m(L)
  holds with probability 1 - o(1), that establishes the direction is
  right and tells us what deviation bound real primes need to satisfy.

  Awaiting user confirmation of scope before execution.

## Session 54 (complete) -- Cramer concentration KILLED

User selected Option C. Concrete probe: characterize drain_Cramer(L)
distribution and test whether |drain_Cramer(L)| < margin(L) holds with
probability -> 1 as L grows.

Infrastructure built:
  - precompute_response(): reduces M_prime Rayleigh quotient from O(dim^2)
    to O(dim) per prime power via precomputed response coefficients.
  - mp_rayleigh_fast(): vectorized over all k=1 terms, validated to 5e-15.
  - session54_cramer_concentration.py: full MC framework.

RESULT: CLEAN KILL.

  Monte Carlo, 2000 trials per L, full Cramer (K=0, count-matched):
  L=3:  std=0.41, margin=0.24, margin/std=0.58sig, P(violate)=60%
  L=4.7: std=0.79, margin=0.26, margin/std=0.33sig, P(violate)=76%
  L=6.5: std=1.29, margin=0.26, margin/std=0.20sig, P(violate)=84%
  L=8:  std=1.73, margin=0.26, margin/std=0.15sig, P(violate)=89%
  L=10: std=2.19, margin=0.26, margin/std=0.12sig, P(violate)=91%
  L=12: std=2.69, margin=0.26, margin/std=0.10sig, P(violate)=93%

  Variance scaling: std ~ 0.091 * L^1.39 (grows much faster than sqrt(L))
  margin/std ~ -0.052*L + 0.63 (linearly decreasing toward 0)

  At L=12, margin is 0.10 standard deviations from center.
  93% of Cramer trials violate |drain| < margin.

Key insight: actual primes satisfy |drain| < margin NOT because random
primes generically do (they don't -- 93% violate), but because the
zeros of zeta enforce the cancellations that keep the drain tame.

Circularity implication: the margin-drain inequality holds for real
primes BECAUSE RH is true. Proving it without RH is circular.
This is deeper than Sessions 35-42 (individual decomposition loops):
Session 54 shows the ENTIRE margin-drain framework is circular.

Commits: ac22387
Memory: project_session54_cramer_kill.md

## Landscape Audit — What's Alive After 54 Sessions (2026-04-05)

### KILLED APPROACHES (comprehensive, by session)

**Pre-v2.0 kills (Sessions 1-31):** 8 approaches
  energy convexity, Gamma confinement, spacing lower bound (GUE delta_min
  -> 0), information-theoretic (explicit formula tautology), Li criterion
  (needs n ~ gamma^2), anti-alignment (max at sigma~0), moment constraint
  (double/sum = 1.003 too soft), spectral Jacobi (trivially SA).

**Connes framework kills (Sessions 32-42):**
  S32: Tracy-Widom dead end
  S33: Sieve bypass (M < 0 on null, 2.6x overcomp), mollifier 41% ceiling
  S34: All analytic approaches circular for null block
       (sieve, trace-norm, Gershgorin, concentration, Pick/Lowner, deformation)
  S35-36: 5 approaches killed, silent eigenvalue exactly zero
  S38: Lefschetz wall (W02=0 on null -> no commutator identity helps),
       spectral gap unbridgeable by generic tools (10^5 to 10^8 x)
  S40: {J,[J,M]}=0 -> Hodge star definiteness impossible on full space

**Margin-drain kills (Sessions 42-54):**
  S42: Smooth barrier negative (primes save it), spectral sum diverges,
       every decomposition -> RH
  S48: Heat kernel K(t) on SL(2,Z)\H != B(L) (RKHS, Selberg, R-S all fail)
  S49-50: Modular angle DECISIVELY DEAD (R^2=1.0 at machine precision)
  S51: Per-prime bound strategy dead (drain is Cramer-typical, tail std
       80x the gap)
  S52: Discrete bound at prime entries dead (p=107 hot spot is collective
       30-prime effect)
  S54: CRAMER CONCENTRATION DEAD (std ~ L^1.39, margin constant, 93%
       violate at L=12). DEEPER CIRCULARITY: margin-drain holds for real
       primes BECAUSE zeros enforce cancellations, not generic randomness.

**Meta-result: Every computational/analytic approach to proving Q_W >= 0
is circular.** The barrier encodes RH (fake zeros flip sign, Session 38).
The drain is tame because the zeros force it (Session 54). Bounding
the drain requires the zeros (Sessions 42-54). There is no analytic
shortcut.

### ALIVE — Structural/geometric (the only non-circular path)

  1. **Hard Lefschetz for Connes' scaling site in characteristic one**
     (Sessions 34, 38, 40)
     Q_W > 0 on null(W02) IS Hodge-Riemann bilinear relations.
     Function field RH was proved this way (Weil/Deligne).
     Adiprasito-Huh-Katz proved Kahler package for matroids (Fields
     Medal 2022) — the scaling site has combinatorial structure.
     This is algebraic geometry, not analytic number theory.

  2. **The missing Hodge star operator** (Sessions 38, 40)
     Classical geometry decomposes H^1 into H^{1,0} + H^{0,1}.
     For the scaling site, this hasn't been constructed. Candidate
     sources: functional equation s <-> 1-s, Hilbert transform,
     scaling operator spectral decomposition.

  3. **Determinant reduction** (Session 40)
     RH reduces to: QW_barrier * QW_null_coupled > M_cross^2
     Four explicit sums from the Weil formula. Proving the barrier
     constant ~0.04 > 0 analytically is the hardest piece.

  All three are aspects of the SAME problem. The structural approach
  has never been pursued computationally in this project.

### ALIVE — Computational tools (not proof paths)

  - Q_W PSD verification at finite lambda (uncapped, to lambda^2=287k)
  - build_all_fast (~100x speedup, validated 1.8e-16)
  - ESPRIT zero extraction from barrier (10x10 operator, gamma_4 error 0.008)
  - mp_rayleigh_fast (O(dim) per prime, Session 54)
  - Lean 4 formalization (4212 lines, 10 files)

### NOT EXPLORED (external to Connes framework)

  - Levinson-Conrey mollifiers (41% ceiling, needs 100%)
  - GUE universality / Katz-Sarnak
  - Rodgers-Tao dual barrier approach

### THE HONEST BOTTOM LINE

After 54 sessions, we've mapped the entire analytic landscape of the
Connes barrier and found it's all one valley leading back to RH.
Every quantitative approach — decomposition, concentration, modular
interpretation, heat kernel, sieve, per-prime bounds — is circular.

The only exit is UP, into geometry: hard Lefschetz for the scaling site.
This is the number field analogue of what Weil proved for function fields
in 1948 and Deligne extended in 1974. It requires constructing the
missing geometric structure (Hodge star, ample class) on Connes'
noncommutative space.

That is primarily a MATHEMATICAL task (reading Connes-Consani,
Adiprasito-Huh-Katz, Morishita 2025), not a computational one.
This project's computational tools are mature and reusable but have
exhausted their proof potential on the analytic side.

Options for the next session:
  A. Literature deep-dive into scaling site geometry (Connes-Consani
     2021, Adiprasito-Huh-Katz 2018, Morishita 2025)
  B. Try to construct the Hodge star computationally (test candidates
     on the finite-dim matrix, check if any produces correct sign pattern)
  C. Pivot entirely away from Connes framework (Levinson-Conrey, GUE,
     Rodgers-Tao — unexplored)
  D. Write up the full landscape as a paper/report (54 sessions of
     results deserve documentation beyond STATE.md + memory files)
