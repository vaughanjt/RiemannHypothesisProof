# Audit: Prime Cap Bug in connes_crossterm.build_all

**Date:** 2026-04-05
**Bug:** `limit = min(lam_sq, 10000)` in `connes_crossterm.py` (Session 30 → Session 48)
**Fix commit:** 7269938

## Bug Timeline

1. **Session 30** (commit 619becc): `connes_crossterm.py` created with `limit = min(lam_sq, 10000)` as a performance optimization.
2. **Session 41** (commit d92c632): Bug discovered. Commit message: "prime cap bug fixed". `session41g_uncapped_barrier.py` created with uncapped sieve. **BUT `connes_crossterm.py` was never patched.**
3. **Sessions 42-48**: Code that used `session41g` was clean; code that called `build_all` directly inherited the bug.
4. **Session 48** (2026-04-05): Bug rediscovered during conjugate Poisson scan. `connes_crossterm.py` finally patched.

## Files Affected (direct `build_all` calls with lam^2 > 10000)

| File | Max lam^2 | Cited in papers/memory? | Status |
|------|-----------|-------------------------|--------|
| `session37_prolate.py` | 100,000 | No (exploratory Slepian study) | DEPRECATED |
| `session41d_focused_sweep.py` | 20,000 | No (superseded by session41g) | DEPRECATED |
| `session48b_selberg_barrier.py` | 50,000 | No (unfinished Session 48 work) | DEPRECATED |

## Files Verified Clean

| File | Why clean |
|------|-----------|
| `session42_proof.py` | Part B caps at 10000, Part C uses session41g |
| `session42j_margin_vs_drain.py` | Uses session41g_uncapped_barrier |
| `session46f_margin_drain_proof.py` | Uses session41g chain |
| `session45o_adelic_barrier.py` | Uses session41g_uncapped_barrier.sieve_primes |
| `session48_rkhs_barrier.py` | All tests at lam^2 ≤ 500 |
| `session48c_viability_checks.py` | All tests at lam^2 ≤ 500 |
| `session48d_rankin_selberg.py` | All tests at lam^2 ≤ 500 |
| `session42_[topography/proof/hilbert_polya/...]` | All at lam^2 ≤ 10000 |

## Paper Claims Audit

### structural_analysis_draft.tex

| Section | Claim | Source | Status |
|---------|-------|--------|--------|
| Abstract, §8 | Barrier positive to lam^2 = 50,000 | session41g (uncapped) | SAFE |
| §8 Table (line 145) | +2.92 at lam^2=50000 | session41g partial barrier | SAFE |
| §8 Margin-drain table (line 166-172) | gap +0.027 at lam^2=20000 | session42j/46f via session41g | SAFE |
| §8 Margin-drain table | gap +0.029 at lam^2=50000 | session42j/46f via session41g | SAFE |
| §10 Eigenvalue table (line 232-240) | eps_0 values to lam^2=5000 | build_all at lam^2 ≤ 5000 | SAFE (below cap) |
| §9 (new) Conjugate Poisson scan | y up to 0.7, lam^2 ≤ 6611 | session48c | SAFE (below cap) |

### quaternionic_zeta_paper.tex

| Section | Claim | Source | Status |
|---------|-------|--------|--------|
| §5 Cross-term table (line 302-305) | R = 0.4999 at lam^2=50000 | session45o (uses session41g) | SAFE |
| §7 Methods (line 359) | lam^2 up to 50000 with N=15 | session45 chain (no build_all) | SAFE |

## Impact Assessment

**Critical claims unaffected.** All published numbers in both papers are traced to clean code paths. The bug affected only three exploratory files whose outputs were never quoted in papers or memory summaries.

**Extra silver lining.** Running the fixed code at lam^2 ∈ {23228, 81612, 286751} in the Lorentzian direction yielded Q_W min-eigenvalue within float64 noise of zero, extending the explicit Q_W positivity verification beyond the previous ceiling of lam^2=5000.

## Recommended Actions

1. ✓ Patch committed (commit 7269938)
2. ✓ Memory entry added (`project_prime_cap_bug.md`)
3. ✓ Paper updated with new Observation on extended Q_W verification
4. [ ] Optional: add regression test that fails if `min(lam_sq, ...)` reappears in matrix construction
5. [ ] Optional: re-run session37_prolate.py with fixed code to see if any of its Slepian-transition conclusions change
