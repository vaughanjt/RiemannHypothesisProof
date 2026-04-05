# Riemann

## What This Is

A hybrid computational research platform and formal proof workbench for proving the Riemann Hypothesis. The project takes an unconventional, cross-disciplinary approach — borrowing tools from physics, information theory, spectral theory, and higher-dimensional geometry rather than following classical proof strategies that have so far fallen short. The user explores interactively through visualizations and "what if" experiments while Claude handles the heavy mathematical formalism.

## Core Value

Discover a novel proof pathway for the Riemann Hypothesis by exploring unconventional cross-disciplinary approaches, with computational tools that can operate in higher-dimensional spaces and project insights down to human-interpretable forms.

## Current Milestone: v2.0 The Modular Barrier

**Goal:** Express the Connes barrier as a heat kernel trace on the modular surface where positivity is automatic (each term positive), then bound the corrections to close the proof.

**Target features:**
- Heat kernel interpretation of the barrier on SL(2,Z)\H
- Modular form parametrization (q-series expansion)
- Laplacian eigenvalue computation on the modular surface
- GL(1)->GL(2) lift (Weil explicit → Selberg trace formula)
- Rankin-Selberg L-value check (Petersson norm positivity)
- CM point evaluation at Heegner numbers
- Correction bounds (heat kernel trace minus actual barrier)
- Formal proof if pathway succeeds

## Requirements

### Validated

- ✓ Computational engine for zeta function and related functions — v1.0 Phase 1
- ✓ Higher-dimensional computation framework — v1.0 Phase 2
- ✓ Interactive visualization tools — v1.0 Phase 1-2
- ✓ Cross-disciplinary exploration modules (spectral, RMT, info-theory, modular forms) — v1.0 Phase 3
- ✓ Research workbench for conjectures, experiments, insights — v1.0 Phase 1
- ✓ Lean 4 formalization pipeline — v1.0 Phase 4
- ✓ Pattern detection and anomaly surfacing — v1.0 Phase 2

### Active

- [ ] Heat kernel interpretation of the Connes barrier as trace on SL(2,Z)\H plus corrections
- [ ] Modular form parametrization with q-series expansion
- [ ] Laplacian eigenvalue computation on the modular surface vs barrier spectrum
- [ ] GL(1)->GL(2) lift: Weil explicit formula as Selberg trace formula
- [ ] Rankin-Selberg L-value: check if B(L) = L(1, f×f̄) (Petersson norm, always positive)
- [ ] CM point evaluation at Heegner numbers for algebraic barrier values
- [ ] Correction bounds between heat kernel trace and actual barrier
- [ ] Rigorous proof assembly if pathway succeeds

### Out of Scope

- Classical proof strategy reproduction — deliberately avoiding well-trodden paths that have not succeeded
- Mobile or web deployment — this is a local research tool
- Publication formatting — focus is on discovery, not typesetting
- Teaching calculus fundamentals — Claude handles the formalism; the user directs exploration
- Direct analytic proof of B(L) > 0 without modular interpretation — every direct approach has been shown circular (Sessions 35-42)

## Context

The Riemann Hypothesis (1859) states that all non-trivial zeros of the Riemann zeta function have real part 1/2. It remains one of the seven Millennium Prize Problems and is arguably the most important unsolved problem in mathematics. Every classical approach — moment methods, random matrix theory, trace formulas, the Selberg zeta function — has produced deep results but not a proof.

The working theory of this project is that the proof may require structures living in dimensions far beyond human spatial intuition. Connections to physics (quantum chaos, spectral theory), information theory, and higher-dimensional geometry (modular forms, adelic spaces, hyperbolic manifolds) suggest the answer may lie at the intersection of disciplines rather than within pure number theory alone.

The user is not a mathematician by training but is technically capable (Python, dev environment ready) and wants to direct exploration as an "explorer" — poking at visualizations, asking "what if," and steering the investigation while Claude does the formal mathematical heavy lifting.

## Constraints

- **Tech stack**: Python for computation/visualization, Lean 4 for formalization
- **Precision**: Must support arbitrary-precision arithmetic (zeta function evaluation requires it)
- **Visualization**: Must handle projection from high-dimensional spaces to 2D/3D
- **Approach**: Unconventional — cross-disciplinary and higher-dimensional, not classical proof strategies

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Hybrid approach (explore then formalize) | Explore fast computationally, formalize in Lean 4 only when we find something promising | — Pending |
| Cross-disciplinary + higher-dimensional focus | Classical pure number theory approaches haven't worked; the proof may live in structures beyond human spatial intuition | — Pending |
| Python + Lean 4 stack | Python for fast exploration and visualization; Lean 4 for machine-verified rigor when needed | — Pending |
| Explorer role for user | User directs investigation and explores visualizations; Claude handles formalism | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-04 after milestone v2.0 start*
