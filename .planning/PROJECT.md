# Riemann

## What This Is

A hybrid computational research platform and formal proof workbench for proving the Riemann Hypothesis. The project takes an unconventional, cross-disciplinary approach — borrowing tools from physics, information theory, spectral theory, and higher-dimensional geometry rather than following classical proof strategies that have so far fallen short. The user explores interactively through visualizations and "what if" experiments while Claude handles the heavy mathematical formalism.

## Core Value

Discover a novel proof pathway for the Riemann Hypothesis by exploring unconventional cross-disciplinary approaches, with computational tools that can operate in higher-dimensional spaces and project insights down to human-interpretable forms.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Computational engine for the Riemann zeta function and related functions (high-precision evaluation, zero-finding)
- [ ] Higher-dimensional computation framework (operate in N-dimensional spaces, project to 2D/3D for visualization)
- [ ] Interactive visualization tools for exploring the zeta function landscape, zero distributions, and derived structures
- [ ] Cross-disciplinary exploration modules (spectral operators, random matrix connections, information-theoretic measures, modular forms)
- [ ] Research workbench for documenting conjectures, tracking proof attempts, and recording insights
- [ ] Lean 4 formalization pipeline for rigorous verification of promising results
- [ ] Pattern detection and anomaly surfacing in zero distributions and related structures

### Out of Scope

- Classical proof strategy reproduction — deliberately avoiding well-trodden paths that have not succeeded
- Mobile or web deployment — this is a local research tool
- Publication formatting — focus is on discovery, not typesetting
- Teaching calculus fundamentals — Claude handles the formalism; the user directs exploration

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

---
*Last updated: 2026-03-18 after initialization*
