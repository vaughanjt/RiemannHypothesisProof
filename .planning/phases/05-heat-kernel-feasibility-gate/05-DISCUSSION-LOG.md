# Phase 5: Heat Kernel Feasibility Gate - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-04
**Phase:** 05-heat-kernel-feasibility-gate
**Areas discussed:** Kill gate criteria, Parameter discovery, Maass eigenvalue source, Dual precision pattern

---

## Kill Gate Criteria

### Q1: Pass/Fail Threshold

| Option | Description | Selected |
|--------|-------------|----------|
| Strict: 6+ digits everywhere | Must agree at ALL tested L values | |
| Statistical: 6+ digits on average | Average agreement, some outliers OK | |
| Tiered: tight core, looser edges | 6+ in core range, 3+ at extremes | |

**User's choice:** "whatever will suffice for a proof" — proof-grade rigor, no arbitrary digit count
**Notes:** User reframed the question: the threshold is determined by what the proof requires, not a preset number.

### Q2: Eisenstein Kill Threshold

| Option | Description | Selected |
|--------|-------------|----------|
| Hard kill: >50% of budget | Kill if continuous spectrum exceeds 0.018 | |
| Soft kill: investigate first | Only kill if it can't be controlled | |
| No kill threshold | Just data — Phase 7 determines if fatal | ✓ |

**User's choice:** No kill threshold
**Notes:** Let the math speak. Don't pre-judge the Eisenstein contribution.

### Q3: Test Point Count

| Option | Description | Selected |
|--------|-------------|----------|
| 50 points | Matching existing barrier range | |
| 100+ points | Dense grid for higher confidence | ✓ |
| You decide | Claude picks based on data density | |

**User's choice:** 100+ points

### Q4: Diagnostic Format

| Option | Description | Selected |
|--------|-------------|----------|
| Summary table + verdict | Quick pass/fail | |
| Full convergence plots | Interactive investigation | |
| Both | Table for verdict, plots for depth | ✓ |

**User's choice:** Both

---

## Parameter Discovery

### Q1: Discovery Method

| Option | Description | Selected |
|--------|-------------|----------|
| Analytic first | Derive from Lorentzian structure | |
| Numerical first | Optimize t for each L, fit form | |
| Both in parallel | Cross-validate analytic vs numerical | ✓ |

**User's choice:** Both in parallel

### Q2: Mapping Complexity

| Option | Description | Selected |
|--------|-------------|----------|
| Accept complexity | Special functions OK if well-defined | |
| Simplicity required | Clean expression for formalization | |
| You decide | Claude judges acceptability | ✓ |

**User's choice:** You decide

---

## Maass Eigenvalue Source

### Q1: Data Source

| Option | Description | Selected |
|--------|-------------|----------|
| LMFDB primary | Query via existing client | |
| Published tables first | Hardcoded high-precision, LMFDB backup | |
| You decide | Claude picks most reliable source | ✓ |

**User's choice:** You decide

### Q2: Eigenvalue Count

| Option | Description | Selected |
|--------|-------------|----------|
| Start small, expand | ~50, add if convergence diagnostics show need | |
| Go big from start | 500+ upfront for small-t convergence | |
| You decide | Claude determines from convergence analysis | ✓ |

**User's choice:** You decide

---

## Dual Precision Pattern

### Q1: Integration Pattern

| Option | Description | Selected |
|--------|-------------|----------|
| Wrapper functions | Explicit compute_X_mpmath() and compute_X_flint() | |
| Backend flag | Single API with dispatch | |
| You decide | Claude picks best integration with existing patterns | ✓ |

**User's choice:** You decide

### Q2: Certification Frequency

| Option | Description | Selected |
|--------|-------------|----------|
| Explore with mpmath, certify key results | python-flint only for proof-cited results | |
| Always dual | Every computation runs both, flagging disagreement | ✓ |
| You decide | Claude determines based on proof chain | |

**User's choice:** Always dual

---

## Claude's Discretion

- Maass eigenvalue data source and count
- Dual-precision integration pattern
- t(L) mapping complexity acceptance
- Heat kernel spectral sum truncation strategy
- Continuous spectrum numerical integration method
- Module organization within `src/riemann/analysis/`
- All performance optimization decisions

## Deferred Ideas

None — discussion stayed within phase scope
