---
phase: 5
slug: heat-kernel-feasibility-gate
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-04
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml (existing) |
| **Quick run command** | `python -m pytest tests/test_heat_kernel.py -x -q` |
| **Full suite command** | `python -m pytest tests/ -x -q` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_heat_kernel.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| TBD | TBD | TBD | HEAT-01 | unit + integration | `pytest tests/test_heat_kernel.py -k "trace"` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | HEAT-02 | integration | `pytest tests/test_heat_kernel.py -k "parameter_mapping"` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | HEAT-03 | unit | `pytest tests/test_heat_kernel.py -k "maass"` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | HEAT-04 | unit + integration | `pytest tests/test_heat_kernel.py -k "dual_precision"` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_heat_kernel.py` — stubs for HEAT-01 through HEAT-04
- [ ] python-flint 0.8.0 added to pyproject.toml and installed

*Existing pytest infrastructure covers framework needs.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Convergence plot visual quality | HEAT-01 | Plot aesthetics can't be automated | Inspect convergence plots in Jupyter for readability |
| Agreement table verdict interpretation | HEAT-02 | Proof-grade threshold is judgment-based | Review table, verify digits of agreement are sufficient for proof |
| Eisenstein budget assessment | HEAT-01 | Go/no-go is a mathematical judgment | Review Eisenstein magnitude vs 0.036 and assess proof viability |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
