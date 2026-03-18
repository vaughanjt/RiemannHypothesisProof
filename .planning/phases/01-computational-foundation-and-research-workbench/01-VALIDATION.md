---
phase: 1
slug: computational-foundation-and-research-workbench
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-18
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >= 7.4 |
| **Config file** | none — Wave 0 installs via pyproject.toml |
| **Quick run command** | `uv run pytest tests/ -x --timeout=30` |
| **Full suite command** | `uv run pytest tests/ -v --timeout=120` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/ -x --timeout=30`
- **After every plan wave:** Run `uv run pytest tests/ -v --timeout=120`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 0 | COMP-01 | unit | `uv run pytest tests/test_engine/test_zeta.py -x` | ❌ W0 | ⬜ pending |
| 01-01-02 | 01 | 0 | COMP-01 | unit | `uv run pytest tests/test_engine/test_zeta.py::test_functional_equation -x` | ❌ W0 | ⬜ pending |
| 01-01-03 | 01 | 0 | COMP-02 | unit | `uv run pytest tests/test_engine/test_zeros.py::test_odlyzko_validation -x` | ❌ W0 | ⬜ pending |
| 01-01-04 | 01 | 0 | COMP-02 | integration | `uv run pytest tests/test_engine/test_zeros.py::test_zero_catalog -x` | ❌ W0 | ⬜ pending |
| 01-01-05 | 01 | 0 | COMP-03 | unit | `uv run pytest tests/test_engine/test_lfunctions.py::test_hardy_z -x` | ❌ W0 | ⬜ pending |
| 01-01-06 | 01 | 0 | COMP-03 | unit | `uv run pytest tests/test_engine/test_lfunctions.py::test_dirichlet_trivial -x` | ❌ W0 | ⬜ pending |
| 01-01-07 | 01 | 0 | COMP-03 | unit | `uv run pytest tests/test_engine/test_lfunctions.py::test_xi_symmetry -x` | ❌ W0 | ⬜ pending |
| 01-01-08 | 01 | 0 | COMP-04 | unit | `uv run pytest tests/test_engine/test_validation.py::test_always_validate -x` | ❌ W0 | ⬜ pending |
| 01-01-09 | 01 | 0 | COMP-04 | integration | `uv run pytest tests/test_engine/test_validation.py::test_stress_rerun -x` | ❌ W0 | ⬜ pending |
| 01-02-01 | 02 | 1 | VIZ-01 | smoke | `uv run pytest tests/test_viz/test_critical_line.py -x` | ❌ W0 | ⬜ pending |
| 01-02-02 | 02 | 1 | VIZ-02 | unit | `uv run pytest tests/test_viz/test_domain_coloring.py -x` | ❌ W0 | ⬜ pending |
| 01-03-01 | 03 | 1 | RSRCH-01 | unit | `uv run pytest tests/test_workbench/test_conjecture.py -x` | ❌ W0 | ⬜ pending |
| 01-03-02 | 03 | 1 | RSRCH-01 | unit | `uv run pytest tests/test_workbench/test_conjecture.py::test_evidence_levels -x` | ❌ W0 | ⬜ pending |
| 01-03-03 | 03 | 1 | RSRCH-02 | unit | `uv run pytest tests/test_workbench/test_experiment.py -x` | ❌ W0 | ⬜ pending |
| 01-03-04 | 03 | 1 | RSRCH-02 | unit | `uv run pytest tests/test_workbench/test_experiment.py::test_checksum -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `pyproject.toml` — project initialization with uv, pytest config, dependency list
- [ ] `tests/conftest.py` — shared fixtures: mpmath precision contexts, temporary SQLite DB, test data paths
- [ ] `data/odlyzko/zeros_100.txt` — first 100 zeros at 1000-digit precision for validation
- [ ] `tests/test_engine/test_zeta.py` — covers COMP-01
- [ ] `tests/test_engine/test_zeros.py` — covers COMP-02
- [ ] `tests/test_engine/test_lfunctions.py` — covers COMP-03
- [ ] `tests/test_engine/test_validation.py` — covers COMP-04
- [ ] `tests/test_viz/test_critical_line.py` — covers VIZ-01
- [ ] `tests/test_viz/test_domain_coloring.py` — covers VIZ-02
- [ ] `tests/test_workbench/test_conjecture.py` — covers RSRCH-01
- [ ] `tests/test_workbench/test_experiment.py` — covers RSRCH-02

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Interactive zoom/pan on critical line plot | VIZ-01 | Requires visual inspection of Plotly widget in JupyterLab | Open notebook, run plot cell, verify zoom/pan controls work |
| Domain coloring visual correctness | VIZ-02 | Coloring quality requires visual inspection | Generate domain coloring, verify zeros appear as points where all colors converge |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
