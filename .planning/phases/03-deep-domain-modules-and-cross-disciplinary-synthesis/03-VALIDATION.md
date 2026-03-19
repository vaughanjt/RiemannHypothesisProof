---
phase: 3
slug: deep-domain-modules-and-cross-disciplinary-synthesis
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-19
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >=9.0.2 |
| **Config file** | `pyproject.toml` [tool.pytest.ini_options] |
| **Quick run command** | `uv run pytest tests/test_analysis/ -x --timeout=60` |
| **Full suite command** | `uv run pytest tests/ -x --timeout=120` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/test_analysis/ -x --timeout=60`
- **After every plan wave:** Run `uv run pytest tests/ -x --timeout=120`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 0 | SPEC-01 | unit | `uv run pytest tests/test_analysis/test_spectral.py -x` | ❌ W0 | ⬜ pending |
| 03-01-02 | 01 | 0 | SPEC-02 | unit | `uv run pytest tests/test_analysis/test_trace_formula.py -x` | ❌ W0 | ⬜ pending |
| 03-01-03 | 01 | 0 | MOD-01 | unit | `uv run pytest tests/test_analysis/test_modular_forms.py -x` | ❌ W0 | ⬜ pending |
| 03-01-04 | 01 | 0 | MOD-02 | unit | `uv run pytest tests/test_analysis/test_lmfdb_client.py -x` | ❌ W0 | ⬜ pending |
| 03-01-05 | 01 | 0 | ADEL-01 | unit | `uv run pytest tests/test_analysis/test_padic.py -x` | ❌ W0 | ⬜ pending |
| 03-01-06 | 01 | 0 | ADEL-02 | unit | `uv run pytest tests/test_analysis/test_padic.py -x` | ❌ W0 | ⬜ pending |
| 03-01-07 | 01 | 0 | XDISC-01 | unit | `uv run pytest tests/test_analysis/test_analogy.py -x` | ❌ W0 | ⬜ pending |
| 03-01-08 | 01 | 0 | XDISC-02 | unit | `uv run pytest tests/test_analysis/test_tda.py -x` | ❌ W0 | ⬜ pending |
| 03-01-09 | 01 | 0 | XDISC-03 | unit | `uv run pytest tests/test_analysis/test_dynamics.py -x` | ❌ W0 | ⬜ pending |
| 03-01-10 | 01 | 0 | XDISC-04 | unit | `uv run pytest tests/test_analysis/test_ncg.py -x` | ❌ W0 | ⬜ pending |
| 03-01-11 | 01 | 0 | RSRCH-03 | unit | `uv run pytest tests/test_analysis/test_conjecture_gen.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_analysis/test_spectral.py` — stubs for SPEC-01
- [ ] `tests/test_analysis/test_trace_formula.py` — stubs for SPEC-02
- [ ] `tests/test_analysis/test_modular_forms.py` — stubs for MOD-01
- [ ] `tests/test_analysis/test_lmfdb_client.py` — stubs for MOD-02 (mock HTTP)
- [ ] `tests/test_analysis/test_padic.py` — stubs for ADEL-01, ADEL-02
- [ ] `tests/test_analysis/test_tda.py` — stubs for XDISC-02
- [ ] `tests/test_analysis/test_dynamics.py` — stubs for XDISC-03
- [ ] `tests/test_analysis/test_ncg.py` — stubs for XDISC-04
- [ ] `tests/test_analysis/test_analogy.py` — stubs for XDISC-01
- [ ] `tests/test_analysis/test_conjecture_gen.py` — stubs for RSRCH-03
- [ ] `uv add ripser persim nolds requests` — new dependencies

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| p-adic fractal tree visual quality | ADEL-02 | Requires visual inspection of plot structure | Run `padic_viz.fractal_tree(p=5, depth=4)`, verify tree has correct branching factor |
| Upper half-plane domain coloring | MOD-01 | Requires visual inspection of color mapping | Run `modular_forms_viz.domain_coloring(...)`, verify colors match phase/magnitude |
| Phase portrait aesthetics | XDISC-03 | Requires visual verification of orbit structure | Run `dynamics_viz.phase_portrait(...)`, verify trajectories show expected topology |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
