---
phase: 4
slug: lean-4-formalization-pipeline
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-19
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >=9.0.2 |
| **Config file** | `pyproject.toml` [tool.pytest.ini_options] |
| **Quick run command** | `uv run pytest tests/test_formalization/ -x --timeout=30` |
| **Full suite command** | `uv run pytest tests/ --timeout=120` |
| **Estimated runtime** | ~20 seconds (unit tests); integration tests depend on WSL build speed |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/test_formalization/ -x --timeout=30`
- **After every plan wave:** Run `uv run pytest tests/ --timeout=120`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds (unit), 60 seconds (integration with WSL build)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 0 | FORM-01 | integration | `uv run pytest tests/test_formalization/test_builder.py::test_wsl_lean_available -x` | ❌ W0 | ⬜ pending |
| 04-01-02 | 01 | 1 | FORM-01 | unit | `uv run pytest tests/test_formalization/test_translator.py -x` | ❌ W0 | ⬜ pending |
| 04-01-03 | 01 | 1 | FORM-01 | unit | `uv run pytest tests/test_formalization/test_parser.py -x` | ❌ W0 | ⬜ pending |
| 04-01-04 | 01 | 1 | FORM-01 | integration | `uv run pytest tests/test_formalization/test_builder.py -x --timeout=60` | ❌ W0 | ⬜ pending |
| 04-02-01 | 02 | 1 | FORM-02 | unit | `uv run pytest tests/test_formalization/test_tracker.py -x` | ❌ W0 | ⬜ pending |
| 04-02-02 | 02 | 1 | FORM-02 | unit | `uv run pytest tests/test_formalization/test_tracker.py::test_auto_promote -x` | ❌ W0 | ⬜ pending |
| 04-02-03 | 02 | 1 | FORM-02 | unit | `uv run pytest tests/test_formalization/test_tracker.py::test_build_history -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_formalization/__init__.py` — package init
- [ ] `tests/test_formalization/test_builder.py` — WSL subprocess build tests
- [ ] `tests/test_formalization/test_parser.py` — Lean output parsing tests
- [ ] `tests/test_formalization/test_translator.py` — Conjecture-to-Lean translation tests
- [ ] `tests/test_formalization/test_tracker.py` — State machine, sorry tracking, auto-promotion tests
- [ ] `tests/test_formalization/conftest.py` — Shared fixtures (temp DB, mock lean output)
- [ ] WSL elan installation validation (Wave 0 task)
- [ ] Lean project creation + Mathlib cache download (Wave 0 task)
- [ ] Hello-world theorem build from Python subprocess (Wave 0 gate)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Mathlib cache download completes | FORM-01 | Network-dependent, slow (~10 min) | Run `wsl -e bash -c "cd /mnt/c/.../lean_proofs && lake exe cache get"` and verify exit code 0 |
| Real conjecture formalization quality | FORM-01 | Requires human math review | Inspect generated .lean files for mathematical correctness |
| Full assault conjecture triage | FORM-02 | Requires human judgment on prioritization | Review Claude's attack order against workbench state |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
