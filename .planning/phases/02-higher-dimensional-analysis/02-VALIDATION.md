---
phase: 2
slug: higher-dimensional-analysis
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-19
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >= 9.0.2 |
| **Config file** | pyproject.toml `[tool.pytest.ini_options]` |
| **Quick run command** | `uv run pytest tests/ -x --timeout=30 -q` |
| **Full suite command** | `uv run pytest tests/ --timeout=120` |
| **Estimated runtime** | ~20 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/ -x --timeout=30 -q`
- **After every plan wave:** Run `uv run pytest tests/ --timeout=120`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 0 | ZERO-01 | unit | `uv run pytest tests/test_analysis/test_spacing.py -x` | ❌ W0 | ⬜ pending |
| 02-01-02 | 01 | 0 | ZERO-01 | unit | `uv run pytest tests/test_analysis/test_spacing.py::test_pair_correlation_gue -x` | ❌ W0 | ⬜ pending |
| 02-01-03 | 01 | 0 | ZERO-02 | unit | `uv run pytest tests/test_analysis/test_anomaly.py -x` | ❌ W0 | ⬜ pending |
| 02-01-04 | 01 | 0 | HDIM-01 | unit | `uv run pytest tests/test_embedding/test_coordinates.py -x` | ❌ W0 | ⬜ pending |
| 02-01-05 | 01 | 0 | HDIM-01 | unit | `uv run pytest tests/test_embedding/test_registry.py -x` | ❌ W0 | ⬜ pending |
| 02-02-01 | 02 | 1 | HDIM-02 | unit | `uv run pytest tests/test_viz/test_projection.py::test_pca -x` | ❌ W0 | ⬜ pending |
| 02-02-02 | 02 | 1 | HDIM-02 | unit | `uv run pytest tests/test_viz/test_projection.py::test_tsne_umap -x` | ❌ W0 | ⬜ pending |
| 02-02-03 | 02 | 1 | VIZ-03 | smoke | `uv run pytest tests/test_viz/test_theater.py -x` | ❌ W0 | ⬜ pending |
| 02-02-04 | 02 | 1 | RMT-01 | unit | `uv run pytest tests/test_analysis/test_rmt.py::test_gue_wigner -x` | ❌ W0 | ⬜ pending |
| 02-02-05 | 02 | 1 | RMT-02 | unit | `uv run pytest tests/test_analysis/test_rmt.py::test_gue_n_scaling -x` | ❌ W0 | ⬜ pending |
| 02-02-06 | 02 | 1 | INFO-01 | unit | `uv run pytest tests/test_analysis/test_information.py::test_entropy -x` | ❌ W0 | ⬜ pending |
| 02-02-07 | 02 | 1 | INFO-02 | unit | `uv run pytest tests/test_analysis/test_information.py::test_cross_comparison -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_analysis/__init__.py` — package init
- [ ] `tests/test_analysis/test_spacing.py` — covers ZERO-01
- [ ] `tests/test_analysis/test_rmt.py` — covers RMT-01, RMT-02
- [ ] `tests/test_analysis/test_information.py` — covers INFO-01, INFO-02
- [ ] `tests/test_analysis/test_anomaly.py` — covers ZERO-02
- [ ] `tests/test_embedding/__init__.py` — package init
- [ ] `tests/test_embedding/test_coordinates.py` — covers HDIM-01
- [ ] `tests/test_embedding/test_registry.py` — covers HDIM-01
- [ ] `tests/test_embedding/test_storage.py` — covers HDIM-01 HDF5
- [ ] `tests/test_viz/test_projection.py` — covers HDIM-02
- [ ] `tests/test_viz/test_theater.py` — covers VIZ-03
- [ ] Dependencies: `uv add scikit-learn umap-learn` — required before tests run

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Projection theater interactive rotation in JupyterLab | VIZ-03 | 3D rotation/zoom requires live browser | Open notebook, render Plotly 3D figure, verify rotation controls |
| Side-by-side projection comparison visual coherence | HDIM-02 | Layout quality requires visual inspection | Generate same data with PCA + t-SNE, verify linked highlighting |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
