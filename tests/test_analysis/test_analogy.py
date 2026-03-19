"""Tests for analogy engine module.

Tests cover:
- AnalogyMapping dataclass: creation, fields, to_dict, from_dict round-trip
- create_analogy_mapping: convenience constructor with defaults
- test_correspondence: KS test with similar/different distributions
- test_correspondence: chi-squared and correlation metrics
- save_analogy_to_workbench / load_analogy_from_workbench: round-trip persistence
- update_analogy_confidence: adjustment based on test results
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# AnalogyMapping dataclass
# ---------------------------------------------------------------------------

class TestAnalogyMapping:
    def test_creation_with_all_fields(self):
        from riemann.analysis.analogy import AnalogyMapping
        mapping = AnalogyMapping(
            source_domain="spectral",
            target_domain="zeta_zeros",
            correspondences={"eigenvalues": "zeros"},
            unknowns=["Hamiltonian"],
            evidence=[],
            confidence=0.0,
        )
        assert mapping.source_domain == "spectral"
        assert mapping.target_domain == "zeta_zeros"
        assert mapping.correspondences == {"eigenvalues": "zeros"}
        assert mapping.unknowns == ["Hamiltonian"]
        assert mapping.evidence == []
        assert mapping.confidence == 0.0

    def test_to_dict_returns_serializable(self):
        from riemann.analysis.analogy import AnalogyMapping
        import json
        mapping = AnalogyMapping(
            source_domain="spectral",
            target_domain="zeta_zeros",
            correspondences={"eigenvalues": "zeros", "trace_formula": "explicit_formula"},
            unknowns=["Hamiltonian"],
            evidence=["exp-001"],
            confidence=0.75,
        )
        d = mapping.to_dict()
        assert isinstance(d, dict)
        # Verify JSON-serializable
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_from_dict_round_trip(self):
        from riemann.analysis.analogy import AnalogyMapping
        original = AnalogyMapping(
            source_domain="spectral",
            target_domain="zeta_zeros",
            correspondences={"eigenvalues": "zeros"},
            unknowns=["Hamiltonian", "boundary_conditions"],
            evidence=["exp-001", "exp-002"],
            confidence=0.85,
        )
        d = original.to_dict()
        restored = AnalogyMapping.from_dict(d)
        assert restored.source_domain == original.source_domain
        assert restored.target_domain == original.target_domain
        assert restored.correspondences == original.correspondences
        assert restored.unknowns == original.unknowns
        assert restored.evidence == original.evidence
        assert restored.confidence == original.confidence


# ---------------------------------------------------------------------------
# create_analogy_mapping
# ---------------------------------------------------------------------------

class TestCreateAnalogyMapping:
    def test_returns_analogy_mapping(self):
        from riemann.analysis.analogy import create_analogy_mapping, AnalogyMapping
        mapping = create_analogy_mapping(
            "spectral", "zeta_zeros", {"eigenvalues": "zeros"}, ["Hamiltonian"]
        )
        assert isinstance(mapping, AnalogyMapping)

    def test_defaults(self):
        from riemann.analysis.analogy import create_analogy_mapping
        mapping = create_analogy_mapping(
            "spectral", "zeta_zeros", {"eigenvalues": "zeros"}
        )
        assert mapping.unknowns == []
        assert mapping.evidence == []
        assert mapping.confidence == 0.0


# ---------------------------------------------------------------------------
# test_correspondence
# ---------------------------------------------------------------------------

class TestTestCorrespondence:
    def test_similar_distributions_high_pvalue(self):
        """Same distribution should yield KS p-value > 0.05."""
        from riemann.analysis.analogy import test_correspondence, create_analogy_mapping
        rng = np.random.default_rng(42)
        source = rng.normal(0, 1, size=500)
        target = rng.normal(0, 1, size=500)
        mapping = create_analogy_mapping("A", "B", {"x": "y"})
        result = test_correspondence(mapping, source, target, metric="ks")
        assert result["pvalue"] > 0.05, (
            f"p-value {result['pvalue']} should be > 0.05 for similar distributions"
        )

    def test_different_distributions_low_pvalue(self):
        """Very different distributions should yield KS p-value < 0.01."""
        from riemann.analysis.analogy import test_correspondence, create_analogy_mapping
        rng = np.random.default_rng(42)
        source = rng.normal(0, 1, size=500)
        target = rng.normal(10, 0.1, size=500)
        mapping = create_analogy_mapping("A", "B", {"x": "y"})
        result = test_correspondence(mapping, source, target, metric="ks")
        assert result["pvalue"] < 0.01, (
            f"p-value {result['pvalue']} should be < 0.01 for very different distributions"
        )

    def test_returns_dict_with_expected_keys(self):
        from riemann.analysis.analogy import test_correspondence, create_analogy_mapping
        rng = np.random.default_rng(42)
        source = rng.normal(0, 1, size=100)
        target = rng.normal(0, 1, size=100)
        mapping = create_analogy_mapping("A", "B", {"x": "y"})
        result = test_correspondence(mapping, source, target, metric="ks")
        assert "metric" in result
        assert "statistic" in result
        assert "pvalue" in result
        assert "n_source" in result
        assert "n_target" in result

    def test_chi_squared_metric(self):
        from riemann.analysis.analogy import test_correspondence, create_analogy_mapping
        rng = np.random.default_rng(42)
        source = rng.normal(0, 1, size=500)
        target = rng.normal(0, 1, size=500)
        mapping = create_analogy_mapping("A", "B", {"x": "y"})
        result = test_correspondence(mapping, source, target, metric="chi_squared")
        assert result["metric"] == "chi_squared"
        assert "statistic" in result
        assert "pvalue" in result

    def test_correlation_metric(self):
        from riemann.analysis.analogy import test_correspondence, create_analogy_mapping
        rng = np.random.default_rng(42)
        # Highly correlated data
        source = np.sort(rng.normal(0, 1, size=200))
        target = np.sort(rng.normal(0, 1, size=200))
        mapping = create_analogy_mapping("A", "B", {"x": "y"})
        result = test_correspondence(mapping, source, target, metric="correlation")
        assert result["metric"] == "correlation"
        assert result["statistic"] > 0.9, (
            f"Sorted normal samples should have high correlation, got {result['statistic']}"
        )


# ---------------------------------------------------------------------------
# save_analogy_to_workbench / load_analogy_from_workbench
# ---------------------------------------------------------------------------

class TestWorkbenchPersistence:
    def test_save_returns_experiment_id(self, temp_db):
        from riemann.analysis.analogy import (
            create_analogy_mapping, save_analogy_to_workbench,
        )
        mapping = create_analogy_mapping(
            "spectral", "zeta_zeros", {"eigenvalues": "zeros"}, ["Hamiltonian"]
        )
        exp_id = save_analogy_to_workbench(mapping, db_path=temp_db)
        assert isinstance(exp_id, str)
        assert len(exp_id) > 0

    def test_round_trip(self, temp_db):
        from riemann.analysis.analogy import (
            create_analogy_mapping, save_analogy_to_workbench,
            load_analogy_from_workbench,
        )
        original = create_analogy_mapping(
            "spectral", "zeta_zeros",
            {"eigenvalues": "zeros", "trace_formula": "explicit_formula"},
            ["Hamiltonian", "boundary_conditions"],
        )
        original.confidence = 0.75
        original.evidence = ["exp-001"]

        exp_id = save_analogy_to_workbench(original, db_path=temp_db)
        loaded = load_analogy_from_workbench(exp_id, db_path=temp_db)

        assert loaded is not None
        assert loaded.source_domain == original.source_domain
        assert loaded.target_domain == original.target_domain
        assert loaded.correspondences == original.correspondences
        assert loaded.unknowns == original.unknowns
        assert loaded.evidence == original.evidence
        assert loaded.confidence == original.confidence

    def test_load_nonexistent_returns_none(self, temp_db):
        from riemann.analysis.analogy import load_analogy_from_workbench
        from riemann.workbench.db import init_db
        init_db(temp_db)
        result = load_analogy_from_workbench("nonexistent-id", db_path=temp_db)
        assert result is None


# ---------------------------------------------------------------------------
# update_analogy_confidence
# ---------------------------------------------------------------------------

class TestUpdateAnalogyConfidence:
    def test_increase_on_high_pvalue(self):
        from riemann.analysis.analogy import (
            create_analogy_mapping, update_analogy_confidence,
        )
        mapping = create_analogy_mapping("A", "B", {"x": "y"})
        mapping.confidence = 0.5
        result = update_analogy_confidence(mapping, {"pvalue": 0.10})
        assert result.confidence == pytest.approx(0.6)

    def test_decrease_on_low_pvalue(self):
        from riemann.analysis.analogy import (
            create_analogy_mapping, update_analogy_confidence,
        )
        mapping = create_analogy_mapping("A", "B", {"x": "y"})
        mapping.confidence = 0.5
        result = update_analogy_confidence(mapping, {"pvalue": 0.005})
        assert result.confidence == pytest.approx(0.4)

    def test_capped_at_1(self):
        from riemann.analysis.analogy import (
            create_analogy_mapping, update_analogy_confidence,
        )
        mapping = create_analogy_mapping("A", "B", {"x": "y"})
        mapping.confidence = 0.95
        result = update_analogy_confidence(mapping, {"pvalue": 0.10})
        assert result.confidence == 1.0

    def test_floored_at_0(self):
        from riemann.analysis.analogy import (
            create_analogy_mapping, update_analogy_confidence,
        )
        mapping = create_analogy_mapping("A", "B", {"x": "y"})
        mapping.confidence = 0.05
        result = update_analogy_confidence(mapping, {"pvalue": 0.005})
        assert result.confidence == 0.0

    def test_no_change_in_middle_pvalue(self):
        """p-value between 0.01 and 0.05 should not change confidence."""
        from riemann.analysis.analogy import (
            create_analogy_mapping, update_analogy_confidence,
        )
        mapping = create_analogy_mapping("A", "B", {"x": "y"})
        mapping.confidence = 0.5
        result = update_analogy_confidence(mapping, {"pvalue": 0.03})
        assert result.confidence == 0.5
