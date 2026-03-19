"""Tests for AI-guided conjecture generation module.

Covers: ExperimentSuggestion dataclass, suggest_experiments (bootstrap + context-aware),
analyze_results (parsing + error handling), generate_conjecture (persistence + evidence linking).
"""
import pytest

from riemann.workbench.db import init_db
from riemann.workbench.experiment import save_experiment, load_experiment, list_experiments
from riemann.workbench.conjecture import list_conjectures, get_conjecture
from riemann.workbench.evidence import get_evidence_for_conjecture


# ---------------------------------------------------------------------------
# ExperimentSuggestion dataclass
# ---------------------------------------------------------------------------

def test_experiment_suggestion_fields():
    """ExperimentSuggestion has all required fields."""
    from riemann.analysis.conjecture_gen import ExperimentSuggestion

    s = ExperimentSuggestion(
        type="spectral",
        description="Run spectral analysis",
        rationale="Explore eigenvalue distributions",
        parameters={"N": 200},
        priority=0.9,
    )
    assert s.type == "spectral"
    assert s.description == "Run spectral analysis"
    assert s.rationale == "Explore eigenvalue distributions"
    assert s.parameters == {"N": 200}
    assert s.priority == 0.9


# ---------------------------------------------------------------------------
# suggest_experiments: empty workbench => bootstrap suggestions
# ---------------------------------------------------------------------------

def test_suggest_experiments_bootstrap(temp_db):
    """Empty workbench returns bootstrap suggestions covering multiple domains."""
    from riemann.analysis.conjecture_gen import suggest_experiments, ExperimentSuggestion

    init_db(temp_db)
    suggestions = suggest_experiments(db_path=temp_db)

    assert isinstance(suggestions, list)
    assert len(suggestions) >= 3  # at least a few bootstrap suggestions
    assert all(isinstance(s, ExperimentSuggestion) for s in suggestions)

    # Should be sorted by priority descending
    priorities = [s.priority for s in suggestions]
    assert priorities == sorted(priorities, reverse=True)

    # Should cover multiple domain types
    types = {s.type for s in suggestions}
    assert len(types) >= 3


def test_suggest_experiments_max_suggestions(temp_db):
    """max_suggestions parameter limits number of results."""
    from riemann.analysis.conjecture_gen import suggest_experiments

    init_db(temp_db)
    suggestions = suggest_experiments(db_path=temp_db, max_suggestions=3)

    assert len(suggestions) <= 3


# ---------------------------------------------------------------------------
# suggest_experiments: with existing experiments => context-aware suggestions
# ---------------------------------------------------------------------------

def test_suggest_experiments_context_aware(temp_db):
    """With existing experiments, suggestions build on prior results."""
    from riemann.analysis.conjecture_gen import suggest_experiments

    init_db(temp_db)

    # Add a few experiments of one type
    for i in range(3):
        save_experiment(
            description=f"Spectral experiment {i}",
            parameters={"type": "spectral", "N": 100 + i * 50},
            result_summary="Eigenvalues follow GUE distribution",
            db_path=temp_db,
        )

    suggestions = suggest_experiments(db_path=temp_db)
    assert isinstance(suggestions, list)
    assert len(suggestions) >= 1

    # Under-explored domains should appear in suggestions
    types = {s.type for s in suggestions}
    # At least some non-spectral domains should be suggested since spectral is well-covered
    non_spectral = types - {"spectral"}
    assert len(non_spectral) >= 1


def test_suggest_experiments_speculative_conjectures(temp_db):
    """Speculative conjectures without supporting experiments get suggestions."""
    from riemann.analysis.conjecture_gen import suggest_experiments
    from riemann.workbench.conjecture import create_conjecture

    init_db(temp_db)

    # Create a speculative conjecture
    create_conjecture(
        statement="The spectral gap narrows for large N",
        status="speculative",
        evidence_level=0,
        db_path=temp_db,
    )

    suggestions = suggest_experiments(db_path=temp_db)
    assert len(suggestions) >= 1


# ---------------------------------------------------------------------------
# analyze_results
# ---------------------------------------------------------------------------

def test_analyze_results_structured_output(temp_db):
    """analyze_results returns dict with all required keys."""
    from riemann.analysis.conjecture_gen import analyze_results

    init_db(temp_db)

    exp_id = save_experiment(
        description="Test spectral experiment",
        parameters={"type": "spectral", "N": 200},
        result_summary="Eigenvalues match GUE distribution. Deviation at N=150 is unexpected. Mean spacing 1.001.",
        db_path=temp_db,
    )

    result = analyze_results(exp_id, db_path=temp_db)

    assert isinstance(result, dict)
    assert "summary" in result
    assert "patterns_detected" in result
    assert "anomalies" in result
    assert "suggested_conjectures" in result
    assert "next_experiments" in result
    assert isinstance(result["patterns_detected"], list)
    assert isinstance(result["anomalies"], list)
    assert isinstance(result["suggested_conjectures"], list)
    assert isinstance(result["next_experiments"], list)


def test_analyze_results_detects_anomaly_keywords(temp_db):
    """analyze_results finds anomaly-like keywords in result_summary."""
    from riemann.analysis.conjecture_gen import analyze_results

    init_db(temp_db)

    exp_id = save_experiment(
        description="Anomalous experiment",
        parameters={"type": "tda"},
        result_summary="Persistence diagram shows unexpected cluster. Deviation from predicted topology is surprising.",
        db_path=temp_db,
    )

    result = analyze_results(exp_id, db_path=temp_db)

    # Should detect at least one anomaly-like finding
    assert len(result["anomalies"]) >= 1


def test_analyze_results_nonexistent_raises(temp_db):
    """analyze_results raises ValueError for nonexistent experiment."""
    from riemann.analysis.conjecture_gen import analyze_results

    init_db(temp_db)

    with pytest.raises(ValueError, match="not found"):
        analyze_results("nonexistent-uuid", db_path=temp_db)


# ---------------------------------------------------------------------------
# generate_conjecture
# ---------------------------------------------------------------------------

def test_generate_conjecture_returns_uuid(temp_db):
    """generate_conjecture returns a conjecture_id string (UUID)."""
    from riemann.analysis.conjecture_gen import generate_conjecture

    init_db(temp_db)

    cid = generate_conjecture(
        observations=["pattern A observed", "pattern B confirmed"],
        db_path=temp_db,
    )

    assert isinstance(cid, str)
    assert len(cid) > 0


def test_generate_conjecture_persists_in_db(temp_db):
    """generate_conjecture saves a conjecture with evidence_level=0 and status=speculative."""
    from riemann.analysis.conjecture_gen import generate_conjecture

    init_db(temp_db)

    cid = generate_conjecture(
        observations=["pattern A observed", "pattern B confirmed"],
        db_path=temp_db,
    )

    conj = get_conjecture(cid, db_path=temp_db)
    assert conj is not None
    assert conj["evidence_level"] == 0
    assert conj["status"] == "speculative"
    assert "pattern A observed" in conj["statement"]
    assert "pattern B confirmed" in conj["statement"]


def test_generate_conjecture_with_evidence_ids(temp_db):
    """generate_conjecture with evidence_ids links the conjecture to experiments."""
    from riemann.analysis.conjecture_gen import generate_conjecture

    init_db(temp_db)

    # Create experiments to link
    exp1 = save_experiment(
        description="Experiment 1",
        parameters={"type": "spectral"},
        db_path=temp_db,
    )
    exp2 = save_experiment(
        description="Experiment 2",
        parameters={"type": "tda"},
        db_path=temp_db,
    )

    cid = generate_conjecture(
        observations=["Cross-domain pattern found"],
        evidence_ids=[exp1, exp2],
        db_path=temp_db,
    )

    # Check evidence links exist
    links = get_evidence_for_conjecture(cid, db_path=temp_db)
    assert len(links) == 2

    linked_exp_ids = {link["experiment_id"] for link in links}
    assert exp1 in linked_exp_ids
    assert exp2 in linked_exp_ids


def test_generate_conjecture_tags(temp_db):
    """generate_conjecture tags the conjecture with ai_generated and computational_evidence."""
    from riemann.analysis.conjecture_gen import generate_conjecture

    init_db(temp_db)

    cid = generate_conjecture(
        observations=["interesting pattern"],
        db_path=temp_db,
    )

    conj = get_conjecture(cid, db_path=temp_db)
    assert conj is not None
    assert "ai_generated" in conj["tags"]
    assert "computational_evidence" in conj["tags"]
