"""AI-guided conjecture generation: experiment suggestion, result analysis, conjecture formalization.

This is the synthesis layer that reads across all domain modules and the workbench
to identify patterns, suggest experiments, and generate formal conjectures. It
transforms raw computational output into actionable mathematical insight.

Key functions:
- suggest_experiments: prioritized next-step suggestions based on workbench state
- analyze_results: structured interpretation of experiment outcomes
- generate_conjecture: synthesize observations into formal conjecture records

Function-based API. Returns dataclass results and dicts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from riemann.workbench.conjecture import create_conjecture, list_conjectures
from riemann.workbench.experiment import list_experiments, load_experiment
from riemann.workbench.evidence import link_evidence
from riemann.workbench.db import init_db


# All recognized experiment/domain types
ALL_DOMAINS = (
    "spectral", "trace", "modular", "padic", "tda",
    "dynamics", "ncg", "analogy", "cross_domain",
)

# Keywords that indicate anomaly-like findings in result text
_ANOMALY_KEYWORDS = ("deviat", "unexpect", "anomal", "surpris")


@dataclass
class ExperimentSuggestion:
    """A prioritized suggestion for the next experiment to run.

    Attributes:
        type: Domain category (e.g., "spectral", "tda", "ncg", "analogy").
        description: Human-readable description of the experiment.
        rationale: Why this experiment is suggested given current state.
        parameters: Suggested parameter values for the experiment.
        priority: 0.0-1.0, higher = more important.
    """

    type: str
    description: str
    rationale: str
    parameters: dict
    priority: float


def _bootstrap_suggestions() -> list[ExperimentSuggestion]:
    """Return default suggestions for an empty workbench."""
    return [
        ExperimentSuggestion(
            type="spectral",
            description="Run Berry-Keating spectral analysis at N=200",
            rationale="Spectral operator eigenvalue statistics are the primary link to zeta zeros",
            parameters={"type": "spectral", "N": 200, "operator": "berry_keating"},
            priority=0.9,
        ),
        ExperimentSuggestion(
            type="analogy",
            description="Test spectral-zeros analogy mapping",
            rationale="Formal correspondence mappings identify proof gaps",
            parameters={"type": "analogy", "source": "spectral_theory", "target": "zeta_zeros"},
            priority=0.85,
        ),
        ExperimentSuggestion(
            type="tda",
            description="Compute persistence diagrams of zero embeddings",
            rationale="Topological features may reveal hidden structure in zero distributions",
            parameters={"type": "tda", "embedding": "zero_spacings", "max_dim": 2},
            priority=0.8,
        ),
        ExperimentSuggestion(
            type="trace",
            description="Compute Weil explicit formula convergence",
            rationale="Trace formula convergence rate constrains zero distribution properties",
            parameters={"type": "trace", "method": "weil_explicit", "num_zeros": 50},
            priority=0.75,
        ),
        ExperimentSuggestion(
            type="ncg",
            description="Scan Bost-Connes phase transition near beta=1",
            rationale="Phase transition structure connects to prime distribution and zeta poles",
            parameters={"type": "ncg", "beta_range": [0.5, 2.0], "steps": 50},
            priority=0.7,
        ),
    ]


def suggest_experiments(
    db_path: str | Path | None = None,
    max_suggestions: int = 5,
) -> list[ExperimentSuggestion]:
    """Suggest prioritized next-step experiments based on current workbench state.

    Strategy A (empty workbench): Returns bootstrap suggestions covering key domains.
    Strategy B (has experiments): Analyzes existing experiments and suggests
    under-explored domains, conjecture-testing experiments, and anomaly investigations.

    Args:
        db_path: Path to workbench database.
        max_suggestions: Maximum number of suggestions to return.

    Returns:
        List of ExperimentSuggestion sorted by priority descending,
        truncated to max_suggestions.
    """
    init_db(db_path)
    experiments = list_experiments(db_path=db_path)
    conjectures = list_conjectures(db_path=db_path)

    if not experiments:
        # Strategy A: bootstrap
        suggestions = _bootstrap_suggestions()
    else:
        # Strategy B: context-aware suggestions
        suggestions = _context_aware_suggestions(experiments, conjectures)

    # Sort by priority descending and truncate
    suggestions.sort(key=lambda s: s.priority, reverse=True)
    return suggestions[:max_suggestions]


def _context_aware_suggestions(
    experiments: list[dict],
    conjectures: list[dict],
) -> list[ExperimentSuggestion]:
    """Generate suggestions based on existing workbench state."""
    suggestions: list[ExperimentSuggestion] = []

    # Count experiments by type
    type_counts: dict[str, int] = {}
    for exp in experiments:
        exp_type = exp.get("parameters", {}).get("type", "unknown")
        type_counts[exp_type] = type_counts.get(exp_type, 0) + 1

    total_experiments = len(experiments)

    # Suggest under-explored domains
    for domain in ALL_DOMAINS:
        count = type_counts.get(domain, 0)
        if count == 0:
            # Completely unexplored domain gets high priority
            suggestions.append(ExperimentSuggestion(
                type=domain,
                description=f"Explore {domain} domain (no experiments yet)",
                rationale=f"Domain '{domain}' has zero experiments; cross-disciplinary coverage needed",
                parameters={"type": domain},
                priority=0.8,
            ))
        elif count < total_experiments / len(ALL_DOMAINS):
            # Under-represented domain
            suggestions.append(ExperimentSuggestion(
                type=domain,
                description=f"Expand {domain} experiments (only {count} so far)",
                rationale=f"Domain '{domain}' is under-represented relative to others",
                parameters={"type": domain},
                priority=0.6,
            ))

    # Suggest testing speculative conjectures
    for conj in conjectures:
        if conj.get("status") == "speculative":
            suggestions.append(ExperimentSuggestion(
                type="cross_domain",
                description=f"Test conjecture: {conj['statement'][:80]}",
                rationale="Speculative conjecture needs supporting or contradicting evidence",
                parameters={"type": "cross_domain", "conjecture_id": conj["id"]},
                priority=0.7,
            ))

    # Suggest investigating anomalies (conjectures tagged "anomaly")
    for conj in conjectures:
        tags = conj.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]
        if tags and "anomaly" in tags:
            suggestions.append(ExperimentSuggestion(
                type="cross_domain",
                description=f"Investigate anomaly: {conj['statement'][:80]}",
                rationale="Anomalous finding requires targeted investigation",
                parameters={"type": "cross_domain", "conjecture_id": conj["id"]},
                priority=0.75,
            ))

    # If no suggestions generated, fall back to bootstrap
    if not suggestions:
        suggestions = _bootstrap_suggestions()

    return suggestions


def analyze_results(
    experiment_id: str,
    db_path: str | Path | None = None,
) -> dict:
    """Produce a structured interpretation of an experiment's outcomes.

    Parses the experiment's parameters and result_summary to extract patterns,
    anomalies, suggested conjectures, and next experiments.

    Args:
        experiment_id: UUID of the experiment to analyze.
        db_path: Path to workbench database.

    Returns:
        Dict with keys:
            - "summary": Brief 1-sentence summary
            - "patterns_detected": List of pattern strings
            - "anomalies": List of anomaly-like findings
            - "suggested_conjectures": List of dicts {"statement": str, "confidence": float}
            - "next_experiments": List of dicts {"type": str, "description": str}

    Raises:
        ValueError: If experiment not found.
    """
    experiment = load_experiment(experiment_id, db_path=db_path)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_id}' not found")

    description = experiment.get("description", "")
    result_summary = experiment.get("result_summary", "") or ""
    parameters = experiment.get("parameters", {})
    exp_type = parameters.get("type", "unknown")

    # Build summary
    summary = f"{exp_type.capitalize()} experiment: {description}"

    # Extract patterns from result_summary (split on sentence boundaries)
    patterns = _extract_patterns(result_summary)

    # Extract anomaly-like findings
    anomalies = _extract_anomalies(result_summary)

    # Generate suggested conjectures from patterns
    suggested_conjectures = []
    for pattern in patterns:
        if len(pattern.strip()) > 10:  # Only substantial patterns
            suggested_conjectures.append({
                "statement": f"Pattern observed: {pattern.strip()}",
                "confidence": 0.1,
            })

    # Suggest next experiments
    next_experiments = _suggest_follow_ups(exp_type, patterns, anomalies)

    return {
        "summary": summary,
        "patterns_detected": patterns,
        "anomalies": anomalies,
        "suggested_conjectures": suggested_conjectures,
        "next_experiments": next_experiments,
    }


def _extract_patterns(result_summary: str) -> list[str]:
    """Extract pattern strings from result_summary by splitting on sentence boundaries."""
    if not result_summary:
        return []

    # Split on periods, semicolons, and newlines
    fragments = re.split(r'[.;\n]+', result_summary)
    # Filter to non-empty, non-whitespace fragments
    return [f.strip() for f in fragments if f.strip()]


def _extract_anomalies(result_summary: str) -> list[str]:
    """Find anomaly-like substrings containing deviation/unexpected/anomaly/surprise keywords."""
    if not result_summary:
        return []

    anomalies = []
    # Split into sentences for context
    sentences = re.split(r'[.;\n]+', result_summary)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        lower = sentence.lower()
        if any(kw in lower for kw in _ANOMALY_KEYWORDS):
            anomalies.append(sentence)

    return anomalies


def _suggest_follow_ups(
    exp_type: str,
    patterns: list[str],
    anomalies: list[str],
) -> list[dict]:
    """Suggest follow-up experiments based on type, patterns, and anomalies."""
    follow_ups = []

    # If anomalies found, suggest deeper investigation
    if anomalies:
        follow_ups.append({
            "type": exp_type,
            "description": f"Re-run {exp_type} with finer parameters to investigate anomalies",
        })

    # Suggest cross-domain comparison
    other_domains = [d for d in ALL_DOMAINS if d != exp_type and d != "cross_domain"]
    if other_domains:
        follow_ups.append({
            "type": other_domains[0],
            "description": f"Compare {exp_type} results with {other_domains[0]} analysis",
        })

    # If patterns detected, suggest conjecture formalization
    if patterns:
        follow_ups.append({
            "type": "cross_domain",
            "description": "Formalize detected patterns into testable conjectures",
        })

    return follow_ups


def generate_conjecture(
    observations: list[str],
    evidence_ids: list[str] | None = None,
    confidence: float = 0.0,
    db_path: str | Path | None = None,
) -> str:
    """Synthesize observations into a formal conjecture record in the workbench.

    Creates a conjecture with evidence_level=0 (OBSERVATION) and status="speculative",
    tagged as AI-generated. Optionally links experiment evidence.

    Args:
        observations: List of observation strings to synthesize.
        evidence_ids: Optional list of experiment UUIDs to link as supporting evidence.
        confidence: Initial confidence score (0.0-1.0).
        db_path: Path to workbench database.

    Returns:
        UUID string of the created conjecture.
    """
    init_db(db_path)

    # Synthesize statement from observations
    obs_text = "; ".join(observations)
    statement = f"Observed: {obs_text}. Conjecture: pattern holds generally"

    conjecture_id = create_conjecture(
        statement=statement,
        description=f"AI-generated conjecture from {len(observations)} observations",
        evidence_level=0,
        status="speculative",
        confidence=confidence,
        tags=["ai_generated", "computational_evidence"],
        db_path=db_path,
    )

    # Link evidence if provided
    if evidence_ids:
        for exp_id in evidence_ids:
            link_evidence(
                conjecture_id=conjecture_id,
                experiment_id=exp_id,
                relationship="supports",
                notes="Linked by AI conjecture generator",
                db_path=db_path,
            )

    return conjecture_id
