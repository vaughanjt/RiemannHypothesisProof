"""Analogy engine: formal correspondence mappings between mathematical domains.

The analogy engine is a synthesis tool that formalizes structure-preserving maps
between mathematical domains (e.g., spectral theory <-> zeta zeros). By encoding
known correspondences and explicitly tracking unknowns, the gaps in the maps
point at what is missing from a proof.

Key objects:
- AnalogyMapping: dataclass encoding source/target domains, known correspondences,
  and unknown gaps
- test_correspondence: statistical comparison of distributions via KS, chi-squared,
  or correlation
- Workbench persistence: save/load mappings as experiments for reproducibility

Function-based API. Returns dataclass results and dicts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, pearsonr

from riemann.workbench.experiment import save_experiment, load_experiment


@dataclass
class AnalogyMapping:
    """A formal correspondence mapping between two mathematical domains.

    Attributes:
        source_domain: Name of the source domain (e.g., "spectral_theory").
        target_domain: Name of the target domain (e.g., "zeta_zeros").
        correspondences: Known mappings between concepts (source -> target).
        unknowns: Concepts in the target domain without known source counterparts.
        evidence: List of experiment IDs supporting this mapping.
        confidence: Overall confidence score (0.0 to 1.0).
    """

    source_domain: str
    target_domain: str
    correspondences: dict[str, str]
    unknowns: list[str]
    evidence: list[str]
    confidence: float = 0.0

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "correspondences": dict(self.correspondences),
            "unknowns": list(self.unknowns),
            "evidence": list(self.evidence),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AnalogyMapping:
        """Reconstruct an AnalogyMapping from a dict."""
        return cls(
            source_domain=d["source_domain"],
            target_domain=d["target_domain"],
            correspondences=dict(d["correspondences"]),
            unknowns=list(d["unknowns"]),
            evidence=list(d["evidence"]),
            confidence=float(d["confidence"]),
        )


def create_analogy_mapping(
    source_domain: str,
    target_domain: str,
    correspondences: dict[str, str],
    unknowns: list[str] | None = None,
    confidence: float = 0.0,
) -> AnalogyMapping:
    """Create an AnalogyMapping with sensible defaults.

    Args:
        source_domain: Name of the source domain.
        target_domain: Name of the target domain.
        correspondences: Known mappings between concepts.
        unknowns: Unknown gaps. Defaults to empty list.
        confidence: Initial confidence score. Defaults to 0.0.

    Returns:
        A new AnalogyMapping instance.
    """
    return AnalogyMapping(
        source_domain=source_domain,
        target_domain=target_domain,
        correspondences=correspondences,
        unknowns=unknowns if unknowns is not None else [],
        evidence=[],
        confidence=confidence,
    )


def test_correspondence(
    mapping: AnalogyMapping,
    source_data: np.ndarray,
    target_data: np.ndarray,
    metric: str = "ks",
) -> dict:
    """Test whether a proposed correspondence holds computationally.

    Compares source_data distribution against target_data using the specified
    statistical test.

    Args:
        mapping: The AnalogyMapping being tested (used for context).
        source_data: 1D array of values from the source domain.
        target_data: 1D array of values from the target domain.
        metric: Statistical test to use:
            - "ks": Kolmogorov-Smirnov two-sample test
            - "chi_squared": Histogram-based chi-squared test
            - "correlation": Pearson correlation of sorted, length-matched arrays

    Returns:
        Dict with keys:
            - "metric": name of the test used
            - "statistic": test statistic value
            - "pvalue": p-value of the test
            - "n_source": number of source data points
            - "n_target": number of target data points
    """
    source_data = np.asarray(source_data, dtype=np.float64).ravel()
    target_data = np.asarray(target_data, dtype=np.float64).ravel()

    result = {
        "metric": metric,
        "n_source": len(source_data),
        "n_target": len(target_data),
    }

    if metric == "ks":
        stat, pvalue = ks_2samp(source_data, target_data)
        result["statistic"] = float(stat)
        result["pvalue"] = float(pvalue)

    elif metric == "chi_squared":
        # Histogram-based chi-squared test
        # Combine data to determine bin edges
        combined = np.concatenate([source_data, target_data])
        n_bins = min(50, max(10, int(np.sqrt(min(len(source_data), len(target_data))))))
        bin_edges = np.histogram_bin_edges(combined, bins=n_bins)

        source_hist, _ = np.histogram(source_data, bins=bin_edges)
        target_hist, _ = np.histogram(target_data, bins=bin_edges)

        # Normalize to same total for fair comparison
        source_norm = source_hist.astype(np.float64)
        target_norm = target_hist.astype(np.float64)

        # Scale target to source total for chi-squared
        if target_norm.sum() > 0:
            target_norm = target_norm * (source_norm.sum() / target_norm.sum())

        # Chi-squared statistic: sum((O - E)^2 / E) where E > 0
        mask = target_norm > 0.5  # Avoid near-zero expected counts
        if mask.sum() > 0:
            chi2 = float(np.sum(
                (source_norm[mask] - target_norm[mask]) ** 2 / target_norm[mask]
            ))
            # Approximate p-value using chi-squared distribution
            from scipy.stats import chi2 as chi2_dist
            dof = max(1, mask.sum() - 1)
            pvalue = float(1.0 - chi2_dist.cdf(chi2, dof))
        else:
            chi2 = 0.0
            pvalue = 1.0

        result["statistic"] = chi2
        result["pvalue"] = pvalue

    elif metric == "correlation":
        # Pearson correlation of sorted, length-matched arrays
        n = min(len(source_data), len(target_data))
        source_sorted = np.sort(source_data)[:n]
        target_sorted = np.sort(target_data)[:n]
        corr, pvalue = pearsonr(source_sorted, target_sorted)
        result["statistic"] = float(corr)
        result["pvalue"] = float(pvalue)

    else:
        raise ValueError(
            f"Unknown metric '{metric}'. Supported: 'ks', 'chi_squared', 'correlation'"
        )

    return result


def save_analogy_to_workbench(
    mapping: AnalogyMapping,
    description: str = "",
    db_path: str | Path | None = None,
) -> str:
    """Persist an AnalogyMapping as an experiment in the workbench.

    Args:
        mapping: The AnalogyMapping to save.
        description: Optional description override.
        db_path: Database path (uses default if None).

    Returns:
        Experiment UUID string.
    """
    if not description:
        description = (
            f"Analogy mapping: {mapping.source_domain} -> {mapping.target_domain}"
        )

    experiment_id = save_experiment(
        description=description,
        parameters=mapping.to_dict(),
        result_summary=f"Confidence: {mapping.confidence:.2f}, "
        f"Correspondences: {len(mapping.correspondences)}, "
        f"Unknowns: {len(mapping.unknowns)}",
        db_path=db_path,
    )

    return experiment_id


def load_analogy_from_workbench(
    experiment_id: str,
    db_path: str | Path | None = None,
) -> AnalogyMapping | None:
    """Load an AnalogyMapping from the workbench by experiment ID.

    Args:
        experiment_id: UUID of the saved experiment.
        db_path: Database path (uses default if None).

    Returns:
        Reconstructed AnalogyMapping, or None if experiment not found.
    """
    experiment = load_experiment(experiment_id, db_path=db_path)
    if experiment is None:
        return None

    return AnalogyMapping.from_dict(experiment["parameters"])


def update_analogy_confidence(
    mapping: AnalogyMapping,
    test_result: dict,
) -> AnalogyMapping:
    """Update mapping confidence based on a test_correspondence result.

    Rules:
    - p-value > 0.05: increase confidence by 0.1 (evidence supports correspondence)
    - p-value < 0.01: decrease confidence by 0.1 (evidence contradicts correspondence)
    - p-value between 0.01 and 0.05: no change (inconclusive)

    Confidence is clamped to [0.0, 1.0].

    Args:
        mapping: The AnalogyMapping to update.
        test_result: Dict from test_correspondence (must contain "pvalue").

    Returns:
        New AnalogyMapping with updated confidence.
    """
    pvalue = test_result["pvalue"]
    new_confidence = mapping.confidence

    if pvalue > 0.05:
        new_confidence = min(1.0, new_confidence + 0.1)
    elif pvalue < 0.01:
        new_confidence = max(0.0, new_confidence - 0.1)

    # Create updated mapping (dataclass is not frozen, but return new for safety)
    return AnalogyMapping(
        source_domain=mapping.source_domain,
        target_domain=mapping.target_domain,
        correspondences=mapping.correspondences,
        unknowns=mapping.unknowns,
        evidence=mapping.evidence,
        confidence=new_confidence,
    )
