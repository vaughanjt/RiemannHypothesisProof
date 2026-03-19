"""Embedding configuration registry for N-dimensional zero representations.

Provides:
- EmbeddingConfig: frozen dataclass defining an embedding coordinate system
- FEATURE_EXTRACTORS: registry mapping feature names to extractor functions
- PRESET_CONFIGS: named preset embedding configurations
- save/load helpers for persisting configs through the workbench experiment system

Feature extractors are stubs in this module (raising NotImplementedError).
Plan 02-03 implements the actual extraction logic. The registry, dataclass,
and preset configurations are fully functional here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from riemann.types import ZetaZero
from riemann.workbench.experiment import load_experiment, save_experiment


@dataclass(frozen=True)
class EmbeddingConfig:
    """Frozen configuration for an N-dimensional zero embedding.

    Defines which features to extract, how to scale them, and which
    zero range to operate on. Immutable so configs can be safely shared
    and stored as experiment parameters.

    Attributes:
        name: Human-readable name for this embedding.
        description: What this embedding captures.
        feature_names: Ordered tuple of feature extractor keys from FEATURE_EXTRACTORS.
        scaling: Scaling method: "standard" (z-score), "robust" (median/IQR), or "none".
        zero_range: (start, end) zero index range (1-based, inclusive).
        dps: Decimal digits of precision for feature extraction.
    """
    name: str
    description: str
    feature_names: tuple[str, ...]
    scaling: str = "standard"
    zero_range: tuple[int, int] = (1, 1000)
    dps: int = 50


# ---------------------------------------------------------------------------
# Feature extractor stubs
# ---------------------------------------------------------------------------
# Each stub documents what the feature computes. Plan 02-03 replaces these
# with real implementations. The registry is populated at module load time.

def _stub_imag_part(zero: ZetaZero) -> float:
    """Extract the imaginary part of the zero (height on critical line)."""
    raise NotImplementedError("imag_part extractor: implemented in Plan 02-03")


def _stub_spacing_left(zero: ZetaZero) -> float:
    """Gap to the previous zero (requires context of neighboring zeros)."""
    raise NotImplementedError("spacing_left extractor: implemented in Plan 02-03")


def _stub_spacing_right(zero: ZetaZero) -> float:
    """Gap to the next zero (requires context of neighboring zeros)."""
    raise NotImplementedError("spacing_right extractor: implemented in Plan 02-03")


def _stub_zeta_derivative_magnitude(zero: ZetaZero) -> float:
    """Magnitude |zeta'(rho)| at the zero -- measures how sharply zeta crosses zero."""
    raise NotImplementedError("zeta_derivative_magnitude extractor: implemented in Plan 02-03")


def _stub_local_density_deviation(zero: ZetaZero) -> float:
    """Deviation of local zero density from Riemann-von Mangoldt prediction."""
    raise NotImplementedError("local_density_deviation extractor: implemented in Plan 02-03")


def _stub_pair_correlation_local(zero: ZetaZero) -> float:
    """Local pair correlation statistic in a window around this zero."""
    raise NotImplementedError("pair_correlation_local extractor: implemented in Plan 02-03")


def _stub_hardy_z_sign_changes(zero: ZetaZero) -> float:
    """Number of sign changes of Hardy Z-function near this zero."""
    raise NotImplementedError("hardy_z_sign_changes extractor: implemented in Plan 02-03")


def _stub_local_entropy(zero: ZetaZero) -> float:
    """Shannon entropy of the local spacing sequence around this zero."""
    raise NotImplementedError("local_entropy extractor: implemented in Plan 02-03")


def _stub_compression_distance(zero: ZetaZero) -> float:
    """Normalized compression distance of the local spacing pattern."""
    raise NotImplementedError("compression_distance extractor: implemented in Plan 02-03")


FEATURE_EXTRACTORS: dict[str, Callable] = {
    "imag_part": _stub_imag_part,
    "spacing_left": _stub_spacing_left,
    "spacing_right": _stub_spacing_right,
    "zeta_derivative_magnitude": _stub_zeta_derivative_magnitude,
    "local_density_deviation": _stub_local_density_deviation,
    "pair_correlation_local": _stub_pair_correlation_local,
    "hardy_z_sign_changes": _stub_hardy_z_sign_changes,
    "local_entropy": _stub_local_entropy,
    "compression_distance": _stub_compression_distance,
}
"""Registry mapping feature names to extractor functions.

All extractors accept a ZetaZero and return a float. Currently stubs
(NotImplementedError) -- Plan 02-03 provides real implementations.
"""


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

PRESET_CONFIGS: dict[str, EmbeddingConfig] = {
    "spectral_basic": EmbeddingConfig(
        name="spectral_basic",
        description="Basic spectral embedding: imaginary part + left/right spacings (3D, fast)",
        feature_names=("imag_part", "spacing_left", "spacing_right"),
        scaling="standard",
    ),
    "spectral_full": EmbeddingConfig(
        name="spectral_full",
        description="Full spectral embedding: all 5 spectral features (5D)",
        feature_names=(
            "imag_part",
            "spacing_left",
            "spacing_right",
            "zeta_derivative_magnitude",
            "local_density_deviation",
        ),
        scaling="standard",
    ),
    "information_space": EmbeddingConfig(
        name="information_space",
        description="Information-theoretic embedding: entropy + compression + spacings (4D)",
        feature_names=(
            "local_entropy",
            "compression_distance",
            "spacing_left",
            "spacing_right",
        ),
        scaling="robust",
    ),
    "kitchen_sink": EmbeddingConfig(
        name="kitchen_sink",
        description="All 9 features -- 'surprise me' mode for maximum view diversity (9D)",
        feature_names=tuple(FEATURE_EXTRACTORS.keys()),
        scaling="standard",
    ),
}
"""Named preset embedding configurations for common analysis scenarios."""


# ---------------------------------------------------------------------------
# Config persistence through workbench
# ---------------------------------------------------------------------------

def _config_to_dict(config: EmbeddingConfig) -> dict:
    """Serialize an EmbeddingConfig to a JSON-compatible dict."""
    return {
        "name": config.name,
        "description": config.description,
        "feature_names": list(config.feature_names),
        "scaling": config.scaling,
        "zero_range": list(config.zero_range),
        "dps": config.dps,
    }


def _dict_to_config(d: dict) -> EmbeddingConfig:
    """Deserialize a dict back to an EmbeddingConfig."""
    return EmbeddingConfig(
        name=d["name"],
        description=d["description"],
        feature_names=tuple(d["feature_names"]),
        scaling=d.get("scaling", "standard"),
        zero_range=tuple(d.get("zero_range", (1, 1000))),
        dps=d.get("dps", 50),
    )


def save_config_as_experiment(
    config: EmbeddingConfig,
    db_path: str | Path | None = None,
) -> str:
    """Save an EmbeddingConfig as a workbench experiment for reproducibility.

    Args:
        config: The embedding configuration to save.
        db_path: Database path (defaults to project DB).

    Returns:
        UUID string of the created experiment.
    """
    params = _config_to_dict(config)
    return save_experiment(
        description=config.description,
        parameters=params,
        result_summary=f"Embedding config: {config.name} ({len(config.feature_names)} features)",
        db_path=db_path,
    )


def load_config_from_experiment(
    experiment_id: str,
    db_path: str | Path | None = None,
) -> EmbeddingConfig:
    """Load an EmbeddingConfig from a stored workbench experiment.

    Args:
        experiment_id: UUID of the experiment to load.
        db_path: Database path (defaults to project DB).

    Returns:
        Reconstructed EmbeddingConfig.

    Raises:
        ValueError: If experiment not found.
    """
    experiment = load_experiment(experiment_id, db_path=db_path)
    if experiment is None:
        raise ValueError(f"Experiment {experiment_id} not found")
    return _dict_to_config(experiment["parameters"])


def list_preset_names() -> list[str]:
    """Return sorted list of available preset configuration names."""
    return sorted(PRESET_CONFIGS.keys())


def get_preset(name: str) -> EmbeddingConfig:
    """Get a preset configuration by name.

    Args:
        name: Key in PRESET_CONFIGS.

    Returns:
        The EmbeddingConfig preset.

    Raises:
        KeyError: If name is not a recognized preset.
    """
    if name not in PRESET_CONFIGS:
        raise KeyError(
            f"Unknown preset '{name}'. Available: {list_preset_names()}"
        )
    return PRESET_CONFIGS[name]
