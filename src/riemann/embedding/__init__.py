"""Embedding configuration and feature extraction for N-dimensional zero representations."""

from riemann.embedding.registry import (
    FEATURE_EXTRACTORS,
    PRESET_CONFIGS,
    EmbeddingConfig,
    get_preset,
    list_preset_names,
    save_config_as_experiment,
    load_config_from_experiment,
)

# Import coordinates AFTER registry to trigger extractor registration.
# This replaces stubs in FEATURE_EXTRACTORS with real implementations.
from riemann.embedding.coordinates import compute_embedding  # noqa: E402
from riemann.embedding.storage import (  # noqa: E402
    list_embeddings,
    load_embedding,
    save_embedding,
)

__all__ = [
    "EmbeddingConfig",
    "FEATURE_EXTRACTORS",
    "PRESET_CONFIGS",
    "compute_embedding",
    "get_preset",
    "list_embeddings",
    "list_preset_names",
    "load_config_from_experiment",
    "load_embedding",
    "save_config_as_experiment",
    "save_embedding",
]
