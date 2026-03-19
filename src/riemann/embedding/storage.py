"""HDF5 read/write for embedding arrays and projections.

Single-writer pattern: always close file before reading.
Uses context managers throughout to ensure file handles are released.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

from riemann.config import DATA_DIR

if TYPE_CHECKING:
    from riemann.embedding.registry import EmbeddingConfig
    from riemann.viz.projection import ProjectionResult


def save_embedding(
    config: EmbeddingConfig,
    embedding: np.ndarray,
    projections: dict[str, ProjectionResult] | None = None,
    base_dir: Path | None = None,
) -> Path:
    """Save an embedding array and optional projections to HDF5.

    Args:
        config: EmbeddingConfig describing the embedding.
        embedding: ndarray of shape (n_points, n_features).
        projections: Optional dict mapping method name to ProjectionResult.
        base_dir: Directory for HDF5 files. Default: DATA_DIR/embeddings.

    Returns:
        Path to the created HDF5 file.
    """
    if base_dir is None:
        base_dir = DATA_DIR / "embeddings"
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    path = base_dir / f"{config.name}.h5"

    with h5py.File(str(path), "w") as f:
        # Main embedding dataset
        f.create_dataset(
            "embedding",
            data=embedding,
            chunks=True,
            compression="gzip",
            compression_opts=4,
        )

        # Config metadata as file attributes
        f.attrs["config_name"] = config.name
        f.attrs["feature_names"] = list(config.feature_names)
        f.attrs["scaling"] = config.scaling
        f.attrs["zero_range"] = list(config.zero_range)
        f.attrs["dps"] = config.dps
        f.attrs["description"] = config.description

        # Projections
        if projections:
            proj_group = f.create_group("projections")
            for method_name, result in projections.items():
                sub = proj_group.create_group(method_name)
                sub.create_dataset("coordinates", data=result.coordinates)
                sub.attrs["method"] = result.method
                sub.attrs["source_dim"] = result.source_dim
                sub.attrs["target_dim"] = result.target_dim

                # Serialize metadata (convert non-serializable to str)
                for key, val in result.metadata.items():
                    try:
                        if isinstance(val, (list, np.ndarray)):
                            sub.attrs[key] = np.array(val) if isinstance(val, list) else val
                        elif isinstance(val, (int, float, str, bool)):
                            sub.attrs[key] = val
                        else:
                            sub.attrs[key] = str(val)
                    except TypeError:
                        sub.attrs[key] = str(val)

    return path


def load_embedding(
    name: str,
    base_dir: Path | None = None,
) -> dict:
    """Load an embedding and optional projections from HDF5.

    Args:
        name: Embedding config name (stem of HDF5 file).
        base_dir: Directory containing HDF5 files. Default: DATA_DIR/embeddings.

    Returns:
        Dict with keys:
        - "embedding": ndarray
        - "config": dict of config attributes
        - "projections": dict mapping method name to ProjectionResult

    Raises:
        FileNotFoundError: If HDF5 file doesn't exist.
    """
    from riemann.viz.projection import ProjectionResult

    if base_dir is None:
        base_dir = DATA_DIR / "embeddings"
    base_dir = Path(base_dir)

    path = base_dir / f"{name}.h5"
    if not path.exists():
        raise FileNotFoundError(f"No embedding file: {path}")

    with h5py.File(str(path), "r") as f:
        # Read embedding into memory (copy before closing file)
        embedding = np.array(f["embedding"])

        # Read config attributes
        config = {
            "config_name": str(f.attrs["config_name"]),
            "feature_names": list(f.attrs["feature_names"]),
            "scaling": str(f.attrs["scaling"]),
            "zero_range": list(f.attrs["zero_range"]),
            "dps": int(f.attrs["dps"]),
        }
        if "description" in f.attrs:
            config["description"] = str(f.attrs["description"])

        # Read projections
        projections = {}
        if "projections" in f:
            for method_name in f["projections"]:
                sub = f["projections"][method_name]
                coords = np.array(sub["coordinates"])

                # Reconstruct metadata from attrs
                metadata = {}
                skip_keys = {"method", "source_dim", "target_dim"}
                for key in sub.attrs:
                    if key not in skip_keys:
                        val = sub.attrs[key]
                        if isinstance(val, np.ndarray):
                            metadata[key] = val.tolist()
                        elif isinstance(val, (np.integer, np.floating)):
                            metadata[key] = val.item()
                        else:
                            metadata[key] = val

                projections[method_name] = ProjectionResult(
                    coordinates=coords,
                    method=str(sub.attrs["method"]),
                    source_dim=int(sub.attrs["source_dim"]),
                    target_dim=int(sub.attrs["target_dim"]),
                    metadata=metadata,
                )

    return {
        "embedding": embedding,
        "config": config,
        "projections": projections,
    }


def list_embeddings(base_dir: Path | None = None) -> list[str]:
    """List available embedding names from HDF5 files on disk.

    Args:
        base_dir: Directory containing HDF5 files. Default: DATA_DIR/embeddings.

    Returns:
        Sorted list of embedding names (file stems).
    """
    if base_dir is None:
        base_dir = DATA_DIR / "embeddings"
    base_dir = Path(base_dir)

    if not base_dir.exists():
        return []

    return sorted(p.stem for p in base_dir.glob("*.h5"))
