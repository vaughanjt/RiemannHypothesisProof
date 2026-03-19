"""Tests for HDF5 embedding storage (save, load, list)."""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config():
    """Create a test EmbeddingConfig."""
    from riemann.embedding.registry import EmbeddingConfig
    return EmbeddingConfig(
        name="test_config",
        description="Test embedding",
        feature_names=("imag_part", "spacing_left", "spacing_right"),
        scaling="standard",
        zero_range=(1, 50),
        dps=50,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSaveEmbedding:
    def test_writes_hdf5_file(self, tmp_path):
        from riemann.embedding.storage import save_embedding
        config = _make_config()
        emb = np.random.default_rng(42).standard_normal((50, 3))
        path = save_embedding(config, emb, base_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".h5"

    def test_hdf5_attributes_contain_config(self, tmp_path):
        import h5py
        from riemann.embedding.storage import save_embedding
        config = _make_config()
        emb = np.random.default_rng(42).standard_normal((50, 3))
        path = save_embedding(config, emb, base_dir=tmp_path)
        with h5py.File(str(path), "r") as f:
            assert f.attrs["config_name"] == "test_config"
            assert list(f.attrs["feature_names"]) == ["imag_part", "spacing_left", "spacing_right"]
            assert f.attrs["scaling"] == "standard"


class TestLoadEmbedding:
    def test_round_trip_embedding_shape(self, tmp_path):
        from riemann.embedding.storage import save_embedding, load_embedding
        config = _make_config()
        emb = np.random.default_rng(42).standard_normal((50, 3))
        save_embedding(config, emb, base_dir=tmp_path)
        data = load_embedding("test_config", base_dir=tmp_path)
        assert "embedding" in data
        assert isinstance(data["embedding"], np.ndarray)
        np.testing.assert_array_almost_equal(data["embedding"], emb)


class TestSaveWithProjections:
    def test_stores_projections_in_groups(self, tmp_path):
        import h5py
        from riemann.embedding.storage import save_embedding
        from riemann.viz.projection import ProjectionResult
        config = _make_config()
        emb = np.random.default_rng(42).standard_normal((50, 3))
        proj = ProjectionResult(
            coordinates=np.random.default_rng(42).standard_normal((50, 2)),
            method="pca",
            source_dim=3,
            target_dim=2,
            metadata={"variance_explained": [0.5, 0.3]},
        )
        path = save_embedding(config, emb, projections={"pca": proj}, base_dir=tmp_path)
        with h5py.File(str(path), "r") as f:
            assert "projections" in f
            assert "pca" in f["projections"]

    def test_load_projections_correct_shape(self, tmp_path):
        from riemann.embedding.storage import save_embedding, load_embedding
        from riemann.viz.projection import ProjectionResult
        config = _make_config()
        emb = np.random.default_rng(42).standard_normal((50, 3))
        proj_coords = np.random.default_rng(42).standard_normal((50, 2))
        proj = ProjectionResult(
            coordinates=proj_coords,
            method="pca",
            source_dim=3,
            target_dim=2,
            metadata={"variance_explained": [0.5, 0.3]},
        )
        save_embedding(config, emb, projections={"pca": proj}, base_dir=tmp_path)
        data = load_embedding("test_config", base_dir=tmp_path)
        assert "projections" in data
        assert "pca" in data["projections"]
        loaded_proj = data["projections"]["pca"]
        assert isinstance(loaded_proj, ProjectionResult)
        np.testing.assert_array_almost_equal(loaded_proj.coordinates, proj_coords)


class TestListEmbeddings:
    def test_lists_saved_embeddings(self, tmp_path):
        from riemann.embedding.storage import save_embedding, list_embeddings
        from riemann.embedding.registry import EmbeddingConfig
        config_a = EmbeddingConfig(
            name="alpha",
            description="A",
            feature_names=("imag_part",),
        )
        config_b = EmbeddingConfig(
            name="beta",
            description="B",
            feature_names=("imag_part",),
        )
        emb = np.random.default_rng(42).standard_normal((10, 1))
        save_embedding(config_a, emb, base_dir=tmp_path)
        save_embedding(config_b, emb, base_dir=tmp_path)
        names = list_embeddings(base_dir=tmp_path)
        assert names == ["alpha", "beta"]
