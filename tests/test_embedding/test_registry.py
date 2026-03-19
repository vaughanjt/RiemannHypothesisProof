"""Tests for embedding configuration registry.

Covers:
- EmbeddingConfig is frozen (immutable)
- FEATURE_EXTRACTORS has all 9 expected keys
- Each stub raises NotImplementedError
- PRESET_CONFIGS has expected preset names
- save/load round-trip through workbench experiment system
- get_preset returns correct config / raises KeyError for unknown
- list_preset_names returns sorted list
"""
import pytest
from mpmath import mpc

from riemann.embedding.registry import (
    FEATURE_EXTRACTORS,
    PRESET_CONFIGS,
    EmbeddingConfig,
    get_preset,
    list_preset_names,
    load_config_from_experiment,
    save_config_as_experiment,
)
from riemann.types import ZetaZero


EXPECTED_FEATURE_KEYS = {
    "imag_part",
    "spacing_left",
    "spacing_right",
    "zeta_derivative_magnitude",
    "local_density_deviation",
    "pair_correlation_local",
    "hardy_z_sign_changes",
    "local_entropy",
    "compression_distance",
}

EXPECTED_PRESET_NAMES = {
    "spectral_basic",
    "spectral_full",
    "information_space",
    "kitchen_sink",
}


class TestEmbeddingConfig:
    def test_is_frozen(self):
        """EmbeddingConfig should be immutable (frozen dataclass)."""
        config = EmbeddingConfig(
            name="test",
            description="test config",
            feature_names=("imag_part",),
        )
        with pytest.raises(AttributeError):
            config.name = "modified"

    def test_defaults(self):
        """Default values for scaling, zero_range, dps."""
        config = EmbeddingConfig(
            name="test",
            description="test",
            feature_names=("imag_part",),
        )
        assert config.scaling == "standard"
        assert config.zero_range == (1, 1000)
        assert config.dps == 50


class TestFeatureExtractors:
    def test_has_all_expected_keys(self):
        assert set(FEATURE_EXTRACTORS.keys()) == EXPECTED_FEATURE_KEYS

    def test_has_9_extractors(self):
        assert len(FEATURE_EXTRACTORS) == 9

    def test_each_stub_raises_not_implemented(self):
        """All feature extractor stubs should raise NotImplementedError."""
        dummy_zero = ZetaZero(
            index=1,
            value=mpc(0.5, 14.13),
            precision_digits=15,
            validated=False,
        )
        for name, extractor in FEATURE_EXTRACTORS.items():
            with pytest.raises(NotImplementedError, match=name):
                extractor(dummy_zero)


class TestPresetConfigs:
    def test_has_expected_presets(self):
        assert set(PRESET_CONFIGS.keys()) == EXPECTED_PRESET_NAMES

    def test_has_4_presets(self):
        assert len(PRESET_CONFIGS) == 4

    def test_spectral_basic_has_3_features(self):
        config = PRESET_CONFIGS["spectral_basic"]
        assert len(config.feature_names) == 3
        assert "imag_part" in config.feature_names

    def test_kitchen_sink_has_all_9_features(self):
        config = PRESET_CONFIGS["kitchen_sink"]
        assert len(config.feature_names) == 9
        assert set(config.feature_names) == EXPECTED_FEATURE_KEYS


class TestGetPreset:
    def test_returns_correct_config(self):
        config = get_preset("spectral_basic")
        assert config.name == "spectral_basic"
        assert isinstance(config, EmbeddingConfig)

    def test_raises_key_error_for_unknown(self):
        with pytest.raises(KeyError, match="nonexistent"):
            get_preset("nonexistent")


class TestListPresetNames:
    def test_returns_sorted_list(self):
        names = list_preset_names()
        assert names == sorted(names)
        assert set(names) == EXPECTED_PRESET_NAMES


class TestConfigRoundTrip:
    def test_save_and_load(self, temp_db):
        """Save a config as experiment, load it back, verify equality."""
        original = EmbeddingConfig(
            name="test_roundtrip",
            description="Round-trip test config",
            feature_names=("imag_part", "spacing_left", "spacing_right"),
            scaling="robust",
            zero_range=(1, 500),
            dps=30,
        )

        experiment_id = save_config_as_experiment(original, db_path=temp_db)
        assert isinstance(experiment_id, str)
        assert len(experiment_id) > 0

        loaded = load_config_from_experiment(experiment_id, db_path=temp_db)
        assert loaded.name == original.name
        assert loaded.description == original.description
        assert loaded.feature_names == original.feature_names
        assert loaded.scaling == original.scaling
        assert loaded.zero_range == original.zero_range
        assert loaded.dps == original.dps

    def test_load_nonexistent_raises(self, temp_db):
        """Loading a non-existent experiment should raise ValueError."""
        from riemann.workbench.db import init_db
        init_db(temp_db)
        with pytest.raises(ValueError, match="not found"):
            load_config_from_experiment("nonexistent-id", db_path=temp_db)
