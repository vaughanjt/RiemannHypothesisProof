"""Tests for RSRCH-02: Experiment reproducibility.

Covers: experiment save/load, parameter serialization, SHA-256 checksums,
numpy result data storage, tamper detection, reproducibility verification.
"""
import numpy as np
import pytest


class TestSaveExperiment:
    """Experiment creation tests."""

    def test_save_experiment_returns_uuid(self, temp_db):
        """save_experiment stores record with id, description, parameters, seed, checksum."""
        from riemann.workbench.experiment import save_experiment

        eid = save_experiment(
            description="Compute first 100 zeros",
            parameters={"n_zeros": 100, "dps": 50, "algorithm": "zetazero"},
            seed=42,
            db_path=temp_db,
        )
        assert eid is not None
        assert isinstance(eid, str)
        assert len(eid) == 36  # UUID format

    def test_save_experiment_records_metadata(self, temp_db):
        """save_experiment records computation_time_ms and precision_digits."""
        from riemann.workbench.experiment import save_experiment, load_experiment

        eid = save_experiment(
            description="High precision test",
            parameters={"dps": 100},
            computation_time_ms=1234.5,
            precision_digits=100,
            db_path=temp_db,
        )

        exp = load_experiment(eid, db_path=temp_db)
        assert exp is not None
        assert exp["computation_time_ms"] == 1234.5
        assert exp["precision_digits"] == 100


class TestLoadExperiment:
    """Experiment retrieval tests."""

    def test_load_experiment_returns_identical_parameters(self, temp_db):
        """load_experiment(id) returns dict with identical parameters to what was saved."""
        from riemann.workbench.experiment import save_experiment, load_experiment

        params = {"n_zeros": 100, "dps": 50, "algorithm": "zetazero"}
        eid = save_experiment(
            description="Compute first 100 zeros",
            parameters=params,
            seed=42,
            db_path=temp_db,
        )

        exp = load_experiment(eid, db_path=temp_db)
        assert exp is not None
        assert exp["description"] == "Compute first 100 zeros"
        assert exp["parameters"]["n_zeros"] == 100
        assert exp["parameters"]["dps"] == 50
        assert exp["parameters"]["algorithm"] == "zetazero"
        assert exp["seed"] == 42

    def test_save_experiment_with_numpy_stores_data(self, temp_db, tmp_path):
        """save_experiment with numpy results stores data file and records path."""
        from riemann.workbench.experiment import save_experiment, load_experiment
        from unittest.mock import patch

        # Patch COMPUTED_DIR to use a temp path for this test
        with patch("riemann.workbench.experiment.COMPUTED_DIR", tmp_path):
            result_data = np.array([14.134725, 21.022040, 25.010858])
            eid = save_experiment(
                description="Zero computation",
                parameters={"n_zeros": 3},
                result_data=result_data,
                db_path=temp_db,
            )

            exp = load_experiment(eid, db_path=temp_db)
            assert exp is not None
            assert len(exp["data_files"]) > 0
            assert exp["result_data"] is not None
            np.testing.assert_array_almost_equal(
                exp["result_data"], result_data
            )


class TestVerifyChecksum:
    """Checksum verification tests."""

    def test_verify_checksum_unmodified(self, temp_db):
        """verify_checksum returns True for unmodified experiment."""
        from riemann.workbench.experiment import save_experiment, verify_checksum

        eid = save_experiment(
            description="Checksum test",
            parameters={"x": 1, "y": 2},
            result_summary="Test result",
            db_path=temp_db,
        )

        assert verify_checksum(eid, db_path=temp_db) is True

    def test_verify_checksum_detects_tampering(self, temp_db):
        """verify_checksum returns False when result data is modified (tamper detection)."""
        import sqlite3
        from riemann.workbench.experiment import save_experiment, verify_checksum

        eid = save_experiment(
            description="Tamper test",
            parameters={"x": 1},
            result_summary="Original result",
            db_path=temp_db,
        )

        # Tamper with result_summary directly in the database
        conn = sqlite3.connect(temp_db)
        try:
            conn.execute(
                "UPDATE experiments SET result_summary = ? WHERE id = ?",
                ("Tampered result", eid),
            )
            conn.commit()
        finally:
            conn.close()

        assert verify_checksum(eid, db_path=temp_db) is False


class TestReproducibility:
    """Experiment reproducibility tests."""

    def test_same_params_same_seed_same_checksum(self, temp_db):
        """Two experiments with same parameters and seed produce same checksum."""
        from riemann.workbench.experiment import save_experiment, load_experiment

        params = {"n": 10, "dps": 50}
        eid1 = save_experiment(
            description="Run 1",
            parameters=params,
            seed=42,
            result_summary="deterministic result",
            db_path=temp_db,
        )
        eid2 = save_experiment(
            description="Run 2",
            parameters=params,
            seed=42,
            result_summary="deterministic result",
            db_path=temp_db,
        )

        exp1 = load_experiment(eid1, db_path=temp_db)
        exp2 = load_experiment(eid2, db_path=temp_db)
        assert exp1["checksum"] == exp2["checksum"]


class TestListExperiments:
    """Experiment listing tests."""

    def test_list_experiments_returns_all(self, temp_db):
        """list_experiments returns all experiments, ordered by creation date."""
        from riemann.workbench.experiment import save_experiment, list_experiments

        save_experiment(
            description="Experiment 1",
            parameters={"n": 1},
            db_path=temp_db,
        )
        save_experiment(
            description="Experiment 2",
            parameters={"n": 2},
            db_path=temp_db,
        )

        results = list_experiments(db_path=temp_db)
        assert len(results) == 2
        # Most recent first
        assert results[0]["description"] == "Experiment 2"
        assert results[1]["description"] == "Experiment 1"
