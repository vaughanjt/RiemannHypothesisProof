"""Test scaffolds for RSRCH-02: Experiment reproducibility.

Tests are marked xfail pending implementation in Plan 01-05.
"""
import pytest


@pytest.mark.xfail(reason="Implementation pending in Plan 01-05")
def test_save_experiment(temp_db):
    """Save experiment with parameters, retrieve with same params."""
    from riemann.workbench.experiment import ExperimentManager

    mgr = ExperimentManager(temp_db)
    eid = mgr.save(
        description="Compute first 100 zeros",
        parameters={"n_zeros": 100, "dps": 50, "algorithm": "zetazero"},
        seed=42,
    )

    exp = mgr.get(eid)
    assert exp is not None
    assert exp["description"] == "Compute first 100 zeros"
    assert exp["parameters"]["n_zeros"] == 100
    assert exp["seed"] == 42


@pytest.mark.xfail(reason="Implementation pending in Plan 01-05")
def test_experiment_checksum(temp_db):
    """Checksum detects result tampering."""
    from riemann.workbench.experiment import ExperimentManager

    mgr = ExperimentManager(temp_db)
    eid = mgr.save(
        description="Test checksum",
        parameters={"x": 1},
        result_data=b"original result bytes",
    )

    assert mgr.verify_checksum(eid) is True

    # Tamper with the stored data
    mgr.tamper_data(eid, b"modified result bytes")
    assert mgr.verify_checksum(eid) is False


@pytest.mark.xfail(reason="Implementation pending in Plan 01-05")
def test_experiment_reproducibility(temp_db):
    """Same params + seed produce same checksum."""
    from riemann.workbench.experiment import ExperimentManager

    mgr = ExperimentManager(temp_db)

    eid1 = mgr.save(
        description="Run 1",
        parameters={"n": 10, "dps": 50},
        seed=42,
        result_data=b"deterministic result",
    )

    eid2 = mgr.save(
        description="Run 2",
        parameters={"n": 10, "dps": 50},
        seed=42,
        result_data=b"deterministic result",
    )

    exp1 = mgr.get(eid1)
    exp2 = mgr.get(eid2)
    assert exp1["checksum"] == exp2["checksum"]
