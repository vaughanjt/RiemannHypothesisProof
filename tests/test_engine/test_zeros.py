"""Test scaffolds for COMP-02: Zero computation and cataloging.

Tests are marked xfail pending implementation in Plan 01-02.
"""
import mpmath
import pytest

from riemann.types import ZetaZero


@pytest.mark.xfail(reason="Implementation pending in Plan 01-02")
def test_odlyzko_validation(odlyzko_zeros, default_precision):
    """First 10 computed zeros match Odlyzko table to 45 digits."""
    from riemann.engine.zeros import compute_zero

    for n in range(1, 11):
        zero = compute_zero(n, dps=50)
        odlyzko_t = odlyzko_zeros[n - 1]

        # Compare imaginary part (Odlyzko table gives t values)
        computed_t = zero.value.imag
        diff = abs(computed_t - odlyzko_t)
        assert diff < mpmath.power(10, -45), (
            f"Zero {n}: computed={mpmath.nstr(computed_t, 20)}, "
            f"odlyzko={mpmath.nstr(odlyzko_t, 20)}, diff={diff}"
        )


@pytest.mark.xfail(reason="Implementation pending in Plan 01-02")
def test_zero_on_critical_line(default_precision):
    """Each computed zero has Re(z) within threshold of 0.5."""
    from riemann.engine.zeros import compute_zero

    for n in range(1, 6):
        zero = compute_zero(n, dps=50)
        assert abs(zero.value.real - mpmath.mpf("0.5")) < mpmath.power(10, -45), (
            f"Zero {n} not on critical line: Re = {zero.value.real}"
        )


@pytest.mark.xfail(reason="Implementation pending in Plan 01-02")
def test_zero_catalog(temp_db):
    """Store and retrieve zeros from SQLite."""
    from riemann.engine.zeros import ZeroCatalog

    catalog = ZeroCatalog(temp_db)
    zero = ZetaZero(
        index=1,
        value=mpmath.mpc(0.5, mpmath.mpf("14.134725141734693790")),
        precision_digits=50,
        validated=True,
        on_critical_line=True,
    )

    catalog.store(zero)
    retrieved = catalog.get(1)
    assert retrieved is not None
    assert retrieved.index == 1
    assert abs(retrieved.value.imag - zero.value.imag) < mpmath.power(10, -40)


@pytest.mark.xfail(reason="Implementation pending in Plan 01-02")
def test_zero_count_riemann_von_mangoldt(default_precision):
    """N(T) matches mpmath.nzeros(T) for several T values."""
    from riemann.engine.zeros import zero_count

    test_T = [50, 100, 200]
    for T in test_T:
        count = zero_count(T)
        expected = mpmath.nzeros(T)
        assert count == expected, f"N({T}): got {count}, expected {expected}"
