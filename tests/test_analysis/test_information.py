"""Tests for information-theoretic analysis of spacing sequences."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def poisson_spacings(rng):
    return rng.exponential(1.0, 1000)


@pytest.fixture
def concentrated_spacings(rng):
    return np.ones(1000) + rng.normal(0, 0.01, 1000)


@pytest.fixture
def correlated_spacings(rng):
    noise = rng.normal(0, 0.1, 1000)
    return np.abs(np.cumsum(noise) - np.cumsum(noise).min() + 0.1)


def test_spacing_entropy_binned_positive(poisson_spacings):
    from riemann.analysis.information import spacing_entropy
    h = spacing_entropy(poisson_spacings, method="binned")
    assert isinstance(h, float)
    assert h > 0


def test_spacing_entropy_kde_positive(poisson_spacings):
    from riemann.analysis.information import spacing_entropy
    h = spacing_entropy(poisson_spacings, method="kde")
    assert isinstance(h, float)
    assert h > 0


def test_spacing_entropy_uniform_higher_than_concentrated(poisson_spacings, concentrated_spacings):
    from riemann.analysis.information import spacing_entropy
    h_uniform = spacing_entropy(poisson_spacings, method="binned")
    h_concentrated = spacing_entropy(concentrated_spacings, method="binned")
    assert h_uniform > h_concentrated


def test_spacing_entropy_empty_raises():
    from riemann.analysis.information import spacing_entropy
    with pytest.raises(ValueError):
        spacing_entropy(np.array([]))


def test_spacing_entropy_unknown_method_raises(poisson_spacings):
    from riemann.analysis.information import spacing_entropy
    with pytest.raises(ValueError):
        spacing_entropy(poisson_spacings, method="unknown")


def test_mutual_information_iid_near_zero(rng):
    from riemann.analysis.information import mutual_information_spacings
    iid = rng.exponential(1.0, 500)
    mi = mutual_information_spacings(iid, lag=1)
    assert isinstance(mi, float)
    assert mi < 0.15


def test_mutual_information_correlated_positive(correlated_spacings):
    from riemann.analysis.information import mutual_information_spacings
    mi = mutual_information_spacings(correlated_spacings[:500], lag=1)
    assert mi > 0.05


def test_lz_complexity_constant_small():
    from riemann.analysis.information import lempel_ziv_complexity
    constant = np.zeros(1000)
    c = lempel_ziv_complexity(constant)
    assert isinstance(c, int)
    assert c <= 3


def test_lz_complexity_random_larger_than_periodic(rng):
    from riemann.analysis.information import lempel_ziv_complexity
    random_seq = rng.random(500)
    periodic = np.tile([0.1, 0.9], 250)
    c_random = lempel_ziv_complexity(random_seq)
    c_periodic = lempel_ziv_complexity(periodic)
    assert c_random > c_periodic


def test_cross_object_comparison_keys(rng):
    from riemann.analysis.information import cross_object_comparison
    zero_s = rng.exponential(1.0, 200)
    gue_s = rng.exponential(1.0, 200)
    result = cross_object_comparison(zero_s, gue_s)
    assert "zeta_zeros" in result
    assert "gue_eigenvalues" in result
    assert "poisson" in result


def test_cross_object_comparison_metric_keys(rng):
    from riemann.analysis.information import cross_object_comparison
    zero_s = rng.exponential(1.0, 200)
    gue_s = rng.exponential(1.0, 200)
    result = cross_object_comparison(zero_s, gue_s)
    expected_keys = {"entropy_binned", "entropy_kde", "mi_lag1", "mi_lag2", "lz_complexity"}
    for obj in result.values():
        assert expected_keys.issubset(obj.keys())


def test_cross_object_comparison_with_primes(rng):
    from riemann.analysis.information import cross_object_comparison
    zero_s = rng.exponential(1.0, 200)
    gue_s = rng.exponential(1.0, 200)
    primes = np.array([1, 2, 2, 4, 2, 4, 2, 4, 6, 2, 6, 4, 2, 4] * 15, dtype=float)
    result = cross_object_comparison(zero_s, gue_s, prime_gaps=primes)
    assert "primes" in result


def test_poisson_higher_entropy_than_gue_like():
    from riemann.analysis.information import spacing_entropy
    rng = np.random.default_rng(123)
    poisson = rng.exponential(1.0, 2000)
    # GUE-like: Wigner surmise sampling via inverse CDF (more concentrated around mode ~0.68)
    u = rng.random(2000)
    gue_like = np.sqrt(-np.log(1 - u) * 4 / np.pi)
    # KDE method captures continuous entropy difference better than binned
    h_poisson = spacing_entropy(poisson, method="kde", bins=100)
    h_gue = spacing_entropy(gue_like, method="kde", bins=100)
    assert h_poisson > h_gue
