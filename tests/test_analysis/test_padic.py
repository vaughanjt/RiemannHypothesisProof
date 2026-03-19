"""Tests for p-adic arithmetic and Kubota-Leopoldt zeta module.

Tests cover:
- PadicNumber construction and representation
- p-adic addition with carry propagation
- p-adic multiplication with carry
- p-adic negation and subtraction
- p-adic norm (|x|_p = p^{-v_p(x)})
- Rational-to-padic conversion
- Kubota-Leopoldt p-adic zeta function
- Fractal tree data generation
"""
import pytest

from riemann.analysis.padic import (
    PadicNumber,
    kubota_leopoldt_zeta,
    padic_fractal_tree_data,
    padic_from_rational,
)


# ---------------------------------------------------------------------------
# PadicNumber construction
# ---------------------------------------------------------------------------


class TestPadicNumberConstruction:
    def test_basic_construction(self):
        """PadicNumber(p=5, digits=[1,2,3], valuation=0, precision=3) is valid."""
        x = PadicNumber(p=5, digits=[1, 2, 3], valuation=0, precision=3)
        assert x.p == 5
        assert x.digits == [1, 2, 3]
        assert x.valuation == 0
        assert x.precision == 3

    def test_repr_contains_p_and_digits(self):
        """repr should show digit expansion with O(p^N) term."""
        x = PadicNumber(p=5, digits=[1, 2, 3], valuation=0, precision=3)
        s = repr(x)
        assert "O(5^3)" in s


# ---------------------------------------------------------------------------
# p-adic arithmetic: addition
# ---------------------------------------------------------------------------


class TestPadicAddition:
    def test_add_with_carry(self):
        """3 + 4 = 7 = 2 + 1*5 in Q_5, so [3,0] + [4,0] = [2,1]."""
        a = PadicNumber(p=5, digits=[3, 0], valuation=0, precision=2)
        b = PadicNumber(p=5, digits=[4, 0], valuation=0, precision=2)
        c = a + b
        assert c.digits[0] == 2
        assert c.digits[1] == 1

    def test_add_same_prime(self):
        """Sum stays in same prime field."""
        a = PadicNumber(p=7, digits=[3, 2], valuation=0, precision=2)
        b = PadicNumber(p=7, digits=[5, 1], valuation=0, precision=2)
        c = a + b
        # 3+5=8=1+1*7, carry 1: 2+1+1=4
        assert c.digits[0] == 1
        assert c.digits[1] == 4

    def test_add_different_primes_raises(self):
        """Cannot add p-adic numbers with different primes."""
        a = PadicNumber(p=5, digits=[1], valuation=0, precision=1)
        b = PadicNumber(p=7, digits=[1], valuation=0, precision=1)
        with pytest.raises(ValueError):
            _ = a + b


# ---------------------------------------------------------------------------
# p-adic arithmetic: multiplication
# ---------------------------------------------------------------------------


class TestPadicMultiplication:
    def test_simple_multiply(self):
        """2 * 3 = 6 = 1*5 + 1 in Q_5, so result starts with digit 1 at val 0
        or digit 1 at val 1 depending on normalization.
        Actually 6 = 1 + 1*5, so digits [1, 1] at valuation 0."""
        a = PadicNumber(p=5, digits=[2], valuation=0, precision=1)
        b = PadicNumber(p=5, digits=[3], valuation=0, precision=1)
        c = a * b
        # 2*3 = 6 = 1 + 1*5, but with precision 1 we only get the first digit
        # The product should represent 6 in Z_5
        # With convolution and carry: 2*3=6, 6 mod 5 = 1, carry 1
        # Result: digits=[1] at valuation 0, OR digits=[1,1] if precision extended
        # Key: the value should represent 6
        # At minimum the leading digit mod 5 should be 1 (6 mod 5 = 1)
        assert c.digits[0] == 1

    def test_multiply_valuations_add(self):
        """Multiplying p^a * p^b should give valuation a+b."""
        # 5^1 * 5^2 = 5^3
        a = PadicNumber(p=5, digits=[1], valuation=1, precision=1)
        b = PadicNumber(p=5, digits=[1], valuation=2, precision=1)
        c = a * b
        assert c.valuation == 3


# ---------------------------------------------------------------------------
# p-adic negation and subtraction
# ---------------------------------------------------------------------------


class TestPadicNegationSubtraction:
    def test_negation_and_add_to_zero(self):
        """x + (-x) should give a zero-like result."""
        x = PadicNumber(p=5, digits=[3, 2], valuation=0, precision=2)
        neg_x = -x
        result = x + neg_x
        # All digits should be 0
        assert all(d == 0 for d in result.digits)

    def test_subtraction(self):
        """a - b = a + (-b)."""
        a = PadicNumber(p=5, digits=[4, 1], valuation=0, precision=2)
        b = PadicNumber(p=5, digits=[2, 0], valuation=0, precision=2)
        c = a - b
        # 4+1*5=9, 2+0*5=2, 9-2=7=2+1*5 -> digits [2,1]
        assert c.digits[0] == 2
        assert c.digits[1] == 1


# ---------------------------------------------------------------------------
# p-adic norm
# ---------------------------------------------------------------------------


class TestPadicNorm:
    def test_norm_of_unit(self):
        """Norm of a unit (valuation 0) is 1.0."""
        x = PadicNumber(p=5, digits=[3, 2], valuation=0, precision=2)
        assert x.norm() == pytest.approx(1.0)

    def test_norm_of_p(self):
        """Norm of p (valuation 1) is 1/p."""
        x = PadicNumber(p=5, digits=[1], valuation=1, precision=1)
        assert x.norm() == pytest.approx(1.0 / 5.0)

    def test_norm_of_zero(self):
        """Norm of zero is 0."""
        x = PadicNumber(p=5, digits=[0, 0], valuation=0, precision=2)
        assert x.norm() == 0.0


# ---------------------------------------------------------------------------
# padic_from_rational
# ---------------------------------------------------------------------------


class TestPadicFromRational:
    def test_integer_conversion(self):
        """3/1 in Q_5 should be [3, 0, 0, ...]."""
        x = padic_from_rational(3, 1, p=5, precision=5)
        assert x.p == 5
        assert x.digits[0] == 3
        assert x.valuation == 0

    def test_one_third_in_q5(self):
        """1/3 in Q_5: 3 * (1/3) = 1. Verify by multiplication."""
        one_third = padic_from_rational(1, 3, p=5, precision=10)
        three = PadicNumber(p=5, digits=[3] + [0] * 9, valuation=0, precision=10)
        product = one_third * three
        # Should represent 1: digits[0] == 1
        assert product.digits[0] == 1

    def test_rational_with_p_in_denominator(self):
        """1/5 in Q_5 should have valuation -1."""
        x = padic_from_rational(1, 5, p=5, precision=5)
        assert x.valuation == -1

    def test_zero_numerator(self):
        """0/1 should give zero."""
        x = padic_from_rational(0, 1, p=5, precision=5)
        assert all(d == 0 for d in x.digits)


# ---------------------------------------------------------------------------
# Kubota-Leopoldt zeta
# ---------------------------------------------------------------------------


class TestKubotaLeopoldtZeta:
    def test_zeta_at_minus_one(self):
        """zeta_5(-1): s=-1, n=2. zeta_p(1-n) = -(1-p^{n-1})*B_n/n.
        n=2: -(1-5)*B_2/2 = -(1-5)*(1/6)/1 ... wait:
        B_2 = 1/6, so -(1-5)*(1/6)/2 = -(-4)/(12) = 4/12 = 1/3.
        The result should be a PadicNumber representing 1/3 in Q_5."""
        result = kubota_leopoldt_zeta(s=-1, p=5, precision=10)
        assert isinstance(result, PadicNumber)
        assert result.p == 5
        # 1/3 in Q_5 has valuation 0 (3 is coprime to 5)
        assert result.valuation == 0

    def test_zeta_rejects_even_negative(self):
        """s=-2 means n=3 which is odd -- should reject (formula needs n even)."""
        with pytest.raises(ValueError):
            kubota_leopoldt_zeta(s=-2, p=5, precision=10)

    def test_zeta_rejects_positive(self):
        """Positive s is not in the domain of the Kubota-Leopoldt formula."""
        with pytest.raises(ValueError):
            kubota_leopoldt_zeta(s=1, p=5, precision=10)

    def test_zeta_at_minus_three(self):
        """zeta_5(-3): s=-3, n=4. zeta_p(1-4) = -(1-5^3)*B_4/4.
        B_4 = -1/30, so -(1-125)*(-1/30)/4 = -(-124)*(-1/120) = -124/120 = -31/30.
        Result should be a PadicNumber."""
        result = kubota_leopoldt_zeta(s=-3, p=5, precision=10)
        assert isinstance(result, PadicNumber)
        assert result.p == 5


# ---------------------------------------------------------------------------
# padic_fractal_tree_data
# ---------------------------------------------------------------------------


class TestPadicFractalTreeData:
    def test_returns_dict_with_nodes_and_edges(self):
        data = padic_fractal_tree_data(p=3, depth=3)
        assert "nodes" in data
        assert "edges" in data
        assert isinstance(data["nodes"], list)
        assert isinstance(data["edges"], list)

    def test_node_count(self):
        """A p-ary tree of depth d has (p^(d+1) - 1) / (p - 1) nodes."""
        data = padic_fractal_tree_data(p=3, depth=2)
        expected_nodes = (3**3 - 1) // (3 - 1)  # = 13
        assert len(data["nodes"]) == expected_nodes

    def test_edge_count(self):
        """Number of edges = number of nodes - 1."""
        data = padic_fractal_tree_data(p=3, depth=2)
        expected_edges = len(data["nodes"]) - 1
        assert len(data["edges"]) == expected_edges

    def test_node_has_expected_fields(self):
        data = padic_fractal_tree_data(p=2, depth=1)
        node = data["nodes"][0]
        assert "x" in node
        assert "y" in node
        assert "label" in node
        assert "depth" in node
