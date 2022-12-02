# -*- coding: utf-8 -*-
import numpy as np

from milad import polynomials

# pylint: disable=invalid-name


# Let's create a polynomial function to be able to compare
def poly_fn(x: np.array):
    return 5 * x[0, 0, 0] * x[0, 1, 0] + 3 * x[0, 1, 0] * x[0, 1, 0] + 9


POLY = polynomials.HomogenousPolynomial(
    2, [5, 3], [[[0, 0, 0], [0, 1, 0]], [[0, 1, 0], [0, 1, 0]]], constant=9
)


def test_polynomial():
    x = np.random.rand(2, 2, 2)
    assert poly_fn(x) == POLY(x)
    x = np.zeros((2, 2, 2))
    assert poly_fn(x) == POLY(x)

    # Check partial derivatives
    deriv = POLY.get_partial_derivative((0, 0, 0))
    assert deriv.degree == POLY.degree - 1
    assert np.all(deriv.prefactors == np.array([5.0]))
    assert np.all(deriv.terms == np.array([[[0, 1, 0]]]))
    assert deriv.constant == 0.0


def test_partial_derivatives():
    deriv = POLY.get_partial_derivative((0, 1, 0))
    assert deriv.degree == POLY.degree - 1
    assert np.all(deriv.prefactors == np.array([5.0, 6.0]))
    assert np.all(deriv.terms == np.array([[[0, 0, 0]], [[0, 1, 0]]]))
    assert deriv.constant == 0.0

    # Let's do the derivative again so we can check how it deals with terms whose power drops to 0
    deriv2 = deriv.get_partial_derivative((0, 1, 0))
    assert deriv2.degree == deriv.degree - 1
    assert np.all(deriv2.prefactors == np.array([]))
    assert np.all(deriv2.terms == np.array([]))
    assert deriv2.constant == 6.0

    # And finally let's check what happens when we do a derivative of a constant
    deriv3 = deriv2.get_partial_derivative((0, 1, 0))
    assert deriv3.degree == deriv.degree - 1
    assert np.all(deriv3.prefactors == np.array([]))
    assert np.all(deriv3.terms == np.array([]))
    assert deriv3.constant == 0.0


def test_gradient():
    grad = POLY.get_gradient()
    assert set(grad.keys()) == POLY.variables


def test_simplify():
    """Test that a polynomial will automatically simplify itself if there are commons factors in the sum"""
    # This is effectively 5 * x[1, 0, 0] + 7 * x[1, 0, 0] which should be simplified
    poly = polynomials.HomogenousPolynomial(1, [5, 7], [[[1, 0, 0]], [[1, 0, 0]]])
    assert np.all(poly.prefactors == np.array([12]))
    assert np.all(poly.terms == np.array([[[1, 0, 0]]]))
