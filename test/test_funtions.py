# -*- coding: utf-8 -*-
import random

import numpy as np
import sympy
from sympy import Symbol, IndexedBase

from milad import functions
from milad import testing


def test_cosine_cutoff():
    # pylint: disable=invalid-name, too-many-locals

    rcut = 1.5  # Larger than the random range so everything is within the cutoff sphere

    # Create some features
    delta = functions.WeightedDelta(np.random.rand(3), weight=random.random())
    gaussian = functions.WeightedGaussian(np.random.rand(3), sigma=random.random(), weight=random.random())
    features = functions.Features(delta, gaussian)

    cos_cut = functions.CosineCutoff(rcut)
    res, jac = cos_cut(features, jacobian=True)

    assert isinstance(res, functions.Features)
    # Order should be preserved
    res_delta = res.features[0]
    assert isinstance(res_delta, functions.WeightedDelta)
    assert np.all(res_delta.pos == delta.pos)
    assert res_delta.weight == delta.weight * cos_cut.fn(np.linalg.norm(delta.pos))

    res_gaussian = res.features[1]
    assert isinstance(res_gaussian, functions.WeightedGaussian)
    assert res_gaussian.weight == gaussian.weight * cos_cut.fn(np.linalg.norm(gaussian.pos))

    # Now let's check derivatives
    r = IndexedBase('r', real=True)
    r_c = IndexedBase('r_c', real=True)
    w = Symbol('w')
    dr = sympy.sqrt(r[0]**2 + r[1]**2 + r[2]**2)

    cut_expr = functions.CosineCutoff.symbolic(dr, w, r_c)
    # Create the numeric substitution for the delta function
    delta_subs = [(r[0], delta.pos[0]), (r[1], delta.pos[1]), (r[2], delta.pos[2]), (w, delta.weight), (r_c, rcut)]

    assert np.all(jac[0:3, 0:3] == np.eye(3))

    for i in range(3):
        np.testing.assert_almost_equal(jac[delta.WEIGHT, i], sympy.diff(cut_expr, r[i]).subs(delta_subs).evalf())
    np.testing.assert_approx_equal(jac[delta.WEIGHT, 3], sympy.diff(cut_expr, w).subs(delta_subs).evalf())

    gaussian_jac = jac[len(delta):, len(delta):]
    gaussian_subs = [(r[0], gaussian.pos[0]), (r[1], gaussian.pos[1]), (r[2], gaussian.pos[2]), (w, gaussian.weight),
                     (r_c, rcut)]

    assert np.all(gaussian_jac[0:3, 0:3] == np.eye(3))

    for i in range(3):
        np.testing.assert_almost_equal(
            gaussian_jac[gaussian.WEIGHT, i],
            sympy.diff(cut_expr, r[i]).subs(gaussian_subs).evalf()
        )

    np.testing.assert_approx_equal(
        gaussian_jac[gaussian.WEIGHT, gaussian.WEIGHT],
        sympy.diff(cut_expr, w).subs(gaussian_subs).evalf()
    )

    testing.test_function(cos_cut, features, check_jacobian=False)


def test_identity():
    # pylint: disable=invalid-name
    x = np.random.rand(10)
    testing.test_function(functions.Identity(), x, expected_output=x)


def test_residuals():
    # pylint: disable=invalid-name
    x = np.random.rand(10)
    testing.test_function(functions.Residuals(x), x, np.zeros(10))


def test_mse():
    # pylint: disable=invalid-name
    x = np.random.rand(10)
    testing.test_function(functions.MeanSquaredError(x), x, np.zeros(10))


def test_map():
    # pylint: disable=invalid-name
    x1 = np.random.rand(10)
    x2 = np.random.rand(10)
    testing.test_function(
        functions.Map(functions.MeanSquaredError(x1), functions.MeanSquaredError(x2), weights=np.random.rand(2)),
        np.random.rand(10),
        check_jacobian=True
    )
