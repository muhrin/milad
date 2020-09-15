# -*- coding: utf-8 -*-
import numpy
import pytest
import sympy

from milad import analytic
from milad import generate
from milad import geometric
from milad import utils

# pylint: disable=invalid-name


def test_multidim_norm_moments():
    sigma = 2.
    mass = 5.

    moms = geometric.gaussian_moments(0, numpy.zeros((3, 1)), sigmas=sigma, weight=mass)
    expected = numpy.empty((3, 1))
    expected.fill(mass)
    assert pytest.approx((moms - expected).max(), 0)

    # Do the odds first
    for order in (1, 3, 5, 7):
        moms = geometric.gaussian_moments(order, numpy.zeros((3, 1)), sigmas=sigma, weight=mass)
        expected = numpy.zeros((3, 1))
        assert pytest.approx((moms - expected).max(), 0)

    moms = geometric.gaussian_moments(2, numpy.zeros((3, 1)), sigmas=sigma, weight=mass)
    expected = numpy.empty((3, 1))
    expected.fill(sigma**2)
    assert pytest.approx((moms - expected).max(), 0)

    moms = geometric.gaussian_moments(4, numpy.zeros((3, 1)), sigmas=sigma, weight=mass)
    expected = numpy.empty((3, 1))
    expected.fill(3 * sigma**4)
    assert pytest.approx((moms - expected).max(), 0)

    moms = geometric.gaussian_moments(6, numpy.zeros((3, 1)), sigmas=sigma, weight=mass)
    expected = numpy.empty((3, 1))
    expected.fill(15 * sigma**6)
    assert pytest.approx((moms - expected).max(), 0)

    moms = geometric.gaussian_moments(8, numpy.zeros((3, 1)), sigmas=sigma, weight=mass)
    expected = numpy.empty((3, 1))
    expected.fill(105 * sigma**8)
    assert pytest.approx((moms - expected).max(), 0)


def test_moment_tensor3d():
    pos = numpy.array((1., 2., 3.))
    sigma = 2.
    mass = 1.
    max_order = 4

    tensor = geometric.from_gaussians(max_order, [pos], sigma, mass)
    assert tensor[0, 0, 0] == mass

    assert tensor[1, 0, 0] == pos[0]
    assert tensor[2, 0, 0] == pos[0]**2 + sigma**2
    assert tensor[3, 0, 0] == pos[0]**3 + 3 * pos[0] * sigma**2

    assert tensor[0, 1, 0] == pos[1]
    assert tensor[0, 2, 0] == pos[1]**2 + sigma**2
    assert tensor[0, 3, 0] == pos[1]**3 + 3 * pos[1] * sigma**2

    assert tensor[0, 0, 1] == pos[2]
    assert tensor[0, 0, 2] == pos[2]**2 + sigma**2
    assert tensor[0, 0, 3] == pos[2]**3 + 3 * pos[2] * sigma**2

    for i in range(max_order + 1):
        for j in range(max_order + 1):
            for k in range(max_order + 1):
                assert tensor[i, j, k] == tensor[i, 0, 0] * tensor[0, j, 0] * tensor[0, 0, k]


def test_moments_symmetric():
    x = 2.
    positions = numpy.array(((-x, 0., 0.), (x, 0., 0.)))
    sigma = 2.
    mass = 1.5
    max_order = 4

    tensor = geometric.from_gaussians(max_order, positions, sigma, mass)
    assert tensor[0, 0, 0] == mass * len(positions)
    assert tensor[1, 0, 0] == 0.
    assert tensor[2, 0, 0] == 2 * mass * (x**2 + sigma**2)


def test_geom_moments_of_deltas():
    num_points = 4
    positions = generate.random_points_in_sphere(num_points, radius=.7)
    weights = numpy.random.rand(num_points)
    max_order = 11

    # Manually calculate the moments
    moms = numpy.zeros((max_order + 1, max_order + 1, max_order + 1))
    for p in utils.from_to(max_order):
        for q in utils.from_to(max_order):
            for r in utils.from_to(max_order):
                for pos, weight in zip(positions, weights):
                    moms[p, q, r] += weight * (pos**(p, q, r)).prod(axis=-1)

    calculated = geometric.from_deltas(max_order, positions, weights=weights)
    numpy.testing.assert_array_almost_equal(moms, calculated.to_matrix())

    # The 0^th moment should always be the sum of the weights
    assert calculated[0, 0, 0] == weights.sum()


def test_delta_moments_derivatives():
    """Test that derivatives of moments calculated from delta functions are correct"""
    order = 5
    NUM_POINTS = 4

    x = sympy.IndexedBase('x')
    w = sympy.IndexedBase('w')
    points = analytic.create_array(x, (NUM_POINTS, 3))
    weights = analytic.create_array(w, NUM_POINTS)

    # Get the moments and derivates wrt to weights and positions
    moments, dw, dx = geometric.from_deltas(order, points, weights=weights, get_derivatives=True)
    for p in range(order):
        for q in range(order):
            for r in range(order):
                moment = moments[p, q, r]

                # Check weight derivatives
                for i in range(NUM_POINTS):
                    dm_dx_calculated = dw[i, p, q, r]
                    if isinstance(moment, float):
                        # We have a constant, so derivative is always 0
                        dw_dx_analytic = 0
                    else:
                        dw_dx_analytic = moment.diff(w[i])

                    assert dm_dx_calculated == dw_dx_analytic

                # Now loop over each point and the x, y, z coordinates
                for i in range(4):
                    for d in range(3):
                        dm_dx_calculated = dx[i, d, p, q, r]
                        if isinstance(moment, (float, sympy.core.numbers.Zero)):
                            # We have a constant, so derivative is always 0
                            dw_dx_analytic = 0
                        else:
                            dw_dx_analytic = moment.diff(x[i, d])

                        assert dm_dx_calculated == dw_dx_analytic
