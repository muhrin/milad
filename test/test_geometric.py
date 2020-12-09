# -*- coding: utf-8 -*-
import numpy
import numpy as np
import pytest
import sympy

from milad import analytic
from milad import functions
from milad import generate
from milad import geometric
from milad import utils

# pylint: disable=invalid-name


def test_multidim_norm_moments():
    sigma = 2.

    moms = geometric.gaussian_moment[0](numpy.zeros((3, 1)), sigma)
    expected = numpy.empty((3, 1))
    expected.fill(1.)
    assert pytest.approx((moms - expected).max(), 0)

    # Do the odds first
    for order in (1, 3, 5, 7):
        moms = geometric.gaussian_moment[order](numpy.zeros((3, 1)))
        expected = numpy.zeros((3, 1))
        assert pytest.approx((moms - expected).max(), 0)

    moms = geometric.gaussian_moment[2](numpy.zeros((3, 1)), sigma)
    expected = numpy.empty((3, 1))
    expected.fill(sigma**2)
    assert pytest.approx((moms - expected).max(), 0)

    moms = geometric.gaussian_moment[4](numpy.zeros((3, 1)), sigma)
    expected = numpy.empty((3, 1))
    expected.fill(3 * sigma**4)
    assert pytest.approx((moms - expected).max(), 0)

    moms = geometric.gaussian_moment[6](numpy.zeros((3, 1)), sigma)
    expected = numpy.empty((3, 1))
    expected.fill(15 * sigma**6)
    assert pytest.approx((moms - expected).max(), 0)

    moms = geometric.gaussian_moment[8](numpy.zeros((3, 1)), sigma)
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
    for p in utils.inclusive(max_order):
        for q in utils.inclusive(max_order):
            for r in utils.inclusive(max_order):
                for pos, weight in zip(positions, weights):
                    moms[p, q, r] += weight * (pos**(p, q, r)).prod(axis=-1)

    calculated = geometric.from_deltas(max_order, positions, weights=weights)
    numpy.testing.assert_array_almost_equal(moms, calculated.to_matrix())

    # The 0^th moment should always be the sum of the weights
    assert calculated[0, 0, 0] == weights.sum()


def test_delta_moments_derivatives():
    """Test that derivatives of moments calculated from delta functions are correct"""
    ORDER = 5
    NUM_POINTS = 4
    UBOUND = ORDER + 1
    Delta = functions.WeightedDelta

    x = sympy.IndexedBase('x')
    w = sympy.IndexedBase('w')

    POINTS = analytic.create_array(x, (NUM_POINTS, 3))
    WEIGHTS = analytic.create_array(w, NUM_POINTS)

    # Get the moments and derivatives wrt to weights and positions
    moments, jac = geometric.from_deltas(ORDER, POINTS, weights=WEIGHTS, get_jacobian=True)

    jac = jac.reshape(UBOUND, UBOUND, UBOUND, -1)

    for i in range(NUM_POINTS):
        delta_idx = i * Delta.LENGTH
        for p in range(ORDER):
            for q in range(ORDER):
                for r in range(ORDER):
                    moment = moments[p, q, r]

                    # Check weight derivatives
                    dm_dx_calculated = jac[p, q, r, delta_idx + Delta.WEIGHT]
                    if isinstance(moment, float):
                        # We have a constant, so derivative is always 0
                        dw_dx_analytic = 0
                    else:
                        dw_dx_analytic = moment.diff(w[i])

                    assert dm_dx_calculated == dw_dx_analytic

                    # Now loop over each x, y, z coordinate
                    for dim in range(3):  # Look over x = 0, y = 1, z = 2
                        dm_dx_calculated = jac[p, q, r, delta_idx + dim]
                        if isinstance(moment, (float, sympy.core.numbers.Zero)):
                            # We have a constant, so derivative is always 0
                            dw_dx_analytic = 0
                        else:
                            dw_dx_analytic = moment.diff(x[i, dim])

                        assert dm_dx_calculated == dw_dx_analytic


def test_geometric_moments_calculator_deltas():
    """Test using the geometric moments calculator with delta functions as features"""
    num_points = 4
    positions = generate.random_points_in_sphere(num_points, radius=.7)
    weights = np.random.rand(num_points)
    max_order = 11

    # First, try a single delta function
    delta = functions.WeightedDelta(positions[0], weights[0])
    calculator = geometric.GeometricMomentsCalculator(max_order)
    moms = calculator(delta)
    moms2 = geometric.from_deltas(max_order, positions[0:1], weights[0:1])
    assert np.all(moms.moments == moms2.moments)

    # Now try all of them
    env = functions.Features(delta)
    for idx in range(1, num_points):
        env.add(functions.WeightedDelta(positions[idx], weights[idx]))

    moms = calculator(env)
    moms2 = geometric.from_deltas(max_order, positions, weights)
    assert np.all(moms.moments == moms2.moments)


def test_geometric_moments_calculator_gaussians():
    """Test using the geometric moments calculator with Gaussian functions as features"""
    num_points = 4
    positions = generate.random_points_in_sphere(num_points, radius=.7)
    sigmas = np.random.rand(num_points)
    weights = np.random.rand(num_points)
    max_order = 11

    # First, try a single delta function
    gaussian = functions.WeightedGaussian(positions[0], sigmas[0], weights[0])
    calculator = geometric.GeometricMomentsCalculator(max_order)
    moms = calculator(gaussian)
    moms2 = geometric.from_gaussians(max_order, positions[0:1], sigmas[0:1], weights[0:1])
    assert np.all(moms.moments == moms2.moments)

    # Now try all of them
    env = functions.Features(gaussian)
    for idx in range(1, num_points):
        env.add(functions.WeightedGaussian(positions[idx], sigmas[idx], weights[idx]))

    moms = calculator(env)
    moms2 = geometric.from_gaussians(max_order, positions, sigmas, weights)
    assert np.all(moms.moments == moms2.moments)
