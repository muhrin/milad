# -*- coding: utf-8 -*-
import numpy as np

from milad import moments
from milad import zernike
from milad import invariants
from milad import generate
from milad import transform


def test_zernike_of_deltas(complex_invariants):
    positions = generate.random_points_in_sphere(4, radius=.7)
    mass = 1.
    max_order = 12

    moms = zernike.zernike_moments_of_deltas(max_order, positions)
    invs_one = invariants.apply_invariants(complex_invariants, moms)

    # Now rotate the system randomly and recalculate the invariants
    rotated = transform.randomly_rotate(positions)
    rot_moms = zernike.zernike_moments_of_deltas(max_order, rotated)
    invs_two = invariants.apply_invariants(complex_invariants, rot_moms)

    np.testing.assert_array_almost_equal(invs_two, invs_one)


def test_zernike_of_gaussians(complex_invariants):
    positions = generate.random_points_in_sphere(4, radius=.7)
    sigma = 0.1
    mass = 1.
    max_order = 12

    geom_moments = moments.geometric_moments_of_gaussians(max_order, positions, sigma, mass)
    zmoments = zernike.calc_zernike_moments(max_order, geom_moments)
    invs_one = invariants.apply_invariants(complex_invariants, zmoments)

    # Now rotate the system randomly and recalculate the invariants
    rotated = transform.randomly_rotate(positions)
    rot_geom_moments = moments.geometric_moments_of_gaussians(max_order, rotated, sigma, mass)
    rot_zmoments = zernike.calc_zernike_moments(max_order, rot_geom_moments)
    invs_two = invariants.apply_invariants(complex_invariants, rot_zmoments)

    np.testing.assert_array_almost_equal(invs_two, invs_one)
