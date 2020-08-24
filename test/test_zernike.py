# -*- coding: utf-8 -*-
import numpy as np

from milad import moments
from milad import zernike
from milad import invariants
from milad import generate
from milad import transform


def test_moment_tensor3d(complex_invariants):
    positions = generate.random_points_in_sphere(4, radius=.7)
    sigma = 0.1
    mass = 1.
    max_order = 12

    geom_moments = moments.calc_raw_moments3d(max_order, positions, sigma,
                                              mass)
    zmoments = zernike.calc_zernike_moments(max_order, geom_moments)
    invs_one = invariants.apply_invariants(complex_invariants, zmoments)

    # Now rotate the system randomly and recalculate the invariants
    rotated = transform.randomly_rotate(positions)
    # rotated = -positions
    rot_geom_moments = moments.calc_raw_moments3d(max_order, rotated, sigma,
                                                  mass)
    rot_zmoments = zernike.calc_zernike_moments(max_order, rot_geom_moments)
    invs_two = invariants.apply_invariants(complex_invariants, rot_zmoments)

    np.testing.assert_array_almost_equal(invs_two, invs_one)
