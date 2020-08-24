# -*- coding: utf-8 -*-
import numpy
from scipy.spatial.transform import Rotation

import milad
from milad import transform

# def center(positions: numpy.array):
#     """Centre the given positions by shifting the centre of mass to zero"""
#     positions -= positions.sum(axis=0) / len(positions)


def test_general(moment_invariants):
    # Settings
    scale = 1.
    sigma = scale * 0.4
    num_rotations = 0
    num_points = 4

    pos1 = 4 * numpy.random.rand(num_points, 3)

    pos1 = transform.center(pos1)

    pos1 *= scale

    invariants = milad.invariants.calc_moment_invariants(
        moment_invariants, pos1, sigma)

    for rot in range(1, num_rotations + 1):
        rot = Rotation.random()
        rotated = rot.apply(pos1)

        # Calculate the moment invariants
        rot_invariants = milad.invariants.calc_moment_invariants(
            moment_invariants, rotated, sigma)

        assert numpy.testing.assert_array_almost_equal(invariants,
                                                       rot_invariants)
