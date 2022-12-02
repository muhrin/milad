# -*- coding: utf-8 -*-
import numpy
from scipy.spatial.transform import Rotation

import milad
from milad import transform
from milad import geometric


def test_general(geometric_invariants):
    # Settings
    scale = 1.0
    sigma = scale * 0.4
    num_rotations = 0
    num_points = 4

    pos1 = 4 * numpy.random.rand(num_points, 3)
    pos1 = transform.center(pos1)
    pos1 *= scale

    moments = geometric.from_gaussians(geometric_invariants.max_order, pos1, sigma)
    invariants = geometric_invariants(moments)

    for rot in range(1, num_rotations + 1):
        rot = Rotation.random()
        rotated = rot.apply(pos1)

        # Calculate the moment invariants
        rot_invariants = milad.invariants.calc_moment_invariants(
            geometric_invariants, rotated, sigma
        )

        assert numpy.testing.assert_array_almost_equal(invariants, rot_invariants)
