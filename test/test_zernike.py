# -*- coding: utf-8 -*-
import numpy as np

import milad
from milad import moments
from milad import zernike
from milad import invariants
from milad import generate
from milad import transform


def test_zernike_reconstruct_deltas():
    num_points = 4
    positions = generate.random_points_in_sphere(num_points, radius=.7)
    weights = 1.
    max_order = 10
    n_samples = 11

    moms = zernike.from_deltas(max_order, positions, weights=weights)
    reconstructed_values = moms.value_at(positions)

    # Now reconstruct a voxel grid
    spacing = np.linspace(-1., 1., n_samples)
    grid_points = []
    for pos in np.array(np.meshgrid(spacing, spacing, spacing)).reshape(3, -1).T:
        if np.linalg.norm(pos) > 1.:
            continue
        grid_points.append(pos)
    grid_points = np.array(grid_points)

    grid_vals = moms.value_at(grid_points)

    # The values at the original delta functions should hold the highest reconstruction values
    # (higher than anywhere else on the grid)
    assert grid_vals.max() < reconstructed_values.max()


def test_zernike_of_deltas(complex_invariants):
    positions = generate.random_points_in_sphere(4, radius=.7)
    weights = 1.
    max_order = 10

    moms = zernike.from_deltas(max_order, positions, weights=weights)
    invs_one = invariants.apply_invariants(complex_invariants, moms)

    # Now rotate the system randomly and recalculate the invariants
    rotated = transform.randomly_rotate(positions)
    rot_moms = zernike.from_deltas(max_order, rotated)
    invs_two = invariants.apply_invariants(complex_invariants, rot_moms)

    np.testing.assert_array_almost_equal(invs_two, invs_one)


def test_zernike_of_gaussians(complex_invariants):
    positions = generate.random_points_in_sphere(4, radius=.7)
    sigma = 0.1
    mass = 1.
    max_order = 10

    geom_moments = moments.geometric_moments_of_gaussians(max_order, positions, sigma, mass)
    zmoments = zernike.from_geometric_moments(max_order, geom_moments)
    invs_one = invariants.apply_invariants(complex_invariants, zmoments)

    # Now rotate the system randomly and recalculate the invariants
    rotated = transform.randomly_rotate(positions)
    rot_geom_moments = moments.geometric_moments_of_gaussians(max_order, rotated, sigma, mass)
    rot_zmoments = zernike.from_geometric_moments(max_order, rot_geom_moments)
    invs_two = invariants.apply_invariants(complex_invariants, rot_zmoments)

    np.testing.assert_array_almost_equal(invs_two, invs_one)


def test_zernike_properties():
    num_particles = 4
    positions = generate.random_points_in_sphere(num_particles, radius=.7)
    sigmas = np.random.rand(num_particles)
    weights = np.random.rand(num_particles)
    max_order = 10

    geom_moments = moments.geometric_moments_of_gaussians(max_order, positions, sigmas=sigmas, weights=weights)
    assert geom_moments[0, 0, 0] == weights.sum()

    moms = zernike.from_geometric_moments(max_order, geom_moments)
    assert moms[0, 0, 0] == 3 / (4 * np.pi) * weights.sum()

    for n in range(max_order + 1):
        for l in range(n + 1):
            if ((n - l) % 2) != 0:
                continue

            for m in range(l):
                assert moms[n, l, -m] == (-1)**m * moms[n, l, m].conjugate()
