# -*- coding: utf-8 -*-
import pytest

import numpy as np
import sympy

from milad import geometric
from milad import zernike
from milad import mathutil
from milad import utils
from milad import generate
from milad import transform


# pylint: disable=invalid-name


def test_zernike_reconstruct_deltas():
    num_points = 4
    positions = generate.random_points_in_sphere(num_points, radius=0.7)
    weights = 1.0
    max_order = 7
    n_samples = 11

    moms = zernike.from_deltas(max_order, positions, weights=weights)
    reconstructed_values = moms.value_at(positions)

    # Now reconstruct a voxel grid
    spacing = np.linspace(-1.0, 1.0, n_samples)
    grid_points = []
    for pos in (
        np.array(np.meshgrid(spacing, spacing, spacing)).reshape(3, -1).T
    ):  # pylint: disable=not-an-iterable
        if np.linalg.norm(pos) > 1.0:
            continue
        grid_points.append(pos)
    grid_points = np.array(grid_points)

    grid_vals = moms.value_at(grid_points)

    # The values at the original delta functions should hold the highest reconstruction values
    # (higher than anywhere else on the grid)
    assert grid_vals.max() < reconstructed_values.max()


def test_zernike_of_deltas(complex_invariants):
    positions = generate.random_points_in_sphere(4, radius=0.7)
    weights = 1.0
    max_order = 10

    moms = zernike.from_deltas(max_order, positions, weights=weights)
    invs_one = complex_invariants.apply(moms)

    # Now rotate the system randomly and recalculate the invariants
    rotated = transform.randomly_rotate(positions)
    rot_moms = zernike.from_deltas(max_order, rotated)
    invs_two = complex_invariants.apply(rot_moms)

    np.testing.assert_array_almost_equal(invs_two, invs_one)


def test_zernike_of_gaussians(complex_invariants):
    positions = generate.random_points_in_sphere(4, radius=0.7)
    sigma = 0.1
    mass = 1.0
    max_order = 10

    geom_moments = geometric.from_gaussians(max_order, positions, sigma, mass)
    zmoments = zernike.from_geometric_moments(max_order, geom_moments)
    invs_one = complex_invariants.apply(zmoments)

    # Now rotate the system randomly and recalculate the invariants
    rotated = transform.randomly_rotate(positions)
    rot_geom_moments = geometric.from_gaussians(max_order, rotated, sigma, mass)
    rot_zmoments = zernike.from_geometric_moments(max_order, rot_geom_moments)
    invs_two = complex_invariants.apply(rot_zmoments)

    np.testing.assert_array_almost_equal(invs_two, invs_one)


def test_zernike_properties():
    num_particles = 4
    positions = generate.random_points_in_sphere(num_particles, radius=0.7)
    sigmas = np.random.rand(num_particles)
    weights = np.random.rand(num_particles)
    max_order = 10

    geom_moments = geometric.from_gaussians(
        max_order, positions, sigmas=sigmas, weights=weights
    )
    assert geom_moments[0, 0, 0] == weights.sum()

    moms = zernike.from_geometric_moments(max_order, geom_moments)
    assert moms[0, 0, 0] == 3 / (4 * np.pi) * weights.sum()

    for n in range(max_order + 1):
        for l in range(n + 1):
            if ((n - l) % 2) != 0:
                continue

            for m in range(l):
                assert moms[n, l, -m] == (-1) ** m * moms[n, l, m].conjugate()


def test_zernike_analytic():
    max_order = 6
    geom = sympy.IndexedBase("m", real=True)  # Geometric moments
    jacobian = zernike.get_jacobian_wrt_geom_moments(max_order)

    # Let's build the Jacobian symbolically
    for _idx, (n, l, m) in enumerate(zernike.iter_indices(max_order, redundant=False)):
        eq = zernike.omega_nl_m(n, l, m, geom)
        for p, q, r in geometric.iter_indices(max_order):
            differentiated = eq.diff(geom[p, q, r])

            zindex = zernike.linear_index((n, l, m))
            gindex = geometric.linear_index(max_order, (p, q, r))
            jacobian_value = jacobian[zindex, gindex]

            assert complex(differentiated) == pytest.approx(
                jacobian_value
            ), f"Omega{n},{l},{m} m{p},{q},{r}"


def test_zernike_indexing():
    max_order = 6

    # Check that linear indexing works correctly
    idx = 0
    for n in utils.inclusive(max_order):
        for l in utils.inclusive(n):
            if not utils.even(n - l):
                continue

            for m in utils.inclusive(l):
                assert zernike.linear_index((n, l, m)) == idx
                idx += 1

            for m in utils.inclusive(-l, -1, 1):
                assert zernike.linear_index((n, l, m)) == idx
                idx += 1


def test_zernike_from_vector():
    """Test converting Zernike moments to a vector and back"""
    num_points = 4
    positions = generate.random_points_in_sphere(num_points, radius=0.7)
    max_order = 7

    moms = zernike.from_deltas(max_order, positions)
    as_vector = moms.vector

    from_vector = zernike.ZernikeMoments.from_vector(max_order, as_vector)
    assert np.allclose(from_vector.vector, as_vector)

    # Double check the round-trip
    assert np.all(
        zernike.ZernikeMoments.from_vector(max_order, from_vector.vector).vector
        == from_vector.vector
    )


def test_zernike_builder():
    """Test that the Zernike builder can create the moments from a vector and the inverse"""
    num_points = 4
    positions = generate.random_points_in_sphere(num_points, radius=0.7)
    max_order = 7

    moms = zernike.from_deltas(max_order, positions)
    builder = moms.get_builder()

    # Get the moments in vector form
    vec = builder.inverse(moms)

    # Now let's create a set of moments from the vector and check that it's all the same
    recreated, _jac = builder(vec, jacobian=True)
    assert recreated == moms


def test_direct_calculation():
    nmax = 4
    lmax = 4

    num_atoms = 4
    pos = generate.random_points_in_sphere(num_atoms)
    from_deltas = zernike.from_deltas(n_max=nmax, l_max=lmax, positions=pos)

    from_deltas_direct = zernike.from_deltas_direct(nmax=nmax, lmax=lmax, positions=pos)

    diff = from_deltas.array - from_deltas_direct.array  # noqa: F841

    spherical = mathutil.cart2sph(pos.T)  # Spherical coordinates
    for n in utils.inclusive(nmax):
        for l in utils.inclusive(n):
            if utils.odd(n - l):
                continue
            for m in utils.inclusive(-l, l):
                vals = (
                    3
                    / (4 * np.pi)
                    * zernike.zernike_poly(n, l, m, spherical).conjugate()
                )
                total = np.sum(vals)  # noqa: F841
                from_deltas_val = from_deltas[n, l, m]  # noqa: F841
