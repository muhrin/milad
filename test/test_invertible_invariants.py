# -*- coding: utf-8 -*-
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import milad
from milad import invertible_invariants
from milad import zernike
from milad import generate

N_MAX = 5
indices = milad.sph.IndexTraits(n_spec=N_MAX, n_minus_l_even=True, l_le_n=True)

# pylint: disable=redefined-outer-name


@pytest.fixture(scope='session')
def inv_invariants():
    """A fixture to create a set of invertible invariants"""
    return invertible_invariants.InvariantsGenerator.generate_all(indices)


def check_inverted_invariants(invariants: milad.MomentInvariants, target: np.array, inverted: np.ndarray):
    diff = target - inverted
    not_zero = np.argwhere(~np.isclose(diff, 0.)).ravel()
    assert not np.any(not_zero), f'First non-zero: {invariants[not_zero[0]]}'


def test_invertible_invariants_basics(inv_invariants):
    """Here we:
        1. Create a fingerprint from a random set of moments
        2. Invert the fingerprint to recover some corresponding moments
        3. Calculate the fingerprint from the inverted moments
        4. Assert that the two fingerprints match
    """
    # Create some random moments and calculate the fingerprint
    pts = generate.random_points_in_sphere(11, radius=1., centre=False)
    moments = zernike.from_deltas(indices.n.max, pts)
    phi = inv_invariants(moments)

    inverted = zernike.ZernikeMoments(indices.n.max, indices.l.max)
    # Perform inversion
    inv_invariants.invert(phi, inverted)
    assert not np.any(np.isnan(inverted.array))

    inverted_phi = inv_invariants(inverted)
    check_inverted_invariants(inv_invariants, phi, inverted_phi)


def test_invertible_invariants_symmetric():
    """Test invertible invariants for symmetric environments"""
    indices = milad.sph.IndexTraits(n_spec=12, l_spec=4, n_minus_l_even=True, l_le_n=True)
    inv_invariants = invertible_invariants.InvariantsGenerator.generate_all(indices)

    pts = np.array([[-0.5, 0, 0], [0.5, 0, 0]])
    # pts = np.array([[-0.5, -0.5, 0], [-0.5, 0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0]])
    moments = zernike.from_deltas(indices.n.max, pts, l_max=indices.l.max)

    phi = inv_invariants(moments)

    inverted = zernike.ZernikeMoments(indices.n.max, indices.l.max)

    # with pytest.raises(ValueError):
    inv_invariants.invert(phi, inverted)

    assert not np.any(np.isnan(inverted.array))
    inverted_phi = inv_invariants(inverted)
    # For now we can't deal with this case...stay tuned
    check_inverted_invariants(inv_invariants, phi, inverted_phi)


def test_invertible_invariants_are_rotation_invariant(inv_invariants):
    """Check that the generated invariants are, indeed, invariant to rotation"""
    num_points = 10
    num_rotations = 10

    pts = generate.random_points_in_sphere(10, radius=1.)
    weights = np.random.rand(num_points)
    moments0 = zernike.from_deltas(N_MAX, pts, weights)
    phi0 = inv_invariants(moments0)
    for _ in range(num_rotations):
        rot = Rotation.random()
        rotated = rot.apply(pts)

        moments = zernike.from_deltas(N_MAX, rotated, weights)
        phi = inv_invariants(moments)

        assert np.allclose(phi0, phi)


def test_against_chiral_tetrahedra(inv_invariants, chiral_tetrahedra):
    minus, plus = chiral_tetrahedra
    minus_phi = inv_invariants(zernike.from_deltas(inv_invariants.max_order, minus)).real
    plus_phi = inv_invariants(zernike.from_deltas(inv_invariants.max_order, plus)).real

    assert np.allclose(minus_phi, plus_phi)
