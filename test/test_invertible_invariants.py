# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.transform import Rotation

from milad import invertible_invariants
from milad import zernike
from milad import generate


def test_invertible_invariants_basics():
    """Here we:
        1. Create a fingerprint from a random set of moments
        2. Invert the fingerprint to recover some corresponding moments
        3. Calculate the fingerprint from the inverted moments
        4. Assert that the two fingerprints match
    """
    n_max = 5

    generator = invertible_invariants.InvariantsGenerator()

    # Generate the invariants
    invs = generator.generate_all(n_max)

    # Create some random moments and calculate the fingerprint
    rand_moms = zernike.rand(n_max)
    phi = invs(rand_moms)

    inverted = zernike.ZernikeMoments(n_max)
    # Perform inversion
    invs.invert(phi, inverted)
    assert not np.any(np.isnan(inverted.array))

    inverted_phi = invs(inverted)
    assert np.allclose(phi, inverted_phi)


def test_invertible_invariants_are_rotation_invariant():
    """Check that the generated invariants are, indeed, invariant to rotation"""
    n_max = 9
    num_points = 10
    num_rotations = 10

    pts = generate.random_points_in_sphere(10, radius=1.)
    weights = np.random.rand(num_points)

    # Create the invariants
    generator = invertible_invariants.InvariantsGenerator()
    invs = generator.generate_all(n_max)

    phi0 = invs(zernike.from_deltas(n_max, pts, weights))
    for _ in range(num_rotations):
        rot = Rotation.random()
        rotated = rot.apply(pts)

        phi = invs(zernike.from_deltas(n_max, rotated, weights))

        assert np.allclose(phi0, phi)
