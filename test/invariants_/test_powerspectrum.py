# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.transform import Rotation

from milad import generate
from milad.invariants import powerspectrum
from milad import zernike

# pylint: disable=invalid-name


def invariants_test(invariants_fn):
    pts = generate.random_points_in_sphere(11)

    phi = invariants_fn(pts)

    for _ in range(10):
        random = Rotation.random()
        rotated = random.apply(pts)

        phi_prime = invariants_fn(rotated)

        assert np.allclose(phi, phi_prime)


def test_powerspectrum_basic():
    ps = powerspectrum.PowerSpectrum()

    def invariants_fn(pts):
        moments = zernike.from_deltas(n_max=7, positions=pts)
        return ps(moments)

    invariants_test(invariants_fn)
