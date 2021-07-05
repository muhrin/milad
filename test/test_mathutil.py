# -*- coding: utf-8 -*-
import numpy as np

from milad import mathutil

# pylint: disable=invalid-name


def test_spherical_cart():
    """Test conversion to and from cartesian/spherical coords works correctly"""
    x = np.random.rand(3)
    r, theta, phi = mathutil.cart2sph(x)
    xprime = np.array(mathutil.sph2cart([r, theta, phi]))
    assert np.allclose(x, xprime)
