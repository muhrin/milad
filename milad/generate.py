# -*- coding: utf-8 -*-
from typing import Tuple

import numpy as np
from scipy.spatial import transform

from . import mathutil


def random_points_in_sphere(
    num: int, radius=1.0, centre=False, minsep=None
) -> np.array:
    """Generate num points within a sphere of the given radius"""
    points = np.empty((num, 3))
    r_sq = radius * radius

    for idx in range(num):
        vec = 2 * radius * (np.random.rand(3) - 0.5)
        while np.dot(vec, vec) >= r_sq:
            vec = 2 * radius * (np.random.rand(3) - 0.5)
        points[idx] = vec

    if minsep is not None:
        pass

    if centre:
        points -= centeroid(points)
        # Make sure that there aren't any outside of the radius now
        radius_sq = (points**2).sum(1).max()
        if radius_sq > r_sq:
            points *= r_sq / radius_sq

    return points


def centeroid(pts: np.ndarray) -> np.ndarray:
    length = len(pts)
    centroid = np.empty((length, 3))
    centroid[:, 0] = np.sum(pts[:, 0]) / length
    centroid[:, 1] = np.sum(pts[:, 1]) / length
    centroid[:, 2] = np.sum(pts[:, 2]) / length
    return centroid


def chiral_tetrahedra() -> Tuple[np.ndarray, np.ndarray]:
    """Create chiral structures that cannot be distinguished by bispectrum from
    https://link.aps.org/doi/10.1103/PhysRevLett.125.166001
    """
    # pylint: disable=invalid-name
    a = np.radians(0)
    b = np.radians(-87)
    c = np.radians(176)

    # First group of points
    b_i = np.zeros((3, 3))
    r = 0.3
    b_i[0, :] = mathutil.sph2cart([r, a, 0.0])
    b_i[1, :] = mathutil.sph2cart([r, b, 0.0])
    b_i[2, :] = mathutil.sph2cart([r, c, 0.0])

    # Second group of points
    rot = transform.Rotation.from_euler("y", 86, degrees=True)
    b_i_prime = rot.apply(b_i)

    pts_dist = 0.4
    b_i[:, 1] += pts_dist
    b_i_prime[:, 1] -= pts_dist

    pt_dist = 0.8

    plus = np.concatenate((np.array([[0, 0, 0], [0, +pt_dist, 0.0]]), b_i, b_i_prime))
    minus = np.concatenate((np.array([[0, 0, 0], [0, -pt_dist, 0.0]]), b_i, b_i_prime))

    return minus, plus
