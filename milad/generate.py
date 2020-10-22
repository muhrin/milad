# -*- coding: utf-8 -*-
import numpy as np


def random_points_in_sphere(num: int, radius=1., centre=False) -> np.array:
    """Generate num points within a sphere of the given radius"""
    points = np.empty((num, 3))
    r_sq = radius * radius

    for idx in range(num):
        vec = 2 * radius * (np.random.rand(1, 3) - 0.5)
        while np.linalg.norm(vec) > r_sq:
            vec = 2 * radius * (np.random.rand(1, 3) - 0.5)
        points[idx] = vec

    if centre:
        points -= centeroid(points)

    return points


def centeroid(pts: np.ndarray) -> np.ndarray:
    length = len(pts)
    centroid = np.empty((length, 3))
    centroid[:, 0] = np.sum(pts[:, 0]) / length
    centroid[:, 1] = np.sum(pts[:, 1]) / length
    centroid[:, 2] = np.sum(pts[:, 2]) / length
    return centroid
