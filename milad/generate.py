# -*- coding: utf-8 -*-
import numpy


def random_points_in_sphere(num: int, radius=1.) -> numpy.array:
    """Generate num points within a sphere of the given radius"""
    points = numpy.empty((num, 3))
    r_sq = radius * radius

    for idx in range(num):
        vec = 2 * radius * (numpy.random.rand(1, 3) - 0.5)
        while numpy.linalg.norm(vec) > r_sq:
            vec = 2 * radius * (numpy.random.rand(1, 3) - 0.5)
        points[idx] = vec

    return points
