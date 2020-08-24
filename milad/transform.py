# -*- coding: utf-8 -*-
import numpy
from scipy.spatial.transform import Rotation


def center(points: numpy.array) -> numpy.array:
    """Centre the given points by shifting the centre of mass to zero"""
    return points - points.sum(axis=0) / len(points)


def randomly_rotate(points: numpy.array) -> numpy.array:
    """Randomly rotate a set of points"""
    rot = Rotation.random()
    return rot.apply(points)
