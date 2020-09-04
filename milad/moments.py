# -*- coding: utf-8 -*-
import numbers
from typing import Union

import numpy

__all__ = 'gaussian_geometric_moments', 'geometric_moments_of_gaussians', \
          'geometric_moments_of_deltas'


def gaussian_moments(
    order: int,
    mu: Union[numbers.Number, numpy.array],
    sigmas: Union[numbers.Number, numpy.array] = 0.4,
    weight: numbers.Number = 1.
) -> numbers.Number:
    """Get the nt^h moment of a n-dim Gaussian (or normal distribution) centred at `mu`
    with a standard deviation of `sigma`.

    Taken from:
    https://en.wikipedia.org/wiki/Normal_distribution#Moments
    Can be generalised to any order using confluent hypergeometric functions of the second kind.

    Another useful reference is:
    http://www.randomservices.org/random/special/Normal.html

    :param mu: the mean of the distribution
    :param sigmas: the standard deviation of the distribution
    :param order: the order of the moment to get
    :param weight: the total probability or mass of the normal distribution.  This is the zero^th
        moment be definition
    """
    if order > 16:
        raise NotImplementedError("Asked for order '{}', only up to order 16 implemented!".format(order))

    mu = numpy.array(mu)
    shape = mu.shape[0]
    sigmas = _to_array(sigmas, shape)

    if order == 0:
        mom = 1.
    if order == 1:
        mom = mu
    if order == 2:
        mom = mu ** 2 + \
              sigmas ** 2
    if order == 3:
        mom = mu ** 3 + \
              3 * mu * sigmas ** 2
    if order == 4:
        mom = mu ** 4 + \
              6 * mu ** 2 * sigmas ** 2 + \
              3 * sigmas ** 4
    if order == 5:
        mom = mu ** 5 + \
              10 * mu ** 3 * sigmas ** 2 + \
              5 * mu * 3 * sigmas ** 4
    if order == 6:
        mom = mu ** 6 + \
              15 * mu ** 4 * sigmas ** 2 + \
              15 * mu ** 2 * 3 * sigmas ** 4 + \
              15 * sigmas ** 6
    if order == 7:
        mom = mu ** 7 + \
              21 * mu ** 5 * sigmas ** 2 + \
              35 * mu ** 3 * 3 * sigmas ** 4 + \
              7 * mu * 15 * sigmas ** 6
    if order == 8:
        mom = mu ** 8 + \
              28 * mu ** 6 * sigmas ** 2 + \
              70 * mu ** 4 * 3 * sigmas ** 4 + \
              28 * mu ** 2 * 15 * sigmas ** 6 + \
              105 * sigmas ** 8
    if order == 9:
        mom = mu ** 9 + \
              36 * mu ** 7 * sigmas ** 2 + \
              126 * mu ** 5 * 3 * sigmas ** 4 + \
              84 * mu ** 3 * 15 * sigmas ** 6 + \
              9 * mu * 105 * sigmas ** 8
    if order == 10:
        mom = mu ** 10 + \
              45 * mu ** 8 * sigmas ** 2 + \
              210 * mu ** 6 * 3 * sigmas ** 4 + \
              210 * mu ** 4 * 15 * sigmas ** 6 + \
              45 * mu ** 2 * 105 * mu * sigmas ** 8 + \
              945 * sigmas ** 10
    if order == 11:
        mom = mu ** 11 + \
              55 * mu ** 9 * sigmas ** 2 + \
              330 * mu ** 7 * 3 * sigmas ** 4 + \
              462 * mu ** 5 * 15 * sigmas ** 6 + \
              165 * mu ** 3 * 105 * sigmas ** 8 + \
              11 * mu * sigmas ** 10
    if order == 12:
        mom = mu ** 12 + \
              66 * mu ** 10 * sigmas ** 2 + \
              495 * mu ** 8 * 3 * sigmas ** 4 + \
              924 * mu ** 6 * 15 * sigmas ** 6 + \
              495 * mu ** 4 * 105 * sigmas ** 8 + \
              66 * mu ** 2 * 945 * sigmas ** 10 + \
              10395 * sigmas ** 12
    if order == 13:
        mom = mu ** 13 + \
              78 * mu ** 11 * sigmas ** 2 + \
              715 * mu ** 9 * 3 * sigmas ** 4 + \
              1716 * mu ** 7 * 15 * sigmas ** 6 + \
              1287 * mu ** 5 * 105 * sigmas ** 8 + \
              286 * mu ** 3 * 945 * sigmas ** 10 + \
              13 * mu * 10395 * sigmas ** 12
    if order == 14:
        mom = mu ** 14 + \
              91 * mu ** 12 * sigmas ** 2 + \
              1001 * mu ** 10 * 3 * sigmas ** 4 + \
              3003 * mu ** 8 * 15 * sigmas ** 6 + \
              3003 * mu ** 6 * 105 * sigmas ** 8 + \
              1001 * mu ** 4 * 945 * sigmas ** 10 + \
              91 * mu ** 2 * 10395 * sigmas ** 12 + \
              135135 * sigmas ** 14
    if order == 15:
        mom = mu ** 15 + \
              105 * mu ** 13 * sigmas ** 2 + \
              1365 * mu ** 11 * 3 * sigmas ** 4 + \
              5005 * mu ** 9 * 15 * sigmas ** 6 + \
              6435 * mu ** 7 * 105 * sigmas ** 8 + \
              3003 * mu ** 5 * 945 * sigmas ** 10 + \
              455 * mu ** 3 * 10395 * sigmas ** 12 + \
              15 * mu * 135135 * sigmas ** 14
    if order == 16:
        mom = mu ** 16 + \
              120 * mu ** 14 * sigmas ** 2 + \
              1820 * mu ** 12 * 3 * sigmas ** 4 + \
              8008 * mu ** 10 * 15 * sigmas ** 6 + \
              12870 * mu ** 8 * 105 * sigmas ** 8 + \
              8008 * mu ** 6 * 945 * sigmas ** 10 + \
              1820 * mu ** 4 * 10395 * sigmas ** 12 + \
              120 * mu ** 2 * 135135 * sigmas ** 14 + \
              2027025 * sigmas ** 16

    return weight * mom


def gaussian_geometric_moments(
    max_order: int, mu: numpy.array, sigma: numbers.Number, weight: numbers.Number
) -> numpy.array:
    """
    Get the geometric moments for a 3D Gaussian

    :param mu: the position of the Gaussian distribution
    :param sigma: the standard deviation (scalar - same in all directions)
    :param max_order: the maximum order to calculate moments for
    :param weight: the total mass of the Gaussian (or equivalently total probability)
    :return: the 3d moment tensor
    """
    ubound = max_order + 1  # Calculate to max order (inclusive)

    moments = numpy.zeros((3, ubound))
    moments[:, 0] = 1.0  # 0^th order, mass is multiplied in at end

    for order in range(1, ubound):
        # Use weight 1 for now and then multiply later
        moments[:, order] = gaussian_moments(order, mu, sigma, weight=1.0)

    moments_3d = numpy.empty((ubound,) * 3)
    # pylint: disable=invalid-name

    for p in range(ubound):
        for q in range(ubound):
            for r in range(ubound):
                moments_3d[p, q, r] = moments[0, p] * moments[1, q] * moments[2, r]

    moments_3d *= weight  # todo: why are the weights here?  Shouldn't they be in 'moments'?
    return moments_3d


def geometric_moments_of_gaussians(
    max_order: int,
    positions: numpy.ndarray,
    sigmas: Union[numbers.Number, numpy.array] = 0.4,
    weights: Union[numbers.Number, numpy.array] = 1.
) -> numpy.ndarray:
    """Calculate the geometric moments for a collection of Gaussians at the given positions with
    the passed parameters.

    :param positions: the positions of the Gaussians
    :param sigmas: the standard deviations
    :param max_order: the maximum order to calculate moments up to
    :param weights: the masses of the Gaussians (or probabilities)
    :return: a max_order * max_order * max_order array of moments
    """
    positions = numpy.array(positions)
    shape = positions.shape[0]
    sigmas = _to_array(sigmas, shape)
    weights = _to_array(weights, shape)

    moments = numpy.zeros((max_order + 1, max_order + 1, max_order + 1))
    for pos, sigma, weight in zip(positions, sigmas, weights):
        moments += gaussian_geometric_moments(max_order, pos, sigma, weight)

    return moments


def geometric_moments_of_deltas(
    max_order: int, positions: numpy.array, weights: Union[numbers.Number, numpy.array] = 1.
) -> numpy.array:
    """
    Calculate the geometric moments for a collection of delta functions at the given positions with
    the given weights.

    :param max_order: the maximum order to calculate moments up to
    :param positions: the positions of the delta functions
    :param weights: the weights of the delta functions
    :return: a max_order * max_order * max_order array of moments
    """
    positions = numpy.array(positions)
    shape = positions.shape[0]
    weights = _to_array(weights, shape)
    ubound = max_order + 1  # Calculate to max order (inclusive)

    # pylint: disable=invalid-name

    moments = numpy.zeros((ubound, ubound, ubound))
    for pos, weight in zip(positions, weights):
        moms = numpy.empty((ubound, 3))
        moms[0] = 1.

        for power in range(1, ubound):
            moms[power] = numpy.multiply(moms[power - 1, :], pos)

        moments += weight * numpy.tensordot(
            numpy.tensordot(moms[:, 0], moms[:, 1], axes=0), moms[:, 2], axes=0)

    return moments



def value_at(geom_moments, x: numpy.array, n_max: int = None) -> float:

    shape = geom_moments.shape
    value = 0.
    for p in range(shape[0]):
        for q in range(shape[1]):
            for r in range(shape[2]):
                value += geom_moments[p, q, r] * (x ** (p, q, r)).prod(axis=-1)
    return value


def _to_array(value: Union[numbers.Number, numpy.array], shape):
    if isinstance(value, numbers.Number):
        sarray = numpy.empty(shape)
        sarray.fill(value)
        return sarray

    return numpy.array(value)
